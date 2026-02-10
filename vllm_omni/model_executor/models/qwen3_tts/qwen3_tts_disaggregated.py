import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .qwen3_tts import Qwen3TTSModel
from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

logger = init_logger(__name__)

_VALID_TASK_TYPES = ("CustomVoice", "VoiceDesign", "Base")
_VALID_STAGES = ("talker", "speech_tokenizer")


class Qwen3TTSForConditionalGenerationDisaggregatedVLLM(nn.Module):
    """Stage-aware wrapper for disaggregated Qwen3-TTS (selects stage via model_stage).
    SpeechTokenizer stage decodes codec->waveform; talker is handled by the AR talker model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.model_stage = getattr(vllm_config.model_config, "model_stage", None)
        self._async_chunk = bool(getattr(vllm_config.model_config, "async_chunk", False))

        if self.model_stage not in _VALID_STAGES:
            raise ValueError(f"Invalid model_stage for Qwen3-TTS disaggregated model: {self.model_stage}")

        if self.model_stage == "talker":
            # Avoid accidental fallback to the HF generate() path.
            raise ValueError(
                "Qwen3-TTS disaggregated wrapper no longer supports model_stage='talker'. "
                "Use model_arch=Qwen3TTSTalkerForConditionalGenerationARVLLM for Stage-0."
            )

        self.have_multimodal_outputs = True
        # Only speech_tokenizer needs preprocess in async_chunk (treat prompt_token_ids as codec codes).
        self.has_preprocess = bool(self.model_stage == "speech_tokenizer" and self._async_chunk)
        if self.model_stage == "speech_tokenizer" and not self._async_chunk:
            raise ValueError(
                "Qwen3-TTS SpeechTokenizer stage no longer supports serial "
                "`additional_information['audio_codes']` mode. Use async_chunk "
                "stage config so Stage-1 consumes codec codes via prompt_token_ids."
            )

        self._talker: Qwen3TTSModel | None = None
        self._speech_tokenizer: Qwen3TTSTokenizer | None = None
        # Only required for Stage-1 streaming decode (to reframe flattened codes).
        self._num_code_groups = 0
        if self.model_stage == "speech_tokenizer":
            try:
                self._num_code_groups = int(vllm_config.model_config.hf_config.talker_config.num_code_groups)
            except Exception as e:
                raise ValueError(f"Failed to read talker_config.num_code_groups from hf_config: {e}") from e
            if self._num_code_groups <= 0:
                raise ValueError(f"Invalid num_code_groups={self._num_code_groups} for Qwen3-TTS.")

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def _ensure_speech_tokenizer_loaded(self) -> Qwen3TTSTokenizer:
        if self._speech_tokenizer is not None:
            return self._speech_tokenizer

        # Locate speech_tokenizer dir from HF cache (or local path).
        speech_tokenizer_path = cached_file(self.model_path, "speech_tokenizer/config.json")
        if speech_tokenizer_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/config.json not found")
        speech_tokenizer_dir = os.path.dirname(speech_tokenizer_path)
        self._speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            torch_dtype=torch.bfloat16,
            load_feature_extractor=False,
        )
        # Run decode on the vLLM worker device, then read back from module.
        if self._speech_tokenizer.model is not None:
            self._speech_tokenizer.model.to(device=self.vllm_config.device_config.device)
            self._speech_tokenizer.device = self._module_device(self._speech_tokenizer.model)
        return self._speech_tokenizer

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Only used in async_chunk speech_tokenizer stage.
        if self.model_stage != "speech_tokenizer" or not self._async_chunk:
            return input_ids, (input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)), {}

        if self._num_code_groups <= 0:
            raise ValueError(f"Invalid talker_config.num_code_groups={self._num_code_groups} for streaming decode.")

        # Optional request id for debugging only (streaming decode keeps no per-request state).
        req_id = str(info_dict.get("_omni_request_id") or "")

        q = int(self._num_code_groups)
        if input_ids.numel() <= 0:
            update = {"model_outputs": None, "sr": None}
            return input_ids, (input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)), update

        tokens = input_ids.reshape(-1).to(torch.long)
        if int(tokens.numel()) % q != 0:
            # Finished requests may still get placeholder tokens; treat as a no-op instead of crashing.
            if bool(info_dict.get("finished", False)) or int(tokens.numel()) <= 1:
                update = {"model_outputs": None, "sr": None}
                return (
                    input_ids,
                    (input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)),
                    update,
                )
            raise ValueError(
                f"Streaming codec token length must be divisible by num_code_groups={q}. "
                f"got={int(tokens.numel())} request_id={req_id or '<unknown>'}"
            )

        frames = int(tokens.numel()) // q
        if frames <= 0:
            update = {"model_outputs": None, "sr": None}
            return input_ids, (input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)), update

        # tokens are codebook-major flattened: [Q, F] flattened row-major.
        codes_qf = tokens.reshape(q, frames)
        codes_fq = codes_qf.transpose(0, 1).contiguous()  # [F, Q]

        ctx_frames = int(info_dict.get("codec_context_frames") or 0)
        if ctx_frames < 0 or ctx_frames > frames:
            raise ValueError(
                f"Invalid codec_context_frames={ctx_frames} for frames={frames} request_id={req_id or '<unknown>'}"
            )

        tok = self._ensure_speech_tokenizer_loaded()
        device = getattr(tok, "device", None) or torch.device("cpu")
        codes_chunk = codes_fq.to(device=device)

        wavs, sr = tok.decode({"audio_codes": codes_chunk})
        if not wavs:
            raise ValueError("SpeechTokenizer streaming decode produced empty waveform list.")
        audio_np = wavs[0].astype(np.float32, copy=False)

        if ctx_frames > 0:
            try:
                upsample = int(tok.get_decode_upsample_rate())
            except Exception as e:
                raise ValueError(f"Failed to get decode upsample rate for streaming trim: {e}") from e
            if upsample <= 0:
                raise ValueError(f"Invalid decode upsample rate: {upsample}")
            cut = ctx_frames * upsample
            if cut >= audio_np.shape[0]:
                raise ValueError(
                    f"Streaming decode context trim exceeds decoded length: cut={cut} decoded={audio_np.shape[0]}"
                )
            audio_np = audio_np[cut:]

        update: dict[str, Any] = {
            "model_outputs": torch.from_numpy(audio_np).to(dtype=torch.float32),
            "sr": torch.tensor(int(sr), dtype=torch.int),
        }
        return input_ids, (input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)), update

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        runtime_info = kwargs.get("runtime_additional_information", [{}])
        if isinstance(runtime_info, list) and runtime_info:
            runtime_info = runtime_info[0]
        if not isinstance(runtime_info, dict):
            runtime_info = {}

        # speech_tokenizer stage: decode in preprocess(); forward returns a dummy tensor for span slicing.
        device = input_ids.device if isinstance(input_ids, torch.Tensor) else torch.device("cpu")
        n = int(input_ids.shape[0]) if isinstance(input_ids, torch.Tensor) else 1
        if n <= 0:
            n = 1
        return torch.zeros((n, 1), dtype=torch.float32, device=device)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        # async_chunk speech_tokenizer: emit the latest decoded chunk from runtime_additional_information.
        if self.model_stage != "speech_tokenizer" or not self._async_chunk:
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})

        runtime_info = kwargs.get("runtime_additional_information", [{}])
        if isinstance(runtime_info, list) and runtime_info:
            runtime_info = runtime_info[0]
        if not isinstance(runtime_info, dict):
            runtime_info = {}

        mo = runtime_info.get("model_outputs")
        sr = runtime_info.get("sr")
        if isinstance(mo, torch.Tensor) and isinstance(sr, torch.Tensor):
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={"model_outputs": mo, "sr": sr})
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        return None

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # SpeechTokenizer ignores token embeddings, but vLLM requires embed_input_ids to select the runner type.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Talker loads weights elsewhere; speech_tokenizer loads `speech_tokenizer/` lazily.
        # Return empty set without consuming weights to avoid vLLM re-loading.
        return set()
