from __future__ import annotations

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

from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

logger = init_logger(__name__)


class Qwen3TTSCode2Wav(nn.Module):
    """Stage-1 code2wav model for Qwen3-TTS (GenerationModelRunner).
    Consumes frame-aligned codec tokens from input_ids and decodes waveform via SpeechTokenizer."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        # Generation-only stage (no logits / sampling).
        self.requires_raw_input_tokens = True

        self._speech_tokenizer: Qwen3TTSTokenizer | None = None
        self._num_quantizers: int | None = None
        self._decode_upsample_rate: int | None = None
        self._output_sample_rate: int | None = None

        # Default streaming window (must match connector config by convention).
        self._stream_chunk_frames = 25
        self._stream_left_context_frames = 25
        self._logged_codec_stats = False

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
        cfg_path = cached_file(self.model_path, "speech_tokenizer/config.json")
        if cfg_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/config.json not found")
        speech_tokenizer_dir = os.path.dirname(cfg_path)

        # Stage-1 only needs decode; skip HF feature extractor to avoid heavy optional deps.
        # Still require preprocessor_config.json (use cached_file so online runs can fetch it).
        prep_cfg = cached_file(self.model_path, "speech_tokenizer/preprocessor_config.json")
        if prep_cfg is None:
            raise ValueError(
                f"{self.model_path}/speech_tokenizer/preprocessor_config.json not found. "
                "Please make sure the checkpoint contains the required HF preprocessing files."
            )

        tok = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            torch_dtype=torch.bfloat16,
            load_feature_extractor=False,
        )

        # Align device with vLLM worker, then read back from module.
        if tok.model is not None:
            tok.model.to(device=self.vllm_config.device_config.device)
            tok.device = self._module_device(tok.model)

        # Derive codec group count and rates from tokenizer config.
        dec_cfg = getattr(tok.model.config, "decoder_config", None)
        num_q = getattr(dec_cfg, "num_quantizers", None) if dec_cfg is not None else None
        if num_q is None:
            raise ValueError("speech_tokenizer decoder_config.num_quantizers not found")
        num_q = int(num_q)
        if num_q <= 0:
            raise ValueError(f"Invalid speech_tokenizer num_quantizers={num_q}")

        try:
            upsample = int(tok.get_decode_upsample_rate())
        except Exception as e:
            raise ValueError(f"Failed to get decode upsample rate: {e}") from e
        if upsample <= 0:
            raise ValueError(f"Invalid decode upsample rate: {upsample}")

        try:
            out_sr = int(tok.get_output_sample_rate())
        except Exception as e:
            raise ValueError(f"Failed to get output sample rate: {e}") from e

        self._speech_tokenizer = tok
        self._num_quantizers = num_q
        self._decode_upsample_rate = upsample
        self._output_sample_rate = out_sr
        return tok

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings. Keep a stable dummy embedding for vLLM runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    @staticmethod
    def _reconstruct_window_codes_fq(
        *,
        chunk_ids: torch.Tensor,
        q: int,
        chunk_frames: int,
        codec_streaming: bool,
        ctx_frames: int,
        ctx_codes: list[int] | None,
    ) -> torch.Tensor:
        """Reconstruct [F, Q] codes from codebook-major flattened chunk ids (and optional left-context)."""
        if q <= 0:
            raise ValueError(f"Invalid q={q} (must be >0).")
        if chunk_frames <= 0:
            raise ValueError(f"Invalid chunk_frames={chunk_frames} (must be >0).")

        if int(chunk_ids.numel()) != int(q) * int(chunk_frames):
            raise ValueError(
                "Invalid chunk_ids length for Qwen3TTSCode2Wav: "
                f"got={int(chunk_ids.numel())} expected={int(q) * int(chunk_frames)} "
                f"(q={q} chunk_frames={chunk_frames})."
            )

        chunk_qf = chunk_ids.reshape(int(q), int(chunk_frames))
        if codec_streaming and ctx_frames > 0:
            if ctx_codes is None:
                raise ValueError("Missing ctx_codes for streaming decode window reconstruction.")
            expected_ctx_tokens = int(q) * int(ctx_frames)
            if len(ctx_codes) != expected_ctx_tokens:
                raise ValueError(
                    "Invalid ctx_codes length for streaming decode window reconstruction: "
                    f"got={len(ctx_codes)} expected={expected_ctx_tokens} (q={q} ctx_frames={ctx_frames})."
                )
            ctx_tensor = torch.tensor(ctx_codes, dtype=torch.long, device=chunk_ids.device)
            ctx_qf = ctx_tensor.reshape(int(q), int(ctx_frames))
            window_qf = torch.cat([ctx_qf, chunk_qf], dim=1)
        else:
            window_qf = chunk_qf

        return window_qf.transpose(0, 1).contiguous()  # [F, Q]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ModelOutput is (audio_tensor, sr_tensor).
        tok = self._ensure_speech_tokenizer_loaded()
        assert self._num_quantizers is not None
        assert self._output_sample_rate is not None

        if input_ids is None:
            # Profile run / placeholder schedule: return empty audio.
            empty = torch.zeros((0,), dtype=torch.float32)
            return empty, torch.tensor(self._output_sample_rate, dtype=torch.int32)

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        q = int(self._num_quantizers)

        if ids.numel() == 0 or ids.numel() < q:
            empty = torch.zeros((0,), dtype=torch.float32)
            return empty, torch.tensor(self._output_sample_rate, dtype=torch.int32)

        # Contract: connector provides codec_streaming + codec_context_frames (left-context frames to trim).
        # Assumes max_batch_size=1 for code2wav (vLLM provides a flattened per-step token stream).
        ctx_frames: int | None = None
        codec_streaming: bool | None = None
        ctx_codes: list[int] | None = None
        chunk_frames: int | None = None
        rt_info = kwargs.get("runtime_additional_information")
        if isinstance(rt_info, list) and len(rt_info) == 1 and isinstance(rt_info[0], dict):
            v = rt_info[0].get("codec_streaming")
            if v is not None:
                try:
                    codec_streaming = bool(v) if not isinstance(v, torch.Tensor) else bool(v.item())
                except Exception:
                    codec_streaming = None
            v = rt_info[0].get("codec_context_frames")
            if v is not None:
                try:
                    ctx_frames = int(v)
                except Exception as e:
                    raise ValueError(f"Invalid codec_context_frames={v!r}: {e}") from e
            v = rt_info[0].get("codec_context_codes")
            if v is not None:
                if isinstance(v, list):
                    ctx_codes = [int(x) for x in v]
                elif isinstance(v, torch.Tensor):
                    ctx_codes = v.detach().to("cpu").reshape(-1).to(dtype=torch.long).tolist()
            v = rt_info[0].get("codec_chunk_frames")
            if v is not None:
                try:
                    chunk_frames = int(v)
                except Exception as e:
                    raise ValueError(f"Invalid codec_chunk_frames={v!r}: {e}") from e

        if codec_streaming is None:
            raise ValueError(
                "Missing codec_streaming in runtime_additional_information for Qwen3TTSCode2Wav. "
                "This indicates the async_chunk connector/adapter contract was not applied."
            )

        if codec_streaming is False:
            ctx_frames = 0
        else:
            if ctx_frames is None:
                raise ValueError(
                    "Missing codec_context_frames in runtime_additional_information for streaming Qwen3TTSCode2Wav. "
                    "This indicates the async_chunk connector/adapter contract was not applied."
                )
            if ctx_frames < 0:
                raise ValueError(f"Invalid codec_context_frames={ctx_frames} (must be >=0).")

        # input_ids may be padded; use codec_chunk_frames to slice the exact chunk (chunk_frames * q) and ignore padding.
        if chunk_frames is None:
            raise ValueError(
                "Missing codec_chunk_frames in runtime_additional_information for Qwen3TTSCode2Wav. "
                "This indicates the async_chunk connector/adapter contract was not applied."
            )
        if chunk_frames < 0:
            raise ValueError(f"Invalid codec_chunk_frames={chunk_frames} (must be >=0).")
        expected_chunk_tokens = int(chunk_frames) * q
        if expected_chunk_tokens == 0:
            empty = torch.zeros((0,), dtype=torch.float32)
            return empty, torch.tensor(self._output_sample_rate, dtype=torch.int32)
        if ids.numel() < expected_chunk_tokens:
            raise ValueError(
                "Code2Wav received fewer tokens than expected for this chunk: "
                f"got={int(ids.numel())} expected={expected_chunk_tokens} "
                f"(chunk_frames={int(chunk_frames)} q={q}). "
                "This indicates vLLM split the chunk across multiple forward calls; "
                "the code2wav stage requires per-step frame-aligned chunks."
            )
        if ids.numel() > expected_chunk_tokens:
            # Extra non-padding tokens beyond expected_chunk_tokens indicate a scheduler/adapter contract violation.
            extra = ids[expected_chunk_tokens:]
            if extra.numel() > 0 and bool((extra != 0).any().item()):
                raise ValueError(
                    "Code2Wav received extra non-padding tokens beyond the expected chunk length: "
                    f"got={int(ids.numel())} expected={expected_chunk_tokens} "
                    f"(chunk_frames={int(chunk_frames)} q={q}). "
                    "This indicates multiple codec chunks were scheduled in a single forward, "
                    "which breaks streaming trim/paste semantics."
                )
            ids = ids[:expected_chunk_tokens]

        chunk_ids = ids
        ctx_frames_i = int(ctx_frames or 0)
        frames = int((ctx_frames_i if codec_streaming else 0) + int(chunk_frames))
        codes_fq = self._reconstruct_window_codes_fq(
            chunk_ids=chunk_ids,
            q=q,
            chunk_frames=int(chunk_frames),
            codec_streaming=bool(codec_streaming),
            ctx_frames=ctx_frames_i,
            ctx_codes=ctx_codes,
        )
        if not self._logged_codec_stats and frames > 1:
            self._logged_codec_stats = True
            try:
                uniq = int(torch.unique(codes_fq).numel())
                cmin = int(codes_fq.min().item())
                cmax = int(codes_fq.max().item())
                head = codes_fq[: min(2, frames), : min(8, q)].detach().to("cpu").tolist()
                logger.info(
                    "Qwen3TTSCode2Wav received codec codes: frames=%d q=%d uniq=%d range=[%d,%d] head=%s",
                    frames,
                    q,
                    uniq,
                    cmin,
                    cmax,
                    head,
                )
            except Exception:
                pass

        wavs, sr = tok.decode({"audio_codes": codes_fq})
        if not wavs:
            raise ValueError("SpeechTokenizer code2wav produced empty waveform list.")
        audio_np = wavs[0].astype(np.float32, copy=False)

        if ctx_frames > 0:
            # Trim waveform samples corresponding to left-context frames in the sliding window.
            upsample = self._decode_upsample_rate
            if upsample is None:
                try:
                    upsample = int(tok.get_decode_upsample_rate())
                except Exception as e:
                    raise ValueError(f"Failed to get decode upsample rate: {e}") from e
                if upsample <= 0:
                    raise ValueError(f"Invalid decode upsample rate: {upsample}")
                self._decode_upsample_rate = upsample

            ctx_frames_i = int(ctx_frames)
            if ctx_frames_i > frames:
                raise ValueError(f"codec_context_frames={ctx_frames_i} exceeds frames={frames}")

            decoded = int(audio_np.shape[0])
            cut = int(ctx_frames_i) * int(upsample)
            if cut > decoded:
                raise ValueError(
                    "Streaming decode context trim exceeds decoded length: "
                    f"cut={cut} decoded={decoded} ctx_frames={ctx_frames_i} frames={frames}"
                )
            audio_np = audio_np[cut:]

        # Return 1D waveform per chunk so the output processor can concatenate along time.
        # Returning [1, T] would stack chunks as channels.
        audio_tensor = torch.from_numpy(audio_np).to(dtype=torch.float32).reshape(-1)
        sr_tensor = torch.tensor(int(sr), dtype=torch.int32)
        return audio_tensor, sr_tensor

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"Qwen3TTSCode2Wav expected (audio_tensor, sr) outputs, got {type(model_outputs)}")

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SpeechTokenizer weights live under `speech_tokenizer/` and are loaded
        # lazily from that directory. Ignore main checkpoint weights.
        return set()
