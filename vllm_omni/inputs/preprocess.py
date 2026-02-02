from typing import Any

from typing_extensions import assert_never
from vllm.inputs.data import EmbedsInputs, SingletonInputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalInputs, MultiModalUUIDDict
from vllm.renderers.inputs import SingletonDictPrompt

from vllm_omni.inputs.data import (
    OmniEmbedsPrompt,
    OmniTextPrompt,
    OmniTokenInputs,
    OmniTokensPrompt,
    token_inputs_omni,
)

logger = init_logger(__name__)


class OmniInputPreprocessor(InputPreprocessor):
    """Input preprocessor for omni models (tokens/embeds/multimodal + additional_information)."""

    def _is_qwen3_tts_talker_ar(self) -> bool:
        archs = getattr(self.model_config, "architectures", None)
        return bool(archs) and "Qwen3TTSTalkerForConditionalGenerationARVLLM" in archs

    def _get_qwen3_tts_codec_pad_id(self) -> int:
        hf_config = getattr(self.model_config, "hf_config", None)
        talker_config = getattr(hf_config, "talker_config", None)
        pad = getattr(talker_config, "codec_pad_id", None)
        try:
            pad_id = int(pad)
        except Exception:
            pad_id = 0
        return max(0, pad_id)

    def _get_qwen3_tts_prompt_len_tokenizer(self):
        # Qwen3-TTS talker prompt length must match HF AutoTokenizer (fix_mistral_regex).
        tok = getattr(self, "_qwen3_tts_prompt_len_tokenizer", None)
        if tok is not None:
            return tok
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            self.model_config.model,
            trust_remote_code=True,
            fix_mistral_regex=True,
            use_fast=True,
        )
        tok.padding_side = "left"
        self._qwen3_tts_prompt_len_tokenizer = tok
        return tok

    def _estimate_qwen3_tts_talker_prompt_len(self, additional_information: dict[str, Any] | None) -> int:
        """Estimate Qwen3-TTS talker placeholder prompt length for vLLM scheduling.
        Real conditioning is carried in additional_information."""
        info = additional_information if isinstance(additional_information, dict) else {}

        def _first(x: object, default: object = "") -> object:
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        task_type = str(_first(info.get("task_type"), "CustomVoice") or "CustomVoice")
        hf_config = getattr(self.model_config, "hf_config", None)
        talker_config = getattr(hf_config, "talker_config", None)
        codec_language_id = getattr(talker_config, "codec_language_id", None)
        spk_is_dialect = getattr(talker_config, "spk_is_dialect", None)

        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker_ar import (
            Qwen3TTSTalkerForConditionalGenerationARVLLM,
        )

        tok = self._get_qwen3_tts_prompt_len_tokenizer()

        def _hf_tokenize_len(s: str) -> list[int]:
            return tok(s, padding=False)["input_ids"]

        return Qwen3TTSTalkerForConditionalGenerationARVLLM.estimate_prompt_len_from_additional_information(
            info,
            task_type=task_type,
            tokenize_prompt=_hf_tokenize_len,
            codec_language_id=codec_language_id,
            spk_is_dialect=spk_is_dialect,
            estimate_ref_code_len=None,
        )

    def _process_text(
        self,
        parsed_content: OmniTextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> OmniTokenInputs | MultiModalInputs:
        prompt_text = parsed_content["prompt"]

        inputs: OmniTokenInputs | MultiModalInputs
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            prompt_embeds = parsed_content.get("prompt_embeds")
            if prompt_embeds is not None:
                inputs["prompt_embeds"] = prompt_embeds
            additional_information = parsed_content.get("additional_information")
            if additional_information is not None:
                inputs["additional_information"] = additional_information
        else:
            if self._is_qwen3_tts_talker_ar():
                # Qwen3-TTS talker uses a small codec vocab; text token ids are OOV.
                # Use in-vocab pad placeholders for scheduling.
                additional_information = parsed_content.get("additional_information")
                prompt_len = self._estimate_qwen3_tts_talker_prompt_len(additional_information)
                pad_id = self._get_qwen3_tts_codec_pad_id()
                prompt_token_ids = [pad_id] * prompt_len
            else:
                prompt_token_ids = self._tokenize_prompt(
                    prompt_text,
                    tokenization_kwargs=tokenization_kwargs,
                )
            inputs = token_inputs_omni(
                prompt_token_ids,
                prompt_embeds=parsed_content.get("prompt_embeds"),
                additional_information=parsed_content.get("additional_information"),
            )

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_tokens(
        self,
        parsed_content: OmniTokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> OmniTokenInputs | MultiModalInputs:
        prompt_token_ids = self._truncate_inputs(parsed_content["prompt_token_ids"], tokenization_kwargs)
        prompt_embeds = parsed_content.get("prompt_embeds")
        additional_information = parsed_content.get("additional_information")

        inputs: OmniTokenInputs | MultiModalInputs
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )
            if prompt_embeds is not None:
                inputs["prompt_embeds"] = prompt_embeds
            if additional_information is not None:
                inputs["additional_information"] = additional_information
        else:
            inputs = token_inputs_omni(
                prompt_token_ids=prompt_token_ids,
                prompt_embeds=prompt_embeds,
                additional_information=additional_information,
            )

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_embeds(
        self,
        parsed_content: OmniEmbedsPrompt,
    ) -> EmbedsInputs:
        """Process embeddings prompt with omni-specific extensions.

        Extends base _process_embeds to handle additional_information payload
        for direct transfer between pipeline stages.
        """
        # Call parent implementation for base embeds processing
        inputs = super()._process_embeds(parsed_content)

        # Add omni-specific additional_information if present
        additional_information = parsed_content.get("additional_information")
        if additional_information is not None:
            inputs["additional_information"] = additional_information  # type: ignore[typeddict-unknown-key]

        return inputs

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> SingletonInputs:
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt

        Returns:

        * [`SingletonInputs`][vllm.inputs.data.SingletonInputs] instance
        """
        if "prompt_token_ids" in prompt:
            return self._process_tokens(
                prompt,  # type: ignore[arg-type]
                mm_uuids=mm_uuids,
            )

        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(prompt)  # type: ignore[arg-type]
