"""Voice cache manager for persisting computed voice-clone embeddings.

Caches speaker embeddings and codec tokens to safetensors files in
/tmp/voice_clones/ so subsequent requests skip the expensive extraction.

Security: safetensors-only (no pickle/torch.load), path confinement.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.metadata_manager import MetadataManager

logger = init_logger(__name__)

VOICE_CLONES_DIR = os.environ.get("SPEECH_VOICE_SAMPLES", "/tmp/voice_clones")


@dataclass
class VoiceClonePromptItem:
    """Container for one sample's voice-clone prompt fed to the model.

    Fields align with Qwen3TTSForConditionalGeneration.generate(voice_clone_prompt=...).
    """

    ref_code: torch.Tensor | None  # (T, Q) or (T,)
    ref_spk_embedding: torch.Tensor  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


class VoiceCacheManager:
    """Manages on-disk voice clone caches (safetensors format).

    Reads speaker metadata from MetadataManager and persists pre-computed
    ref_code + speaker embeddings so they can be reloaded on subsequent
    requests without re-encoding the reference audio.
    """

    def __init__(
        self,
        speech_voice_samples_dir: str | None = None,
        metadata_manager: MetadataManager | None = None,
    ):
        self.speech_voice_samples_dir = speech_voice_samples_dir or VOICE_CLONES_DIR

        if metadata_manager is not None:
            self.metadata_manager = metadata_manager
        else:
            metadata_file = Path(self.speech_voice_samples_dir) / "metadata.json"
            self.metadata_manager = MetadataManager(metadata_file)

    def load_uploaded_speakers_from_metadata(self) -> dict[str, Any] | None:
        try:
            return self.metadata_manager.get_uploaded_speakers()
        except Exception as e:
            logger.warning("Failed to load uploaded speakers from metadata: %s", e)
            return None

    def update_metadata_cache_info(
        self, speaker: str, cache_file_path: Path, status: str = "ready", **extra: Any
    ) -> bool:
        try:
            return self.metadata_manager.update_cache_info(
                speaker_key=speaker.lower(),
                cache_file_path=cache_file_path,
                status=status,
                **extra,
            )
        except Exception as e:
            logger.error("Failed to update metadata cache info: %s", e)
            return False

    def save_voice_cache(
        self,
        speaker: str,
        audio_file_path: Path,
        prompt_items: list[VoiceClonePromptItem],
    ) -> bool:
        """Save voice cache using safetensors (no pickle, no RCE)."""
        try:
            cache_file_path = audio_file_path.with_suffix(".safetensors")

            tensors: dict[str, torch.Tensor] = {}
            metadata: dict[str, str] = {}

            tensors["__len__"] = torch.tensor(len(prompt_items), dtype=torch.int64)

            for i, item in enumerate(prompt_items):
                prefix = f"item_{i}_"
                tensors[prefix + "ref_spk_embedding"] = item.ref_spk_embedding.detach().cpu().contiguous()
                has_ref_code = item.ref_code is not None
                tensors[prefix + "has_ref_code"] = torch.tensor(int(has_ref_code), dtype=torch.int8)
                if has_ref_code:
                    tensors[prefix + "ref_code"] = item.ref_code.detach().cpu().contiguous()
                tensors[prefix + "x_vector_only_mode"] = torch.tensor(int(item.x_vector_only_mode), dtype=torch.int8)
                tensors[prefix + "icl_mode"] = torch.tensor(int(item.icl_mode), dtype=torch.int8)
                if item.ref_text is not None:
                    metadata[prefix + "ref_text"] = item.ref_text

            save_file(tensors, str(cache_file_path), metadata=metadata)

            extra: dict[str, int | None] = {}
            if prompt_items and prompt_items[0].ref_code is not None:
                extra["ref_code_len"] = int(prompt_items[0].ref_code.shape[0])

            return self.update_metadata_cache_info(
                speaker=speaker,
                cache_file_path=cache_file_path,
                status="ready",
                **extra,
            )
        except Exception as e:
            logger.error("Failed to save safetensors cache for speaker %s: %s", speaker, e)
            self.update_metadata_cache_info(speaker, Path(""), "failed")
            return False

    def load_cached_voice_prompt(
        self,
        speaker: str,
        device: str | None = None,
    ) -> list[VoiceClonePromptItem] | None:
        """Load cached VoiceClonePromptItem list from safetensors."""
        try:
            uploaded_speakers = self.load_uploaded_speakers_from_metadata()
            if not uploaded_speakers:
                return None

            speaker_key = speaker.lower()
            if speaker_key not in uploaded_speakers:
                return None

            speaker_info = uploaded_speakers[speaker_key]
            if speaker_info.get("cache_status") != "ready":
                return None

            cache_file_path = Path(speaker_info.get("cache_file", "")).resolve()
            base_dir = Path(self.speech_voice_samples_dir).resolve()

            if not str(cache_file_path).startswith(str(base_dir)):
                logger.error("Illegal cache path outside base dir: %s", cache_file_path)
                return None
            if not cache_file_path.exists():
                return None
            if cache_file_path.suffix != ".safetensors":
                logger.error("Legacy or unsafe cache format rejected: %s", cache_file_path)
                return None

            with safe_open(cache_file_path, framework="pt", device="cpu") as f:
                meta = f.metadata()
                num_items = int(f.get_tensor("__len__").item())
                result: list[VoiceClonePromptItem] = []

                for i in range(num_items):
                    prefix = f"item_{i}_"
                    has_ref_code = bool(f.get_tensor(prefix + "has_ref_code").item())
                    ref_code = f.get_tensor(prefix + "ref_code").to(device) if has_ref_code else None
                    ref_spk_embedding = f.get_tensor(prefix + "ref_spk_embedding").to(device)
                    x_vector_only_mode = bool(f.get_tensor(prefix + "x_vector_only_mode").item())
                    icl_mode = bool(f.get_tensor(prefix + "icl_mode").item())
                    ref_text = meta.get(prefix + "ref_text")

                    result.append(
                        VoiceClonePromptItem(
                            ref_code=ref_code,
                            ref_spk_embedding=ref_spk_embedding,
                            x_vector_only_mode=x_vector_only_mode,
                            icl_mode=icl_mode,
                            ref_text=ref_text,
                        )
                    )

            logger.info("Safetensors cache loaded for speaker: %s", speaker)
            return result
        except Exception as e:
            logger.warning("Failed to load safetensors cache for speaker %s: %s", speaker, e)
            return None

    def get_speaker_audio_path(self, speaker: str) -> Path | None:
        uploaded_speakers = self.load_uploaded_speakers_from_metadata()
        if not uploaded_speakers:
            return None

        speaker_key = speaker.lower()
        if speaker_key not in uploaded_speakers:
            return None

        audio_file_path = Path(uploaded_speakers[speaker_key]["file_path"])
        if audio_file_path.exists():
            return audio_file_path

        logger.warning("Audio file not found for speaker %s: %s", speaker, audio_file_path)
        return None
