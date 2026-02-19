import asyncio
import base64
import io
import ipaddress
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import soundfile as sf
from fastapi import Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.metadata_manager import MetadataManager
from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    CreateAudio,
    OpenAICreateSpeechRequest,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_REF_AUDIO_TIMEOUT_S = 15
_REF_AUDIO_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_REF_AUDIO_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# TTS Configuration (currently supports Qwen3-TTS)
_TTS_MODEL_STAGES: set[str] = {"qwen3_tts"}
_TTS_LANGUAGES: set[str] = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Only allows alphanumeric characters, underscores, hyphens, and dots.
    Replaces any other characters with underscores.
    """
    filename = os.path.basename(filename)
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    if not sanitized:
        sanitized = "file"
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def _validate_path_within_directory(file_path: Path, directory: Path) -> bool:
    """Validate that file_path is within the specified directory.

    Prevents path traversal attacks by ensuring the resolved path
    is within the target directory.
    """
    try:
        file_path_resolved = file_path.resolve()
        directory_resolved = directory.resolve()
        return directory_resolved in file_path_resolved.parents or directory_resolved == file_path_resolved
    except Exception:
        return False


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize uploaded speakers storage
        speech_voice_samples_dir = os.environ.get("SPEECH_VOICE_SAMPLES", "/tmp/voice_samples")
        self.uploaded_speakers_dir = Path(speech_voice_samples_dir)
        self.uploaded_speakers_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.uploaded_speakers_dir / "metadata.json"

        # Initialize metadata manager
        self.metadata_manager = MetadataManager(self.metadata_file)

        # Load supported speakers
        self.supported_speakers = self._load_supported_speakers()
        # Load uploaded speakers (in-memory cache, updated via metadata_manager)
        self.uploaded_speakers = {}
        self._refresh_uploaded_speakers_cache()
        self.supported_speakers.update(self.uploaded_speakers.keys())

        logger.info(f"Loaded {len(self.supported_speakers)} supported speakers: {sorted(self.supported_speakers)}")
        logger.info(f"Loaded {len(self.uploaded_speakers)} uploaded speakers")
        self._tts_tokenizer = None

    def _refresh_uploaded_speakers_cache(self):
        """Refresh in-memory cache of uploaded speakers from metadata manager."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file) as f:
                    metadata = json.load(f)
                self.uploaded_speakers = metadata.get("uploaded_speakers", {})
        except Exception as e:
            logger.warning(f"Could not refresh uploaded speakers cache: {e}")
            self.uploaded_speakers = {}

    def _load_supported_speakers(self) -> set[str]:
        """Load supported speakers (case-insensitive) from the model configuration."""
        try:
            talker_config = self.engine_client.model_config.hf_config.talker_config

            # Check for speakers in either spk_id or speaker_id
            for attr_name in ["spk_id", "speaker_id"]:
                speakers_dict = getattr(talker_config, attr_name, None)
                if speakers_dict and isinstance(speakers_dict, dict):
                    # Normalize to lowercase for case-insensitive matching
                    return {speaker.lower() for speaker in speakers_dict.keys()}

            logger.warning("No speakers found in talker_config (checked spk_id and speaker_id)")
        except Exception as e:
            logger.warning(f"Could not load speakers from model config: {e}")

        return set()

    def _get_uploaded_audio_data(self, voice_name: str) -> str | None:
        """Get base64 encoded audio data for uploaded voice.

        Non-WAV formats (m4a, mp3, aac, etc.) are converted to WAV on the fly
        using torchaudio so that downstream code (soundfile) can always read them.
        """
        voice_name_lower = voice_name.lower()
        if voice_name_lower not in self.uploaded_speakers:
            return None

        speaker_info = self.uploaded_speakers[voice_name_lower]
        file_path = Path(speaker_info["file_path"])

        if not file_path.exists():
            logger.warning(f"Audio file not found for voice {voice_name}: {file_path}")
            return None

        try:
            mime_type = speaker_info.get("mime_type", "audio/wav")
            needs_conversion = mime_type not in ("audio/wav", "audio/x-wav", "audio/flac", "audio/ogg")

            if needs_conversion:
                from pydub import AudioSegment

                audio_seg = AudioSegment.from_file(str(file_path))
                buf = io.BytesIO()
                audio_seg.export(buf, format="wav")
                audio_bytes = buf.getvalue()
                mime_type = "audio/wav"
            else:
                with open(file_path, "rb") as f:
                    audio_bytes = f.read()

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            return f"data:{mime_type};base64,{audio_b64}"
        except Exception as e:
            logger.error(f"Could not read audio file for voice {voice_name}: {e}")
            return None

    async def upload_voice(
        self,
        audio_file: UploadFile,
        consent: str,
        name: str,
        ref_text: str | None = None,
    ) -> dict:
        """Upload a new voice sample."""
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        audio_file.file.seek(0, 2)
        file_size = audio_file.file.tell()
        audio_file.file.seek(0)

        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum limit of 10MB. Got {file_size} bytes.")

        # Detect MIME type from filename if content_type is generic
        mime_type = audio_file.content_type
        if mime_type == "application/octet-stream":
            filename_lower = audio_file.filename.lower()
            if filename_lower.endswith(".wav"):
                mime_type = "audio/wav"
            elif filename_lower.endswith((".mp3", ".mpeg")):
                mime_type = "audio/mpeg"
            elif filename_lower.endswith(".flac"):
                mime_type = "audio/flac"
            elif filename_lower.endswith(".ogg"):
                mime_type = "audio/ogg"
            elif filename_lower.endswith(".aac"):
                mime_type = "audio/aac"
            elif filename_lower.endswith(".webm"):
                mime_type = "audio/webm"
            elif filename_lower.endswith(".mp4"):
                mime_type = "audio/mp4"
            else:
                mime_type = "audio/wav"

        allowed_mime_types = {
            "audio/mpeg",
            "audio/wav",
            "audio/x-wav",
            "audio/ogg",
            "audio/aac",
            "audio/flac",
            "audio/webm",
            "audio/mp4",
        }

        if mime_type not in allowed_mime_types:
            raise ValueError(f"Unsupported MIME type: {mime_type}. Allowed: {allowed_mime_types}")

        voice_name_lower = name.lower()

        if voice_name_lower in self.uploaded_speakers:
            raise ValueError(f"Voice '{name}' already exists")

        sanitized_name = _sanitize_filename(name)
        sanitized_consent = _sanitize_filename(consent)

        timestamp = int(time.time())
        file_suffix = Path(audio_file.filename).suffix
        file_ext = file_suffix[1:] if file_suffix and len(file_suffix) > 1 else "wav"
        sanitized_ext = _sanitize_filename(file_ext)
        if not sanitized_ext or sanitized_ext == "file":
            sanitized_ext = "wav"

        filename = f"{sanitized_name}_{sanitized_consent}_{timestamp}.{sanitized_ext}"
        file_path = self.uploaded_speakers_dir / filename

        if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
            raise ValueError("Invalid file path: potential path traversal attack detected")

        try:
            with open(file_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
        except Exception as e:
            raise ValueError(f"Failed to save audio file: {e}")

        speaker_data = {
            "name": name,
            "consent": consent,
            "file_path": str(file_path),
            "created_at": timestamp,
            "mime_type": mime_type,
            "original_filename": audio_file.filename,
            "file_size": file_size,
            "ref_text": ref_text,
            "cache_status": "pending",
            "cache_file": None,
            "cache_generated_at": None,
        }

        success = self.metadata_manager.create_speaker(voice_name_lower, speaker_data)
        if not success:
            try:
                file_path.unlink()
            except Exception:
                pass
            raise ValueError(f"Failed to create metadata for voice '{name}' (possibly already exists)")

        self.uploaded_speakers[voice_name_lower] = speaker_data
        self.supported_speakers.add(voice_name_lower)

        logger.info(f"Uploaded new voice '{name}' with consent ID '{consent}'")

        return {
            "name": name,
            "consent": consent,
            "created_at": timestamp,
            "mime_type": mime_type,
            "file_size": file_size,
            "ref_text": ref_text,
        }

    async def delete_voice(self, name: str) -> bool:
        """Delete an uploaded voice."""
        voice_name_lower = name.lower()

        if voice_name_lower not in self.uploaded_speakers:
            logger.warning(f"Voice '{name}' not found in memory cache")
            return False

        deleted_info = self.metadata_manager.delete_speaker(voice_name_lower)
        if not deleted_info:
            logger.error(f"Failed to delete voice '{name}' from metadata")
            return False

        if voice_name_lower in self.uploaded_speakers:
            del self.uploaded_speakers[voice_name_lower]
        if voice_name_lower in self.supported_speakers:
            self.supported_speakers.remove(voice_name_lower)

        logger.info(f"Deleted voice '{name}' and associated files")
        return True

    def _estimate_prompt_len(self, tts_params: dict[str, Any]) -> int:
        """Estimate prompt length so the placeholder matches model-side embeddings."""
        try:
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
            )

            if self._tts_tokenizer is None:
                from transformers import AutoTokenizer

                model_name = self.engine_client.model_config.model
                self._tts_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    padding_side="left",
                )
            hf_config = self.engine_client.model_config.hf_config
            talker_config = hf_config.talker_config
            task_type = (tts_params.get("task_type") or ["CustomVoice"])[0]

            codec_rate = getattr(talker_config, "codec_rate", 12)

            cached_len = tts_params.get("_cached_ref_code_len")
            if isinstance(cached_len, list) and cached_len:
                cached_len = cached_len[0]

            def _estimate_ref_code_len(ref_audio_data: Any) -> int | None:
                """Estimate codec frame count from resolved ref_audio or cache metadata."""
                if isinstance(cached_len, int):
                    return cached_len
                if not isinstance(ref_audio_data, list) or not ref_audio_data:
                    return None
                val = ref_audio_data[0]
                if isinstance(val, list) and len(val) == 2:
                    wav_samples, sr = val
                    if sr > 0:
                        duration = len(wav_samples) / sr
                        return int(duration * codec_rate)
                return None

            return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
                additional_information=tts_params,
                task_type=task_type,
                tokenize_prompt=lambda t: self._tts_tokenizer(t, padding=False)["input_ids"],
                codec_language_id=getattr(talker_config, "codec_language_id", None),
                spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
                estimate_ref_code_len=_estimate_ref_code_len,
            )
        except Exception as e:
            logger.warning("Failed to estimate TTS prompt length, using fallback 2048: %s", e)
            return 2048

    def _is_tts_model(self) -> bool:
        """Check if the current model is a supported TTS model."""
        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list:
            for stage in stage_list:
                model_stage = getattr(stage, "model_stage", None)
                if model_stage in _TTS_MODEL_STAGES:
                    return True
        return False

    def _validate_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate TTS request parameters. Returns error message or None."""
        task_type = request.task_type or "CustomVoice"

        # Normalize voice to lowercase for case-insensitive matching
        if request.voice is not None:
            request.voice = request.voice.lower()

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language
        if request.language is not None and request.language not in _TTS_LANGUAGES:
            return f"Invalid language '{request.language}'. Supported: {', '.join(sorted(_TTS_LANGUAGES))}"

        # Validate speaker for CustomVoice task
        if task_type == "CustomVoice" and request.voice is not None:
            if self.supported_speakers and request.voice not in self.supported_speakers:
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(self.supported_speakers))}"

        # Validate Base task requirements
        if task_type == "Base":
            if request.voice is None:
                if request.ref_audio is None:
                    return "Base task requires 'ref_audio' for voice cloning"
                # Validate ref_audio format
                if not (request.ref_audio.startswith(("http://", "https://")) or request.ref_audio.startswith("data:")):
                    return "ref_audio must be a URL (http/https) or base64 data URL (data:...)"
            else:
                voice_lower = request.voice.lower()
                if voice_lower in self.uploaded_speakers:
                    speaker_info = self.uploaded_speakers[voice_lower]
                    file_path = Path(speaker_info["file_path"])
                    if not file_path.exists():
                        return f"Audio file for uploaded speaker '{request.voice}' not found on disk"
                else:
                    if request.ref_audio is None:
                        return (
                            f"Base task with built-in speaker '{request.voice}' requires 'ref_audio' for voice cloning"
                        )
                    if not (
                        request.ref_audio.startswith(("http://", "https://")) or request.ref_audio.startswith("data:")
                    ):
                        return "ref_audio must be a URL (http/https) or base64 data URL (data:...)"

        # Validate cross-parameter dependencies
        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        # Validate VoiceDesign task requirements
        if task_type == "VoiceDesign" and not request.instructions:
            return "VoiceDesign task requires 'instructions' to describe the voice"

        # Validate instructions length
        if request.instructions and len(request.instructions) > _TTS_MAX_INSTRUCTIONS_LENGTH:
            return f"Instructions too long (max {_TTS_MAX_INSTRUCTIONS_LENGTH} characters)"

        # Validate max_new_tokens range
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    @staticmethod
    async def _resolve_ref_audio(ref_audio_str: str) -> tuple[list[float], int]:
        """Resolve ref_audio URL/base64 to (wav_samples, sample_rate)."""
        parsed = urlparse(ref_audio_str)

        def _check_ssrf(url: str) -> None:
            host = urlparse(url).hostname
            if not host:
                raise ValueError("ref_audio URL must include a hostname")
            for info in socket.getaddrinfo(host, None):
                ip_str = str(info[4][0]).split("%", 1)[0]
                addr = ipaddress.ip_address(ip_str)
                if any(addr in net for net in _REF_AUDIO_BLOCKED_NETWORKS):
                    raise ValueError(f"ref_audio URL resolves to blocked address: {addr}")

        def _fetch_sync() -> tuple[np.ndarray, int]:
            if parsed.scheme in ("http", "https"):
                _check_ssrf(ref_audio_str)
                with urlopen(ref_audio_str, timeout=_REF_AUDIO_TIMEOUT_S) as resp:
                    data = resp.read(_REF_AUDIO_MAX_BYTES + 1)
                    if len(data) > _REF_AUDIO_MAX_BYTES:
                        raise ValueError(f"ref_audio URL exceeds {_REF_AUDIO_MAX_BYTES} bytes")
                buf = io.BytesIO(data)
            elif ref_audio_str.startswith("data:"):
                b64 = ref_audio_str
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                buf = io.BytesIO(base64.b64decode(b64))
            else:
                raise ValueError("ref_audio must be an http(s) URL or data: base64 URI")
            audio, sr = sf.read(buf, dtype="float32", always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            return np.asarray(audio, dtype=np.float32), int(sr)

        loop = asyncio.get_running_loop()
        wav_np, sr = await loop.run_in_executor(None, _fetch_sync)
        return wav_np.tolist(), sr

    @staticmethod
    def _extract_audio_from_output(output: OmniRequestOutput) -> tuple[np.ndarray, int] | None:
        """Extract audio tensor and sample rate from an OmniRequestOutput.

        Returns (audio_numpy, sample_rate) or None if no audio found.
        """
        audio_output = None
        if hasattr(output, "multimodal_output") and output.multimodal_output:
            audio_output = output.multimodal_output
        if not audio_output and hasattr(output, "request_output"):
            if output.request_output and hasattr(output.request_output, "multimodal_output"):
                audio_output = output.request_output.multimodal_output

        if not audio_output:
            return None

        # Check for audio data using either "audio" or "model_outputs" key
        audio_key = None
        if "audio" in audio_output:
            audio_key = "audio"
        elif "model_outputs" in audio_output:
            audio_key = "model_outputs"

        if audio_key is None:
            return None

        audio_tensor = audio_output[audio_key]
        sample_rate = audio_output.get("sr", 24000)
        if hasattr(sample_rate, "item"):
            sample_rate = sample_rate.item()

        # Streaming accumulates chunks as a list; concat first.
        if isinstance(audio_tensor, list):
            import torch

            audio_tensor = torch.cat(audio_tensor, dim=-1)
        # Convert tensor to numpy
        if hasattr(audio_tensor, "float"):
            audio_tensor = audio_tensor.float().detach().cpu().numpy()

        # Squeeze batch dimension if present, but preserve channel dimension for stereo
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.squeeze()

        return audio_tensor, int(sample_rate)

    def _build_tts_params(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build TTS parameters from request.

        Processes each parameter if present, skips if not.
        Values are wrapped in lists as required by the model.
        """
        params: dict[str, Any] = {}

        # Text content (always required)
        params["text"] = [request.input]

        # Task type
        if request.task_type is not None:
            params["task_type"] = [request.task_type]
        else:
            params["task_type"] = ["CustomVoice"]

        # Language
        if request.language is not None:
            params["language"] = [request.language]
        else:
            params["language"] = ["Auto"]

        # Speaker (voice)
        if request.voice is not None:
            params["speaker"] = [request.voice]

            # If voice is an uploaded speaker and no ref_audio provided, auto-set it
            if request.voice.lower() in self.uploaded_speakers and request.ref_audio is None:
                speaker_info = self.uploaded_speakers[request.voice.lower()]

                # Re-read metadata if cache may have been warmed by the model since upload
                if speaker_info.get("cache_status") != "ready":
                    self._refresh_uploaded_speakers_cache()
                    speaker_info = self.uploaded_speakers.get(request.voice.lower(), speaker_info)

                cache_ready = speaker_info.get("cache_status") == "ready"

                stored_ref_text = speaker_info.get("ref_text")
                if stored_ref_text and request.ref_text is None:
                    params["ref_text"] = [stored_ref_text]
                    params["x_vector_only_mode"] = [False]
                else:
                    params["x_vector_only_mode"] = [True]

                if cache_ready:
                    # Cache warm: skip audio file I/O entirely.
                    # The model will load pre-computed ref_code + speaker
                    # embedding from safetensors via VoiceCacheManager.
                    cached_ref_code_len = speaker_info.get("ref_code_len")
                    if cached_ref_code_len is not None:
                        params["_cached_ref_code_len"] = [int(cached_ref_code_len)]
                    icl = params.get("x_vector_only_mode", [True])[0] is False
                    mode = "ICL" if icl else "x_vector_only"
                    logger.info(
                        "Using cached voice for '%s' (%s mode, ref_code_len=%s) — skipping audio I/O",
                        request.voice, mode, cached_ref_code_len,
                    )
                else:
                    # Cache cold: fall through to audio file read
                    audio_data = self._get_uploaded_audio_data(request.voice)
                    if audio_data:
                        params["ref_audio"] = [audio_data]
                        icl = params.get("x_vector_only_mode", [True])[0] is False
                        mode = "ICL" if icl else "x_vector_only"
                        logger.info(f"Auto-set ref_audio ({mode} mode) for uploaded voice: {request.voice}")
                    else:
                        raise ValueError(f"Audio file for uploaded voice '{request.voice}' is missing or corrupted")

        elif params["task_type"][0] == "CustomVoice":
            params["speaker"] = ["Vivian"]  # Default for CustomVoice

        # Instructions for style/emotion control
        if request.instructions is not None:
            params["instruct"] = [request.instructions]
        else:
            params["instruct"] = [""]

        # Voice clone: ref_audio resolved in create_speech(), not here.
        if request.ref_text is not None:
            params["ref_text"] = [request.ref_text]
        if request.x_vector_only_mode is not None:
            params["x_vector_only_mode"] = [request.x_vector_only_mode]

        # Generation parameters
        if request.max_new_tokens is not None:
            params["max_new_tokens"] = [request.max_new_tokens]
        else:
            params["max_new_tokens"] = [2048]

        return params

    async def _prepare_tts_generator(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> tuple:
        """Validate request, build prompt, and return the async generator.

        Shared by both _generate_audio_bytes() and the streaming output path
        in create_speech().

        Returns:
            Tuple of (async_generator, icl_trim_info) where icl_trim_info
            is a dict with ref_audio_duration and ref_text/target_text lengths
            for ICL voice-clone trimming, or None when not applicable.

        Raises:
            ValueError: If validation fails.
        """
        icl_trim_info: dict | None = None

        if self._is_tts_model():
            validation_error = self._validate_tts_request(request)
            if validation_error:
                raise ValueError(validation_error)

            tts_params = self._build_tts_params(request)

            if request.ref_audio is not None:
                wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
                tts_params["ref_audio"] = [[wav_list, sr]]
            elif "ref_audio" in tts_params:
                ref_val = tts_params["ref_audio"]
                if isinstance(ref_val, list) and ref_val and isinstance(ref_val[0], str):
                    wav_list, sr = await self._resolve_ref_audio(ref_val[0])
                    tts_params["ref_audio"] = [[wav_list, sr]]

            # Collect ICL trim metadata: in ICL mode the model generates audio
            # that overlaps with the tail of the reference text.  We estimate
            # the overlap from the speaker's rate and trim it after generation.
            is_icl = (
                tts_params.get("task_type", [""])[0] == "Base"
                and tts_params.get("x_vector_only_mode", [True])[0] is False
            )
            if is_icl:
                ref_audio_val = tts_params.get("ref_audio")
                ref_text_val = tts_params.get("ref_text", [None])[0]
                if ref_text_val:
                    ref_duration: float | None = None
                    if isinstance(ref_audio_val, list) and ref_audio_val:
                        inner = ref_audio_val[0]
                        if isinstance(inner, list) and len(inner) == 2:
                            wav_samples, wav_sr = inner
                            if wav_sr > 0:
                                ref_duration = len(wav_samples) / wav_sr
                    if ref_duration is None:
                        # Cache-warm path: estimate duration from cached ref_code_len
                        cached_rcl = tts_params.get("_cached_ref_code_len")
                        if isinstance(cached_rcl, list) and cached_rcl:
                            hf_config = self.engine_client.model_config.hf_config
                            codec_rate = getattr(hf_config.talker_config, "codec_rate", 12)
                            ref_duration = int(cached_rcl[0]) / codec_rate
                    if ref_duration is not None:
                        icl_trim_info = {
                            "ref_duration": ref_duration,
                            "ref_text_len": len(ref_text_val),
                            "target_text_len": len(request.input),
                        }

            ph_len = self._estimate_prompt_len(tts_params)

            # Strip private estimation hints before sending to engine
            tts_params.pop("_cached_ref_code_len", None)

            prompt = {
                "prompt_token_ids": [1] * ph_len,
                "additional_information": tts_params,
            }
        else:
            tts_params = {}
            prompt = {"prompt": request.input}

        request_id = f"speech-{random_uuid()}"

        logger.info(
            "TTS speech request %s: text=%r, task_type=%s",
            request_id,
            request.input[:50] + "..." if len(request.input) > 50 else request.input,
            tts_params.get("task_type", ["unknown"])[0],
        )

        sampling_params_list = self.engine_client.default_sampling_params_list

        generator = self.engine_client.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=["audio"],
        )

        return generator, icl_trim_info

    async def _generate_audio_bytes(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> tuple[bytes, str]:
        """Core TTS generation logic: validate, generate, and encode audio.

        Extracted from create_speech() so it can be reused by the streaming
        WebSocket handler for per-sentence generation.

        Args:
            request: The speech request with text and parameters.

        Returns:
            Tuple of (audio_bytes, media_type).

        Raises:
            ValueError: If validation fails or generation produces no output.
        """
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        generator, icl_trim_info = await self._prepare_tts_generator(request)

        audio_chunks: list[tuple[np.ndarray, int]] = []
        final_output: OmniRequestOutput | None = None
        async for res in generator:
            final_output = res
            chunk = self._extract_audio_from_output(res)
            if chunk is not None:
                audio_chunks.append(chunk)

        if final_output is None:
            raise ValueError("No output generated from the model.")

        if not audio_chunks:
            raise ValueError("TTS model did not produce audio output.")

        sample_rate = audio_chunks[0][1]
        if len(audio_chunks) == 1:
            audio_tensor = audio_chunks[0][0]
        else:
            audio_tensor = np.concatenate([c[0] for c in audio_chunks])

        # ICL voice-clone trim: the model generates audio that overlaps with
        # the tail of the reference text.  Estimate the target-text portion
        # from the reference speaker's rate and keep only that.
        if icl_trim_info and audio_tensor.size > 0:
            ref_dur = icl_trim_info["ref_duration"]
            ref_len = icl_trim_info["ref_text_len"]
            tgt_len = icl_trim_info["target_text_len"]
            if ref_dur > 0 and ref_len > 0 and tgt_len > 0:
                chars_per_sec = ref_len / ref_dur
                target_duration = tgt_len / chars_per_sec
                # 15% buffer so we don't clip the trailing syllable
                target_samples = int(target_duration * 1.15 * sample_rate)
                total_samples = audio_tensor.size
                trim_samples = max(0, total_samples - target_samples)
                if 0 < trim_samples < total_samples:
                    logger.info(
                        "ICL trim: speaker_rate=%.1f chars/s  "
                        "target_dur=%.2fs  trim=%.2fs  keep=%.2fs",
                        chars_per_sec,
                        target_duration,
                        trim_samples / sample_rate,
                        target_samples / sample_rate,
                    )
                    audio_tensor = audio_tensor[trim_samples:]

        audio_obj = CreateAudio(
            audio_tensor=audio_tensor,
            sample_rate=int(sample_rate),
            response_format=request.response_format or "wav",
            speed=request.speed or 1.0,
            stream_format=request.stream_format,
            base64_encode=False,
        )

        audio_response: AudioResponse = self.create_audio(audio_obj)
        return audio_response.audio_data, audio_response.media_type

    async def _generate_audio_stream(
        self,
        request: OpenAICreateSpeechRequest,
    ):
        """Stream raw PCM audio chunks as they are produced by the model.

        Yields (pcm_bytes, sample_rate) for each Code2Wav output chunk.
        This is used by the streaming WebSocket handler so the first audio
        frame can be sent after just the first codec chunk rather than waiting
        for the entire sentence to be generated.

        Raises:
            ValueError: If validation fails.
        """
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        generator, _icl_trim_info = await self._prepare_tts_generator(request)
        speed = request.speed or 1.0

        async for res in generator:
            audio_chunk = self._extract_audio_from_output(res)
            if audio_chunk is None:
                continue
            audio_tensor, sample_rate = audio_chunk
            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=int(sample_rate),
                response_format="pcm",
                speed=speed,
                stream_format="audio",
                base64_encode=False,
            )
            audio_response: AudioResponse = self.create_audio(audio_obj)
            yield audio_response.audio_data, int(sample_rate)

    async def create_speech(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI
        Create Speech API.

        For Qwen3-TTS models, additional parameters are supported:
        - task_type: "CustomVoice", "VoiceDesign", or "Base"
        - language: Language code (e.g., "Chinese", "English", "Auto")
        - voice: Speaker name (e.g., "Vivian", "Ryan") for CustomVoice
        - instructions: Voice style/emotion instructions
        - ref_audio: Reference audio for voice cloning (Base task)
        - ref_text: Transcript of reference audio (Base task)
        - x_vector_only_mode: Use speaker embedding only (Base task)

        When stream=True, audio chunks are yielded as raw bytes
        (PCM/WAV per response_format) via a StreamingResponse.
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        try:
            # --- Streaming audio output path ---
            if request.stream:
                if self.engine_client.errored:
                    raise self.engine_client.dead_error

                generator, _icl_trim_info = await self._prepare_tts_generator(request)

                response_format = request.response_format or "wav"
                speed = request.speed or 1.0

                media_type_map = {
                    "wav": "audio/wav",
                    "pcm": "audio/pcm",
                    "flac": "audio/flac",
                    "mp3": "audio/mpeg",
                    "aac": "audio/aac",
                    "opus": "audio/ogg",
                }
                media_type = media_type_map.get(response_format, "audio/wav")

                async def audio_stream_generator():
                    async for res in generator:
                        audio_chunk = self._extract_audio_from_output(res)
                        if audio_chunk is None:
                            continue
                        audio_tensor, sample_rate = audio_chunk
                        audio_obj = CreateAudio(
                            audio_tensor=audio_tensor,
                            sample_rate=int(sample_rate),
                            response_format=response_format,
                            speed=speed,
                            stream_format=request.stream_format,
                            base64_encode=False,
                        )
                        audio_response: AudioResponse = self.create_audio(audio_obj)
                        yield audio_response.audio_data

                return StreamingResponse(
                    audio_stream_generator(),
                    media_type=media_type,
                )

            # --- Non-streaming path (default) ---
            audio_data, media_type = await self._generate_audio_bytes(request)
            return Response(content=audio_data, media_type=media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)
        except Exception as e:
            logger.exception("Speech generation failed: %s", e)
            return self.create_error_response(f"Speech generation failed: {e}")
