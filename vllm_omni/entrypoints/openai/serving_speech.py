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
from fastapi import Request
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
    """Sanitize filename to prevent path traversal attacks."""
    filename = os.path.basename(filename)
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    if not sanitized:
        sanitized = "file"
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def _validate_path_within_directory(file_path: Path, directory: Path) -> bool:
    """Validate that file_path is within the specified directory."""
    try:
        file_path_resolved = file_path.resolve()
        directory_resolved = directory.resolve()
        return directory_resolved in file_path_resolved.parents or directory_resolved == file_path_resolved
    except Exception:
        return False


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        speech_voice_samples_dir = os.environ.get("SPEECH_VOICE_SAMPLES", "/tmp/voice_clones")
        self.uploaded_speakers_dir = Path(speech_voice_samples_dir)
        self.uploaded_speakers_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.uploaded_speakers_dir / "metadata.json"
        self.metadata_manager = MetadataManager(self.metadata_file)

        self.supported_speakers = self._load_supported_speakers()
        self.uploaded_speakers: dict[str, dict[str, Any]] = {}
        self._refresh_uploaded_speakers_cache()
        self.supported_speakers.update(self.uploaded_speakers.keys())

        logger.info("Loaded %d supported speakers: %s", len(self.supported_speakers), sorted(self.supported_speakers))
        logger.info("Loaded %d uploaded/cached speakers", len(self.uploaded_speakers))
        self._tts_tokenizer = None

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

    def _refresh_uploaded_speakers_cache(self):
        """Refresh in-memory cache of uploaded speakers from metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file) as f:
                    metadata = json.load(f)
                self.uploaded_speakers = metadata.get("uploaded_speakers", {})
        except Exception as e:
            logger.warning("Could not refresh uploaded speakers cache: %s", e)
            self.uploaded_speakers = {}

    def _get_uploaded_audio_data(self, voice_name: str) -> str | None:
        """Get base64 encoded audio data for an uploaded voice.

        Non-WAV formats are converted on the fly via pydub.
        """
        voice_name_lower = voice_name.lower()
        if voice_name_lower not in self.uploaded_speakers:
            return None

        speaker_info = self.uploaded_speakers[voice_name_lower]
        file_path = Path(speaker_info["file_path"])
        if not file_path.exists():
            logger.warning("Audio file not found for voice %s: %s", voice_name, file_path)
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
            logger.error("Could not read audio file for voice %s: %s", voice_name, e)
            return None

    def register_voice_clone(
        self,
        voice_name: str,
        audio_data_uri: str,
        ref_text: str | None = None,
    ) -> dict[str, Any]:
        """Register a voice clone from a base64 data URI and save it to disk.

        Called by the WebSocket handler when a client provides ref_audio +
        voice_name in session.config.  If the voice already exists, returns
        the existing speaker info without overwriting.

        Returns the speaker info dict.
        """
        voice_key = voice_name.lower()

        if voice_key in self.uploaded_speakers:
            existing = self.uploaded_speakers[voice_key]
            if Path(existing["file_path"]).exists():
                logger.info("Voice '%s' already registered, reusing", voice_name)
                return existing

        if "," in audio_data_uri:
            header, b64_data = audio_data_uri.split(",", 1)
        else:
            header, b64_data = "", audio_data_uri

        audio_bytes = base64.b64decode(b64_data)

        mime_type = "audio/wav"
        if header.startswith("data:"):
            mime_part = header[5:]
            if ";" in mime_part:
                mime_type = mime_part.split(";")[0]

        ext_map = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mpeg": "mp3",
            "audio/mp4": "m4a",
            "audio/flac": "flac",
            "audio/ogg": "ogg",
            "audio/aac": "aac",
            "audio/webm": "webm",
        }
        ext = ext_map.get(mime_type, "wav")

        sanitized_name = _sanitize_filename(voice_name)
        timestamp = int(time.time())
        filename = f"{sanitized_name}_{timestamp}.{ext}"
        file_path = self.uploaded_speakers_dir / filename

        if not _validate_path_within_directory(file_path, self.uploaded_speakers_dir):
            raise ValueError("Invalid voice name: potential path traversal")

        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        speaker_data: dict[str, Any] = {
            "name": voice_name,
            "file_path": str(file_path),
            "created_at": timestamp,
            "mime_type": mime_type,
            "file_size": len(audio_bytes),
            "ref_text": ref_text,
            "cache_status": "pending",
            "cache_file": None,
            "cache_generated_at": None,
        }

        self.metadata_manager.create_speaker(voice_key, speaker_data)
        self.uploaded_speakers[voice_key] = speaker_data
        self.supported_speakers.add(voice_key)

        logger.info("Registered voice clone '%s' (%d bytes, %s)", voice_name, len(audio_bytes), mime_type)
        return speaker_data

    def delete_voice_clone(self, voice_name: str) -> bool:
        """Delete a cached voice clone and its associated files.

        Removes the audio file, safetensors cache, and metadata entry.
        Returns True if the voice was found and deleted.
        """
        voice_key = voice_name.lower()

        if voice_key not in self.uploaded_speakers:
            return False

        deleted_info = self.metadata_manager.delete_speaker(voice_key)
        if deleted_info is None:
            logger.warning("Voice '%s' not found in metadata", voice_name)
            return False

        self.uploaded_speakers.pop(voice_key, None)
        self.supported_speakers.discard(voice_key)

        logger.info("Deleted voice clone '%s' and associated files", voice_name)
        return True

    def _estimate_prompt_len(self, tts_params: dict[str, Any]) -> int:
        """Estimate prompt length so the placeholder matches model-side embeddings."""
        try:
            task_type = (tts_params.get("task_type") or ["CustomVoice"])[0]
            hf_config = self.engine_client.model_config.hf_config
            talker_config = hf_config.talker_config

            # Fast path for streaming Base tasks with a warm cache.
            # The streaming formula doesn't depend on tokenized text length,
            # so we can skip the tokenizer entirely (~2-5 ms saved per request).
            cached_ref_code_len = (tts_params.get("_cached_ref_code_len") or [None])[0]
            if task_type == "Base" and isinstance(cached_ref_code_len, (int, float)):
                return self._fast_estimate_base_streaming(
                    tts_params, int(cached_ref_code_len), hf_config, talker_config
                )

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

            codec_fps = getattr(hf_config, "codec_frame_rate_hz", None)
            if codec_fps is None:
                codec_fps = float(getattr(talker_config, "position_id_per_seconds", 12))

            def _estimate_ref_code_len(ref_audio_raw: object) -> int | None:
                """Derive codec frame count from resolved [wav_samples, sr] pairs."""
                if not isinstance(ref_audio_raw, list) or not ref_audio_raw:
                    return None
                item = ref_audio_raw[0]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    wav, sr = item
                    n_samples = len(wav) if isinstance(wav, (list, np.ndarray)) else 0
                    if n_samples > 0 and sr > 0:
                        return int(n_samples / sr * codec_fps)
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

    def _fast_estimate_base_streaming(
        self,
        tts_params: dict[str, Any],
        ref_code_len: int,
        hf_config: Any,
        talker_config: Any,
    ) -> int:
        """Tokenizer-free prompt length for streaming Base tasks with warm cache.

        For streaming (non_streaming_mode=False), the prompt is:
            role(3) + codec_prefix + {codec_lens if ICL else 1}
        None of these depend on the tokenized text, so we skip the tokenizer.
        """
        language = (tts_params.get("language") or ["Auto"])[0]
        if not isinstance(language, str):
            language = "Auto"

        codec_language_id = getattr(talker_config, "codec_language_id", None)
        spk_is_dialect = getattr(talker_config, "spk_is_dialect", None)
        speaker = (tts_params.get("speaker") or [""])[0]

        language_id = None
        if language.lower() != "auto" and codec_language_id:
            language_id = codec_language_id.get(language.lower())
        if (
            language_id is None
            and codec_language_id
            and spk_is_dialect
            and isinstance(language, str)
            and language.lower() in ("chinese", "auto")
            and isinstance(speaker, str)
            and speaker.strip()
        ):
            dialect = spk_is_dialect.get(speaker.lower())
            if isinstance(dialect, str) and dialect:
                language_id = codec_language_id.get(dialect)

        prefill_len = 3 if language_id is None else 4
        codec_prefix_len = prefill_len + 1 + 2 - 1  # +speaker +[pad,bos] -1

        xvec_only = bool((tts_params.get("x_vector_only_mode") or [False])[0])
        role_len = 3
        if xvec_only:
            prompt_len = role_len + codec_prefix_len + 1
        else:
            codec_lens = 1 + ref_code_len
            prompt_len = role_len + codec_prefix_len + codec_lens

        return max(2, int(prompt_len))

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
            self._refresh_uploaded_speakers_cache()
            if request.voice is not None:
                voice_lower = request.voice.lower()
                if voice_lower in self.uploaded_speakers:
                    speaker_info = self.uploaded_speakers[voice_lower]
                    if not Path(speaker_info["file_path"]).exists():
                        return f"Audio file for cached voice '{request.voice}' not found on disk"
                elif request.ref_audio is None:
                    return (
                        f"Voice '{request.voice}' is not cached. "
                        "Provide ref_audio (and optionally ref_text) to register a voice clone first."
                    )
                elif not (
                    request.ref_audio.startswith(("http://", "https://")) or request.ref_audio.startswith("data:")
                ):
                    return "ref_audio must be a URL (http/https) or base64 data URL (data:...)"
            elif request.ref_audio is None:
                return "Base task requires 'ref_audio' for voice cloning"
            elif not (request.ref_audio.startswith(("http://", "https://")) or request.ref_audio.startswith("data:")):
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
        if isinstance(sample_rate, list):
            sample_rate = sample_rate[-1]
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

            # Auto-populate ref_audio from cached uploaded voice for Base task
            if request.voice.lower() in self.uploaded_speakers and request.ref_audio is None:
                speaker_info = self.uploaded_speakers[request.voice.lower()]

                # Re-read metadata in case cache was warmed since upload
                if speaker_info.get("cache_status") != "ready":
                    self._refresh_uploaded_speakers_cache()
                    speaker_info = self.uploaded_speakers.get(request.voice.lower(), speaker_info)

                cache_ready = speaker_info.get("cache_status") == "ready"

                stored_ref_text = speaker_info.get("ref_text")
                if request.x_vector_only_mode is True:
                    params["x_vector_only_mode"] = [True]
                elif stored_ref_text and request.ref_text is None:
                    params["ref_text"] = [stored_ref_text]
                    params["x_vector_only_mode"] = [False]
                else:
                    params["x_vector_only_mode"] = [True]

                if cache_ready:
                    cached_ref_code_len = speaker_info.get("ref_code_len")
                    if cached_ref_code_len is not None:
                        params["_cached_ref_code_len"] = [int(cached_ref_code_len)]
                    icl = params.get("x_vector_only_mode", [True])[0] is False
                    mode = "ICL" if icl else "x_vector_only"
                    logger.info(
                        "Using cached voice for '%s' (%s mode, ref_code_len=%s)",
                        request.voice,
                        mode,
                        cached_ref_code_len,
                    )
                else:
                    audio_data = self._get_uploaded_audio_data(request.voice)
                    if audio_data:
                        params["ref_audio"] = [audio_data]
                        icl = params.get("x_vector_only_mode", [True])[0] is False
                        mode = "ICL" if icl else "x_vector_only"
                        logger.info("Auto-set ref_audio (%s mode) for cached voice: %s", mode, request.voice)
                    else:
                        raise ValueError(f"Audio file for cached voice '{request.voice}' is missing or corrupted")

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

        # VoiceDesign requires non_streaming_mode (match offline script behaviour).
        if params["task_type"][0] == "VoiceDesign":
            params["non_streaming_mode"] = [True]

        return params

    async def _prepare_tts_generator(
        self,
        request: OpenAICreateSpeechRequest,
    ):
        """Validate request, build prompt, and return the async generator.

        Shared by both _generate_audio_bytes() and the streaming output path
        in create_speech().

        Returns:
            The async generator from engine_client.generate().

        Raises:
            ValueError: If validation fails.
        """
        if self._is_tts_model():
            validation_error = self._validate_tts_request(request)
            if validation_error:
                raise ValueError(validation_error)

            # Must use prompt_token_ids (not text prompt): the AR Talker
            # operates on codec tokens; text token IDs exceed codec vocab.
            # model.preprocess replaces all embeddings, so placeholder value
            # is irrelevant -- but length must match to avoid excess padding.
            tts_params = self._build_tts_params(request)

            if request.ref_audio is not None:
                wav_list, sr = await self._resolve_ref_audio(request.ref_audio)
                tts_params["ref_audio"] = [[wav_list, sr]]
            elif "ref_audio" in tts_params:
                # ref_audio was auto-populated from cache as a data URI string
                ref_val = tts_params["ref_audio"]
                if isinstance(ref_val, list) and ref_val and isinstance(ref_val[0], str):
                    wav_list, sr = await self._resolve_ref_audio(ref_val[0])
                    tts_params["ref_audio"] = [[wav_list, sr]]

            ph_len = self._estimate_prompt_len(tts_params)
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

        return generator

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

        generator = await self._prepare_tts_generator(request)

        final_output: OmniRequestOutput | None = None
        async for res in generator:
            final_output = res

        if final_output is None:
            raise ValueError("No output generated from the model.")

        audio_chunk = self._extract_audio_from_output(final_output)
        if audio_chunk is None:
            raise ValueError("TTS model did not produce audio output.")

        audio_tensor, sample_rate = audio_chunk

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

        Yields (pcm_bytes, sample_rate) for each new Code2Wav output chunk.
        Uses delta-slicing so only new audio is emitted on each iteration,
        avoiding O(n^2) re-concatenation of the full accumulated audio.

        The engine generator is explicitly closed in a finally block so that
        ``GeneratorExit`` propagates into ``async_omni.generate()``, which
        calls ``abort()`` on all pipeline stages.  Without this, a client
        disconnect can leave Stage-1 stuck in an SHM retry loop.

        Raises:
            ValueError: If validation fails.
        """
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        generator = await self._prepare_tts_generator(request)
        speed = request.speed or 1.0
        prev_count = 0
        sample_rate_val = 24000

        try:
            async for res in generator:
                audio_output = None
                if hasattr(res, "multimodal_output") and res.multimodal_output:
                    audio_output = res.multimodal_output
                if not audio_output and hasattr(res, "request_output"):
                    if res.request_output and hasattr(res.request_output, "multimodal_output"):
                        audio_output = res.request_output.multimodal_output
                if not audio_output:
                    continue

                audio_key = "audio" if "audio" in audio_output else (
                    "model_outputs" if "model_outputs" in audio_output else None
                )
                if audio_key is None:
                    continue

                sr_raw = audio_output.get("sr")
                if sr_raw is not None:
                    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
                    sample_rate_val = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

                audio_val = audio_output[audio_key]
                if isinstance(audio_val, list):
                    new_chunks = audio_val[prev_count:]
                    prev_count = len(audio_val)
                else:
                    if audio_val is not None:
                        new_chunks = [audio_val]
                        prev_count += 1
                    else:
                        new_chunks = []

                for chunk_tensor in new_chunks:
                    if hasattr(chunk_tensor, "float"):
                        chunk_np = chunk_tensor.float().detach().cpu().numpy()
                    else:
                        chunk_np = chunk_tensor
                    if chunk_np.ndim > 1:
                        chunk_np = chunk_np.squeeze()
                    audio_obj = CreateAudio(
                        audio_tensor=chunk_np,
                        sample_rate=sample_rate_val,
                        response_format="pcm",
                        speed=speed,
                        stream_format="audio",
                        base64_encode=False,
                    )
                    audio_response: AudioResponse = self.create_audio(audio_obj)
                    yield audio_response.audio_data, sample_rate_val
        finally:
            await generator.aclose()

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

                generator = await self._prepare_tts_generator(request)

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
                    try:
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
                    finally:
                        await generator.aclose()

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
