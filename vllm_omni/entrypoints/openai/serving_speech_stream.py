"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input
        {"type": "voice.delete", "voice_name": "..."}  # Delete a cached voice

    Server -> Client:
        {"type": "voice.registered", "voice_name": "...", "cached": true}  # if voice_name provided
        {"type": "voice.deleted", "voice_name": "...", "success": true}
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0, "sample_rate": 24000}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
"""

import asyncio
import json
from contextlib import aclosing

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.text_splitter import (
    SPLIT_CLAUSE,
    SPLIT_SENTENCE,
    SentenceSplitter,
)

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_MAX_CONFIG_MESSAGE_SIZE = 16 * 1024 * 1024  # allow large base64-encoded ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is split at sentence boundaries, and audio is generated
    per sentence using the existing OmniOpenAIServingSpeech pipeline.

    Args:
        speech_service: The existing TTS serving instance (reused for
            validation and audio generation).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        speech_service: OmniOpenAIServingSpeech,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._speech_service = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent, connection closing

            # If voice_name is provided, map it to the voice field for cache lookup
            if config.voice_name:
                config.voice = config.voice_name

            # Validate model if specified
            if config.model and hasattr(self._speech_service, "_check_model"):
                error = await self._speech_service._check_model(
                    OpenAICreateSpeechRequest(input="ping", model=config.model)
                )
                if error is not None:
                    await self._send_error(websocket, str(error))
                    return

            # Check if voice is already cached and notify client
            if config.voice_name and self._speech_service._has_voice_cache(config.voice_name):
                await websocket.send_json({
                    "type": "voice.registered",
                    "voice_name": config.voice_name,
                    "cached": True,
                })

            boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
            splitter = SentenceSplitter(
                min_sentence_length=20,
                boundary_re=boundary_re,
                max_buffered_words=config.max_buffered_words,
            )
            sentence_index = 0

            # 2. Receive text chunks until input.done
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout: no message received")
                    return

                if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                    await self._send_error(websocket, "input.text message too large")
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                if not isinstance(msg, dict):
                    await self._send_error(websocket, "WebSocket messages must be JSON objects")
                    continue

                msg_type = msg.get("type")

                if msg_type == "input.text":
                    text = msg.get("text", "")
                    if not isinstance(text, str):
                        await self._send_error(websocket, "input.text requires a string value")
                        continue
                    sentences = splitter.add_text(text)
                    for sentence in sentences:
                        await self._generate_and_send(websocket, config, sentence, sentence_index)
                        sentence_index += 1

                elif msg_type == "input.done":
                    # Flush remaining buffer
                    remaining = splitter.flush()
                    if remaining:
                        await self._generate_and_send(websocket, config, remaining, sentence_index)
                        sentence_index += 1

                    # Send session.done
                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "total_sentences": sentence_index,
                        }
                    )
                    return

                elif msg_type == "voice.delete":
                    await self._handle_voice_delete(websocket, msg)

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )

        except WebSocketDisconnect:
            logger.info("Streaming speech: client disconnected")
        except Exception as e:
            logger.exception("Streaming speech session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                logger.debug("Failed to send error to streaming speech client", exc_info=True)

    async def _receive_config(self, websocket: WebSocket) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        if len(raw) > _MAX_CONFIG_MESSAGE_SIZE:
            await self._send_error(websocket, "session.config message too large")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict):
            await self._send_error(websocket, "session.config must be a JSON object")
            return None

        if msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type')}",
            )
            return None

        try:
            config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"

        request = OpenAICreateSpeechRequest(
            input=sentence_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            stream=config.stream_audio,
        )

        start_payload = {
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        }
        if config.stream_audio and response_format == "pcm":
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        request_id = None
        try:
            if config.stream_audio:
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
                async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                    async for chunk in stream:
                        total_bytes += len(chunk)
                        await websocket.send_bytes(chunk)
            else:
                audio_bytes, _ = await self._speech_service._generate_audio_bytes(request)
                total_bytes = len(audio_bytes)
                await websocket.send_bytes(audio_bytes)
        except WebSocketDisconnect:
            if request_id is not None:
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    logger.debug("Failed to abort streaming speech request %s", request_id, exc_info=True)
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Generation failed for sentence %d: %s", sentence_index, e)
            await self._send_error(websocket, f"Generation failed for sentence {sentence_index}: {e}")
        finally:
            try:
                done_payload: dict = {
                    "type": "audio.done",
                    "sentence_index": sentence_index,
                    "total_bytes": total_bytes,
                    "sample_rate": _PCM_SAMPLE_RATE,
                    "error": generation_failed,
                }
                await websocket.send_json(done_payload)
            except Exception:
                logger.debug("Failed to send audio.done for sentence %d", sentence_index, exc_info=True)

            # After first sentence, check if voice was newly cached and notify
            if (
                sentence_index == 0
                and config.voice_name
                and not generation_failed
                and self._speech_service._has_voice_cache(config.voice_name)
            ):
                try:
                    await websocket.send_json({
                        "type": "voice.registered",
                        "voice_name": config.voice_name,
                        "cached": True,
                    })
                    # Clear ref_audio for subsequent sentences — cache will be used
                    # Keep ref_text as the talker still needs it for tokenization
                    config.ref_audio = None
                except Exception:
                    logger.debug("Failed to send voice.registered", exc_info=True)

    async def _handle_voice_delete(self, websocket: WebSocket, msg: dict) -> None:
        """Handle a voice.delete request.

        Deletes from both uploaded_speakers registry and voice cache (safetensors).
        """
        voice_name = msg.get("voice_name")
        if not voice_name or not isinstance(voice_name, str):
            await self._send_error(websocket, "voice.delete requires a 'voice_name' string")
            return

        try:
            # Try uploaded speakers first
            deleted_uploaded = await self._speech_service.delete_voice(voice_name)

            # Also clean up voice cache files (safetensors + wav)
            deleted_cache = self._delete_voice_cache_files(voice_name)

            success = deleted_uploaded or deleted_cache
            await websocket.send_json({
                "type": "voice.deleted",
                "voice_name": voice_name,
                "success": success,
            })
            if success:
                logger.info("Voice %r deleted via WebSocket (uploaded=%s, cache=%s)",
                            voice_name, deleted_uploaded, deleted_cache)
            else:
                logger.warning("Voice %r not found for deletion via WebSocket", voice_name)
        except Exception as e:
            logger.error("Failed to delete voice %r: %s", voice_name, e)
            await self._send_error(websocket, f"Failed to delete voice '{voice_name}': {e}")

    @staticmethod
    def _delete_voice_cache_files(voice_name: str) -> bool:
        """Remove safetensors + wav voice cache files from disk."""
        import os
        from pathlib import Path

        cache_dir = Path(os.environ.get("SPEECH_VOICE_SAMPLES", "/tmp/voice_samples"))
        deleted = False
        for ext in (".safetensors", ".wav"):
            path = cache_dir / f"{voice_name.lower()}{ext}"
            if path.exists():
                try:
                    path.unlink()
                    deleted = True
                    logger.info("Deleted voice cache file: %s", path)
                except Exception as e:
                    logger.warning("Failed to delete %s: %s", path, e)
        return deleted

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": message,
                }
            )
        except Exception:
            pass  # Connection may already be closed; safe to ignore
