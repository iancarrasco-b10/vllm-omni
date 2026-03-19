"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client:
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
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
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
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
            initial_message = await self._receive_initial_message(websocket)
            if initial_message is None:
                return  # Error already sent, connection closing

            if isinstance(initial_message, dict):
                await self._handle_voice_delete(websocket, initial_message)
                return

            config = initial_message
            await self._register_uploaded_voice_if_needed(websocket, config)

            # Validate model if specified
            if config.model and hasattr(self._speech_service, "_check_model"):
                error = await self._speech_service._check_model(
                    OpenAICreateSpeechRequest(input="ping", model=config.model)
                )
                if error is not None:
                    await self._send_error(websocket, str(error))
                    return

            boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
            splitter = SentenceSplitter(boundary_re=boundary_re)
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

    async def _receive_initial_message(
        self,
        websocket: WebSocket,
    ) -> StreamingSpeechSessionConfig | dict[str, str] | None:
        """Wait for and validate the first client message."""
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

        msg_type = msg.get("type")
        if msg_type == "voice.delete":
            voice_name = msg.get("voice_name")
            if not isinstance(voice_name, str) or not voice_name.strip():
                await self._send_error(websocket, "voice.delete requires a non-empty 'voice_name'")
                return None
            return {"type": "voice.delete", "voice_name": voice_name}

        if msg_type != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg_type}",
            )
            return None

        try:
            config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _register_uploaded_voice_if_needed(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        """Persist a cloned voice for later sessions when requested by the client."""
        voice_name = config.voice_name
        if not voice_name or config.ref_audio is None:
            return

        voice_key = voice_name.lower()
        if voice_key not in self._speech_service.uploaded_speakers:
            await self._speech_service.upload_voice_from_data_uri(config.ref_audio, voice_name)

        await websocket.send_json(
            {
                "type": "voice.registered",
                "voice_name": voice_name,
            }
        )

    async def _handle_voice_delete(self, websocket: WebSocket, msg: dict[str, str]) -> None:
        """Delete a stored uploaded voice and acknowledge the result."""
        voice_name = msg["voice_name"]
        deleted = await self._speech_service.delete_voice(voice_name)
        if not deleted:
            await self._send_error(websocket, f"Voice '{voice_name}' not found")
            return

        await websocket.send_json(
            {
                "type": "voice.deleted",
                "voice_name": voice_name,
            }
        )

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
                async with aclosing(self._speech_service._generate_audio_chunks(generator, request_id)) as stream:
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
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
            except Exception:
                logger.debug("Failed to send audio.done for sentence %d", sentence_index, exc_info=True)

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
