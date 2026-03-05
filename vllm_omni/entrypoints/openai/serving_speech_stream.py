"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Voice cloning:
    The session.config message supports inline voice cloning.  Provide
    ``task_type: "Base"`` with ``ref_audio`` (base64 data URI) and
    optionally ``ref_text`` (transcript).  Set ``voice_name`` to cache
    the voice clone in /tmp so subsequent sessions with the same name
    skip the expensive embedding extraction.

    When ``ref_text`` is provided, text is buffered and generated as a
    single request using in-context learning (ICL) for higher voice
    fidelity.  Set ``x_vector_only_mode: true`` to skip ICL and enable
    per-sentence streaming, which gives much lower TTFA at the cost of
    slightly reduced voice similarity.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input
        {"type": "voice.delete", "voice_name": "..."}  # Delete a cached voice
        {"type": "voice.list"}                         # List available voices

    Server -> Client:
        {"type": "voice.registered", "voice_name": "...", "cached": true}
        {"type": "voice.deleted", "voice_name": "..."}
        {"type": "voice.list", "voices": [...], "uploaded_voices": [...]}
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
"""

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.text_splitter import SentenceSplitter

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds


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
            # 1. Wait for session.config (or a voice.delete command)
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent, or voice.delete handled

            # 1b. Handle voice clone registration / cache lookup when
            # voice_name is provided in session.config.
            if config.voice_name and config.ref_audio:
                try:
                    self._speech_service.register_voice_clone(
                        voice_name=config.voice_name,
                        audio_data_uri=config.ref_audio,
                        ref_text=config.ref_text,
                    )
                    await websocket.send_json(
                        {
                            "type": "voice.registered",
                            "voice_name": config.voice_name,
                            "cached": True,
                        }
                    )
                except Exception as e:
                    logger.error("Voice clone registration failed: %s", e)
                    await self._send_error(websocket, f"Voice clone registration failed: {e}")
                    return

                # Point generation at the cached voice; the serving layer
                # will pull ref_audio / ref_text from disk automatically.
                config.voice = config.voice_name
                config.task_type = "Base"
                config.ref_audio = None
                config.ref_text = None

            elif config.voice_name:
                # Reuse a previously cached voice clone — only voice_name
                # is needed; no ref_audio or task_type required.
                self._speech_service._refresh_uploaded_speakers_cache()
                voice_key = config.voice_name.lower()
                if voice_key not in self._speech_service.uploaded_speakers:
                    await self._send_error(
                        websocket,
                        f"Voice '{config.voice_name}' is not cached. "
                        "Provide ref_audio (and optionally ref_text) in session.config to register it first.",
                    )
                    return

                config.voice = config.voice_name
                config.task_type = "Base"
                config.ref_audio = None
                config.ref_text = None

            elif (config.task_type or "").lower() == "base" and config.voice and not config.ref_audio:
                # Base task with voice but no ref_audio: voice must be cached.
                self._speech_service._refresh_uploaded_speakers_cache()
                voice_key = config.voice.lower()
                if voice_key not in self._speech_service.uploaded_speakers:
                    await self._send_error(
                        websocket,
                        f"Voice '{config.voice}' is not cached. "
                        "Provide ref_audio (and optionally ref_text) to register a voice clone first.",
                    )
                    return

            # ref_audio without ref_text can only use speaker embedding.
            if (
                (config.task_type or "").lower() == "base"
                and config.ref_audio
                and not config.ref_text
                and not config.x_vector_only_mode
            ):
                config.x_vector_only_mode = True

            # Determine text buffering strategy.
            # ICL mode interleaves ref_code with text in the prompt, so it
            # requires all text upfront.  x_vector_only mode only uses the
            # speaker embedding (like CustomVoice) and can generate sentences
            # independently — enabling per-sentence streaming with lower TTFA.
            icl_mode = False
            if (config.task_type or "").lower() == "base":
                if config.x_vector_only_mode:
                    icl_mode = False
                elif config.voice_name:
                    voice_key = config.voice_name.lower()
                    speaker_info = self._speech_service.uploaded_speakers.get(voice_key, {})
                    icl_mode = bool(speaker_info.get("ref_text"))
                else:
                    icl_mode = config.ref_text is not None

            splitter = SentenceSplitter()
            sentence_index = 0
            text_buffer: list[str] = []

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

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue

                msg_type = msg.get("type")

                if msg_type == "input.text":
                    text = msg.get("text", "")
                    if icl_mode:
                        text_buffer.append(text)
                    else:
                        sentences = splitter.add_text(text)
                        for sentence in sentences:
                            await self._generate_and_send(websocket, config, sentence, sentence_index)
                            sentence_index += 1

                elif msg_type == "input.done":
                    if icl_mode:
                        full_text = "".join(text_buffer).strip()
                        if full_text:
                            await self._generate_and_send(websocket, config, full_text, sentence_index)
                            sentence_index += 1
                    else:
                        remaining = splitter.flush()
                        if remaining:
                            await self._generate_and_send(websocket, config, remaining, sentence_index)
                            sentence_index += 1

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
            logger.info("Streaming speech: client disconnected — in-flight requests aborted")
        except Exception as e:
            if "close message has been sent" in str(e):
                logger.info("Streaming speech: client disconnected (close already sent)")
            else:
                logger.exception("Streaming speech session error: %s", e)
                try:
                    await self._send_error(websocket, f"Internal error: {e}")
                except Exception:
                    pass

    async def _receive_config(self, websocket: WebSocket) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message.

        Also handles ``voice.delete`` as a one-shot command: deletes the
        cached voice, sends a response, and returns None so the session
        closes cleanly.
        """
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        # Handle voice.delete as a one-shot command
        if msg.get("type") == "voice.delete":
            voice_name = msg.get("voice_name", "")
            if not voice_name:
                await self._send_error(websocket, "voice.delete requires 'voice_name'")
                return None
            deleted = self._speech_service.delete_voice_clone(voice_name)
            if deleted:
                await websocket.send_json(
                    {"type": "voice.deleted", "voice_name": voice_name}
                )
            else:
                await self._send_error(
                    websocket,
                    f"Voice '{voice_name}' not found in cache",
                )
            return None

        # Handle voice.list as a one-shot command
        if msg.get("type") == "voice.list":
            self._speech_service._refresh_uploaded_speakers_cache()
            builtin = sorted(self._speech_service.supported_speakers - set(self._speech_service.uploaded_speakers.keys()))
            uploaded = []
            for key, info in self._speech_service.uploaded_speakers.items():
                uploaded.append({
                    "name": info.get("name", key),
                    "cache_status": info.get("cache_status", "pending"),
                    "has_ref_text": bool(info.get("ref_text")),
                    "created_at": info.get("created_at", 0),
                })
            await websocket.send_json({
                "type": "voice.list",
                "voices": builtin,
                "uploaded_voices": uploaded,
            })
            return None

        if msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, voice.delete, or voice.list, got: {msg.get('type')}",
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
        """Generate audio for a single sentence and stream it over WebSocket.

        Audio is sent as a series of raw PCM binary frames as each codec chunk
        is produced by the model, rather than waiting for the full sentence to
        be generated.  The client assembles the frames into a final audio file
        using the sample_rate and chunk_count fields on the audio.done message.

        On client disconnect, the audio generator is explicitly closed so that
        the engine abort propagates immediately to all pipeline stages.
        """
        response_format = config.response_format or "wav"

        await websocket.send_json(
            {
                "type": "audio.start",
                "sentence_index": sentence_index,
                "sentence_text": sentence_text,
                "format": response_format,
            }
        )

        sample_rate: int | None = None
        chunk_count = 0
        disconnected = False
        audio_gen = None

        try:
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
                ref_audio=config.ref_audio,
                ref_text=config.ref_text,
                x_vector_only_mode=config.x_vector_only_mode,
            )

            audio_gen = self._speech_service._generate_audio_stream(request)
            async for pcm_bytes, sr in audio_gen:
                if sample_rate is None:
                    sample_rate = sr
                try:
                    await websocket.send_bytes(pcm_bytes)
                except (WebSocketDisconnect, RuntimeError):
                    disconnected = True
                    break
                chunk_count += 1

        except WebSocketDisconnect:
            disconnected = True
        except Exception as e:
            logger.error("Generation failed for sentence %d: %s", sentence_index, e)
            await self._send_error(
                websocket,
                f"Generation failed for sentence {sentence_index}: {e}",
            )
        finally:
            if audio_gen is not None:
                await audio_gen.aclose()

        if disconnected:
            raise WebSocketDisconnect()

        await websocket.send_json(
            {
                "type": "audio.done",
                "sentence_index": sentence_index,
                "sample_rate": sample_rate or 24000,
                "chunk_count": chunk_count,
            }
        )

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
            pass  # Connection may already be closed
