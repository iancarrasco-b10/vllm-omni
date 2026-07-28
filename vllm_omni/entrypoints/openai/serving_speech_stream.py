"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket and supports buffered generation,
token-level extension, or explicit sentence commits to one persistent TTS
request.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.commit", "commit_id": "..."} # Commit buffered text
        {"type": "input.done"}            # End of input

    Server -> Client (default, word_timestamps=false):
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        {"type": "input.committed", "commit_id": "...", "sentence_index": 0}
        <binary frame: audio bytes>
        ...
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}

    Server -> Client (when word_timestamps=true):
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "pcm"}
        {"type": "audio.chunk", "sentence_index": 0, "chunk_id": 0, "audio_b64": "<base64 PCM>", "timestamps": null}
        ...
        {"type": "audio.chunk", "audio_b64": "", "timestamps": [{"word", "start_ms", "end_ms"}, ...]}
        {"type": "audio.done", "sentence_index": 0}
        # Audio is JSON base64 PCM (not binary). A trailing empty-audio chunk carries the
        # full sentence-relative alignment. timestamps: list = aligned, [] = silence, null = failed.
"""

import asyncio
import base64
import json
from contextlib import aclosing

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechInputCommit,
    StreamingSpeechInputCommitted,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.utils.forced_aligner import ForcedAlignerLoadError
from vllm_omni.utils.forced_aligner import align as forced_align

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_BYTES_PER_SAMPLE = 2  # 16-bit mono PCM
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024
_MAX_BUFFER_SIZE = 100_000  # max accumulated text chars for token-level mode


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is buffered until input.done, and audio is generated once
    for the buffered input using the existing OmniOpenAIServingSpeech pipeline.

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

            # Validate model if specified
            if config.model and hasattr(self._speech_service, "_check_model"):
                error = await self._speech_service._check_model(
                    OpenAICreateSpeechRequest(input="ping", model=config.model)
                )
                if error is not None:
                    await self._send_error(websocket, str(error))
                    return

            # Single-request streaming modes require engine-level text updates.
            if config.streaming_mode in ("token_level", "sentence_commit"):
                server_enabled = getattr(
                    self._speech_service.engine_client.model_config,
                    "streaming_text_enabled",
                    False,
                )
                if not server_enabled:
                    await self._send_error(
                        websocket,
                        f"{config.streaming_mode} streaming is disabled on this server "
                        "(set streaming_text_enabled: true in stage config to enable)",
                    )
                    return
                if config.streaming_mode == "sentence_commit":
                    await self._handle_sentence_commit_session(websocket, config)
                else:
                    await self._handle_token_level_session(websocket, config)
                return

            text_parts: list[str] = []

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
                    text_parts.append(text)

                elif msg_type == "input.done":
                    full_text = "".join(text_parts).strip()
                    total_sentences = 0
                    if full_text:
                        await self._generate_and_send(websocket, config, full_text, 0)
                        total_sentences = 1

                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "total_sentences": total_sentences,
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

    async def _handle_token_level_session(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        """True engine-level streaming: text arrives incrementally, audio
        generation starts immediately and runs concurrently.

        1. Collect minimum text buffer (MIN_INITIAL_CHARS)
        2. Submit initial TTS request to the engine (audio generation starts)
        3. Concurrently: read text from WebSocket -> extend_text messages to orchestrator
                         stream audio from engine -> send to WebSocket
        4. On input.done -> send text_finished signal
        """
        response_format = config.response_format or "pcm"
        all_text = ""
        input_done = False

        MIN_INITIAL_CHARS = 60
        while len(all_text) < MIN_INITIAL_CHARS:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=self._idle_timeout,
                )
            except asyncio.TimeoutError:
                await self._send_error(websocket, "Idle timeout")
                return
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type")
            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                all_text += text
                if len(all_text) > _MAX_BUFFER_SIZE:
                    await self._send_error(websocket, "input.text buffer exceeded limit")
                    return
            elif msg_type == "input.done":
                input_done = True
                break

        if not all_text.strip():
            await websocket.send_json({"type": "session.done", "total_sentences": 0})
            return

        initial_request = OpenAICreateSpeechRequest(
            input=all_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            non_streaming_mode=False,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=True,
            streaming_text_input=True,
            streaming_drain_max_steps=config.streaming_drain_max_steps,
        )
        request_id, generator, _ = await self._speech_service._prepare_speech_generation(
            initial_request,
        )

        start_payload: dict = {
            "type": "audio.start",
            "sentence_index": 0,
            "sentence_text": all_text[:80] + ("..." if len(all_text) > 80 else ""),
            "format": response_format,
        }
        if response_format == "pcm":
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False

        _extend_count = 0
        _extend_chars_total = 0

        finished_sent = False
        input_error: str | None = None

        def _send_extend(new_text: str, finished: bool) -> None:
            nonlocal _extend_count, _extend_chars_total
            _extend_count += 1
            _extend_chars_total += len(new_text) if new_text else 0
            logger.info(
                "[WS][extend] req=%s chunk#%d text_len=%d finished=%s cumulative_chars=%d",
                request_id,
                _extend_count,
                len(new_text) if new_text else 0,
                finished,
                _extend_chars_total,
            )
            self._speech_service.engine_client.extend_streaming_text(
                request_id,
                new_text=new_text,
                finished=finished,
            )

        def _finish_text() -> None:
            nonlocal finished_sent
            if not finished_sent:
                finished_sent = True
                _send_extend("", finished=True)

        async def feed_text() -> None:
            nonlocal all_text, input_error
            if input_done:
                _finish_text()
                return
            try:
                text_chars_total = len(all_text)
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=self._idle_timeout,
                        )
                    except asyncio.TimeoutError:
                        break
                    if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                        input_error = "input.text message too large"
                        break
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    msg_type = msg.get("type")
                    if msg_type == "input.text":
                        new_text = msg.get("text", "")
                        if not isinstance(new_text, str):
                            input_error = "input.text requires a string value"
                            break
                        if not new_text:
                            continue
                        text_chars_total += len(new_text)
                        if text_chars_total > _MAX_BUFFER_SIZE:
                            input_error = "input.text buffer exceeded limit"
                            break
                        _send_extend(new_text, finished=False)
                    elif msg_type == "input.done":
                        break
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.debug("feed_text error", exc_info=True)
                input_error = "input.text streaming failed"
            if input_error is None:
                _finish_text()

        text_task = asyncio.create_task(feed_text())

        try:
            async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                async for chunk in stream:
                    total_bytes += len(chunk)
                    await websocket.send_bytes(chunk)
            if not text_task.done():
                text_task.cancel()
                try:
                    await text_task
                except asyncio.CancelledError:
                    pass
            if input_error is None:
                _finish_text()
        except WebSocketDisconnect:
            text_task.cancel()
            try:
                await self._speech_service.engine_client.abort(request_id)
            except Exception:
                pass
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Token-level generation failed: %s", e)
            await self._send_error(websocket, f"Generation failed: {e}")
        finally:
            if not text_task.done():
                text_task.cancel()
                try:
                    await text_task
                except asyncio.CancelledError:
                    pass
                except WebSocketDisconnect:
                    pass
            if input_error is not None:
                generation_failed = True
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    pass
                await self._send_error(websocket, input_error)
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": 0,
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
                await websocket.send_json(
                    {
                        "type": "session.done",
                        "total_sentences": 1,
                    }
                )
            except Exception:
                pass

    async def _handle_sentence_commit_session(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        """Stream committed sentences through one persistent TTS request.

        ``input.text`` messages are buffered until ``input.commit``. The first
        commit creates the request; later commits extend it without resetting
        model state. Only ``input.done`` marks the engine request as finished.
        """
        response_format = config.response_format or "pcm"
        pending_parts: list[str] = []
        total_input_chars = 0
        sentence_count = 0
        input_done = False
        first_commit: StreamingSpeechInputCommit | None = None
        first_text = ""

        async def receive_message() -> dict | None:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=self._idle_timeout,
                )
            except asyncio.TimeoutError:
                await self._send_error(websocket, "Idle timeout")
                return None
            if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                await self._send_error(websocket, "input.text message too large")
                return {}
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON message")
                return {}
            if not isinstance(msg, dict):
                await self._send_error(websocket, "WebSocket messages must be JSON objects")
                return {}
            return msg

        # Wait for the first commit before creating the engine request.
        while first_commit is None and not input_done:
            msg = await receive_message()
            if msg is None:
                return
            if not msg:
                continue
            msg_type = msg.get("type")
            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                total_input_chars += len(text)
                if total_input_chars > _MAX_BUFFER_SIZE:
                    await self._send_error(websocket, "input.text buffer exceeded limit")
                    return
                pending_parts.append(text)
            elif msg_type in ("input.commit", "input.done"):
                text = "".join(pending_parts)
                if not text.strip():
                    if msg_type == "input.done":
                        await websocket.send_json({"type": "session.done", "total_sentences": 0})
                        return
                    await self._send_error(websocket, "input.commit requires buffered text")
                    continue
                if msg_type == "input.commit":
                    try:
                        first_commit = StreamingSpeechInputCommit.model_validate(msg)
                    except ValidationError as e:
                        await self._send_error(websocket, f"Invalid input.commit: {e}")
                        continue
                else:
                    first_commit = StreamingSpeechInputCommit(type="input.commit")
                    input_done = True
                first_text = text
                pending_parts.clear()
            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")

        if first_commit is None:
            return

        initial_request = OpenAICreateSpeechRequest(
            input=first_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            non_streaming_mode=False,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=True,
            streaming_text_input=True,
            streaming_drain_max_steps=config.streaming_drain_max_steps,
        )
        request_id, generator, _ = await self._speech_service._prepare_speech_generation(initial_request)

        sentence_count = 1
        start_payload: dict = {
            "type": "audio.start",
            "sentence_index": 0,
            "sentence_text": first_text[:80] + ("..." if len(first_text) > 80 else ""),
            "format": response_format,
        }
        if response_format == "pcm":
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        finished_sent = False
        input_error: str | None = None

        async def send_extend(new_text: str, *, finished: bool) -> None:
            accepted = await self._speech_service.engine_client.extend_streaming_text_confirmed_async(
                request_id,
                new_text=new_text,
                finished=finished,
            )
            if not accepted:
                raise RuntimeError("TTS engine rejected the streaming text update")

        async def wait_for_engine_acceptance() -> None:
            deadline = asyncio.get_running_loop().time() + 10.0
            while True:
                accepted = await self._speech_service.engine_client.extend_streaming_text_confirmed_async(
                    request_id,
                    new_text="",
                    finished=False,
                )
                if accepted:
                    return
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError("TTS engine did not accept the initial request")
                await asyncio.sleep(0.01)

        async def finish_text() -> None:
            nonlocal finished_sent
            if not finished_sent:
                await send_extend("", finished=True)
                finished_sent = True

        async def commit_pending(commit: StreamingSpeechInputCommit) -> bool:
            nonlocal sentence_count
            text = "".join(pending_parts)
            if not text.strip():
                await self._send_error(websocket, "input.commit requires buffered text")
                return False
            pending_parts.clear()
            await send_extend(text, finished=False)
            await websocket.send_json(
                StreamingSpeechInputCommitted(
                    commit_id=commit.commit_id,
                    sentence_index=sentence_count,
                    chars_committed=len(text),
                ).model_dump()
            )
            sentence_count += 1
            return True

        async def feed_commits() -> None:
            nonlocal input_done, input_error, total_input_chars
            if input_done:
                await finish_text()
                return
            try:
                while True:
                    msg = await receive_message()
                    if msg is None:
                        input_error = "Idle timeout"
                        break
                    if not msg:
                        continue
                    msg_type = msg.get("type")
                    if msg_type == "input.text":
                        text = msg.get("text", "")
                        if not isinstance(text, str):
                            await self._send_error(websocket, "input.text requires a string value")
                            continue
                        total_input_chars += len(text)
                        if total_input_chars > _MAX_BUFFER_SIZE:
                            input_error = "input.text buffer exceeded limit"
                            break
                        pending_parts.append(text)
                    elif msg_type == "input.commit":
                        try:
                            commit = StreamingSpeechInputCommit.model_validate(msg)
                        except ValidationError as e:
                            await self._send_error(websocket, f"Invalid input.commit: {e}")
                            continue
                        await commit_pending(commit)
                    elif msg_type == "input.done":
                        input_done = True
                        if any(part.strip() for part in pending_parts):
                            await commit_pending(StreamingSpeechInputCommit(type="input.commit"))
                        await finish_text()
                        break
                    else:
                        await self._send_error(websocket, f"Unknown message type: {msg_type}")
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.debug("sentence commit feed error", exc_info=True)
                input_error = "sentence commit streaming failed"

        engine_accepted = asyncio.Event()

        async def stream_audio() -> None:
            nonlocal total_bytes
            async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                async for chunk in stream:
                    await engine_accepted.wait()
                    total_bytes += len(chunk)
                    await websocket.send_bytes(chunk)

        audio_task = asyncio.create_task(stream_audio())
        input_task: asyncio.Task | None = None
        try:
            # Start consuming the async generator first so its add_request message
            # is enqueued before the acceptance probe.
            await asyncio.sleep(0)
            await wait_for_engine_acceptance()
            await websocket.send_json(
                StreamingSpeechInputCommitted(
                    commit_id=first_commit.commit_id,
                    sentence_index=0,
                    chars_committed=len(first_text),
                ).model_dump()
            )
            engine_accepted.set()

            input_task = asyncio.create_task(feed_commits())
            done, _ = await asyncio.wait(
                {input_task, audio_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if input_task in done:
                await input_task
                if input_error is not None:
                    try:
                        await self._speech_service.engine_client.abort(request_id)
                    finally:
                        if not audio_task.done():
                            audio_task.cancel()
                    try:
                        await audio_task
                    except asyncio.CancelledError:
                        pass
                else:
                    await audio_task
            else:
                await audio_task
                if not input_task.done():
                    input_error = "Generation ended before input.done"
                    input_task.cancel()
                    try:
                        await input_task
                    except asyncio.CancelledError:
                        pass
        except WebSocketDisconnect:
            if input_task is not None and not input_task.done():
                input_task.cancel()
            if not audio_task.done():
                audio_task.cancel()
            try:
                await self._speech_service.engine_client.abort(request_id)
            except Exception:
                pass
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Sentence-commit generation failed: %s", e)
            try:
                await self._speech_service.engine_client.abort(request_id)
            except Exception:
                pass
            await self._send_error(websocket, f"Generation failed: {e}")
        finally:
            if input_task is not None and not input_task.done():
                input_task.cancel()
            if input_task is not None:
                try:
                    await input_task
                except (asyncio.CancelledError, Exception):
                    pass
            if not audio_task.done():
                audio_task.cancel()
            try:
                await audio_task
            except (asyncio.CancelledError, Exception):
                pass
            if input_error is not None:
                generation_failed = True
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    pass
                await self._send_error(websocket, input_error)
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": 0,
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
                await websocket.send_json(
                    {
                        "type": "session.done",
                        "total_sentences": sentence_count,
                    }
                )
            except Exception:
                pass

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"

        # Reject unmet word-timestamps preconditions early with a clear reason.
        if config.word_timestamps:
            if self._speech_service.forced_aligner_config is None:
                await self._send_error(
                    websocket,
                    "word_timestamps=true but the server was launched without "
                    "--forced-aligner; either restart the server with that flag "
                    "or set word_timestamps=false in session.config.",
                )
                return
            if not (config.stream_audio and response_format == "pcm"):
                await self._send_error(
                    websocket,
                    "word_timestamps=true requires stream_audio=true and "
                    "response_format='pcm' (the aligner consumes raw PCM).",
                )
                return

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
            non_streaming_mode=config.non_streaming_mode,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=config.stream_audio,
            word_timestamps=config.word_timestamps,
        )

        start_payload = {
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        }
        if config.stream_audio and response_format == "pcm":
            # Nominal stream rate; each audio.chunk carries the authoritative
            # per-chunk sample_rate.
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        if config.word_timestamps:
            start_payload["word_timestamps"] = True
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        request_id = None
        try:
            if config.stream_audio:
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
                if config.word_timestamps:
                    total_bytes = await self._stream_audio_with_alignments(
                        websocket=websocket,
                        request_id=request_id,
                        generator=generator,
                        sentence_text=sentence_text,
                        sentence_index=sentence_index,
                        language=config.language,
                    )
                else:
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

    async def _stream_audio_with_alignments(
        self,
        *,
        websocket: WebSocket,
        request_id: str,
        generator,
        sentence_text: str,
        sentence_index: int,
        language: str | None = None,
    ) -> int:
        """Stream PCM as JSON ``audio.chunk`` frames, aligned per sentence.

        Forward each PCM chunk live (``timestamps: null``) while buffering the
        sentence audio, then run the forced aligner once over the whole
        sentence and emit a final empty-audio ``audio.chunk`` with the word
        timestamps. On aligner failure timestamps is ``null``; for silence it
        is ``[]`` (audio always flows regardless).
        """
        aligner_config = self._speech_service.forced_aligner_config
        assert aligner_config is not None  # gated by the precondition check

        sentence_audio = bytearray()
        total_bytes = 0
        sample_rate = _PCM_SAMPLE_RATE
        chunk_id = 0

        async def send_chunk(
            chunk: bytes,
            chunk_sample_rate: int,
            timestamps_payload: list[dict] | None,
            chunk_start_ms: int,
            chunk_end_ms: int,
        ) -> None:
            nonlocal chunk_id
            await websocket.send_json(
                {
                    "type": "audio.chunk",
                    "sentence_index": sentence_index,
                    "chunk_id": chunk_id,
                    "chunk_start_ms": chunk_start_ms,
                    "chunk_end_ms": chunk_end_ms,
                    "sample_rate": chunk_sample_rate,
                    "audio_b64": base64.b64encode(chunk).decode("ascii"),
                    "timestamps": timestamps_payload,
                }
            )
            chunk_id += 1

        async with aclosing(
            self._speech_service._generate_pcm_chunks(
                generator,
                request_id,
                include_sample_rate=True,
            )
        ) as stream:
            async for chunk, chunk_sample_rate in stream:
                sample_rate = chunk_sample_rate
                chunk_start_ms = int(round((len(sentence_audio) / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
                sentence_audio.extend(chunk)
                chunk_end_ms = int(round((len(sentence_audio) / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
                total_bytes += len(chunk)
                # Audio first, timestamps after the whole sentence is aligned.
                await send_chunk(chunk, chunk_sample_rate, None, chunk_start_ms, chunk_end_ms)

        # Single alignment pass over the full sentence, then emit timestamps.
        # A load/config failure is permanent, so surface the reason once; audio
        # has already streamed, so the trailing frame still carries null.
        try:
            timestamps_payload = await self._align_sentence(
                audio=bytes(sentence_audio),
                text=sentence_text,
                sample_rate=sample_rate,
                config=aligner_config,
                language=language,
            )
        except ForcedAlignerLoadError as exc:
            await self._send_error(websocket, f"forced aligner unavailable: {exc}")
            timestamps_payload = None
        sentence_end_ms = int(round((len(sentence_audio) / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
        await send_chunk(b"", sample_rate, timestamps_payload, 0, sentence_end_ms)

        return total_bytes

    @staticmethod
    async def _align_sentence(
        *,
        audio: bytes,
        text: str,
        sample_rate: int,
        config,
        language: str | None = None,
    ) -> list[dict] | None:
        """Convert a sentence alignment into JSON word-timestamp dicts.

        Returns ``None`` on aligner failure, ``[]`` when it ran but produced no
        tokens. Monotonic, non-overlapping bounds are guaranteed by the decoder.
        ``language`` is forwarded to word segmentation.
        """
        aligned = await forced_align(
            audio=audio,
            text=text,
            sample_rate=sample_rate,
            config=config,
            language=language,
        )
        if aligned is None:
            return None
        return [{"word": ts.word, "start_ms": ts.start_ms, "end_ms": ts.end_ms} for ts in aligned]

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
