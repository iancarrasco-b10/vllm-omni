"""WebSocket streaming TTS handler for incremental text input.

Implements a WebSocket endpoint that accepts text incrementally (simulating
real-time STT output) and streams audio back per sentence.

Protocol (Client -> Server):
    {"type": "session.config", ...}   - Session configuration
    {"type": "input.text", "text": "..."} - Incremental text
    {"type": "input.done"}            - Flush remaining buffer

Protocol (Server -> Client):
    {"type": "audio.start", "sentence_index": N, "sentence_text": "...", "format": "pcm"}
    Binary frame: audio chunk bytes (multiple per sentence with streaming)
    {"type": "audio.done", "sentence_index": N}
    {"type": "session.done", "total_sentences": N}
    {"type": "error", "message": "..."}
"""

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.text_splitter import SentenceSplitter

logger = init_logger(__name__)


class OmniStreamingSpeechHandler:
    """WebSocket handler for streaming text-to-speech.

    Accepts incremental text via WebSocket, splits into sentences,
    and streams audio back per sentence using the TTS engine.
    """

    def __init__(self, speech_serving):
        """Initialize with a reference to the speech serving instance.

        Args:
            speech_serving: OmniOpenAIServingSpeech instance
        """
        self._speech = speech_serving

    async def handle(self, websocket: WebSocket):
        """Main WebSocket handler loop."""
        await websocket.accept()

        config = None
        splitter = SentenceSplitter()
        sentence_idx = 0

        try:
            # Wait for session config
            config = await self._wait_for_config(websocket, timeout=10.0)
            if config is None:
                await self._send_error(websocket, "No session config received within timeout")
                await websocket.close()
                return

            # Process messages
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    await self._send_error(websocket, "Idle timeout (30s)")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON")
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "input.text":
                    text = msg.get("text", "")
                    if text:
                        sentences = splitter.feed(text)
                        for sentence in sentences:
                            await self._process_sentence(
                                websocket, config, sentence, sentence_idx
                            )
                            sentence_idx += 1

                elif msg_type == "input.done":
                    remaining = splitter.flush()
                    if remaining:
                        await self._process_sentence(
                            websocket, config, remaining, sentence_idx
                        )
                        sentence_idx += 1

                    await websocket.send_json({
                        "type": "session.done",
                        "total_sentences": sentence_idx,
                    })
                    break

                elif msg_type == "session.config":
                    # Allow reconfiguration mid-session
                    config = self._parse_config(msg)

                else:
                    await self._send_error(websocket, f"Unknown message type: {msg_type}")

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.exception("WebSocket handler error: %s", e)
            try:
                await self._send_error(websocket, str(e))
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    async def _wait_for_config(
        self, websocket: WebSocket, timeout: float = 10.0
    ) -> StreamingSpeechSessionConfig | None:
        """Wait for initial session configuration message."""
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
            msg = json.loads(raw)
            if msg.get("type") == "session.config":
                return self._parse_config(msg)
            return None
        except (asyncio.TimeoutError, json.JSONDecodeError):
            return None

    def _parse_config(self, msg: dict) -> StreamingSpeechSessionConfig:
        """Parse session configuration from message."""
        return StreamingSpeechSessionConfig(
            voice=msg.get("voice", "Vivian"),
            task_type=msg.get("task_type", "CustomVoice"),
            language=msg.get("language", "Auto"),
            instructions=msg.get("instructions"),
            response_format=msg.get("response_format", "pcm"),
            ref_audio=msg.get("ref_audio"),
            ref_text=msg.get("ref_text"),
            x_vector_only_mode=msg.get("x_vector_only_mode"),
            max_new_tokens=msg.get("max_new_tokens"),
        )

    async def _process_sentence(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence: str,
        sentence_idx: int,
    ):
        """Generate and stream audio for a single sentence."""
        try:
            # Send audio.start
            await websocket.send_json({
                "type": "audio.start",
                "sentence_index": sentence_idx,
                "sentence_text": sentence,
                "format": config.response_format,
            })

            # Build request from config
            request = OpenAICreateSpeechRequest(
                input=sentence,
                voice=config.voice,
                task_type=config.task_type,
                language=config.language,
                instructions=config.instructions,
                response_format=config.response_format,
                ref_audio=config.ref_audio,
                ref_text=config.ref_text,
                x_vector_only_mode=config.x_vector_only_mode,
                max_new_tokens=config.max_new_tokens,
                stream=True,
            )

            # Stream audio chunks
            async for chunk_bytes, is_finished in self._speech._generate_audio_chunks(request):
                if chunk_bytes:
                    await websocket.send_bytes(chunk_bytes)
                if is_finished:
                    break

            # Send audio.done
            await websocket.send_json({
                "type": "audio.done",
                "sentence_index": sentence_idx,
            })

        except Exception as e:
            logger.warning(
                "Error processing sentence %d: %s", sentence_idx, e
            )
            await self._send_error(
                websocket,
                f"Error generating audio for sentence {sentence_idx}: {e}",
            )

    async def _send_error(self, websocket: WebSocket, message: str):
        """Send an error message to the client."""
        try:
            await websocket.send_json({
                "type": "error",
                "message": message,
            })
        except Exception:
            pass
