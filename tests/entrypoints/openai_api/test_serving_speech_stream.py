"""Integration tests for WebSocket streaming speech handler."""

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self._send_queue: list = []
        self._recv_queue: asyncio.Queue = asyncio.Queue()
        self.accepted = False
        self.closed = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.close_code = code

    async def send_json(self, data: dict):
        self._send_queue.append(("json", data))

    async def send_bytes(self, data: bytes):
        self._send_queue.append(("bytes", data))

    async def send_text(self, data: str):
        self._send_queue.append(("text", data))

    async def receive_text(self) -> str:
        return await self._recv_queue.get()

    def inject_message(self, msg: str | dict):
        """Inject a message to be received by the handler."""
        if isinstance(msg, dict):
            msg = json.dumps(msg)
        self._recv_queue.put_nowait(msg)

    def get_sent_messages(self) -> list:
        return list(self._send_queue)

    def get_json_messages(self) -> list[dict]:
        return [data for kind, data in self._send_queue if kind == "json"]

    def get_binary_messages(self) -> list[bytes]:
        return [data for kind, data in self._send_queue if kind == "bytes"]


class TestOmniStreamingSpeechHandler:
    @pytest.fixture
    def mock_speech_serving(self):
        serving = MagicMock()

        async def mock_generate_chunks(request):
            yield b"\x00\x01\x02\x03", False
            yield b"\x04\x05\x06\x07", True

        serving._generate_audio_chunks = mock_generate_chunks
        serving.create_error_response = MagicMock(return_value=MagicMock())
        return serving

    @pytest.fixture
    def handler(self, mock_speech_serving):
        return OmniStreamingSpeechHandler(mock_speech_serving)

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, handler):
        """Test basic session: config -> text -> done."""
        ws = MockWebSocket()

        # Queue up messages
        ws.inject_message({
            "type": "session.config",
            "voice": "Vivian",
            "task_type": "CustomVoice",
            "language": "Auto",
            "response_format": "pcm",
        })
        ws.inject_message({
            "type": "input.text",
            "text": "Hello world. ",
        })
        ws.inject_message({"type": "input.done"})

        await handler.handle(ws)

        assert ws.accepted
        json_msgs = ws.get_json_messages()
        types = [m["type"] for m in json_msgs]

        # Should have at least audio.start, audio.done, session.done
        assert "audio.start" in types
        assert "audio.done" in types
        assert "session.done" in types

    @pytest.mark.asyncio
    async def test_multi_sentence(self, handler):
        """Test processing multiple sentences."""
        ws = MockWebSocket()

        ws.inject_message({
            "type": "session.config",
            "voice": "Vivian",
        })
        ws.inject_message({
            "type": "input.text",
            "text": "First sentence. Second sentence. ",
        })
        ws.inject_message({"type": "input.done"})

        await handler.handle(ws)

        json_msgs = ws.get_json_messages()
        start_msgs = [m for m in json_msgs if m["type"] == "audio.start"]
        done_msgs = [m for m in json_msgs if m["type"] == "audio.done"]

        # Should have audio.start/done for each sentence
        assert len(start_msgs) >= 2
        assert len(done_msgs) >= 2

    @pytest.mark.asyncio
    async def test_incremental_text(self, handler):
        """Test incremental text input (simulating STT)."""
        ws = MockWebSocket()

        ws.inject_message({
            "type": "session.config",
            "voice": "Vivian",
        })
        # Send text word by word
        ws.inject_message({"type": "input.text", "text": "Hello "})
        ws.inject_message({"type": "input.text", "text": "world. "})
        ws.inject_message({"type": "input.text", "text": "How are you?"})
        ws.inject_message({"type": "input.done"})

        await handler.handle(ws)

        json_msgs = ws.get_json_messages()
        session_done = [m for m in json_msgs if m["type"] == "session.done"]
        assert len(session_done) == 1

    @pytest.mark.asyncio
    async def test_flush_on_done(self, handler):
        """Test that input.done flushes remaining buffer."""
        ws = MockWebSocket()

        ws.inject_message({
            "type": "session.config",
            "voice": "Vivian",
        })
        # Text without sentence ending
        ws.inject_message({"type": "input.text", "text": "No period here"})
        ws.inject_message({"type": "input.done"})

        await handler.handle(ws)

        json_msgs = ws.get_json_messages()
        start_msgs = [m for m in json_msgs if m["type"] == "audio.start"]
        # The remaining buffer should be flushed as a sentence
        assert len(start_msgs) == 1
        assert start_msgs[0]["sentence_text"] == "No period here"

    @pytest.mark.asyncio
    async def test_missing_config_timeout(self, handler):
        """Test that missing config results in error."""
        ws = MockWebSocket()

        # Send non-config message first
        ws.inject_message({"type": "input.text", "text": "Hello"})

        # Override timeout to be very short for testing
        with patch.object(handler, '_wait_for_config', return_value=None):
            await handler.handle(ws)

        json_msgs = ws.get_json_messages()
        error_msgs = [m for m in json_msgs if m["type"] == "error"]
        assert len(error_msgs) >= 1

    @pytest.mark.asyncio
    async def test_invalid_json(self, handler):
        """Test that invalid JSON is handled gracefully."""
        ws = MockWebSocket()

        ws.inject_message({
            "type": "session.config",
            "voice": "Vivian",
        })
        ws.inject_message("not valid json {{{")
        ws.inject_message({"type": "input.done"})

        await handler.handle(ws)

        json_msgs = ws.get_json_messages()
        error_msgs = [m for m in json_msgs if m["type"] == "error"]
        assert len(error_msgs) >= 1

    @pytest.mark.asyncio
    async def test_binary_audio_sent(self, handler):
        """Test that binary audio chunks are sent."""
        ws = MockWebSocket()

        ws.inject_message({
            "type": "session.config",
            "voice": "Vivian",
            "response_format": "pcm",
        })
        ws.inject_message({"type": "input.text", "text": "Hello world. "})
        ws.inject_message({"type": "input.done"})

        await handler.handle(ws)

        binary_msgs = ws.get_binary_messages()
        assert len(binary_msgs) >= 1
