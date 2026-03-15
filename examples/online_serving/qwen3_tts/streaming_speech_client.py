"""WebSocket client for streaming text-input TTS.

Connects to the /v1/audio/speech/stream endpoint, sends text incrementally
(simulating real-time STT output), and saves per-sentence audio files.

Usage:
    # Send full text at once
    python streaming_speech_client.py --text "Hello world. How are you? I am fine."

    # Simulate STT: send text word-by-word with delay
    python streaming_speech_client.py \
        --text "Hello world. How are you? I am fine." \
        --simulate-stt --stt-delay 0.1

    # VoiceDesign task
    python streaming_speech_client.py \
        --text "Today is a great day. The weather is nice." \
        --task-type VoiceDesign \
        --instructions "A cheerful young female voice"

    # Base task (voice cloning)
    python streaming_speech_client.py \
        --text "Hello world. How are you?" \
        --task-type Base \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of reference audio"

Requirements:
    pip install websockets
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import struct
import time

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


async def stream_tts(
    url: str,
    text: str,
    config: dict,
    output_dir: str,
    simulate_stt: bool = False,
    stt_delay: float = 0.1,
) -> None:
    """Connect to the streaming TTS endpoint and process audio responses."""
    os.makedirs(output_dir, exist_ok=True)

    async with websockets.connect(url) as ws:
        # 1. Send session config
        config_msg = {"type": "session.config", **config}
        await ws.send(json.dumps(config_msg))
        print(f"Sent session config: {config}")

        # 2. Send text (either all at once or word-by-word)
        async def send_text():
            if simulate_stt:
                words = text.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input.text",
                                "text": chunk,
                            }
                        )
                    )
                    print(f"  Sent: {chunk!r}")
                    await asyncio.sleep(stt_delay)
            else:
                await ws.send(
                    json.dumps(
                        {
                            "type": "input.text",
                            "text": text,
                        }
                    )
                )
                print(f"Sent full text: {text!r}")

            # 3. Signal end of input
            await ws.send(json.dumps({"type": "input.done"}))
            print("Sent input.done")

        # Run sender and receiver concurrently
        sender_task = asyncio.create_task(send_text())

        response_format = config.get("response_format", "wav")
        current_sentence_index = 0
        current_chunks: list[bytes] = []
        session_t0 = time.perf_counter()
        sentence_t0: float | None = None
        first_audio_received = False
        total_bytes_written = 0

        # Combined output file: append all PCM chunks as they arrive, wrapped in WAV
        combined_path = os.path.join(output_dir, "combined.wav")
        combined_file = open(combined_path, "wb")
        # Write WAV header with placeholder sizes (patched on close)
        sample_rate = 24000
        bits_per_sample = 16
        num_channels = 1
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        combined_file.write(struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 0xFFFFFFFF,  # placeholder RIFF size
            b"WAVE",
            b"fmt ", 16,  # PCM fmt chunk
            1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
            b"data", 0xFFFFFFFF,  # placeholder data size
        ))

        try:
            while True:
                message = await ws.recv()

                if isinstance(message, bytes):
                    if not first_audio_received:
                        first_audio_received = True
                        ttfa = time.perf_counter() - session_t0
                        sentence_ttfa = time.perf_counter() - sentence_t0 if sentence_t0 else ttfa
                        print(f"  TTFA: {ttfa:.3f}s (session) / {sentence_ttfa:.3f}s (sentence)")
                    current_chunks.append(message)
                    # Append to combined file immediately
                    combined_file.write(message)
                    combined_file.flush()
                    total_bytes_written += len(message)
                    print(f"  Received audio chunk for sentence {current_sentence_index}: {len(message)} bytes")
                else:
                    # JSON frame
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "voice.registered":
                        print(f"  Voice registered: {msg.get('voice_name')!r} (cached={msg.get('cached')})")
                    elif msg_type == "voice.deleted":
                        print(f"  Voice deleted: {msg.get('voice_name')!r} (success={msg.get('success')})")
                    elif msg_type == "audio.start":
                        current_sentence_index = msg["sentence_index"]
                        current_chunks = []
                        first_audio_received = False
                        sentence_t0 = time.perf_counter()
                        print(f"  [sentence {msg['sentence_index']}] Generating: {msg['sentence_text']!r}")
                    elif msg_type == "audio.done":
                        sentence_elapsed = time.perf_counter() - sentence_t0 if sentence_t0 else 0
                        filename = os.path.join(
                            output_dir,
                            f"sentence_{msg['sentence_index']:03d}.{response_format}",
                        )
                        with open(filename, "wb") as f:
                            f.write(b"".join(current_chunks))
                        print(
                            f"  [sentence {msg['sentence_index']}] Done"
                            f" bytes={msg.get('total_bytes', len(b''.join(current_chunks)))}"
                            f" time={sentence_elapsed:.2f}s"
                            f" error={msg.get('error', False)}"
                            f" -> {filename}"
                        )
                        current_chunks = []
                    elif msg_type == "session.done":
                        total_elapsed = time.perf_counter() - session_t0
                        print(
                            f"\nSession complete: {msg['total_sentences']} sentence(s)"
                            f" in {total_elapsed:.2f}s"
                        )
                        break
                    elif msg_type == "error":
                        print(f"  ERROR: {msg['message']}")
                    else:
                        print(f"  Unknown message: {msg}")
        finally:
            # Patch WAV header with actual sizes
            data_size = total_bytes_written
            riff_size = 36 + data_size
            combined_file.seek(4)
            combined_file.write(struct.pack("<I", riff_size))
            combined_file.seek(40)
            combined_file.write(struct.pack("<I", data_size))
            combined_file.close()
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass  # Task cancellation is expected during shutdown

    print(f"\nAudio files saved to: {output_dir}/")
    print(f"Combined stream: {combined_path} ({total_bytes_written} bytes)")


async def delete_voice(url: str, voice_name: str) -> None:
    """Connect and send a voice.delete request."""
    async with websockets.connect(url) as ws:
        # Must send session.config first (protocol requirement)
        await ws.send(json.dumps({"type": "session.config"}))

        # Send delete request
        await ws.send(json.dumps({"type": "voice.delete", "voice_name": voice_name}))
        print(f"Sent voice.delete for {voice_name!r}")

        # Wait for response
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            msg_type = msg.get("type")
            if msg_type == "voice.deleted":
                success = msg.get("success", False)
                print(f"Voice {voice_name!r} deleted: success={success}")
                break
            elif msg_type == "error":
                print(f"Error: {msg.get('message')}")
                break
            # Ignore other messages (e.g. session config ack)

        # Close cleanly
        await ws.send(json.dumps({"type": "input.done"}))


def main():
    parser = argparse.ArgumentParser(description="Streaming text-input TTS client")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/v1/audio/speech/stream",
        help="WebSocket endpoint URL",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output-dir",
        default="streaming_tts_output",
        help="Directory to save audio files (default: streaming_tts_output)",
    )

    # Session config options
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--voice", default="Vivian", help="Speaker voice")
    parser.add_argument(
        "--task-type",
        default="CustomVoice",
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="TTS task type",
    )
    parser.add_argument("--language", default="Auto", help="Language")
    parser.add_argument("--instructions", default=None, help="Voice style instructions")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "flac", "mp3", "aac", "opus"],
        help="Audio format",
    )
    parser.add_argument(
        "--stream-audio",
        action="store_true",
        help="Receive one or more PCM chunks per sentence (requires --response-format pcm)",
    )
    parser.add_argument(
        "--max-buffered-words",
        type=int,
        default=None,
        help="Flush text to TTS after this many words even without sentence boundary",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (0.25-4.0)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max tokens")

    # Base task options
    parser.add_argument("--ref-audio", default=None, help="Reference audio (file path or data URI)")
    parser.add_argument("--ref-text", default=None, help="Reference text")
    parser.add_argument("--voice-name", default=None, help="Voice cache name for voice cloning")
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        default=False,
        help="Speaker embedding only mode",
    )

    # Voice management
    parser.add_argument(
        "--delete-voice",
        default=None,
        help="Delete a cached voice by name (sends voice.delete, then exits)",
    )

    # STT simulation
    parser.add_argument(
        "--simulate-stt",
        action="store_true",
        help="Simulate STT by sending text word-by-word",
    )
    parser.add_argument(
        "--stt-delay",
        type=float,
        default=0.1,
        help="Delay between words in STT simulation (seconds)",
    )

    args = parser.parse_args()

    # Voice deletion mode
    if args.delete_voice:
        asyncio.run(delete_voice(url=args.url, voice_name=args.delete_voice))
        return

    # TTS mode requires --text
    if not args.text:
        parser.error("--text is required (unless using --delete-voice)")

    # Build session config (only include non-None values)
    config = {}
    for key in [
        "model",
        "voice",
        "task_type",
        "language",
        "instructions",
        "response_format",
        "speed",
        "max_new_tokens",
        "ref_audio",
        "ref_text",
        "voice_name",
        "max_buffered_words",
    ]:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            config[key] = val
    # Encode local ref_audio file as data URI if it's a file path
    if "ref_audio" in config and not config["ref_audio"].startswith(("data:", "http://", "https://", "file://")):
        path = config["ref_audio"]
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        config["ref_audio"] = f"data:{mime};base64,{b64}"
        print(f"Encoded ref_audio as data URI ({mime}, {len(b64)} chars)")

    if args.stream_audio:
        config["stream_audio"] = True
    if args.x_vector_only_mode:
        config["x_vector_only_mode"] = True

    asyncio.run(
        stream_tts(
            url=args.url,
            text=args.text,
            config=config,
            output_dir=args.output_dir,
            simulate_stt=args.simulate_stt,
            stt_delay=args.stt_delay,
        )
    )


if __name__ == "__main__":
    main()
