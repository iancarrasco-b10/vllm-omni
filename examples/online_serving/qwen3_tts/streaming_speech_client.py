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
import io
import json
import mimetypes
import os
import struct
import time
import wave

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

    def _write_wav_header(file_obj, sample_rate: int) -> None:
        bits_per_sample = 16
        num_channels = 1
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        file_obj.write(struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 0xFFFFFFFF,
            b"WAVE",
            b"fmt ", 16,
            1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
            b"data", 0xFFFFFFFF,
        ))

    def _finalize_wav_file(file_obj, data_size: int) -> None:
        riff_size = 36 + data_size
        file_obj.seek(4)
        file_obj.write(struct.pack("<I", riff_size))
        file_obj.seek(40)
        file_obj.write(struct.pack("<I", data_size))
        file_obj.close()

    def _extract_pcm_for_combined(audio_bytes: bytes, fmt: str) -> bytes:
        """Return raw PCM payload for the combined WAV output."""
        if fmt == "pcm":
            return audio_bytes
        if fmt != "wav":
            return b""

        # Each sentence arrives as a complete WAV file. Strip the container header
        # before appending so `combined.wav` is valid continuous PCM.
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())

    async with websockets.connect(url, ping_interval=None) as ws:
        # 1. Send session config
        config_msg = {"type": "session.config", **config}
        await ws.send(json.dumps(config_msg))
        config_display = {k: (f"<{len(v)} chars>" if k == "ref_audio" and isinstance(v, str) and len(v) > 200 else v) for k, v in config.items()}
        print(f"Sent session config: {config_display}")

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
        current_sentence_file = None
        current_sentence_path: str | None = None
        current_sentence_pcm_bytes = 0
        current_sentence_sample_rate = 24000

        # Combined output file: append all PCM chunks as they arrive, wrapped in WAV
        combined_path = os.path.join(output_dir, "combined.wav")
        combined_file = open(combined_path, "wb")
        _write_wav_header(combined_file, sample_rate=24000)

        interrupted = False
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
                    # Append only raw PCM to the combined WAV output.
                    pcm_bytes = _extract_pcm_for_combined(message, response_format)
                    if pcm_bytes:
                        combined_file.write(pcm_bytes)
                        combined_file.flush()
                        total_bytes_written += len(pcm_bytes)
                        if current_sentence_file is not None:
                            current_sentence_file.write(pcm_bytes)
                            current_sentence_file.flush()
                            current_sentence_pcm_bytes += len(pcm_bytes)
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
                        current_sentence_sample_rate = int(msg.get("sample_rate", 24000))
                        current_sentence_pcm_bytes = 0
                        current_sentence_path = None
                        if current_sentence_file is not None:
                            _finalize_wav_file(current_sentence_file, current_sentence_pcm_bytes)
                            current_sentence_file = None
                        if response_format == "pcm":
                            current_sentence_path = os.path.join(
                                output_dir,
                                f"sentence_{msg['sentence_index']:03d}.wav",
                            )
                            current_sentence_file = open(current_sentence_path, "wb")
                            _write_wav_header(current_sentence_file, sample_rate=current_sentence_sample_rate)
                        print(f"  [sentence {msg['sentence_index']}] Generating: {msg['sentence_text']!r}")
                    elif msg_type == "audio.done":
                        sentence_elapsed = time.perf_counter() - sentence_t0 if sentence_t0 else 0
                        if current_sentence_file is not None:
                            _finalize_wav_file(current_sentence_file, current_sentence_pcm_bytes)
                            current_sentence_file = None
                            filename = current_sentence_path or os.path.join(
                                output_dir,
                                f"sentence_{msg['sentence_index']:03d}.wav",
                            )
                        else:
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
        except (KeyboardInterrupt, asyncio.CancelledError):
            interrupted = True
            elapsed = time.perf_counter() - session_t0
            print(f"\n  Interrupted after {elapsed:.2f}s")
        finally:
            # Patch WAV header with actual sizes — always runs so partial audio is usable
            if current_sentence_file is not None:
                _finalize_wav_file(current_sentence_file, current_sentence_pcm_bytes)
            data_size = total_bytes_written
            _finalize_wav_file(combined_file, data_size)
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass  # Task cancellation is expected during shutdown

    label = "Interrupted — partial" if interrupted else "Audio files saved to"
    print(f"\n{label}: {output_dir}/")
    print(f"Combined stream: {combined_path} ({total_bytes_written} bytes)")


async def delete_voice(url: str, voice_name: str) -> None:
    """Connect and send a voice.delete request."""
    async with websockets.connect(url, ping_interval=None) as ws:
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
        default="pcm",
        choices=["wav", "pcm", "flac", "mp3", "aac", "opus"],
        help="Audio format (default: pcm for progressive WebSocket streaming)",
    )
    parser.add_argument(
        "--stream-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Receive progressive PCM chunks per sentence (default: enabled, requires --response-format pcm)",
    )
    parser.add_argument(
        "--max-buffered-words",
        type=int,
        default=None,
        help="Flush text to TTS after this many words even without sentence boundary",
    )
    parser.add_argument(
        "--split-granularity",
        default="sentence",
        choices=["sentence", "clause"],
        help="Text splitting granularity for websocket TTS (default: sentence)",
    )
    parser.add_argument(
        "--min-sentence-length",
        type=int,
        default=40,
        help="Minimum characters before the websocket splitter emits a segment (default: 40)",
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
        "split_granularity",
        "min_sentence_length",
    ]:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            config[key] = val

    if 'ref_text' in config and os.path.isfile(config['ref_text']):
        with open(config['ref_text'], 'r', encoding='utf-8') as f:
            config['ref_text'] = f.read().strip()
        print(f"Read ref_text from file: {config['ref_text']}")

    # Encode local ref_audio file as data URI if it's a file path
    if "ref_audio" in config and not config["ref_audio"].startswith(("data:", "http://", "https://", "file://")):
        path = config["ref_audio"]
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        config["ref_audio"] = f"data:{mime};base64,{b64}"
        print(f"Encoded ref_audio as data URI ({mime}, {len(b64)} chars)")

    config["stream_audio"] = args.stream_audio
    if args.x_vector_only_mode:
        config["x_vector_only_mode"] = True

    try:
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
    except KeyboardInterrupt:
        pass  # stream_tts already saved partial output in its finally block


if __name__ == "__main__":
    main()
