#!/usr/bin/env python3
"""WebSocket streaming TTS client example.

Usage:
    # Basic usage - send text all at once
    python streaming_speech_client.py --text "Hello world. How are you? I am fine."

    # Simulate real-time STT by sending text word-by-word
    python streaming_speech_client.py --text "Hello world. How are you?" --simulate-stt --stt-delay 0.1

    # Use specific voice and language
    python streaming_speech_client.py --text "你好世界" --voice Vivian --language Chinese

    # Save per-sentence audio files
    python streaming_speech_client.py --text "Hello. World." --output-dir ./audio_output

Requirements:
    pip install websockets
"""

import argparse
import asyncio
import json
import os
import struct
import time

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise


async def stream_tts(
    url: str,
    text: str,
    voice: str = "Vivian",
    task_type: str = "CustomVoice",
    language: str = "Auto",
    instructions: str | None = None,
    response_format: str = "pcm",
    simulate_stt: bool = False,
    stt_delay: float = 0.1,
    output_dir: str | None = None,
):
    """Connect to WebSocket TTS endpoint and stream audio."""

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    async with websockets.connect(url) as ws:
        # Send session config
        config = {
            "type": "session.config",
            "voice": voice,
            "task_type": task_type,
            "language": language,
            "response_format": response_format,
        }
        if instructions:
            config["instructions"] = instructions

        await ws.send(json.dumps(config))
        print(f"[Config] Sent session config: voice={voice}, task_type={task_type}")

        # Start receiver task
        audio_buffers: dict[int, bytearray] = {}
        sentence_texts: dict[int, str] = {}
        done_event = asyncio.Event()

        async def receiver():
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        # Binary frame = audio data
                        # Associate with current sentence
                        current_idx = max(audio_buffers.keys()) if audio_buffers else 0
                        if current_idx not in audio_buffers:
                            audio_buffers[current_idx] = bytearray()
                        audio_buffers[current_idx].extend(message)
                        print(f"  [Audio] Received {len(message)} bytes for sentence {current_idx}")
                    else:
                        msg = json.loads(message)
                        msg_type = msg.get("type", "")

                        if msg_type == "audio.start":
                            idx = msg["sentence_index"]
                            audio_buffers[idx] = bytearray()
                            sentence_texts[idx] = msg.get("sentence_text", "")
                            fmt = msg.get("format", "pcm")
                            print(f"[Start] Sentence {idx}: {sentence_texts[idx]!r} (format={fmt})")

                        elif msg_type == "audio.done":
                            idx = msg["sentence_index"]
                            total = len(audio_buffers.get(idx, b""))
                            print(f"[Done]  Sentence {idx}: {total} bytes total")

                            if output_dir and idx in audio_buffers:
                                fname = os.path.join(output_dir, f"sentence_{idx}.{response_format}")
                                if response_format == "pcm":
                                    # Wrap PCM in WAV for playback
                                    fname = os.path.join(output_dir, f"sentence_{idx}.wav")
                                    _write_pcm_as_wav(fname, audio_buffers[idx], sample_rate=24000)
                                else:
                                    with open(fname, "wb") as f:
                                        f.write(audio_buffers[idx])
                                print(f"  Saved: {fname}")

                        elif msg_type == "session.done":
                            total_sentences = msg.get("total_sentences", 0)
                            print(f"[Session Done] Total sentences: {total_sentences}")
                            done_event.set()
                            return

                        elif msg_type == "error":
                            print(f"[Error] {msg.get('message', 'Unknown error')}")

            except websockets.exceptions.ConnectionClosed:
                done_event.set()

        recv_task = asyncio.create_task(receiver())

        # Send text
        t0 = time.time()
        if simulate_stt:
            # Simulate real-time STT by sending word by word
            words = text.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                await ws.send(json.dumps({"type": "input.text", "text": chunk}))
                print(f"[Send] '{chunk}'")
                await asyncio.sleep(stt_delay)
        else:
            # Send all text at once
            await ws.send(json.dumps({"type": "input.text", "text": text}))
            print(f"[Send] Sent full text ({len(text)} chars)")

        # Signal input done
        await ws.send(json.dumps({"type": "input.done"}))
        print("[Send] input.done")

        # Wait for session completion
        await done_event.wait()
        recv_task.cancel()

        elapsed = time.time() - t0
        total_bytes = sum(len(b) for b in audio_buffers.values())
        print(f"\n[Summary] {elapsed:.2f}s, {total_bytes} bytes audio, "
              f"{len(audio_buffers)} sentences")


def _write_pcm_as_wav(path: str, pcm_data: bytes, sample_rate: int = 24000):
    """Write raw PCM (int16, mono) as a WAV file."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_data)


def main():
    parser = argparse.ArgumentParser(description="Streaming TTS WebSocket client")
    parser.add_argument("--url", default="ws://localhost:8000/v1/audio/speech/stream",
                        help="WebSocket endpoint URL")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default="Vivian", help="Voice name")
    parser.add_argument("--task-type", default="CustomVoice",
                        choices=["CustomVoice", "VoiceDesign", "Base"])
    parser.add_argument("--language", default="Auto", help="Language code")
    parser.add_argument("--instructions", default=None, help="Style instructions")
    parser.add_argument("--response-format", default="pcm",
                        choices=["pcm", "wav", "mp3", "flac"])
    parser.add_argument("--simulate-stt", action="store_true",
                        help="Send text word-by-word to simulate STT")
    parser.add_argument("--stt-delay", type=float, default=0.1,
                        help="Delay between words in STT simulation mode")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save per-sentence audio files")
    args = parser.parse_args()

    asyncio.run(stream_tts(
        url=args.url,
        text=args.text,
        voice=args.voice,
        task_type=args.task_type,
        language=args.language,
        instructions=args.instructions,
        response_format=args.response_format,
        simulate_stt=args.simulate_stt,
        stt_delay=args.stt_delay,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
