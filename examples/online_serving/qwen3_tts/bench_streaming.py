"""Benchmark concurrent streaming TTS sessions.

Launches N parallel WebSocket connections, each sending a text payload,
and reports per-session TTFA / total time plus aggregate statistics.

Usage:
    # 1 concurrent stream (baseline)
    python bench_streaming.py --concurrency 1

    # 4 concurrent streams, 3 rounds each
    python bench_streaming.py --concurrency 4 --rounds 3

    # Custom text (same for all sessions)
    python bench_streaming.py --concurrency 2 \
        --text "The quick brown fox jumps over the lazy dog."

    # Different prompts per concurrent stream (round-robin assignment)
    python bench_streaming.py --concurrency 3 \
        --text "First prompt." --text "Second prompt." --text "Third prompt."

    # Load prompts from a file (one per line)
    python bench_streaming.py --concurrency 4 --text-file prompts.txt

    # Save output audio for each stream
    python bench_streaming.py --concurrency 4 --output-dir ./bench_output

    # Voice cloning (Base task) with a reference audio file
    python bench_streaming.py --concurrency 2 \
        --task-type Base \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of reference audio" \
        --text "Clone my voice saying this."
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import time
import wave

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise SystemExit(1)


_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
}


def _encode_audio(path: str) -> str:
    """Encode a local audio file to a base64 data-URL, or pass through URLs."""
    if path.startswith(("http://", "https://", "data:")):
        return path
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME_TYPES.get(ext, "audio/wav")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _write_wav(path: str, pcm_data: bytes, sample_rate: int) -> None:
    """Write raw PCM-16LE bytes to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


async def run_one_session(
    url: str,
    text: str,
    config: dict,
    session_id: int,
    round_num: int = 0,
    output_dir: str | None = None,
) -> dict:
    """Run a single streaming TTS session, return timing metrics."""
    async with websockets.connect(url) as ws:
        config_msg = {"type": "session.config", **config}
        t0 = time.perf_counter()
        await ws.send(json.dumps(config_msg))
        await ws.send(json.dumps({"type": "input.text", "text": text}))
        await ws.send(json.dumps({"type": "input.done"}))

        ttfa = None
        sentence_count = 0
        total_pcm_bytes = 0
        chunk_count = 0
        sample_rate = 24000
        all_pcm: list[bytes] = []

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                total_pcm_bytes += len(message)
                chunk_count += 1
                all_pcm.append(message)
            else:
                msg = json.loads(message)
                if msg.get("type") == "audio.done":
                    sample_rate = msg.get("sample_rate", 24000)
                    sentence_count += 1
                elif msg.get("type") == "session.done":
                    break
                elif msg.get("type") == "error":
                    return {
                        "session_id": session_id,
                        "error": msg["message"],
                    }

        t_total = time.perf_counter() - t0
        duration_s = total_pcm_bytes / (sample_rate * 2) if total_pcm_bytes else 0

        if output_dir and all_pcm:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"round{round_num}_session{session_id}.wav")
            _write_wav(path, b"".join(all_pcm), sample_rate)

        return {
            "session_id": session_id,
            "ttfa_ms": ttfa * 1000 if ttfa else None,
            "total_ms": t_total * 1000,
            "sentences": sentence_count,
            "chunks": chunk_count,
            "pcm_bytes": total_pcm_bytes,
            "audio_duration_s": duration_s,
            "rtf": t_total / duration_s if duration_s > 0 else None,
        }


async def run_round(
    url: str,
    texts: list[str],
    config: dict,
    concurrency: int,
    round_num: int,
    output_dir: str | None = None,
) -> list[dict]:
    tasks = [
        run_one_session(url, texts[i % len(texts)], config, session_id=i, round_num=round_num, output_dir=output_dir)
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            processed.append({"session_id": i, "error": str(r)})
        else:
            processed.append(r)
    return processed


def print_results(all_results: list[dict], concurrency: int):
    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]

    if failed:
        print(f"\n  Failed sessions: {len(failed)}")
        for f in failed:
            print(f"    session {f['session_id']}: {f['error']}")

    if not successful:
        print("  No successful sessions.")
        return

    ttfas = [r["ttfa_ms"] for r in successful if r["ttfa_ms"] is not None]
    totals = [r["total_ms"] for r in successful]
    rtfs = [r["rtf"] for r in successful if r["rtf"] is not None]

    def fmt_stats(vals, unit="ms"):
        if not vals:
            return "n/a"
        if len(vals) == 1:
            return f"{vals[0]:.1f}{unit}"
        return (
            f"avg={statistics.mean(vals):.1f}{unit}  "
            f"p50={statistics.median(vals):.1f}{unit}  "
            f"min={min(vals):.1f}{unit}  "
            f"max={max(vals):.1f}{unit}"
        )

    print(f"\n  Concurrency: {concurrency}  |  Sessions: {len(successful)}")
    print(f"  TTFA:   {fmt_stats(ttfas)}")
    print(f"  Total:  {fmt_stats(totals)}")
    print(f"  RTF:    {fmt_stats(rtfs, unit='x')}")

    total_audio = sum(r["audio_duration_s"] for r in successful)
    wall_clock = max(totals) / 1000
    print(f"  Throughput: {total_audio:.1f}s audio in {wall_clock:.1f}s wall = {total_audio/wall_clock:.2f}x realtime")


def _resolve_texts(args) -> list[str]:
    """Build the list of prompts from --text and/or --text-file."""
    texts: list[str] = []
    if args.text_file:
        with open(args.text_file) as f:
            texts.extend(line.strip() for line in f if line.strip())
    if args.text:
        texts.extend(args.text)
    if not texts:
        texts.append("Hello world. How are you today? I am doing very well, thank you for asking.")
    return texts


async def main_async(args):
    config: dict = {
        "voice": args.voice,
        "task_type": args.task_type,
        "language": "Auto",
        "response_format": "wav",
        "speed": 1.0,
    }
    if args.ref_audio:
        config["ref_audio"] = _encode_audio(args.ref_audio)
    if args.ref_text:
        config["ref_text"] = args.ref_text
    if args.instructions:
        config["instructions"] = args.instructions

    texts = _resolve_texts(args)

    print(f"Target: {args.url}")
    print(f"Task: {args.task_type}")
    if args.ref_audio:
        print(f"Ref audio: {args.ref_audio}")
    if args.ref_text:
        print(f"Ref text: {args.ref_text!r}")
    print(f"Prompts: {len(texts)}")
    for i, t in enumerate(texts):
        label = f"  [{i}] "
        preview = t if len(t) <= 80 else t[:77] + "..."
        print(f"{label}{preview!r}")
    print(f"Concurrency: {args.concurrency}  |  Rounds: {args.rounds}")
    if args.output_dir:
        print(f"Output: {args.output_dir}/")
    print("=" * 60)

    all_results = []
    for rnd in range(args.rounds):
        print(f"\n--- Round {rnd + 1}/{args.rounds} ---")
        results = await run_round(
            args.url, texts, config, args.concurrency, rnd,
            output_dir=args.output_dir,
        )
        for r in results:
            if "error" not in r:
                print(
                    f"  session {r['session_id']}: "
                    f"TTFA={r['ttfa_ms']:.0f}ms  "
                    f"total={r['total_ms']:.0f}ms  "
                    f"sentences={r['sentences']}  "
                    f"chunks={r['chunks']}  "
                    f"RTF={r['rtf']:.2f}x"
                )
            else:
                print(f"  session {r['session_id']}: ERROR {r['error']}")
        all_results.extend(results)

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print_results(all_results, args.concurrency)


def main():
    parser = argparse.ArgumentParser(description="Benchmark streaming TTS")
    parser.add_argument("--url", default="ws://localhost:8091/v1/audio/speech/stream")
    parser.add_argument(
        "--text", action="append", default=None,
        help="Text prompt. Can be specified multiple times for different prompts "
             "per session (assigned round-robin). Falls back to a default if omitted.",
    )
    parser.add_argument(
        "--text-file", default=None,
        help="Path to a file with one prompt per line. "
             "Combined with any --text prompts.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save WAV files. Files are named round<R>_session<S>.wav. "
             "If not set, audio is not saved.",
    )
    parser.add_argument("--voice", default="Vivian", help="Speaker voice name")
    parser.add_argument(
        "--task-type", default="CustomVoice",
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="TTS task type (use Base for voice cloning)",
    )
    parser.add_argument(
        "--ref-audio", default=None,
        help="Reference audio for voice cloning (local path or URL). "
             "Requires --task-type Base.",
    )
    parser.add_argument(
        "--ref-text", default=None,
        help="Transcript of the reference audio for voice cloning.",
    )
    parser.add_argument(
        "--instructions", default=None,
        help="Voice style instructions (for VoiceDesign / CustomVoice).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
