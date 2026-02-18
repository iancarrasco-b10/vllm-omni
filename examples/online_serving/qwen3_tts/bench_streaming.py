"""Benchmark concurrent streaming TTS sessions.

Launches N parallel WebSocket connections, each sending a text payload,
and reports per-session TTFA / total time plus aggregate statistics.

Usage:
    # 1 concurrent stream (baseline)
    python bench_streaming.py --concurrency 1

    # 4 concurrent streams, 3 rounds each
    python bench_streaming.py --concurrency 4 --rounds 3

    # Custom text
    python bench_streaming.py --concurrency 2 \
        --text "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
"""

import argparse
import asyncio
import json
import statistics
import time

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise SystemExit(1)


async def run_one_session(
    url: str,
    text: str,
    config: dict,
    session_id: int,
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

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                total_pcm_bytes += len(message)
                chunk_count += 1
            else:
                msg = json.loads(message)
                if msg.get("type") == "audio.done":
                    sentence_count += 1
                elif msg.get("type") == "session.done":
                    break
                elif msg.get("type") == "error":
                    return {
                        "session_id": session_id,
                        "error": msg["message"],
                    }

        t_total = time.perf_counter() - t0
        duration_s = total_pcm_bytes / (24000 * 2) if total_pcm_bytes else 0
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
    text: str,
    config: dict,
    concurrency: int,
    round_num: int,
) -> list[dict]:
    tasks = [
        run_one_session(url, text, config, session_id=i)
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


async def main_async(args):
    config = {
        "voice": args.voice,
        "task_type": args.task_type,
        "language": "Auto",
        "response_format": "wav",
        "speed": 1.0,
    }

    print(f"Target: {args.url}")
    print(f"Text: {args.text!r}")
    print(f"Concurrency: {args.concurrency}  |  Rounds: {args.rounds}")
    print("=" * 60)

    all_results = []
    for rnd in range(args.rounds):
        print(f"\n--- Round {rnd + 1}/{args.rounds} ---")
        results = await run_round(
            args.url, args.text, config, args.concurrency, rnd
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
        "--text",
        default="Hello world. How are you today? I am doing very well, thank you for asking.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--voice", default="Vivian")
    parser.add_argument("--task-type", default="CustomVoice")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
