"""Concurrency sweep: measure TTFA and RTF at varying concurrency levels.

Usage:
    # Bulk mode (send full text at once — default)
    python concurrency_sweep.py

    # Simulate STT (drip text word-by-word, 25ms/word default)
    python concurrency_sweep.py --simulate-stt

    # Custom STT pace and ref_text for 1.7B Base model
    python concurrency_sweep.py --simulate-stt --stt-delay 0.03 \
        --ref-text examples/online_serving/qwen3_tts/finn_ref_text.txt

    # Environment overrides
    TTS_WS_URL=ws://host:port/v1/audio/speech/stream
    TTS_VOICE=finn-test-17b
    TTS_STAGGER_S=0.05
    TTS_SKIP_WARMUP=1
"""

import argparse
import asyncio
import json
import os
import time

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise SystemExit(1)

WS_URL = os.environ.get("TTS_WS_URL", "ws://localhost:8091/v1/audio/speech/stream")
VOICE_NAME = os.environ.get("TTS_VOICE", "jfinn1")
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2

PROMPTS = [
    "I hear you, and I want you to know that what you're feeling is completely normal. A lot of people go through this exact same thing, and it doesn't mean anything is wrong with you. Let's take it one step at a time, okay?",
    "Okay so here's how this works. First, you'll want to open the app and go to your profile settings. Then scroll down to the privacy section and tap on data sharing.",
    "That reminds me of something my mom always used to say. She'd tell me, don't worry so much about getting it perfect the first time. Just start somewhere and figure it out as you go.",
    "Hmm, that's interesting. So what ended up happening after that? Did they come back with a counter offer, or did they just drop it entirely?",
    "Oh wow, that's amazing! Congratulations, seriously, that is such great news. You should be really proud of yourself.",
    "I see where you're coming from, but I actually think there might be a better way to handle this. Instead of jumping straight to that conversation, it could help to take a day or two.",
    "Hmm, I'm honestly not sure about that one. I mean, it could work, but there's also a chance it backfires.",
    "I'm really sorry to hear that. That's a tough situation, and I know it probably doesn't feel like it right now, but things will get better.",
]

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 20, 32, 64]
MIN_REQUESTS_PER_LEVEL = 16
STAGGER_S = float(os.environ.get("TTS_STAGGER_S", "0.05"))
SKIP_WARMUP = os.environ.get("TTS_SKIP_WARMUP", "0") == "1"

SIMULATE_STT = False
STT_DELAY_S = 0.025
REF_TEXT: str | None = None


def percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def run_single(prompt: str, request_id: int) -> dict:
    """Run one TTS request, return timing metrics."""
    config: dict = {
        "task_type": "Base",
        "voice_name": VOICE_NAME,
        "response_format": "pcm",
        "stream_audio": True,
        "min_sentence_length": 40,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }
    if REF_TEXT is not None:
        config["ref_text"] = REF_TEXT

    total_bytes = 0
    t0 = time.perf_counter()
    ttfa = None
    error = None

    try:
        async with websockets.connect(WS_URL, ping_interval=None, close_timeout=120) as ws:
            await ws.send(json.dumps({"type": "session.config", **config}))

            async def send_text():
                if SIMULATE_STT:
                    words = prompt.split(" ")
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        await ws.send(json.dumps({"type": "input.text", "text": chunk}))
                        await asyncio.sleep(STT_DELAY_S)
                else:
                    await ws.send(json.dumps({"type": "input.text", "text": prompt}))
                await ws.send(json.dumps({"type": "input.done"}))

            sender = asyncio.create_task(send_text())

            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                if isinstance(msg, bytes):
                    if ttfa is None:
                        ttfa = time.perf_counter() - t0
                    total_bytes += len(msg)
                else:
                    data = json.loads(msg)
                    if data.get("type") == "session.done":
                        break
                    elif data.get("type") == "error":
                        error = data.get("message", str(data))
                        break

            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
    except Exception as e:
        error = str(e)

    elapsed = time.perf_counter() - t0
    audio_secs = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE) if total_bytes else 0
    rtf = elapsed / audio_secs if audio_secs > 0 else 0

    return {
        "request_id": request_id,
        "ttfa_ms": round(ttfa * 1000, 1) if ttfa else None,
        "elapsed_s": round(elapsed, 3),
        "audio_secs": round(audio_secs, 2),
        "rtf": round(rtf, 2),
        "error": error,
    }


async def run_level(concurrency: int, total_requests: int, stagger: float = 0.0) -> list[dict]:
    """Run total_requests at the given concurrency with optional stagger between launches."""
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def worker(idx: int):
        if stagger > 0 and idx > 0:
            await asyncio.sleep(idx * stagger)
        async with sem:
            prompt = PROMPTS[idx % len(PROMPTS)]
            r = await run_single(prompt, idx)
            results.append(r)

    tasks = [asyncio.create_task(worker(i)) for i in range(total_requests)]
    await asyncio.gather(*tasks)
    return results


async def main():
    mode = f"simulate-stt ({STT_DELAY_S * 1000:.0f}ms/word)" if SIMULATE_STT else "bulk (full text at once)"
    print(f"Endpoint: {WS_URL}")
    print(f"Voice:    {VOICE_NAME}")
    print(f"Mode:     {mode}")
    print(f"Ref text: {'yes' if REF_TEXT else 'no'}")
    print(f"Min requests per level: {MIN_REQUESTS_PER_LEVEL}")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"Stagger: {STAGGER_S}s")
    print()

    if not SKIP_WARMUP:
        print(f"Warmup (c={CONCURRENCY_LEVELS})...")
        for wc in CONCURRENCY_LEVELS:
            await run_level(wc, wc)
        print("Warmup done.\n")
    else:
        print("Warmup skipped (TTS_SKIP_WARMUP=1)\n")

    all_results = {}

    header = f"{'Conc':>6}  {'TTFA p50':>10}  {'TTFA p90':>10}  {'TTFA p99':>10}  {'RTF p50':>9}  {'Elapsed':>9}  {'Errors':>6}"
    print(header)
    print("-" * len(header))

    for c in CONCURRENCY_LEVELS:
        n_requests = max(MIN_REQUESTS_PER_LEVEL, c)
        results = await run_level(c, n_requests, stagger=STAGGER_S)

        ttfas = [r["ttfa_ms"] for r in results if r["ttfa_ms"] is not None]
        rtfs = [r["rtf"] for r in results if r["error"] is None]
        elapsed_total = max(r["elapsed_s"] for r in results) if results else 0
        errors = sum(1 for r in results if r["error"])
        if errors:
            for r in results:
                if r["error"]:
                    print(f"  [c={c} req={r['request_id']}] ERROR: {r['error'][:200]}")

        row = {
            "concurrency": c,
            "n": len(results),
            "errors": errors,
            "ttfa_p50_ms": round(percentile(ttfas, 50), 0) if ttfas else None,
            "ttfa_p90_ms": round(percentile(ttfas, 90), 0) if ttfas else None,
            "ttfa_p99_ms": round(percentile(ttfas, 99), 0) if ttfas else None,
            "rtf_p50": round(percentile(rtfs, 50), 2) if rtfs else None,
            "wall_s": round(elapsed_total, 1),
        }
        all_results[c] = row

        p50 = f"{row['ttfa_p50_ms']:.0f}ms" if row["ttfa_p50_ms"] else "n/a"
        p90 = f"{row['ttfa_p90_ms']:.0f}ms" if row["ttfa_p90_ms"] else "n/a"
        p99 = f"{row['ttfa_p99_ms']:.0f}ms" if row["ttfa_p99_ms"] else "n/a"
        rtf = f"{row['rtf_p50']:.2f}x" if row["rtf_p50"] else "n/a"

        print(f"{c:>6}  {p50:>10}  {p90:>10}  {p99:>10}  {rtf:>9}  {row['wall_s']:>8.1f}s  {errors:>6}")

    print()
    suffix = "_stt" if SIMULATE_STT else ""
    out_path = os.path.join(os.path.dirname(__file__) or ".", f"concurrency_sweep{suffix}_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Concurrency sweep for TTS")
    parser.add_argument("--simulate-stt", action="store_true", help="Send text word-by-word (default: 25ms/word)")
    parser.add_argument("--stt-delay", type=float, default=0.025, help="Seconds between words in STT mode (default: 0.025)")
    parser.add_argument("--ref-text", default=None, help="Path to ref_text file or literal string (required for 1.7B Base)")
    parser.add_argument("--voice", default=None, help="Voice name override (default: TTS_VOICE env or jfinn1)")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup phase")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.simulate_stt:
        SIMULATE_STT = True
        STT_DELAY_S = args.stt_delay
    if args.voice:
        VOICE_NAME = args.voice
    if args.skip_warmup:
        SKIP_WARMUP = True
    if args.ref_text:
        if os.path.isfile(args.ref_text):
            with open(args.ref_text, encoding="utf-8") as f:
                REF_TEXT = f.read().strip()
        else:
            REF_TEXT = args.ref_text

    asyncio.run(main())
