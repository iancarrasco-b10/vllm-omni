"""Sweep concurrency levels and plot p50/p90/p99 TTFA curves.

Runs the streaming TTS benchmark at concurrency 2, 4, 6, …, max_concurrency,
collects TTFA percentiles, and saves a plot + CSV.

Usage:
    python sweep_concurrency.py
    python sweep_concurrency.py --max-concurrency 20 --step 2 --rounds 5
    python sweep_concurrency.py --output-plot ttfa_sweep.png
"""

import argparse
import asyncio
import csv
import json
import math
import os
import time

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise SystemExit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib")
    raise SystemExit(1)


async def run_one_session(url: str, text: str, config: dict) -> dict:
    async with websockets.connect(url) as ws:
        config_msg = {"type": "session.config", **config}
        t0 = time.perf_counter()
        await ws.send(json.dumps(config_msg))
        await ws.send(json.dumps({"type": "input.text", "text": text}))
        await ws.send(json.dumps({"type": "input.done"}))

        ttfa = None
        total_pcm_bytes = 0

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                total_pcm_bytes += len(message)
            else:
                msg = json.loads(message)
                if msg.get("type") == "session.done":
                    break
                elif msg.get("type") == "error":
                    return {"error": msg["message"]}

        t_total = time.perf_counter() - t0
        duration_s = total_pcm_bytes / (24000 * 2) if total_pcm_bytes else 0
        return {
            "ttfa_ms": ttfa * 1000 if ttfa else None,
            "total_ms": t_total * 1000,
            "audio_duration_s": duration_s,
        }


async def run_at_concurrency(
    url: str, text: str, config: dict, concurrency: int, rounds: int,
) -> list[dict]:
    all_results: list[dict] = []
    for _ in range(rounds):
        tasks = [run_one_session(url, text, config) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                all_results.append({"error": str(r)})
            else:
                all_results.append(r)
    return all_results


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


async def sweep(args):
    config = {
        "voice": args.voice,
        "task_type": args.task_type,
        "language": "Auto",
        "response_format": "wav",
        "speed": 1.0,
    }
    text = args.text

    concurrencies = list(range(args.step, args.max_concurrency + 1, args.step))

    print(f"Target: {args.url}")
    print(f"Text: {text!r}")
    print(f"Concurrencies: {concurrencies}")
    print(f"Rounds per level: {args.rounds}")
    print(f"Warmup rounds: {args.warmup}")
    print("=" * 70)

    if args.warmup > 0:
        print(f"\n  Warmup ({args.warmup} round(s) at c=1)...", end="", flush=True)
        await run_at_concurrency(args.url, text, config, 1, args.warmup)
        print(" done")

    rows: list[dict] = []

    for conc in concurrencies:
        print(f"\n  [c={conc:>2}] benchmarking ({args.rounds} rounds × {conc} sessions)...", end="", flush=True)
        results = await run_at_concurrency(args.url, text, config, conc, args.rounds)
        print(" done")

        ok = [r for r in results if "error" not in r]
        ttfas = [r["ttfa_ms"] for r in ok if r.get("ttfa_ms") is not None]
        totals = [r["total_ms"] for r in ok]
        audio_secs = [r["audio_duration_s"] for r in ok]
        rtfs = [r["total_ms"] / 1000 / r["audio_duration_s"]
                for r in ok if r.get("audio_duration_s", 0) > 0]
        errors = len(results) - len(ok)

        p50 = percentile(ttfas, 50)
        p90 = percentile(ttfas, 90)
        p99 = percentile(ttfas, 99)
        rtf_p50 = percentile(rtfs, 50)
        rtf_p90 = percentile(rtfs, 90)
        rtf_p99 = percentile(rtfs, 99)
        total_p50 = percentile(totals, 50)
        total_audio = sum(audio_secs)
        wall = max(totals) / 1000 if totals else 0
        throughput = total_audio / wall if wall > 0 else 0

        row = {
            "concurrency": conc,
            "samples": len(ttfas),
            "errors": errors,
            "ttfa_p50": p50,
            "ttfa_p90": p90,
            "ttfa_p99": p99,
            "rtf_p50": rtf_p50,
            "rtf_p90": rtf_p90,
            "rtf_p99": rtf_p99,
            "total_p50": total_p50,
            "throughput_x": throughput,
        }
        rows.append(row)

        print(
            f"           TTFA  p50={p50:>7.0f}ms  p90={p90:>7.0f}ms  p99={p99:>7.0f}ms  "
            f"| RTF  p50={rtf_p50:.2f}x  p90={rtf_p90:.2f}x  p99={rtf_p99:.2f}x  "
            f"| throughput={throughput:.1f}x"
            f"{f'  errors={errors}' if errors else ''}"
        )

    # --- Save CSV ---
    csv_path = args.output_csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # --- Plot ---
    concs = [r["concurrency"] for r in rows]
    p50s = [r["ttfa_p50"] for r in rows]
    p90s = [r["ttfa_p90"] for r in rows]
    p99s = [r["ttfa_p99"] for r in rows]
    rp50s = [r["rtf_p50"] for r in rows]
    rp90s = [r["rtf_p90"] for r in rows]
    rp99s = [r["rtf_p99"] for r in rows]
    throughputs = [r["throughput_x"] for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.plot(concs, p50s, "o-", label="p50", linewidth=2, markersize=6)
    ax1.plot(concs, p90s, "s-", label="p90", linewidth=2, markersize=6)
    ax1.plot(concs, p99s, "^-", label="p99", linewidth=2, markersize=6)
    ax1.set_xlabel("Concurrency", fontsize=12)
    ax1.set_ylabel("TTFA (ms)", fontsize=12)
    ax1.set_title("Time to First Audio vs Concurrency", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(concs)

    ax2.plot(concs, rp50s, "o-", label="p50", linewidth=2, markersize=6)
    ax2.plot(concs, rp90s, "s-", label="p90", linewidth=2, markersize=6)
    ax2.plot(concs, rp99s, "^-", label="p99", linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="realtime")
    ax2.set_xlabel("Concurrency", fontsize=12)
    ax2.set_ylabel("RTF (lower = faster)", fontsize=12)
    ax2.set_title("Real-Time Factor vs Concurrency", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(concs)

    ax3.plot(concs, throughputs, "D-", color="green", linewidth=2, markersize=6)
    ax3.set_xlabel("Concurrency", fontsize=12)
    ax3.set_ylabel("Throughput (× realtime)", fontsize=12)
    ax3.set_title("Aggregate Throughput vs Concurrency", fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(concs)

    fig.tight_layout()
    plot_path = args.output_plot
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sweep concurrency & plot TTFA percentiles")
    parser.add_argument("--url", default="ws://localhost:8091/v1/audio/speech/stream")
    parser.add_argument(
        "--text",
        default="Hello world. How are you today? I am doing very well, thank you for asking.",
    )
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3, help="Measured rounds per concurrency level")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds (discarded) per level")
    parser.add_argument("--voice", default="Vivian")
    parser.add_argument("--task-type", default="CustomVoice")
    parser.add_argument("--output-plot", default="ttfa_sweep.png")
    parser.add_argument("--output-csv", default="ttfa_sweep.csv")
    args = parser.parse_args()
    asyncio.run(sweep(args))


if __name__ == "__main__":
    main()
