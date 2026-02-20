"""Concurrency sweep for voice-cloning (Base/ICL) streaming TTS.

Runs bench_streaming at concurrency levels 1..N, collects per-level
aggregate stats, writes a CSV, and renders a latency curve plot.

Usage:
    python bench_clone_sweep.py                     # default sweep 1,2,4,8
    python bench_clone_sweep.py --levels 1 2 4 8 16
    python bench_clone_sweep.py --voice finn --rounds 5
"""

import argparse
import asyncio
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from bench_streaming import run_round  # noqa: E402

DEFAULT_TEXT = (
    "Hello, this is my cloned voice speaking. "
    "What is the laziest animal in the world?"
)


def percentile(data: list[float], p: float) -> float:
    """Simple linear-interpolation percentile (like numpy)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])


async def sweep(args):
    config: dict = {
        "voice": args.voice,
        "task_type": "Base",
        "language": "Auto",
        "response_format": "wav",
        "speed": 1.0,
    }
    texts = [args.text]
    url = args.url

    print(f"Voice-clone concurrency sweep")
    print(f"  URL:    {url}")
    print(f"  Voice:  {args.voice}")
    print(f"  Text:   {args.text!r}")
    print(f"  Rounds: {args.rounds}  |  Levels: {args.levels}")
    print("=" * 70)

    # Warmup round (not counted)
    print("\nWarmup...", end="", flush=True)
    await run_round(url, texts, config, concurrency=1, round_num=-1)
    print(" done")

    rows: list[dict] = []

    for level in args.levels:
        all_results: list[dict] = []
        for rnd in range(args.rounds):
            results = await run_round(
                url, texts, config,
                concurrency=level, round_num=rnd,
            )
            all_results.extend(r for r in results if "error" not in r)

        ttfas = [r["ttfa_ms"] for r in all_results if r.get("ttfa_ms")]
        totals = [r["total_ms"] for r in all_results]
        rtfs = [r["rtf"] for r in all_results if r.get("rtf")]
        durations = [r["audio_duration_s"] for r in all_results]

        row = {
            "concurrency": level,
            "sessions": len(all_results),
            "ttfa_avg": statistics.mean(ttfas) if ttfas else 0,
            "ttfa_p50": percentile(ttfas, 50) if ttfas else 0,
            "ttfa_p95": percentile(ttfas, 95) if ttfas else 0,
            "ttfa_max": max(ttfas) if ttfas else 0,
            "total_avg": statistics.mean(totals) if totals else 0,
            "total_p50": percentile(totals, 50) if totals else 0,
            "total_p95": percentile(totals, 95) if totals else 0,
            "total_max": max(totals) if totals else 0,
            "rtf_avg": statistics.mean(rtfs) if rtfs else 0,
            "audio_dur_avg": statistics.mean(durations) if durations else 0,
            "throughput_x": (
                sum(durations) / (max(totals) / 1000)
                if totals and durations else 0
            ),
        }
        rows.append(row)

        print(
            f"  C={level:>2}  "
            f"TTFA avg={row['ttfa_avg']:>7.0f}ms  p50={row['ttfa_p50']:>7.0f}ms  p95={row['ttfa_p95']:>7.0f}ms  "
            f"Total avg={row['total_avg']:>7.0f}ms  "
            f"Throughput={row['throughput_x']:.2f}x RT"
        )

    # Write CSV
    csv_path = os.path.join(os.path.dirname(__file__), "clone_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        concurrencies = [r["concurrency"] for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(concurrencies, [r["ttfa_avg"] for r in rows], "o-", label="TTFA avg", linewidth=2)
        ax1.plot(concurrencies, [r["ttfa_p50"] for r in rows], "s--", label="TTFA p50", linewidth=1.5)
        ax1.plot(concurrencies, [r["ttfa_p95"] for r in rows], "^--", label="TTFA p95", linewidth=1.5)
        ax1.set_xlabel("Concurrent Sessions")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Voice Clone Streaming Latency vs Concurrency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(concurrencies)

        ax2.plot(concurrencies, [r["rtf_avg"] for r in rows], "o-", color="green", linewidth=2)
        ax2.set_xlabel("Concurrent Sessions")
        ax2.set_ylabel("RTF (lower is better)")
        ax2.set_title("Voice Clone RTF vs Concurrency")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(concurrencies)

        fig.tight_layout()
        png_path = os.path.join(os.path.dirname(__file__), "clone_sweep.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {png_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")


def main():
    parser = argparse.ArgumentParser(description="Voice clone concurrency sweep")
    parser.add_argument("--url", default="ws://localhost:8091/v1/audio/speech/stream")
    parser.add_argument("--voice", default="finn")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 4, 8],
        help="Concurrency levels to sweep (default: 1 2 4 8)",
    )
    args = parser.parse_args()
    asyncio.run(sweep(args))


if __name__ == "__main__":
    main()
