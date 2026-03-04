"""Concurrency sweep for CustomVoice streaming TTS.

Sweeps concurrency levels using the CustomVoice model (1.7B-CustomVoice),
measures TTFA (Time To First Audio), RTF (Real-Time Factor), and throughput,
then saves results as JSON + optional CSV and plots.

Usage:
    # Default sweep (concurrency 1,2,4,8)
    python sweep_customvoice.py

    # Custom levels, more rounds
    python sweep_customvoice.py --levels 1 2 4 8 16 32 --rounds 5

    # Use a specific voice with style instructions
    python sweep_customvoice.py --voice Chelsie --instructions "Speak slowly and calmly"

    # Multiple text prompts of varying length
    python sweep_customvoice.py --text-file prompts.txt

    # Save audio output + plots
    python sweep_customvoice.py --save-audio --plot

Requirements:
    pip install websockets
    pip install matplotlib  # optional, for plots
"""

import argparse
import asyncio
import csv
import json
import math
import os
import time
import wave

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise SystemExit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


SAMPLE_TEXTS = [
    "Hello world. How are you today? I am doing very well, thank you for asking.",
    (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence contains every letter of the English alphabet."
    ),
    (
        "In a quiet village nestled among rolling hills, a young inventor "
        "spent her evenings tinkering with old clockwork and dreaming of "
        "machines that could fly."
    ),
]


def _write_wav(path: str, pcm_data: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


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


async def run_one_session(
    url: str,
    text: str,
    config: dict,
    session_id: int = 0,
    stagger_delay: float = 0.0,
    save_path: str | None = None,
    timeout: float | None = None,
) -> dict:
    if stagger_delay > 0:
        await asyncio.sleep(stagger_delay * session_id)

    async def _do() -> dict:
        async with websockets.connect(url) as ws:
            config_msg = {"type": "session.config", **config}
            t0 = time.perf_counter()
            await ws.send(json.dumps(config_msg))
            await ws.send(json.dumps({"type": "input.text", "text": text}))
            await ws.send(json.dumps({"type": "input.done"}))

            ttfa = None
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
                    if save_path:
                        all_pcm.append(message)
                else:
                    msg = json.loads(message)
                    if msg.get("type") == "audio.done":
                        sample_rate = msg.get("sample_rate", 24000)
                    elif msg.get("type") == "session.done":
                        break
                    elif msg.get("type") == "error":
                        return {"error": msg.get("message", "unknown"), "session_id": session_id}

            t_total = time.perf_counter() - t0
            duration_s = total_pcm_bytes / (sample_rate * 2) if total_pcm_bytes else 0

            if save_path and all_pcm:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                _write_wav(save_path, b"".join(all_pcm), sample_rate)

            return {
                "session_id": session_id,
                "ttfa_ms": ttfa * 1000 if ttfa else None,
                "total_ms": t_total * 1000,
                "audio_duration_s": duration_s,
                "rtf": t_total / duration_s if duration_s > 0 else None,
                "chunks": chunk_count,
                "text_len": len(text),
            }

    try:
        if timeout is not None:
            return await asyncio.wait_for(_do(), timeout=timeout)
        return await _do()
    except asyncio.TimeoutError:
        return {"error": f"timeout after {timeout}s", "session_id": session_id}
    except Exception as exc:
        return {"error": str(exc), "session_id": session_id}


async def run_at_concurrency(
    url: str,
    texts: list[str],
    config: dict,
    concurrency: int,
    rounds: int,
    stagger: float = 0.0,
    output_dir: str | None = None,
    timeout: float | None = None,
) -> list[dict]:
    all_results: list[dict] = []
    for rnd in range(rounds):
        tasks = []
        for i in range(concurrency):
            text = texts[i % len(texts)]
            save_path = None
            if output_dir:
                save_path = os.path.join(output_dir, f"c{concurrency}_r{rnd}_s{i}.wav")
            tasks.append(
                run_one_session(
                    url, text, config,
                    session_id=i,
                    stagger_delay=stagger,
                    save_path=save_path,
                    timeout=timeout,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                all_results.append({"error": str(r), "session_id": i, "round": rnd})
            else:
                r["round"] = rnd
                r["concurrency"] = concurrency
                all_results.append(r)
    return all_results


def summarise(results: list[dict], concurrency: int) -> dict:
    ok = [r for r in results if "error" not in r]
    errors = len(results) - len(ok)
    ttfas = [r["ttfa_ms"] for r in ok if r.get("ttfa_ms") is not None]
    totals = [r["total_ms"] for r in ok]
    rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
    audio_secs = [r["audio_duration_s"] for r in ok]

    total_audio = sum(audio_secs)
    wall = max(totals) / 1000 if totals else 0
    throughput = total_audio / wall if wall > 0 else 0

    return {
        "concurrency": concurrency,
        "samples": len(ok),
        "errors": errors,
        "ttfa_p50": percentile(ttfas, 50),
        "ttfa_p90": percentile(ttfas, 90),
        "ttfa_p95": percentile(ttfas, 95),
        "ttfa_p99": percentile(ttfas, 99),
        "ttfa_min": min(ttfas) if ttfas else float("nan"),
        "ttfa_max": max(ttfas) if ttfas else float("nan"),
        "rtf_p50": percentile(rtfs, 50),
        "rtf_p90": percentile(rtfs, 90),
        "rtf_p95": percentile(rtfs, 95),
        "total_p50": percentile(totals, 50),
        "total_p90": percentile(totals, 90),
        "throughput_x": throughput,
        "total_audio_s": total_audio,
        "wall_clock_s": wall,
    }


def print_table(summaries: list[dict]) -> None:
    header = (
        f"{'Conc':>5}  {'N':>4}  {'Err':>3}  "
        f"{'TTFA p50':>9}  {'TTFA p90':>9}  {'TTFA p99':>9}  "
        f"{'RTF p50':>8}  {'RTF p90':>8}  "
        f"{'Tput':>6}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("CustomVoice CONCURRENCY SWEEP RESULTS")
    print(sep)
    print(header)
    print("-" * len(header))

    for s in summaries:
        def _ms(v):
            return f"{v:.0f}ms" if not math.isnan(v) else "n/a"

        def _x(v):
            return f"{v:.2f}x" if not math.isnan(v) else "n/a"

        err_str = str(s["errors"]) if s["errors"] else ""
        print(
            f"{s['concurrency']:>5}  {s['samples']:>4}  {err_str:>3}  "
            f"{_ms(s['ttfa_p50']):>9}  {_ms(s['ttfa_p90']):>9}  {_ms(s['ttfa_p99']):>9}  "
            f"{_x(s['rtf_p50']):>8}  {_x(s['rtf_p90']):>8}  "
            f"{_x(s['throughput_x']):>6}"
        )

    print(sep)


def save_plot(summaries: list[dict], path: str) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plot")
        return

    concs = [s["concurrency"] for s in summaries]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.plot(concs, [s["ttfa_p50"] for s in summaries], "o-", label="p50", linewidth=2)
    ax1.plot(concs, [s["ttfa_p90"] for s in summaries], "s-", label="p90", linewidth=2)
    ax1.plot(concs, [s["ttfa_p99"] for s in summaries], "^-", label="p99", linewidth=2)
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("TTFA (ms)")
    ax1.set_title("CustomVoice: TTFA vs Concurrency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(concs)

    ax2.plot(concs, [s["rtf_p50"] for s in summaries], "o-", label="p50", linewidth=2)
    ax2.plot(concs, [s["rtf_p90"] for s in summaries], "s-", label="p90", linewidth=2)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="realtime")
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("RTF (lower = faster)")
    ax2.set_title("CustomVoice: RTF vs Concurrency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(concs)

    ax3.plot(concs, [s["throughput_x"] for s in summaries], "D-", color="green", linewidth=2)
    ax3.set_xlabel("Concurrency")
    ax3.set_ylabel("Throughput (x realtime)")
    ax3.set_title("CustomVoice: Throughput vs Concurrency")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(concs)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Plot saved: {path}")
    plt.close(fig)


async def sweep(args):
    config: dict = {
        "voice": args.voice,
        "task_type": "CustomVoice",
        "language": "Auto",
        "response_format": "wav",
        "speed": args.speed,
    }
    if args.instructions:
        config["instructions"] = args.instructions

    texts = []
    if args.text_file:
        with open(args.text_file) as f:
            texts.extend(line.strip() for line in f if line.strip())
    if args.text:
        texts.extend(args.text)
    if not texts:
        texts = list(SAMPLE_TEXTS)

    print("=" * 70)
    print("CustomVoice CONCURRENCY SWEEP")
    print("=" * 70)
    print(f"  URL:          {args.url}")
    print(f"  Voice:        {args.voice}")
    if args.instructions:
        print(f"  Instructions: {args.instructions!r}")
    print(f"  Speed:        {args.speed}")
    print(f"  Prompts:      {len(texts)}")
    for i, t in enumerate(texts):
        preview = t if len(t) <= 70 else t[:67] + "..."
        print(f"    [{i}] ({len(t)} chars) {preview!r}")
    print(f"  Levels:       {args.levels}")
    print(f"  Rounds:       {args.rounds}")
    print(f"  Stagger:      {args.stagger}s")
    print(f"  Warmup:       {args.warmup}")
    print("=" * 70)

    # Warmup
    if args.warmup > 0:
        print(f"\nWarming up ({args.warmup} session(s))...", end="", flush=True)
        for w in range(args.warmup):
            r = await run_one_session(args.url, texts[0], config, session_id=w)
            if "error" in r:
                print(f"\n  warmup {w}: ERROR {r['error']}")
            else:
                print(
                    f"\n  warmup {w}: TTFA={r['ttfa_ms']:.0f}ms  "
                    f"RTF={r['rtf']:.2f}x  audio={r['audio_duration_s']:.1f}s",
                    end="",
                )
        print(" done\n")

    all_summaries: list[dict] = []
    all_raw: list[dict] = []

    for level in args.levels:
        print(
            f"  [c={level:>3}] benchmarking ({args.rounds} rounds x {level} sessions)...",
            end="",
            flush=True,
        )
        results = await run_at_concurrency(
            args.url, texts, config, level, args.rounds,
            stagger=args.stagger,
            output_dir=args.output_dir if args.save_audio else None,
            timeout=args.session_timeout,
        )
        all_raw.extend(results)
        print(" done")

        summary = summarise(results, level)
        all_summaries.append(summary)

        ttfa = summary["ttfa_p50"]
        rtf = summary["rtf_p50"]
        tput = summary["throughput_x"]
        errs = summary["errors"]
        ttfa_str = f"{ttfa:.0f}" if not math.isnan(ttfa) else "n/a"
        rtf_str = f"{rtf:.2f}" if not math.isnan(rtf) else "n/a"
        err_str = f"  errors={errs}" if errs else ""
        print(
            f"           TTFA p50={ttfa_str}ms  "
            f"RTF p50={rtf_str}x  "
            f"throughput={tput:.1f}x{err_str}"
        )

    print_table(all_summaries)

    # Save JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, args.json)
    payload = {
        "config": {
            "url": args.url,
            "task_type": "CustomVoice",
            "voice": args.voice,
            "instructions": args.instructions,
            "speed": args.speed,
            "levels": args.levels,
            "rounds": args.rounds,
            "stagger": args.stagger,
            "warmup": args.warmup,
            "texts": texts,
        },
        "summaries": all_summaries,
        "raw": all_raw,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nJSON results: {json_path}")

    # Save CSV
    if args.csv:
        csv_path = os.path.join(script_dir, args.csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"CSV results:  {csv_path}")

    # Save plot
    if args.plot:
        plot_path = os.path.join(script_dir, args.plot_path)
        save_plot(all_summaries, plot_path)


def main():
    parser = argparse.ArgumentParser(
        description="Concurrency sweep for CustomVoice streaming TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--url",
        default="ws://localhost:8091/v1/audio/speech/stream",
        help="WebSocket endpoint URL",
    )

    # Sweep parameters
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Concurrency levels to sweep (default: 1 2 4 8)",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per level (default: 3)")
    parser.add_argument("--stagger", type=float, default=0.0, help="Stagger between connections (default: 0)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup sessions (default: 1)")
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=None,
        metavar="SECS",
        help="Per-session timeout in seconds",
    )

    # Text input
    parser.add_argument("--text", action="append", default=None, help="Text prompt (repeatable)")
    parser.add_argument("--text-file", default=None, help="File with one prompt per line")

    # CustomVoice config
    parser.add_argument("--voice", default="Vivian", help="Speaker voice name (default: Vivian)")
    parser.add_argument(
        "--instructions",
        default=None,
        help="Voice style instructions, e.g. 'Speak with excitement'",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")

    # Output
    parser.add_argument(
        "--json",
        default="customvoice_sweep.json",
        help="JSON results file (default: customvoice_sweep.json)",
    )
    parser.add_argument("--csv", default=None, metavar="PATH", help="Also save summary CSV")
    parser.add_argument("--plot", action="store_true", help="Save plots (requires matplotlib)")
    parser.add_argument("--plot-path", default="customvoice_sweep.png", help="Plot filename")
    parser.add_argument("--save-audio", action="store_true", help="Save generated WAV files")
    parser.add_argument("--output-dir", default="customvoice_audio", help="Directory for WAV files")

    args = parser.parse_args()
    asyncio.run(sweep(args))


if __name__ == "__main__":
    main()
