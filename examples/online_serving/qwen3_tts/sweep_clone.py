"""Concurrency sweep for voice-cloning (Base/ICL) streaming TTS.

Registers a voice clone on the first warmup request, then sweeps
concurrency levels using the cached voice_name. Measures TTFA, RTF,
and throughput at each level.

Usage:
    # Minimal: use default voice file + transcript
    python sweep_clone.py

    # Custom reference audio + transcript
    python sweep_clone.py \
        --ref-audio /path/to/reference.wav \
        --ref-text /path/to/transcript.txt \
        --voice-name my_voice

    # Custom concurrency levels
    python sweep_clone.py --levels 1 2 4 8 16

    # More rounds for stable percentiles
    python sweep_clone.py --rounds 5 --warmup 2
"""

import argparse
import asyncio
import base64
import csv
import json
import math
import mimetypes
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

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DEFAULT_TEXT = (
    "Hello, this is my cloned voice speaking. "
    "What is the laziest animal in the world?"
)


def _encode_audio_file(path: str) -> str:
    """Read a local audio file and return a base64 data URI."""
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".aac": "audio/aac",
            ".webm": "audio/webm",
        }
        mime_type = mime_map.get(ext, "audio/wav")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


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


async def run_one_session(url: str, text: str, config: dict) -> dict:
    """Run a single streaming TTS session, return timing metrics."""
    async with websockets.connect(url) as ws:
        config_msg = {"type": "session.config", **config}
        t0 = time.perf_counter()
        await ws.send(json.dumps(config_msg))

        ttfa = None
        total_pcm_bytes = 0
        chunk_count = 0
        sample_rate = 24000
        registered = False

        # Read messages until we get past any voice.registered / error
        # before sending text, since registration happens before input.
        while True:
            # Send text + done right after config (server processes
            # registration synchronously before reading input)
            if not registered:
                await ws.send(json.dumps({"type": "input.text", "text": text}))
                await ws.send(json.dumps({"type": "input.done"}))
                registered = True

            message = await ws.recv()
            if isinstance(message, bytes):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                total_pcm_bytes += len(message)
                chunk_count += 1
            else:
                msg = json.loads(message)
                msg_type = msg.get("type")
                if msg_type == "session.done":
                    break
                elif msg_type == "audio.done":
                    sample_rate = msg.get("sample_rate", 24000)
                elif msg_type == "error":
                    return {"error": msg.get("message", "unknown")}
                # voice.registered, audio.start — just continue

        t_total = time.perf_counter() - t0
        duration_s = total_pcm_bytes / (sample_rate * 2) if total_pcm_bytes else 0
        return {
            "ttfa_ms": ttfa * 1000 if ttfa else None,
            "total_ms": t_total * 1000,
            "audio_duration_s": duration_s,
            "rtf": t_total / duration_s if duration_s > 0 else None,
            "chunks": chunk_count,
        }


async def run_at_concurrency(
    url: str, text: str, config: dict, concurrency: int, rounds: int
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


async def register_voice(
    url: str,
    voice_name: str,
    ref_audio_uri: str,
    ref_text: str | None,
) -> None:
    """Register the voice clone (cold path) with a short dummy generation."""
    config: dict = {"voice_name": voice_name, "ref_audio": ref_audio_uri}
    if ref_text:
        config["ref_text"] = ref_text

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "session.config", **config}))
        await ws.send(json.dumps({"type": "input.text", "text": "Hello."}))
        await ws.send(json.dumps({"type": "input.done"}))

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                continue
            msg = json.loads(message)
            if msg.get("type") == "session.done":
                break
            elif msg.get("type") == "error":
                raise RuntimeError(f"Registration failed: {msg.get('message')}")


async def sweep(args):
    script_dir = os.path.dirname(__file__)

    # Resolve ref_audio
    ref_audio_uri = None
    if args.ref_audio:
        if not os.path.isfile(args.ref_audio):
            print(f"Error: --ref-audio file not found: {args.ref_audio}")
            raise SystemExit(1)
        ref_audio_uri = _encode_audio_file(args.ref_audio)
        size_kb = os.path.getsize(args.ref_audio) / 1024
        print(f"Reference audio: {args.ref_audio} ({size_kb:.1f} KB)")

    # Resolve ref_text (can be a file path)
    ref_text = args.ref_text
    if ref_text and os.path.isfile(ref_text):
        with open(ref_text) as f:
            ref_text = f.read().strip()
        print(f"Reference text (from file): {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")

    voice_name = args.voice_name

    print(f"\nVoice-clone concurrency sweep")
    print(f"  URL:        {args.url}")
    print(f"  Voice name: {voice_name}")
    print(f"  Text:       {args.text!r}")
    print(f"  Rounds:     {args.rounds}  |  Warmup: {args.warmup}")
    print(f"  Levels:     {args.levels}")
    print("=" * 70)

    # Step 1: Register the voice clone (cold path)
    if ref_audio_uri:
        print("\nRegistering voice clone (cold path)...", end="", flush=True)
        t0 = time.perf_counter()
        await register_voice(args.url, voice_name, ref_audio_uri, ref_text)
        reg_ms = (time.perf_counter() - t0) * 1000
        print(f" done ({reg_ms:.0f} ms)")
    else:
        print(f"\nNo --ref-audio provided; assuming '{voice_name}' is already cached.")

    # Config for cached-voice sessions (no ref_audio needed)
    config: dict = {
        "voice_name": voice_name,
        "language": "Auto",
        "response_format": "wav",
        "speed": 1.0,
    }

    # Step 2: Warmup
    if args.warmup > 0:
        print(f"\nWarmup ({args.warmup} round(s) at c=1)...", end="", flush=True)
        await run_at_concurrency(args.url, args.text, config, 1, args.warmup)
        print(" done")

    # Step 3: Sweep
    rows: list[dict] = []

    for level in args.levels:
        print(
            f"\n  [c={level:>2}] benchmarking ({args.rounds} rounds x {level} sessions)...",
            end="",
            flush=True,
        )
        results = await run_at_concurrency(
            args.url, args.text, config, level, args.rounds
        )
        print(" done")

        ok = [r for r in results if "error" not in r]
        errors = len(results) - len(ok)
        ttfas = [r["ttfa_ms"] for r in ok if r.get("ttfa_ms") is not None]
        totals = [r["total_ms"] for r in ok]
        rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
        audio_secs = [r["audio_duration_s"] for r in ok]

        p50 = percentile(ttfas, 50)
        p90 = percentile(ttfas, 90)
        p99 = percentile(ttfas, 99)
        rtf_p50 = percentile(rtfs, 50)
        rtf_p90 = percentile(rtfs, 90)
        total_audio = sum(audio_secs)
        wall = max(totals) / 1000 if totals else 0
        throughput = total_audio / wall if wall > 0 else 0

        row = {
            "concurrency": level,
            "samples": len(ttfas),
            "errors": errors,
            "ttfa_p50": p50,
            "ttfa_p90": p90,
            "ttfa_p99": p99,
            "rtf_p50": rtf_p50,
            "rtf_p90": rtf_p90,
            "total_p50": percentile(totals, 50),
            "throughput_x": throughput,
        }
        rows.append(row)

        print(
            f"           TTFA  p50={p50:>7.0f}ms  p90={p90:>7.0f}ms  p99={p99:>7.0f}ms  "
            f"| RTF  p50={rtf_p50:.2f}x  p90={rtf_p90:.2f}x  "
            f"| throughput={throughput:.1f}x"
            f"{'  errors=' + str(errors) if errors else ''}"
        )

    # --- Save CSV ---
    csv_path = os.path.join(script_dir, args.output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # --- Plot ---
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plot")
        return

    concs = [r["concurrency"] for r in rows]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.plot(concs, [r["ttfa_p50"] for r in rows], "o-", label="p50", linewidth=2)
    ax1.plot(concs, [r["ttfa_p90"] for r in rows], "s-", label="p90", linewidth=2)
    ax1.plot(concs, [r["ttfa_p99"] for r in rows], "^-", label="p99", linewidth=2)
    ax1.set_xlabel("Concurrency", fontsize=12)
    ax1.set_ylabel("TTFA (ms)", fontsize=12)
    ax1.set_title("Voice Clone: TTFA vs Concurrency", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(concs)

    ax2.plot(concs, [r["rtf_p50"] for r in rows], "o-", label="p50", linewidth=2)
    ax2.plot(concs, [r["rtf_p90"] for r in rows], "s-", label="p90", linewidth=2)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="realtime")
    ax2.set_xlabel("Concurrency", fontsize=12)
    ax2.set_ylabel("RTF (lower = faster)", fontsize=12)
    ax2.set_title("Voice Clone: RTF vs Concurrency", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(concs)

    ax3.plot(
        concs,
        [r["throughput_x"] for r in rows],
        "D-",
        color="green",
        linewidth=2,
    )
    ax3.set_xlabel("Concurrency", fontsize=12)
    ax3.set_ylabel("Throughput (x realtime)", fontsize=12)
    ax3.set_title("Voice Clone: Throughput vs Concurrency", fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(concs)

    fig.tight_layout()
    plot_path = os.path.join(script_dir, args.output_plot)
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(__file__)
    default_ref_audio = os.path.join(script_dir, "finn-20s.m4a")
    default_ref_text = os.path.join(script_dir, "transcript.txt")

    parser = argparse.ArgumentParser(
        description="Voice clone concurrency sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="ws://localhost:8091/v1/audio/speech/stream")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice-name", default="finn")
    parser.add_argument(
        "--ref-audio",
        default=default_ref_audio if os.path.isfile(default_ref_audio) else None,
        help="Reference audio file for cloning (default: finn-20s.m4a if present)",
    )
    parser.add_argument(
        "--ref-text",
        default=default_ref_text if os.path.isfile(default_ref_text) else None,
        help="Transcript of reference audio, or path to .txt file (default: transcript.txt if present)",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Concurrency levels to sweep (default: 1 2 4 8)",
    )
    parser.add_argument("--output-csv", default="clone_sweep.csv")
    parser.add_argument("--output-plot", default="clone_sweep.png")
    args = parser.parse_args()

    asyncio.run(sweep(args))


if __name__ == "__main__":
    main()
