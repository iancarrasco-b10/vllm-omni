"""Concurrency sweep benchmark for streaming TTS with voice cloning.

Runs the streaming TTS WebSocket client at increasing concurrency levels
and reports TTFA (Time To First Audio) and RTF (Real-Time Factor) statistics
at each level.  Supports staggered connections so that sessions ramp up
gradually rather than hammering the server all at once.

TTFA with voice cloning (Base): If you pass --ref-audio and --voice-name,
the first request is used to register the voice (cold path); all sweep
sessions then use only voice_name (cached). This matches sweep_clone.py
and keeps TTFA low. If you omit --voice-name and only pass --ref-audio,
every session re-sends the reference audio and pays speaker-embedding
cost, so TTFA will be much higher.

Usage:
    # Sweep 1,2,4,8 concurrent sessions (default levels)
    python concurrency_sweep.py

    # Custom concurrency levels with staggered start
    python concurrency_sweep.py --levels 1 2 4 8 16 --stagger 0.1

    # Voice cloning sweep
    python concurrency_sweep.py \
        --levels 1 2 4 --rounds 3 --stagger 0.05 \
        --task-type Base \
        --ref-audio /path/to/reference.wav \
        --ref-text "Transcript of the reference audio." \
        --text "Clone my voice saying this sentence."

    # Save results to JSON
    python concurrency_sweep.py --levels 1 2 4 8 --json results.json

Requirements:
    pip install websockets
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
    if path.startswith(("http://", "https://", "data:")):
        return path
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME_TYPES.get(ext, "audio/wav")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _write_wav(path: str, pcm_data: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ---------------------------------------------------------------------------
# Single session
# ---------------------------------------------------------------------------

async def run_one_session(
    url: str,
    text: str,
    config: dict,
    session_id: int,
    stagger_delay: float = 0.0,
    output_dir: str | None = None,
    concurrency_tag: int = 0,
    round_num: int = 0,
    session_timeout: float | None = None,
) -> dict:
    """Run a single streaming TTS session after an optional stagger delay."""
    if stagger_delay > 0:
        await asyncio.sleep(stagger_delay * session_id)

    async def _do_session() -> dict:
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
                            "concurrency": concurrency_tag,
                            "error": msg["message"],
                        }

            t_total = time.perf_counter() - t0
            duration_s = total_pcm_bytes / (sample_rate * 2) if total_pcm_bytes else 0

            if output_dir and all_pcm:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(
                    output_dir,
                    f"c{concurrency_tag}_r{round_num}_s{session_id}.wav",
                )
                _write_wav(path, b"".join(all_pcm), sample_rate)

            return {
                "session_id": session_id,
                "concurrency": concurrency_tag,
                "ttfa_ms": ttfa * 1000 if ttfa else None,
                "total_ms": t_total * 1000,
                "sentences": sentence_count,
                "chunks": chunk_count,
                "pcm_bytes": total_pcm_bytes,
                "audio_duration_s": duration_s,
                "rtf": t_total / duration_s if duration_s > 0 else None,
            }

    try:
        if session_timeout is not None:
            return await asyncio.wait_for(_do_session(), timeout=session_timeout)
        return await _do_session()
    except asyncio.TimeoutError:
        return {
            "session_id": session_id,
            "concurrency": concurrency_tag,
            "error": f"session timed out after {session_timeout}s",
        }
    except Exception as exc:
        return {
            "session_id": session_id,
            "concurrency": concurrency_tag,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Round runner (one concurrency level, one round)
# ---------------------------------------------------------------------------

async def run_round(
    url: str,
    texts: list[str],
    config: dict,
    concurrency: int,
    stagger: float,
    round_num: int = 0,
    output_dir: str | None = None,
    session_timeout: float | None = None,
) -> list[dict]:
    tasks = [
        run_one_session(
            url,
            texts[i % len(texts)],
            config,
            session_id=i,
            stagger_delay=stagger,
            output_dir=output_dir,
            concurrency_tag=concurrency,
            round_num=round_num,
            session_timeout=session_timeout,
        )
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            processed.append({"session_id": i, "concurrency": concurrency, "error": str(r)})
        else:
            processed.append(r)
    return processed


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stat_line(values: list[float], unit: str = "ms") -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.1f}{unit}"
    return (
        f"avg={statistics.mean(values):.1f}{unit}  "
        f"p50={statistics.median(values):.1f}{unit}  "
        f"min={min(values):.1f}{unit}  "
        f"max={max(values):.1f}{unit}  "
        f"stdev={statistics.stdev(values):.1f}{unit}"
    )


def _percentile(values: list[float], p: float) -> float:
    """Simple linear-interpolation percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def summarise_level(results: list[dict], concurrency: int) -> dict:
    """Compute summary statistics for one concurrency level."""
    ok = [r for r in results if "error" not in r]
    errors = len(results) - len(ok)
    ttfas = [r["ttfa_ms"] for r in ok if r["ttfa_ms"] is not None]
    totals = [r["total_ms"] for r in ok]
    rtfs = [r["rtf"] for r in ok if r["rtf"] is not None]
    audio_durations = [r["audio_duration_s"] for r in ok]

    total_audio = sum(audio_durations)
    wall_clock = max(totals) / 1000 if totals else 0

    summary: dict = {
        "concurrency": concurrency,
        "sessions": len(ok),
        "errors": errors,
    }

    for name, vals, unit in [
        ("ttfa", ttfas, "ms"),
        ("total", totals, "ms"),
        ("rtf", rtfs, "x"),
    ]:
        if vals:
            summary[f"{name}_avg"] = statistics.mean(vals)
            summary[f"{name}_p50"] = statistics.median(vals)
            summary[f"{name}_p95"] = _percentile(vals, 0.95)
            summary[f"{name}_p99"] = _percentile(vals, 0.99)
            summary[f"{name}_min"] = min(vals)
            summary[f"{name}_max"] = max(vals)
            if len(vals) > 1:
                summary[f"{name}_stdev"] = statistics.stdev(vals)

    if wall_clock > 0:
        summary["throughput_x"] = total_audio / wall_clock
    summary["total_audio_s"] = total_audio
    summary["wall_clock_s"] = wall_clock

    return summary


def print_level_summary(summary: dict) -> None:
    c = summary["concurrency"]
    n = summary["sessions"]
    errs = summary["errors"]
    err_str = f"  ({errs} errors)" if errs else ""

    print(f"\n  Concurrency {c}  |  {n} sessions{err_str}")

    if "ttfa_avg" in summary:
        print(
            f"  TTFA:  avg={summary['ttfa_avg']:.0f}ms  "
            f"p50={summary['ttfa_p50']:.0f}ms  "
            f"p95={summary['ttfa_p95']:.0f}ms  "
            f"min={summary['ttfa_min']:.0f}ms  "
            f"max={summary['ttfa_max']:.0f}ms"
        )
    if "rtf_avg" in summary:
        print(
            f"  RTF:   avg={summary['rtf_avg']:.2f}x  "
            f"p50={summary['rtf_p50']:.2f}x  "
            f"p95={summary['rtf_p95']:.2f}x  "
            f"min={summary['rtf_min']:.2f}x  "
            f"max={summary['rtf_max']:.2f}x"
        )
    if "throughput_x" in summary:
        print(
            f"  Throughput: {summary['total_audio_s']:.1f}s audio in "
            f"{summary['wall_clock_s']:.1f}s wall = "
            f"{summary['throughput_x']:.2f}x realtime"
        )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(summaries: list[dict]) -> None:
    header = (
        f"{'Conc':>5}  {'Sess':>5}  "
        f"{'TTFA avg':>9}  {'TTFA p50':>9}  {'TTFA p95':>9}  {'TTFA max':>9}  "
        f"{'RTF avg':>8}  {'RTF p95':>8}  {'RTF max':>8}  "
        f"{'Tput':>6}"
    )
    print("\n" + "=" * len(header))
    print("SWEEP SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for s in summaries:
        ttfa_avg = f"{s.get('ttfa_avg', 0):.0f}ms" if "ttfa_avg" in s else "n/a"
        ttfa_p50 = f"{s.get('ttfa_p50', 0):.0f}ms" if "ttfa_p50" in s else "n/a"
        ttfa_p95 = f"{s.get('ttfa_p95', 0):.0f}ms" if "ttfa_p95" in s else "n/a"
        ttfa_max = f"{s.get('ttfa_max', 0):.0f}ms" if "ttfa_max" in s else "n/a"
        rtf_avg = f"{s.get('rtf_avg', 0):.2f}x" if "rtf_avg" in s else "n/a"
        rtf_p95 = f"{s.get('rtf_p95', 0):.2f}x" if "rtf_p95" in s else "n/a"
        rtf_max = f"{s.get('rtf_max', 0):.2f}x" if "rtf_max" in s else "n/a"
        tput = f"{s.get('throughput_x', 0):.1f}x" if "throughput_x" in s else "n/a"

        print(
            f"{s['concurrency']:>5}  {s['sessions']:>5}  "
            f"{ttfa_avg:>9}  {ttfa_p50:>9}  {ttfa_p95:>9}  {ttfa_max:>9}  "
            f"{rtf_avg:>8}  {rtf_p95:>8}  {rtf_max:>8}  "
            f"{tput:>6}"
        )

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _resolve_texts(args) -> list[str]:
    texts: list[str] = []
    if args.text_file:
        with open(args.text_file) as f:
            texts.extend(line.strip() for line in f if line.strip())
    if args.text:
        texts.extend(args.text)
    if not texts:
        texts.append(
            "Hello world. How are you today? I am doing very well, "
            "thank you for asking."
        )
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    if args.voice_name:
        config["voice_name"] = args.voice_name
    if args.x_vector_only_mode:
        config["x_vector_only_mode"] = True
    if args.instructions:
        config["instructions"] = args.instructions
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens

    # For Base/voice-clone: register once then sweep with voice_name only, so TTFA
    # is comparable to sweep_clone (otherwise every session re-sends ref_audio).
    sweep_config = dict(config)
    do_register_first = (
        args.task_type == "Base"
        and args.ref_audio
        and args.voice_name
    )
    if do_register_first:
        sweep_config = {k: v for k, v in config.items() if k not in ("ref_audio", "ref_text")}

    texts = _resolve_texts(args)

    print("=" * 60)
    print("CONCURRENCY SWEEP")
    print("=" * 60)
    print(f"Target:      {args.url}")
    print(f"Task type:   {args.task_type}")
    if args.ref_audio:
        print(f"Ref audio:   {args.ref_audio}")
    if args.ref_text:
        preview = args.ref_text if len(args.ref_text) <= 60 else args.ref_text[:57] + "..."
        print(f"Ref text:    {preview!r}")
    if args.voice_name:
        print(f"Voice name:  {args.voice_name}")
    print(f"Prompts:     {len(texts)}")
    for i, t in enumerate(texts):
        preview = t if len(t) <= 70 else t[:67] + "..."
        print(f"  [{i}] {preview!r}")
    print(f"Levels:      {args.levels}")
    print(f"Rounds:      {args.rounds}")
    print(f"Stagger:     {args.stagger}s between connections")
    if args.warmup:
        print(f"Warmup:      {args.warmup} session(s)")
    print("=" * 60)

    # One-time registration for Base + ref_audio + voice_name (so sweep uses cache)
    if do_register_first:
        print("\nRegistering voice (one session with ref_audio)...", end="", flush=True)
        reg_result = await run_one_session(
            args.url, texts[0], config, session_id=0,
            concurrency_tag=0, round_num=-1,
        )
        if "error" in reg_result:
            print(f" ERROR: {reg_result['error']}")
            print("Sweep will still run but each session may re-send ref_audio.")
        else:
            print(" done. Sweep will use cached voice_name.")

    # Optional warmup: send a few requests so the server is primed
    if args.warmup > 0:
        print(f"\nWarming up with {args.warmup} sequential session(s)...")
        for w in range(args.warmup):
            result = await run_one_session(
                args.url, texts[0], sweep_config, session_id=w,
                concurrency_tag=0, round_num=-1,
            )
            if "error" in result:
                print(f"  warmup {w}: ERROR {result['error']}")
            else:
                print(
                    f"  warmup {w}: TTFA={result['ttfa_ms']:.0f}ms  "
                    f"RTF={result['rtf']:.2f}x"
                )

    all_summaries: list[dict] = []
    all_raw: list[dict] = []

    for level in args.levels:
        print(f"\n{'─' * 60}")
        print(f"CONCURRENCY LEVEL: {level}  (stagger={args.stagger}s)")
        print(f"{'─' * 60}")

        level_results: list[dict] = []
        for rnd in range(args.rounds):
            print(f"\n  Round {rnd + 1}/{args.rounds}")
            results = await run_round(
                args.url,
                texts,
                sweep_config,
                concurrency=level,
                stagger=args.stagger,
                round_num=rnd,
                output_dir=args.output_dir,
                session_timeout=args.session_timeout,
            )
            for r in results:
                r["round"] = rnd
                if "error" not in r:
                    print(
                        f"    s{r['session_id']:>2}: "
                        f"TTFA={r['ttfa_ms']:>7.0f}ms  "
                        f"total={r['total_ms']:>7.0f}ms  "
                        f"RTF={r['rtf']:.2f}x"
                    )
                else:
                    print(f"    s{r['session_id']:>2}: ERROR {r['error']}")
            level_results.extend(results)
            all_raw.extend(results)

        summary = summarise_level(level_results, level)
        all_summaries.append(summary)
        print_level_summary(summary)

    print_summary_table(all_summaries)

    if args.json:
        payload = {
            "config": {
                "url": args.url,
                "task_type": args.task_type,
                "voice": args.voice,
                "levels": args.levels,
                "rounds": args.rounds,
                "stagger": args.stagger,
                "warmup": args.warmup,
                "texts": texts,
            },
            "summaries": all_summaries,
            "raw": all_raw,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nResults written to {args.json}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep concurrency levels for streaming TTS, "
        "tracking TTFA and RTF with optional staggered connections.",
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
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Rounds per concurrency level (default: 3)",
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=0.0,
        help="Seconds to stagger between each connection within a round. "
        "Session i starts after stagger*i seconds. 0 = all at once (default: 0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of sequential warmup requests before sweeping (default: 1)",
    )

    # Text input
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Text prompt (repeatable). Round-robin assignment across sessions.",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="File with one prompt per line.",
    )

    # TTS config
    parser.add_argument("--voice", default="Vivian", help="Speaker voice name")
    parser.add_argument(
        "--task-type",
        default="CustomVoice",
        choices=["CustomVoice", "VoiceDesign", "Base"],
        help="TTS task type (use Base for voice cloning)",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Reference audio for voice cloning (local path or URL)",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of reference audio for voice cloning. "
        "Can be inline text or a path to a .txt file.",
    )
    parser.add_argument(
        "--voice-name",
        default=None,
        help="Cache name for the cloned voice (avoids re-extracting embeddings)",
    )
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        default=False,
        help="Speaker-embedding only mode (no ICL). Lower TTFA, slightly "
        "reduced similarity.",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Voice style instructions (for VoiceDesign / CustomVoice)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max tokens to generate per session. Guards against runaway "
        "generation from the code predictor.",
    )
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=None,
        metavar="SECS",
        help="Per-session wall-clock timeout in seconds. Sessions exceeding "
        "this are killed and marked as errors.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save WAV files (c<C>_r<R>_s<S>.wav). "
        "If not set, audio is not saved.",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Write full results (summaries + raw) to a JSON file.",
    )

    args = parser.parse_args()

    if args.ref_text and os.path.isfile(args.ref_text):
        with open(args.ref_text) as f:
            args.ref_text = f.read().strip()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
