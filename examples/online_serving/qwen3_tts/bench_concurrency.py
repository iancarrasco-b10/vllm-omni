#!/usr/bin/env python3
"""Concurrency benchmark for TTS endpoint. Measures RTF at increasing concurrency levels."""

import argparse
import asyncio
import struct
import time

import httpx

DEFAULT_BASE_URL = "http://localhost:8091"
TEXT = "Hey, is there something you want to talk about?"
VOICE = "t7"


def parse_audio_duration(raw: bytes) -> float:
    if len(raw) < 44 or raw[:4] != b"RIFF":
        return len(raw) / (24000 * 2)
    sr = struct.unpack_from("<I", raw, 24)[0]
    offset = 12
    while offset + 8 <= len(raw):
        cid = raw[offset : offset + 4]
        csz = struct.unpack_from("<I", raw, offset + 4)[0]
        if cid == b"data":
            return csz / (sr * 2)
        offset += 8 + csz
    return (len(raw) - 44) / (sr * 2)


async def tts_request(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=120.0)
        elapsed = time.perf_counter() - t0
        if resp.status_code != 200:
            return {"ok": False, "elapsed": elapsed, "error": f"HTTP {resp.status_code}"}
        audio_dur = parse_audio_duration(resp.content)
        return {"ok": True, "elapsed": elapsed, "audio_dur": audio_dur}
    except Exception as e:
        return {"ok": False, "elapsed": time.perf_counter() - t0, "error": str(e)[:80]}


async def run_batch(n: int, concurrency: int, url: str, payload: dict) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def limited(client):
        async with sem:
            return await tts_request(client, url, payload)

    async with httpx.AsyncClient() as client:
        tasks = [limited(client) for _ in range(n)]
        return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="TTS concurrency sweep")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--runs", type=int, default=6, help="Min requests per level")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds between levels")
    args = parser.parse_args()

    url = f"{args.base_url}/v1/audio/speech"
    payload = {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "input": args.text,
        "voice": args.voice,
        "response_format": "wav",
    }

    # Warmup
    print(f"Warming up ({args.warmup} sequential requests)...")
    for i in range(args.warmup):
        r = asyncio.run(run_batch(1, 1, url, payload))
        if r[0]["ok"]:
            print(f"  warmup {i+1}: {r[0]['elapsed']*1000:.0f}ms, audio={r[0]['audio_dur']:.2f}s")
        else:
            print(f"  warmup {i+1}: FAILED - {r[0].get('error','?')}")

    concurrencies = []
    c = 1
    while c <= args.max_concurrency:
        concurrencies.append(c)
        c *= 2

    results = []
    hdr = (f"{'Conc':>5} {'Reqs':>5} {'Wall(s)':>8} {'AvgE2E':>8} {'AvgRTF':>7} "
           f"{'Audio':>7} {'Tput(xRT)':>9} {'AvgC/s':>7} {'AggC/s':>8}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for conc in concurrencies:
        n = max(conc, args.runs)

        time.sleep(args.cooldown)

        t_wall_start = time.perf_counter()
        batch = asyncio.run(run_batch(n, conc, url, payload))
        t_wall = time.perf_counter() - t_wall_start

        ok = [r for r in batch if r["ok"]]
        fail = len(batch) - len(ok)

        if not ok:
            errs = set(r.get("error", "?") for r in batch if not r["ok"])
            print(f"{conc:>5} {n:>5}  ALL FAILED: {'; '.join(errs)}")
            continue

        avg_e2e = sum(r["elapsed"] for r in ok) / len(ok)
        avg_audio = sum(r["audio_dur"] for r in ok) / len(ok)
        avg_rtf = avg_e2e / avg_audio if avg_audio > 0 else float("inf")
        total_audio = sum(r["audio_dur"] for r in ok)
        throughput = total_audio / t_wall

        fail_str = f" ({fail} fail)" if fail else ""
        text_len = len(args.text)
        avg_cps = text_len / avg_e2e if avg_e2e > 0 else 0
        total_chars = text_len * len(ok)
        agg_cps = total_chars / t_wall if t_wall > 0 else 0
        results.append({
            "concurrency": conc,
            "n": len(ok),
            "wall_s": t_wall,
            "avg_e2e": avg_e2e,
            "avg_rtf": avg_rtf,
            "avg_audio": avg_audio,
            "throughput": throughput,
            "avg_cps": avg_cps,
            "agg_cps": agg_cps,
        })

        print(f"{conc:>5} {len(ok):>5} {t_wall:>7.2f}s {avg_e2e:>7.2f}s "
              f"{avg_rtf:>6.2f}x {avg_audio:>6.2f}s {throughput:>8.2f}x "
              f"{avg_cps:>6.1f}c/s {agg_cps:>7.1f}c/s{fail_str}")

    if not results:
        print("\nNo successful results to plot.")
        return

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        concs = [r["concurrency"] for r in results]
        rtfs = [r["avg_rtf"] for r in results]
        tputs = [r["throughput"] for r in results]
        avg_cps_list = [r["avg_cps"] for r in results]
        agg_cps_list = [r["agg_cps"] for r in results]

        from matplotlib.ticker import FixedLocator, FixedFormatter
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        tick_labels = [str(c) for c in concs]

        ax1.plot(concs, rtfs, "o-", linewidth=2, markersize=8, color="#2563eb")
        ax1.set_xlabel("Concurrency", fontsize=12)
        ax1.set_ylabel("Avg RTF (lower = faster)", fontsize=12)
        ax1.set_title("RTF vs Concurrency", fontsize=13)
        ax1.set_xscale("log", base=2)
        ax1.xaxis.set_major_locator(FixedLocator(concs))
        ax1.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax1.grid(True, alpha=0.3)
        for x, y in zip(concs, rtfs):
            ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

        ax2.plot(concs, tputs, "s-", color="#16a34a", linewidth=2, markersize=8)
        ax2.set_xlabel("Concurrency", fontsize=12)
        ax2.set_ylabel("Throughput (x realtime)", fontsize=12)
        ax2.set_title("Throughput vs Concurrency", fontsize=13)
        ax2.set_xscale("log", base=2)
        ax2.xaxis.set_major_locator(FixedLocator(concs))
        ax2.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax2.grid(True, alpha=0.3)
        for x, y in zip(concs, tputs):
            ax2.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

        ax3.plot(concs, avg_cps_list, "o-", linewidth=2, markersize=8, color="#dc2626",
                 label="Per-request")
        ax3.plot(concs, agg_cps_list, "D-", linewidth=2, markersize=8, color="#9333ea",
                 label="Aggregate")
        ax3.set_xlabel("Concurrency", fontsize=12)
        ax3.set_ylabel("Chars / sec", fontsize=12)
        ax3.set_title("Chars/sec vs Concurrency", fontsize=13)
        ax3.set_xscale("log", base=2)
        ax3.xaxis.set_major_locator(FixedLocator(concs))
        ax3.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        for x, y in zip(concs, avg_cps_list):
            ax3.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=9, color="#dc2626")
        for x, y in zip(concs, agg_cps_list):
            ax3.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, color="#9333ea")

        text_short = args.text[:40] + ("..." if len(args.text) > 40 else "")
        fig.suptitle(f"TTS Concurrency Benchmark — \"{text_short}\"", fontsize=11)
        fig.tight_layout()
        out = "bench_concurrency.png"
        fig.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
    except ImportError:
        print("\nInstall matplotlib for plots: pip install matplotlib")


if __name__ == "__main__":
    main()
