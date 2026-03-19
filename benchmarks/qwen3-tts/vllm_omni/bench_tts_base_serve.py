"""Benchmark client for Qwen3-TTS Base (voice clone) via /v1/audio/speech.

Measures TTFP (Time-to-First-Packet), E2E latency, and RTF (Real-Time Factor)
across configurable concurrency levels for the Base voice-clone task.
Saves results as JSON for plotting.

By default, the reference audio is uploaded once as a named voice clone, then
subsequent benchmark requests reference it by name with ICL mode
(x_vector_only_mode=false).  Use --no-upload to fall back to sending
ref_audio per request.

Supports --ws mode which uses the WebSocket streaming endpoint
(/v1/audio/speech/stream) with full text sent at once, and --simulate-stt
mode which additionally drips text word-by-word with a configurable delay.

Usage:
    # Upload-once ICL mode via HTTP (default):
    python bench_tts_base_serve.py \
        --host 127.0.0.1 --port 8000 \
        --num-prompts 50 \
        --max-concurrency 1 4 10 \
        --ref-audio "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav" \
        --ref-text "Okay. Yeah. I resent you. I love you. ..." \
        --result-dir results/

    # WebSocket streaming (full text, no STT delay):
    python bench_tts_base_serve.py --ws \
        --num-prompts 50 --max-concurrency 1 4 10 ...

    # Simulate STT streaming (word-by-word with delay):
    python bench_tts_base_serve.py --simulate-stt --stt-delay 0.08 \
        --num-prompts 20 --max-concurrency 1 4 ...

    # Legacy per-request ref_audio mode:
    python bench_tts_base_serve.py --no-upload \
        --ref-audio "https://..." --ref-text "..." ...
"""

import argparse
import asyncio
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

PROMPTS = [
    "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time and visit ancient civilizations?",
    "The restaurant on the corner serves the best pasta I have ever tasted in my entire life.",
    "After the meeting, we should discuss the quarterly results and plan for the next phase.",
    "Learning a new language takes patience, practice, and a genuine curiosity about other cultures.",
    "The train leaves at half past seven, so we need to arrive at the station before then.",
    "Could you please turn down the music a little bit, I'm trying to concentrate on my work.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock at the door.",
]

REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)


@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0
    e2e: float = 0.0
    audio_bytes: int = 0
    audio_duration: float = 0.0
    rtf: float = 0.0
    prompt: str = ""
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list = field(default_factory=list)


def pcm_bytes_to_duration(num_bytes: int, sample_rate: int = 24000, sample_width: int = 2) -> float:
    """Convert raw PCM byte count to duration in seconds."""
    return num_bytes / sample_width / sample_rate


async def upload_voice(
    host: str,
    port: int,
    ref_audio_url: str,
    clone_name: str,
    consent: str = "benchmark",
) -> bool:
    """Download ref_audio and upload it as a named voice clone via POST /v1/audio/voices."""
    voices_url = f"http://{host}:{port}/v1/audio/voices"
    tmp_path = None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            print(f"  Downloading reference audio from {ref_audio_url} ...")
            async with session.get(ref_audio_url) as resp:
                if resp.status != 200:
                    print(f"  [ERROR] Failed to download ref_audio: HTTP {resp.status}")
                    return False
                audio_bytes = await resp.read()

            suffix = ".wav"
            if ref_audio_url.lower().endswith(".mp3"):
                suffix = ".mp3"
            elif ref_audio_url.lower().endswith(".flac"):
                suffix = ".flac"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(audio_bytes)
            tmp.close()
            tmp_path = tmp.name

            data = aiohttp.FormData()
            data.add_field("audio_sample", open(tmp_path, "rb"), filename=f"ref{suffix}")
            data.add_field("consent", consent)
            data.add_field("name", clone_name)

            print(f"  Uploading voice as '{clone_name}' ...")
            async with session.post(voices_url, data=data) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"  [ERROR] Upload failed: HTTP {resp.status}: {body[:300]}")
                    return False
                result = await resp.json()
                print(f"  Voice uploaded: {result}")
                return True
    except Exception as e:
        print(f"  [ERROR] upload_voice failed: {e}")
        return False
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


async def delete_voice(host: str, port: int, clone_name: str) -> bool:
    """Delete an uploaded voice via DELETE /v1/audio/voices/{name}."""
    url = f"http://{host}:{port}/v1/audio/voices/{clone_name}"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.delete(url) as resp:
                if resp.status == 200:
                    print(f"  Deleted uploaded voice '{clone_name}'")
                    return True
                body = await resp.text()
                print(f"  [WARN] Delete voice returned HTTP {resp.status}: {body[:200]}")
                return False
    except Exception as e:
        print(f"  [WARN] delete_voice failed: {e}")
        return False


async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    ref_text: str,
    language: str = "Auto",
    clone_name: str | None = None,
    ref_audio: str | None = None,
    x_vector_only_mode: bool = False,
    initial_chunk_frames: int | None = None,
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a streaming Base voice-clone TTS request and measure latency.

    Two modes:
    - clone_name set: use uploaded voice by name (server loads ref_audio from disk).
      Sends x_vector_only_mode=false + ref_text for ICL cloning.
    - clone_name unset: send ref_audio URL per request (legacy).
    """
    payload: dict = {
        "input": prompt,
        "task_type": "Base",
        "ref_text": ref_text,
        "language": language,
        "stream": True,
        "response_format": "pcm",
    }
    if clone_name is not None:
        payload["voice"] = clone_name
        payload["x_vector_only_mode"] = False
    else:
        payload["ref_audio"] = ref_audio
        if x_vector_only_mode:
            payload["x_vector_only_mode"] = True
    if initial_chunk_frames is not None:
        payload["initial_codec_chunk_frames"] = initial_chunk_frames

    result = RequestResult(prompt=prompt)
    st = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                result.success = False
                return result

            first_chunk = True
            total_bytes = 0

            async for chunk in response.content.iter_any():
                if first_chunk and len(chunk) > 0:
                    result.ttfp = time.perf_counter() - st
                    first_chunk = False
                total_bytes += len(chunk)

            result.e2e = time.perf_counter() - st
            result.audio_bytes = total_bytes
            result.audio_duration = pcm_bytes_to_duration(total_bytes)

            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False
        result.e2e = time.perf_counter() - st

    if pbar:
        pbar.update(1)
    return result


async def send_tts_request_ws(
    ws_url: str,
    prompt: str,
    ref_text: str,
    language: str = "Auto",
    clone_name: str | None = None,
    ref_audio: str | None = None,
    x_vector_only_mode: bool = False,
    stt_delay: float = 0.0,
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a WebSocket streaming Base voice-clone TTS request.

    When stt_delay > 0, text is sent word-by-word with a delay to simulate
    real-time STT output. When stt_delay == 0 (default), the full text is
    sent in one shot followed immediately by input.done.
    """
    import websockets

    config: dict = {
        "type": "session.config",
        "task_type": "Base",
        "language": language,
        "response_format": "pcm",
        "stream_audio": True,
        "ref_text": ref_text,
    }
    if clone_name is not None:
        config["voice"] = clone_name
        config["x_vector_only_mode"] = False
    else:
        if ref_audio:
            config["ref_audio"] = ref_audio
        config["x_vector_only_mode"] = x_vector_only_mode

    result = RequestResult(prompt=prompt)
    st = time.perf_counter()

    try:
        async with websockets.connect(ws_url, max_size=16 * 1024 * 1024) as ws:
            await ws.send(json.dumps(config))

            async def send_text():
                if stt_delay > 0:
                    words = prompt.split(" ")
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        await ws.send(json.dumps({"type": "input.text", "text": chunk}))
                        if i < len(words) - 1:
                            await asyncio.sleep(stt_delay)
                else:
                    await ws.send(json.dumps({"type": "input.text", "text": prompt}))
                await ws.send(json.dumps({"type": "input.done"}))

            sender = asyncio.create_task(send_text())

            first_audio = True
            total_bytes = 0
            try:
                while True:
                    message = await asyncio.wait_for(ws.recv(), timeout=60)
                    if isinstance(message, bytes):
                        if first_audio and len(message) > 0:
                            result.ttfp = time.perf_counter() - st
                            first_audio = False
                        total_bytes += len(message)
                    else:
                        msg = json.loads(message)
                        if msg.get("type") == "session.done":
                            break
                        if msg.get("type") == "error":
                            result.error = msg.get("message", "unknown ws error")
                            result.success = False
                            if pbar:
                                pbar.update(1)
                            return result
            finally:
                sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass

            result.e2e = time.perf_counter() - st
            result.audio_bytes = total_bytes
            result.audio_duration = pcm_bytes_to_duration(total_bytes)
            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False
        result.e2e = time.perf_counter() - st

    if pbar:
        pbar.update(1)
    return result


async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    ref_text: str,
    language: str = "Auto",
    clone_name: str | None = None,
    ref_audio: str | None = None,
    x_vector_only_mode: bool = False,
    initial_chunk_frames: int | None = None,
    num_warmups: int = 3,
    use_ws: bool = False,
    simulate_stt: bool = False,
    stt_delay: float = 0.08,
) -> BenchmarkResult:
    """Run benchmark at a given concurrency level."""
    use_ws = use_ws or simulate_stt
    api_url = f"http://{host}:{port}/v1/audio/speech"
    ws_url = f"ws://{host}:{port}/v1/audio/speech/stream"
    ws_stt_delay = stt_delay if simulate_stt else 0.0

    session = None
    if not use_ws:
        connector = aiohttp.TCPConnector(
            limit=max_concurrency,
            limit_per_host=max_concurrency,
            keepalive_timeout=60,
        )
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=600),
        )

    if num_warmups > 0:
        print(f"  Warming up with {num_warmups} requests...")
        warmup_tasks = []
        for i in range(num_warmups):
            prompt = PROMPTS[i % len(PROMPTS)]
            if use_ws:
                warmup_tasks.append(
                    send_tts_request_ws(
                        ws_url, prompt, ref_text, language,
                        clone_name, ref_audio, x_vector_only_mode, ws_stt_delay,
                    )
                )
            else:
                warmup_tasks.append(
                    send_tts_request(
                        session, api_url, prompt, ref_text, language,
                        clone_name, ref_audio, x_vector_only_mode, initial_chunk_frames,
                    )
                )
        await asyncio.gather(*warmup_tasks)
        print("  Warmup done.")

    request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]

    if simulate_stt:
        mode_label = f"WS+STT({stt_delay*1000:.0f}ms)"
    elif use_ws:
        mode_label = "WS"
    else:
        mode_label = "HTTP"
    print(f"  Running {num_prompts} requests ({mode_label}) with concurrency={max_concurrency}...")
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

    async def limited_request(prompt):
        async with semaphore:
            if use_ws:
                return await send_tts_request_ws(
                    ws_url, prompt, ref_text, language,
                    clone_name, ref_audio, x_vector_only_mode, ws_stt_delay, pbar,
                )
            return await send_tts_request(
                session, api_url, prompt, ref_text, language,
                clone_name, ref_audio, x_vector_only_mode, initial_chunk_frames, pbar,
            )

    start_time = time.perf_counter()
    tasks = [asyncio.create_task(limited_request(p)) for p in request_prompts]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()

    if session is not None:
        await session.close()

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if successful:
        ttfps = [r.ttfp * 1000 for r in successful]
        e2es = [r.e2e * 1000 for r in successful]
        rtfs = [r.rtf for r in successful]
        audio_durs = [r.audio_duration for r in successful]

        bench.mean_ttfp_ms = float(np.mean(ttfps))
        bench.median_ttfp_ms = float(np.median(ttfps))
        bench.std_ttfp_ms = float(np.std(ttfps))
        bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
        bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
        bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))

        bench.mean_e2e_ms = float(np.mean(e2es))
        bench.median_e2e_ms = float(np.median(e2es))
        bench.std_e2e_ms = float(np.std(e2es))
        bench.p90_e2e_ms = float(np.percentile(e2es, 90))
        bench.p95_e2e_ms = float(np.percentile(e2es, 95))
        bench.p99_e2e_ms = float(np.percentile(e2es, 99))

        bench.mean_rtf = float(np.mean(rtfs))
        bench.median_rtf = float(np.median(rtfs))
        bench.std_rtf = float(np.std(rtfs))
        bench.p99_rtf = float(np.percentile(rtfs, 99))

        bench.mean_audio_duration_s = float(np.mean(audio_durs))
        bench.total_audio_duration_s = float(np.sum(audio_durs))
        bench.audio_throughput = bench.total_audio_duration_s / duration
        bench.request_throughput = len(successful) / duration

        bench.per_request = [
            {
                "ttfp_ms": r.ttfp * 1000,
                "e2e_ms": r.e2e * 1000,
                "rtf": r.rtf,
                "audio_duration_s": r.audio_duration,
                "prompt": r.prompt,
            }
            for r in successful
        ]

    W = 50
    mode = "x_vector_only" if x_vector_only_mode else "ICL"
    if simulate_stt:
        mode += f" + STT sim {stt_delay*1000:.0f}ms"
    elif use_ws:
        mode += " + WS"
    print("")
    print(f"{'=' * W}")
    print(f"{'Base Voice Clone Benchmark Result':^{W}}")
    print(f"{'(' + mode + ')':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{max_concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{bench.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{bench.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{bench.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{bench.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{bench.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")

    if failed:
        for r in failed[:3]:
            print(f"  [ERROR] {r.error[:200]}")

    return bench


async def main(args):
    clone_name: str | None = None
    ref_audio: str | None = args.ref_audio

    # Upload voice once so benchmark requests reference it by name.
    if not args.no_upload:
        clone_name = args.clone_name or f"bench_clone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ok = await upload_voice(args.host, args.port, args.ref_audio, clone_name)
        if not ok:
            print("  [FATAL] Voice upload failed, aborting benchmark.")
            return []
        ref_audio = None  # not needed per-request when using clone_name

    all_results = []
    try:
        for concurrency in args.max_concurrency:
            result = await run_benchmark(
                host=args.host,
                port=args.port,
                num_prompts=args.num_prompts,
                max_concurrency=concurrency,
                ref_text=args.ref_text,
                language=args.language,
                clone_name=clone_name,
                ref_audio=ref_audio,
                x_vector_only_mode=args.x_vector_only,
                initial_chunk_frames=args.initial_chunk_frames,
                num_warmups=args.num_warmups,
                use_ws=args.ws,
                simulate_stt=args.simulate_stt,
                stt_delay=args.stt_delay,
            )
            result.config_name = args.config_name
            all_results.append(asdict(result))
    finally:
        if clone_name is not None:
            await delete_voice(args.host, args.port, clone_name)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Base Voice Clone Benchmark Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts per concurrency level")
    parser.add_argument(
        "--max-concurrency", type=int, nargs="+", default=[1, 4, 10], help="Concurrency levels to test"
    )
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=REF_AUDIO_URL,
        help="Reference audio URL or data URI for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=REF_TEXT,
        help="Transcript of the reference audio",
    )
    parser.add_argument("--language", type=str, default="Auto", help="Language hint (default: Auto)")
    parser.add_argument(
        "--clone-name", type=str, default=None,
        help="Name for the uploaded voice clone (auto-generated if omitted)",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip voice upload; send ref_audio per request (legacy mode)",
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use x-vector only mode (skip in-context learning, lower quality but faster)",
    )
    parser.add_argument(
        "--initial-chunk-frames", type=int, default=None,
        help="Override server-side dynamic IC with a fixed initial codec chunk size (in frames)",
    )
    parser.add_argument(
        "--ws", action="store_true",
        help="Use WebSocket endpoint (full text sent at once, no STT delay)",
    )
    parser.add_argument(
        "--simulate-stt", action="store_true",
        help="Use WebSocket endpoint and send text word-by-word to simulate real-time STT input (implies --ws)",
    )
    parser.add_argument(
        "--stt-delay", type=float, default=0.08,
        help="Delay in seconds between words in STT simulation (default: 0.08 = 80ms)",
    )
    parser.add_argument(
        "--config-name", type=str, default="base_voice_clone", help="Label for this config (used in filenames)"
    )
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
