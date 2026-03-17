"""Quick smoke test: iterate manifest.json prompts over WebSocket TTS."""

import asyncio
import json
import os
import struct
import time

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise SystemExit(1)

def _write_wav(path: str, pcm_data: bytes, sample_rate: int) -> None:
    bits = 16
    ch = 1
    data_size = len(pcm_data)
    with open(path, "wb") as f:
        f.write(struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE",
            b"fmt ", 16, 1, ch, sample_rate,
            sample_rate * ch * bits // 8, ch * bits // 8, bits,
            b"data", data_size,
        ))
        f.write(pcm_data)


WS_URL = os.environ.get("TTS_WS_URL", "ws://localhost:8091/v1/audio/speech/stream")
VOICE_NAME = os.environ.get("TTS_VOICE", "jfinn1")
MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "manifest_test_output")
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2


async def run_one(entry: dict, idx: int) -> dict:
    text = entry["text"]
    label = entry.get("label", entry["id"])
    print(f"\n[{idx+1}] {label}")
    print(f"    text: {text[:80]}{'...' if len(text) > 80 else ''}")

    config = {
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

    total_bytes = 0
    sentence_count = 0
    sentences = []
    pcm_chunks: list[bytes] = []
    t0 = time.perf_counter()
    ttfa = None
    error_msg = None

    try:
        async with websockets.connect(WS_URL, ping_interval=None, close_timeout=60) as ws:
            await ws.send(json.dumps({"type": "session.config", **config}))
            await ws.send(json.dumps({"type": "input.text", "text": text}))
            await ws.send(json.dumps({"type": "input.done"}))

            cur_sentence = {"index": 0, "text": "", "bytes": 0, "ttfa": None, "t0": t0}

            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)

                if isinstance(msg, bytes):
                    if ttfa is None:
                        ttfa = time.perf_counter() - t0
                    if cur_sentence["ttfa"] is None:
                        cur_sentence["ttfa"] = time.perf_counter() - cur_sentence["t0"]
                    total_bytes += len(msg)
                    cur_sentence["bytes"] += len(msg)
                    pcm_chunks.append(msg)
                else:
                    data = json.loads(msg)
                    mtype = data.get("type")
                    if mtype == "audio.start":
                        cur_sentence = {
                            "index": data["sentence_index"],
                            "text": data.get("sentence_text", ""),
                            "bytes": 0,
                            "ttfa": None,
                            "t0": time.perf_counter(),
                        }
                    elif mtype == "audio.done":
                        secs = cur_sentence["bytes"] / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                        cur_sentence["duration_s"] = round(secs, 2)
                        sentences.append(cur_sentence)
                        sentence_count += 1
                        flag = "OK" if secs > 0.5 else "SHORT"
                        s_ttfa = f"{cur_sentence['ttfa']:.3f}s" if cur_sentence["ttfa"] else "n/a"
                        print(f"    [{flag}] sentence {cur_sentence['index']}: "
                              f"{cur_sentence['bytes']:,}B = {secs:.2f}s  "
                              f"ttfa={s_ttfa}  \"{cur_sentence['text'][:60]}\"")
                    elif mtype == "session.done":
                        break
                    elif mtype == "error":
                        error_msg = data.get("message", str(data))
                        print(f"    ERROR: {error_msg}")
                        break
    except Exception as exc:
        error_msg = str(exc)
        print(f"    EXCEPTION: {exc}")

    # Save combined WAV
    if pcm_chunks:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        wav_path = os.path.join(OUTPUT_DIR, f"{entry['id']}.wav")
        pcm_data = b"".join(pcm_chunks)
        _write_wav(wav_path, pcm_data, SAMPLE_RATE)

    elapsed = time.perf_counter() - t0
    total_secs = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    rtf = total_secs / elapsed if elapsed > 0 else 0
    result = {
        "id": entry["id"],
        "label": label,
        "sentences": sentence_count,
        "total_bytes": total_bytes,
        "audio_secs": round(total_secs, 2),
        "expected_secs": entry.get("audio_secs"),
        "elapsed_s": round(elapsed, 2),
        "ttfa_s": round(ttfa, 3) if ttfa else None,
        "rtf": round(rtf, 2),
        "error": error_msg,
    }
    status = "PASS" if not error_msg and total_secs > 1.0 else "FAIL"
    print(f"    => {status}  audio={total_secs:.2f}s  wall={elapsed:.2f}s  RTF={rtf:.2f}x  "
          f"TTFA={result['ttfa_s']}s  sentences={sentence_count}")
    return result


async def main():
    with open(MANIFEST) as f:
        entries = json.load(f)

    print(f"Manifest: {len(entries)} prompts")
    print(f"Endpoint: {WS_URL}")
    print(f"Voice:    {VOICE_NAME}")
    print("=" * 72)

    results = []
    for i, entry in enumerate(entries):
        r = await run_one(entry, i)
        results.append(r)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    passes = sum(1 for r in results if not r["error"] and (r["audio_secs"] or 0) > 1.0)
    fails = len(results) - passes
    total_audio = sum(r["audio_secs"] or 0 for r in results)
    total_wall = sum(r["elapsed_s"] or 0 for r in results)
    avg_ttfa = [r["ttfa_s"] for r in results if r["ttfa_s"] is not None]

    print(f"  Passed:     {passes}/{len(results)}")
    print(f"  Failed:     {fails}/{len(results)}")
    print(f"  Total audio: {total_audio:.1f}s")
    print(f"  Total wall:  {total_wall:.1f}s")
    if avg_ttfa:
        print(f"  Avg TTFA:    {sum(avg_ttfa)/len(avg_ttfa):.3f}s")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
