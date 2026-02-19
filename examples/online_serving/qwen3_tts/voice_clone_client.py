"""Voice cloning example for Qwen3-TTS Base model.

Demonstrates the full voice cloning workflow:
  1. Upload a custom voice sample via POST /v1/audio/voices
  2. List voices to confirm the upload via GET /v1/audio/voices
  3. Generate speech using the cloned voice via POST /v1/audio/speech
  4. (Optionally) delete the uploaded voice via DELETE /v1/audio/voices/{name}

Prerequisites:
  - A running vLLM-Omni server with the Base model:
      ./examples/online_serving/qwen3_tts/run_server.sh Base
  - A reference audio file (.wav, .mp3, .flac, etc.)

Usage:
    # Minimal: upload a voice and generate speech (x_vector_only mode)
    python voice_clone_client.py \\
        --ref-audio /path/to/reference.wav \\
        --text "Hello, this is my cloned voice speaking."

    # With reference text for ICL mode (higher-quality cloning)
    python voice_clone_client.py \\
        --ref-audio /path/to/reference.wav \\
        --ref-text "Transcript of the reference audio." \\
        --text "Hello, this is my cloned voice speaking."

    # Reference text from a file
    python voice_clone_client.py \\
        --ref-audio /path/to/reference.wav \\
        --ref-text /path/to/transcript.txt \\
        --text "Hello, this is my cloned voice speaking."

    # Full options
    python voice_clone_client.py \\
        --ref-audio /path/to/reference.wav \\
        --ref-text /path/to/transcript.txt \\
        --voice-name my_voice \\
        --consent user_consent_123 \\
        --text "Hello, this is my cloned voice speaking." \\
        --output cloned_output.wav \\
        --language English \\
        --cleanup

    # Use a previously uploaded voice (no upload needed)
    python voice_clone_client.py \\
        --skip-upload \\
        --voice-name my_voice \\
        --text "Hello, this is my cloned voice speaking."

    # Warm the cache after upload (pre-compute speaker embedding + ref_code)
    # so subsequent requests have much lower TTFA
    python voice_clone_client.py \\
        --ref-audio /path/to/reference.wav \\
        --ref-text /path/to/transcript.txt \\
        --text "Hello, this is my cloned voice speaking." \\
        --warm-cache

    # Disable streaming (use the original HTTP endpoint)
    python voice_clone_client.py \\
        --ref-audio /path/to/reference.wav \\
        --text "Hello, this is my cloned voice speaking." \\
        --no-stream
"""

import argparse
import asyncio
import json
import os
import sys
import time
import wave

import httpx

try:
    import websockets
except ImportError:
    websockets = None

DEFAULT_API_BASE = "http://localhost:8091"


def upload_voice(
    api_base: str,
    audio_path: str,
    name: str,
    consent: str,
    ref_text: str | None = None,
) -> dict:
    """Upload a voice sample to the server.

    When ref_text (transcript of the audio) is provided, the server uses
    ICL mode (x_vector_only_mode=False) for higher-quality voice cloning.
    Without ref_text, x_vector_only_mode=True is used.
    """
    url = f"{api_base}/v1/audio/voices"

    with open(audio_path, "rb") as f:
        files = {"audio_sample": (audio_path.split("/")[-1], f)}
        data = {"name": name, "consent": consent}
        if ref_text is not None:
            data["ref_text"] = ref_text

        mode = "ICL mode (ref_text provided)" if ref_text else "x_vector_only mode"
        print(f"Uploading voice '{name}' from {audio_path} [{mode}]...")
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, files=files, data=data)

    if response.status_code != 200:
        print(f"Upload failed ({response.status_code}): {response.text}")
        sys.exit(1)

    result = response.json()
    if not result.get("success"):
        print(f"Upload failed: {result}")
        sys.exit(1)

    voice_info = result["voice"]
    print(f"  Voice uploaded successfully:")
    print(f"    Name:       {voice_info['name']}")
    print(f"    MIME type:  {voice_info['mime_type']}")
    print(f"    File size:  {voice_info['file_size']} bytes")
    print(f"    Created at: {voice_info['created_at']}")
    if voice_info.get("ref_text"):
        print(f"    Ref text:   {voice_info['ref_text'][:80]}{'...' if len(voice_info['ref_text']) > 80 else ''}")
    return voice_info


def list_voices(api_base: str) -> dict:
    """List all available voices."""
    url = f"{api_base}/v1/audio/voices"

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)

    if response.status_code != 200:
        print(f"List voices failed ({response.status_code}): {response.text}")
        sys.exit(1)

    result = response.json()
    voices = result.get("voices", [])
    uploaded = result.get("uploaded_voices", [])

    print(f"\nAvailable voices ({len(voices)} total):")
    print(f"  Built-in + uploaded: {', '.join(voices[:10])}")
    if len(voices) > 10:
        print(f"  ... and {len(voices) - 10} more")

    if uploaded:
        print(f"\n  Uploaded voices ({len(uploaded)}):")
        for v in uploaded:
            print(f"    - {v['name']} (type: {v['mime_type']}, size: {v['file_size']} bytes)")

    return result


def _write_wav(path: str, pcm_data: bytes, sample_rate: int) -> None:
    """Write raw PCM-16LE bytes to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


async def _stream_speech(
    ws_url: str,
    text: str,
    voice: str,
    language: str | None = None,
    task_type: str = "Base",
) -> dict:
    """Run a single streaming TTS session over WebSocket, return audio + metrics."""
    config_msg = {
        "type": "session.config",
        "voice": voice,
        "task_type": task_type,
        "language": language or "Auto",
        "response_format": "wav",
        "speed": 1.0,
    }

    async with websockets.connect(ws_url) as ws:
        t0 = time.perf_counter()
        await ws.send(json.dumps(config_msg))
        await ws.send(json.dumps({"type": "input.text", "text": text}))
        await ws.send(json.dumps({"type": "input.done"}))

        ttfa = None
        t_audio_start = None
        prefill_s = None
        total_pcm_bytes = 0
        chunk_count = 0
        sample_rate = 24000
        all_pcm: list[bytes] = []

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                    if t_audio_start is not None:
                        prefill_s = time.perf_counter() - t_audio_start
                total_pcm_bytes += len(message)
                chunk_count += 1
                all_pcm.append(message)
            else:
                msg = json.loads(message)
                if msg.get("type") == "audio.start":
                    t_audio_start = time.perf_counter()
                elif msg.get("type") == "audio.done":
                    sample_rate = msg.get("sample_rate", 24000)
                elif msg.get("type") == "session.done":
                    break
                elif msg.get("type") == "error":
                    raise RuntimeError(msg.get("message", "unknown stream error"))

        t_total = time.perf_counter() - t0

    return {
        "ttfa_s": ttfa,
        "prefill_s": prefill_s,
        "queue_s": (t_audio_start - t0) if t_audio_start is not None else None,
        "total_s": t_total,
        "chunks": chunk_count,
        "pcm_bytes": total_pcm_bytes,
        "audio_duration_s": total_pcm_bytes / (sample_rate * 2) if total_pcm_bytes else 0,
        "sample_rate": sample_rate,
        "pcm_data": b"".join(all_pcm),
    }


def generate_speech(
    api_base: str,
    text: str,
    voice: str,
    output_path: str,
    language: str | None = None,
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    stream: bool = True,
) -> None:
    """Generate speech using an uploaded voice (voice cloning).

    When stream=True (default), uses the WebSocket streaming endpoint and
    logs TTFA (Time To First Audio).  Falls back to the HTTP endpoint when
    stream=False or the websockets package is unavailable.
    """
    print(f"\nGenerating speech with voice '{voice}'...")
    print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

    if stream and websockets is None:
        print("  [warn] websockets not installed, falling back to non-streaming")
        stream = False

    if stream:
        ws_url = (
            api_base.replace("http://", "ws://").replace("https://", "wss://")
            + "/v1/audio/speech/stream"
        )
        print(f"  Mode: streaming (WebSocket)")

        result = asyncio.run(
            _stream_speech(ws_url, text, voice, language=language)
        )

        if result["pcm_data"]:
            _write_wav(output_path, result["pcm_data"], result["sample_rate"])

        size_kb = os.path.getsize(output_path) / 1024 if os.path.exists(output_path) else 0
        print(f"  Audio saved to: {output_path} ({size_kb:.1f} KB)")
        print(f"  Audio duration: {result['audio_duration_s']:.2f}s")
        if result["queue_s"] is not None:
            print(f"  Queue time:     {result['queue_s'] * 1000:.0f} ms")
        if result["prefill_s"] is not None:
            print(f"  Prefill time:   {result['prefill_s'] * 1000:.0f} ms")
        if result["ttfa_s"] is not None:
            print(f"  TTFA:           {result['ttfa_s'] * 1000:.0f} ms")
        print(f"  Total time:     {result['total_s']:.2f}s ({result['chunks']} chunks)")
        if result["audio_duration_s"] > 0:
            rtf = result["total_s"] / result["audio_duration_s"]
            print(f"  RTF:            {rtf:.2f}x")
    else:
        url = f"{api_base}/v1/audio/speech"
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "task_type": "Base",
            "response_format": "wav",
        }
        if language:
            payload["language"] = language

        headers = {"Content-Type": "application/json"}
        print(f"  Mode: non-streaming (HTTP)")

        t0 = time.perf_counter()
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json=payload, headers=headers)
        elapsed = time.perf_counter() - t0

        if response.status_code != 200:
            try:
                err = response.json()
                print(f"Generation failed ({response.status_code}): {json.dumps(err, indent=2)}")
            except Exception:
                print(f"Generation failed ({response.status_code}): {response.text[:500]}")
            sys.exit(1)

        try:
            text_body = response.content.decode("utf-8")
            if text_body.startswith('{"error"'):
                print(f"Generation error: {text_body}")
                sys.exit(1)
        except UnicodeDecodeError:
            pass

        with open(output_path, "wb") as f:
            f.write(response.content)

        size_kb = len(response.content) / 1024
        print(f"  Audio saved to: {output_path} ({size_kb:.1f} KB)")
        print(f"  Generation time: {elapsed:.2f}s")


def warm_cache(
    api_base: str,
    voice: str,
    language: str | None = None,
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
) -> None:
    """Send a short generation request to eagerly warm the server-side voice cache.

    After this returns, subsequent requests skip the expensive speaker-embedding
    extraction and codec encoding, using the pre-computed safetensors cache instead.
    """
    url = f"{api_base}/v1/audio/speech"
    payload = {
        "model": model,
        "input": "Hello.",
        "voice": voice,
        "task_type": "Base",
        "response_format": "wav",
    }
    if language:
        payload["language"] = language

    print(f"\nWarming voice cache for '{voice}'...")
    t0 = time.perf_counter()
    with httpx.Client(timeout=300.0) as client:
        response = client.post(url, json=payload, headers={"Content-Type": "application/json"})
    elapsed = time.perf_counter() - t0

    if response.status_code != 200:
        print(f"  Cache warm failed ({response.status_code}): {response.text[:200]}")
    else:
        print(f"  Cache warmed in {elapsed:.2f}s (subsequent requests will skip speaker embedding extraction)")


def delete_voice(api_base: str, name: str) -> None:
    """Delete an uploaded voice."""
    url = f"{api_base}/v1/audio/voices/{name}"

    print(f"\nDeleting voice '{name}'...")
    with httpx.Client(timeout=30.0) as client:
        response = client.delete(url)

    if response.status_code == 200:
        print(f"  Voice '{name}' deleted successfully.")
    elif response.status_code == 404:
        print(f"  Voice '{name}' not found (already deleted?).")
    else:
        print(f"  Delete failed ({response.status_code}): {response.text}")


def main():
    parser = argparse.ArgumentParser(
        description="Voice cloning example for Qwen3-TTS Base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Server base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Path to reference audio file for voice cloning (required unless --skip-upload)",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of the reference audio, or path to a .txt file containing it. "
        "When provided, enables ICL mode (x_vector_only_mode=False) for higher-quality voice cloning.",
    )
    parser.add_argument(
        "--voice-name",
        default="my_cloned_voice",
        help="Name for the uploaded voice (default: my_cloned_voice)",
    )
    parser.add_argument(
        "--consent",
        default="user_consent_default",
        help="Consent recording ID (default: user_consent_default)",
    )
    parser.add_argument(
        "--text",
        default="Hello, this is a demonstration of voice cloning using a custom voice sample.",
        help="Text to synthesize with the cloned voice",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="cloned_output.wav",
        help="Output audio file path (default: cloned_output.wav)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language: Auto, Chinese, English, etc.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model name/path",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the uploaded voice after generation",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip upload step (use a previously uploaded voice)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use the non-streaming HTTP endpoint instead of WebSocket streaming",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="After upload, send a short request to eagerly warm the server-side voice cache. "
        "This pre-computes ref_code + speaker embedding so subsequent requests have lower TTFA.",
    )

    args = parser.parse_args()

    if not args.skip_upload and not args.ref_audio:
        parser.error("--ref-audio is required unless --skip-upload is set")

    # Resolve --ref-text: if it looks like a file path, read its contents
    if args.ref_text and os.path.isfile(args.ref_text):
        with open(args.ref_text) as f:
            args.ref_text = f.read().strip()
        print(f"Read ref_text from file: {args.ref_text[:80]}{'...' if len(args.ref_text) > 80 else ''}")

    print("=" * 60)
    print("  Qwen3-TTS Voice Cloning Example")
    print("=" * 60)

    # Step 1: Upload voice
    if not args.skip_upload:
        upload_voice(args.api_base, args.ref_audio, args.voice_name, args.consent, ref_text=args.ref_text)
    else:
        print(f"Skipping upload, using existing voice '{args.voice_name}'")

    # Step 1b: Optionally warm the server-side cache (pre-compute speaker
    # embedding + ref_code so subsequent requests skip the expensive extraction)
    if args.warm_cache:
        warm_cache(args.api_base, args.voice_name, language=args.language, model=args.model)

    # Step 2: List voices to confirm
    list_voices(args.api_base)

    # Step 3: Generate speech
    generate_speech(
        api_base=args.api_base,
        text=args.text,
        voice=args.voice_name,
        output_path=args.output,
        language=args.language,
        model=args.model,
        stream=not args.no_stream,
    )

    # Step 4: Optionally clean up
    if args.cleanup:
        delete_voice(args.api_base, args.voice_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
