"""HTTP client for TTS via /v1/audio/speech endpoint.

Mirrors the CLI of call.py but uses the REST endpoint instead of WebSocket.
Supports streaming (chunked transfer) for TTFA measurement and non-streaming
for simple E2E timing.

Voice registration: when --voice-name and --ref-audio are both provided, the
voice is uploaded to the server via POST /v1/audio/voices so subsequent calls
with just --voice-name work without resending audio data.

Usage:
    # Voice cloning (first time: registers voice on server)
    python call_http.py \
        --text "Hello world. How are you?" \
        --ref-audio /path/to/reference.wav \
        --ref-text /path/to/transcript.txt \
        --voice-name my_voice

    # Voice cloning (subsequent: reuses registered voice)
    python call_http.py \
        --text "Hello world." \
        --voice-name my_voice

    # CustomVoice (built-in speakers, no ref audio needed)
    python call_http.py --text "Hello world." --voice vivian

    # Non-streaming (wait for full response)
    python call_http.py --text "Hello world." --voice vivian --no-stream

    # Play audio after generation (requires: pip install sounddevice numpy)
    python call_http.py --text "Hello world." --voice vivian --play

Requirements:
    pip install httpx
    pip install sounddevice numpy  # optional, for --play
"""

import argparse
import base64
import mimetypes
import os
import struct
import time
import wave

import httpx

DEFAULT_BASE_URL = "http://localhost:8091"

try:
    import numpy as np
    import sounddevice as sd

    HAS_PLAYBACK = True
except ImportError:
    HAS_PLAYBACK = False


def _write_wav(path: str, pcm_data: bytes, sample_rate: int, channels: int = 1) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def _strip_wav_header(data: bytes) -> tuple[bytes, int]:
    """Strip WAV header from chunked response, return (pcm, sample_rate)."""
    if len(data) < 44 or data[:4] != b"RIFF":
        return data, 24000
    sr = struct.unpack_from("<I", data, 24)[0]
    data_offset = 12
    while data_offset + 8 <= len(data):
        chunk_id = data[data_offset : data_offset + 4]
        chunk_size = struct.unpack_from("<I", data, data_offset + 4)[0]
        if chunk_id == b"data":
            return data[data_offset + 8 :], sr
        data_offset += 8 + chunk_size
    return data[44:], sr


def _voice_is_registered(base_url: str, voice_name: str) -> bool:
    """Check if a voice is already registered (uploaded) on the server."""
    try:
        resp = httpx.get(f"{base_url}/v1/audio/voices", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            for v in data.get("uploaded_voices", []):
                if v.get("name", "").lower() == voice_name.lower():
                    return True
    except Exception:
        pass
    return False


def _register_voice(
    base_url: str, voice_name: str, audio_path: str, ref_text: str | None = None
) -> bool:
    """Upload a voice sample (and optional ref_text) to the server via POST /v1/audio/voices."""
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        mime_type = "audio/wav"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    form_data: dict[str, str] = {"name": voice_name, "consent": "cli_upload"}
    if ref_text:
        form_data["ref_text"] = ref_text

    try:
        resp = httpx.post(
            f"{base_url}/v1/audio/voices",
            files={"audio_sample": (os.path.basename(audio_path), audio_bytes, mime_type)},
            data=form_data,
            timeout=30.0,
        )
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                mode = "ICL" if ref_text else "x-vector only"
                print(f"  Voice '{voice_name}' registered on server ({mode}).")
                return True
        error_text = resp.text
        if "already exists" in error_text:
            print(f"  Voice '{voice_name}' already registered.")
            return True
        print(f"  Warning: voice upload returned {resp.status_code}: {error_text}")
        return False
    except Exception as e:
        print(f"  Warning: voice upload failed: {e}")
        return False


def run_streaming(args, payload: dict) -> None:
    """Stream audio via chunked transfer encoding with TTFA measurement."""
    url = f"{args.base_url}/v1/audio/speech"
    headers = {"Content-Type": "application/json"}

    t_start = time.perf_counter()
    ttfa: float | None = None
    chunks: list[bytes] = []
    total_bytes = 0

    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                resp.read()
                print(f"Error {resp.status_code}: {resp.text}")
                return

            for chunk in resp.iter_bytes(chunk_size=4096):
                if not chunk:
                    continue
                if ttfa is None:
                    ttfa = time.perf_counter() - t_start
                chunks.append(chunk)
                total_bytes += len(chunk)

    t_total = time.perf_counter() - t_start
    raw = b"".join(chunks)

    if not raw:
        print("No audio data received.")
        return

    if raw[:4] == b"RIFF":
        pcm_data, sample_rate = _strip_wav_header(raw)
        is_wav = True
    elif payload.get("response_format") == "pcm":
        pcm_data = raw
        sample_rate = 24000
        is_wav = False
    else:
        with open(args.output, "wb") as f:
            f.write(raw)
        print(f"Saved {args.output} ({len(raw)} bytes)")
        print(f"  Total time: {t_total * 1000:.1f} ms")
        if ttfa is not None:
            print(f"  TTFA:       {ttfa * 1000:.1f} ms")
        return

    audio_duration = len(pcm_data) / (sample_rate * 2)
    _write_wav(args.output, pcm_data, sample_rate)

    text_len = len(args.text) if hasattr(args, "text") else len(payload.get("input", ""))
    cps = text_len / t_total if t_total > 0 else 0

    print(f"\nSaved {args.output} ({len(pcm_data)} PCM bytes, {audio_duration:.2f}s audio)")
    if ttfa is not None:
        print(f"  TTFA:       {ttfa * 1000:.1f} ms")
    print(f"  Total time: {t_total * 1000:.1f} ms")
    if audio_duration > 0:
        print(f"  RTF:        {t_total / audio_duration:.2f}x")
    print(f"  Chars/sec:  {cps:.1f} ({text_len} chars)")

    if is_wav and args.play and HAS_PLAYBACK:
        print("  Playing audio...")
        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
        sd.play(audio_np, samplerate=sample_rate, blocking=True)


def run_non_streaming(args, payload: dict) -> None:
    """Single request/response, no chunked streaming."""
    url = f"{args.base_url}/v1/audio/speech"
    headers = {"Content-Type": "application/json"}

    t_start = time.perf_counter()
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(url, json=payload, headers=headers)
    t_total = time.perf_counter() - t_start

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    try:
        text = resp.content.decode("utf-8")
        if text.startswith('{"error"'):
            print(f"Error: {text}")
            return
    except UnicodeDecodeError:
        pass

    raw = resp.content
    if not raw:
        print("No audio data received.")
        return

    if raw[:4] == b"RIFF":
        pcm_data, sample_rate = _strip_wav_header(raw)
    elif payload.get("response_format") == "pcm":
        pcm_data = raw
        sample_rate = 24000
    else:
        with open(args.output, "wb") as f:
            f.write(raw)
        print(f"Saved {args.output} ({len(raw)} bytes, {t_total * 1000:.1f} ms)")
        return

    audio_duration = len(pcm_data) / (sample_rate * 2)
    _write_wav(args.output, pcm_data, sample_rate)

    text_len = len(args.text) if hasattr(args, "text") else len(payload.get("input", ""))
    cps = text_len / t_total if t_total > 0 else 0

    print(f"\nSaved {args.output} ({len(pcm_data)} PCM bytes, {audio_duration:.2f}s audio)")
    print(f"  Total time: {t_total * 1000:.1f} ms")
    if audio_duration > 0:
        print(f"  RTF:        {t_total / audio_duration:.2f}x")
    print(f"  Chars/sec:  {cps:.1f} ({text_len} chars)")

    if args.play and HAS_PLAYBACK:
        print("  Playing audio...")
        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
        sd.play(audio_np, samplerate=sample_rate, blocking=True)


def main():
    parser = argparse.ArgumentParser(
        description="HTTP client for TTS via /v1/audio/speech",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Server base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--output",
        default="http_tts_output.wav",
        help="Output WAV file path (default: http_tts_output.wav)",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from server if omitted)",
    )
    parser.add_argument(
        "--task-type",
        default=None,
        choices=["Base", "CustomVoice", "VoiceDesign"],
        help="Task type (default: Base when --ref-audio given, else CustomVoice)",
    )
    parser.add_argument("--language", default=None, help="Language (default: Auto)")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "flac", "mp3", "aac", "opus"],
        help="Audio format (default: wav)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max tokens")

    # Voice / speaker
    parser.add_argument(
        "--voice-name",
        default=None,
        help="Speaker name for voice cloning. On first call provide --ref-audio to "
        "register the voice; subsequent calls reuse it without resending audio.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Alias for --voice-name",
    )

    # Voice cloning (Base task)
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Reference audio: local path, URL, or data URI",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of reference audio (inline text or path to .txt file)",
    )
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        default=False,
        help="Speaker embedding only (no ICL voice cloning)",
    )

    # Style
    parser.add_argument(
        "--instructions",
        default=None,
        help="Voice style/emotion instructions",
    )

    # Streaming
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable chunked streaming (single request/response)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio after generation (requires: pip install sounddevice numpy)",
    )
    parser.add_argument(
        "--delete-voice",
        default=None,
        metavar="NAME",
        help="Delete a registered voice by name and exit.",
    )

    args = parser.parse_args()

    # Handle --delete-voice
    if args.delete_voice:
        resp = httpx.request("DELETE", f"{args.base_url}/v1/audio/voices/{args.delete_voice}", timeout=10.0)
        if resp.status_code == 200:
            print(f"Voice '{args.delete_voice}' deleted.")
        else:
            print(f"Delete failed: {resp.status_code} {resp.text}")
        return

    if args.play and not HAS_PLAYBACK:
        print("Error: --play requires sounddevice and numpy. pip install sounddevice numpy")
        raise SystemExit(1)

    # Read ref_text from file if it's a path
    if args.ref_text and os.path.isfile(args.ref_text):
        with open(args.ref_text) as f:
            args.ref_text = f.read().strip()
        print(f"Read ref_text from file: {args.ref_text[:80]}{'...' if len(args.ref_text) > 80 else ''}")

    # Resolve voice name
    voice = args.voice_name or args.voice
    if voice is None and args.ref_audio is None:
        voice = "vivian"

    # Resolve ref_audio to a local path (for voice registration) and a URI (for API)
    ref_audio_local_path: str | None = None
    ref_audio_uri: str | None = None
    if args.ref_audio:
        if args.ref_audio.startswith(("http://", "https://", "data:")):
            ref_audio_uri = args.ref_audio
            print(f"Reference audio URL: {args.ref_audio[:80]}")
        else:
            if not os.path.isfile(args.ref_audio):
                print(f"Error: reference audio file not found: {args.ref_audio}")
                raise SystemExit(1)
            ref_audio_local_path = os.path.abspath(args.ref_audio)
            ref_audio_uri = f"file://{ref_audio_local_path}"
            size_kb = os.path.getsize(args.ref_audio) / 1024
            print(f"Reference audio: {ref_audio_local_path} ({size_kb:.1f} KB)")

    # Voice registration: if --voice-name + --ref-audio (local), upload to server
    # so subsequent calls don't need --ref-audio or --ref-text.
    voice_registered = False
    if voice and ref_audio_local_path:
        if _voice_is_registered(args.base_url, voice):
            print(f"  Voice '{voice}' already registered on server.")
            voice_registered = True
        else:
            voice_registered = _register_voice(
                args.base_url, voice, ref_audio_local_path, ref_text=args.ref_text
            )

    # Infer task type
    task_type = args.task_type
    if task_type is None:
        if args.ref_audio or args.ref_text or voice_registered:
            task_type = "Base"
        elif voice and _voice_is_registered(args.base_url, voice):
            task_type = "Base"
        else:
            task_type = "CustomVoice"

    # For registered voices, we don't need to send ref_audio in the payload
    if voice_registered and not ref_audio_uri:
        pass  # server will auto-set from uploaded voice
    elif task_type == "Base" and not ref_audio_uri:
        if voice and _voice_is_registered(args.base_url, voice):
            pass  # server will auto-set
        else:
            print(
                "Error: Base voice cloning requires --ref-audio (first time) or a "
                "previously registered --voice-name."
            )
            raise SystemExit(1)

    # Build payload
    payload: dict = {
        "input": args.text,
        "response_format": args.response_format,
        "task_type": task_type,
    }
    if args.model:
        payload["model"] = args.model
    if voice:
        payload["voice"] = voice
    if args.language:
        payload["language"] = args.language
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.max_new_tokens:
        payload["max_new_tokens"] = args.max_new_tokens
    if ref_audio_uri and not voice_registered:
        payload["ref_audio"] = ref_audio_uri
        if args.ref_text:
            payload["ref_text"] = args.ref_text
    elif args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.x_vector_only_mode:
        payload["x_vector_only_mode"] = True

    print(f"Task: {task_type}, Voice: {voice or '(none)'}, Format: {args.response_format}")
    print(f"Text: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print()

    if args.no_stream:
        run_non_streaming(args, payload)
    else:
        payload["stream"] = True
        run_streaming(args, payload)


if __name__ == "__main__":
    main()
