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
    # Minimal: upload a voice and generate speech
    python voice_clone_client.py \\
        --ref-audio /path/to/reference.wav \\
        --text "Hello, this is my cloned voice speaking."

    # Full options
    python voice_clone_client.py \\
        --ref-audio /path/to/reference.wav \\
        --voice-name my_voice \\
        --consent user_consent_123 \\
        --text "Hello, this is my cloned voice speaking." \\
        --output cloned_output.wav \\
        --language English \\
        --cleanup
"""

import argparse
import json
import sys
import time

import httpx

DEFAULT_API_BASE = "http://localhost:8091"


def upload_voice(api_base: str, audio_path: str, name: str, consent: str) -> dict:
    """Upload a voice sample to the server."""
    url = f"{api_base}/v1/audio/voices"

    with open(audio_path, "rb") as f:
        files = {"audio_sample": (audio_path.split("/")[-1], f)}
        data = {"name": name, "consent": consent}

        print(f"Uploading voice '{name}' from {audio_path}...")
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


def generate_speech(
    api_base: str,
    text: str,
    voice: str,
    output_path: str,
    language: str | None = None,
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
) -> None:
    """Generate speech using an uploaded voice (voice cloning)."""
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

    print(f"\nGenerating speech with voice '{voice}'...")
    print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

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

    # Check for JSON error in response body
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
        required=True,
        help="Path to reference audio file for voice cloning",
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

    args = parser.parse_args()

    print("=" * 60)
    print("  Qwen3-TTS Voice Cloning Example")
    print("=" * 60)

    # Step 1: Upload voice
    if not args.skip_upload:
        upload_voice(args.api_base, args.ref_audio, args.voice_name, args.consent)
    else:
        print(f"Skipping upload, using existing voice '{args.voice_name}'")

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
    )

    # Step 4: Optionally clean up
    if args.cleanup:
        delete_voice(args.api_base, args.voice_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
