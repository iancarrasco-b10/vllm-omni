"""Clear all uploaded/cloned TTS voices from the server.

Lists voices via GET /v1/audio/voices, then deletes each uploaded voice
via the WebSocket voice.delete command.

Usage:
    python clear_clones.py
    python clear_clones.py --url ws://localhost:8091/v1/audio/speech/stream
"""

import argparse
import asyncio
import json

try:
    import websockets
except ImportError:
    print("pip install websockets")
    raise SystemExit(1)

try:
    import httpx
except ImportError:
    print("pip install httpx")
    raise SystemExit(1)


def _api_base_from_ws_url(ws_url: str) -> str:
    """Convert ws://host:port/path to http://host:port."""
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[6:].split("/", 1)[0]
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[5:].split("/", 1)[0]
    return "http://localhost:8091"


async def list_uploaded_voices(api_base: str) -> list[str]:
    """Return list of uploaded voice names (clones)."""
    url = f"{api_base}/v1/audio/voices"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        print(f"Failed to list voices: {e}")
        return []

    uploaded = data.get("uploaded_voices") or []
    # API returns list of {"name": "...", ...} for uploaded_voices; name may be key
    names = []
    for v in uploaded:
        name = v.get("name") or v.get("voice_name")
        if name:
            names.append(name)
    return names


async def delete_voice(ws_url: str, voice_name: str) -> bool:
    """Send voice.delete for one voice. Returns True if deleted."""
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "voice.delete", "voice_name": voice_name}))
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "voice.deleted":
                return True
            if msg.get("type") == "error":
                print(f"  {voice_name}: {msg.get('message', 'unknown')}")
                return False
            print(f"  {voice_name}: unexpected {msg}")
            return False
    except Exception as e:
        print(f"  {voice_name}: {e}")
        return False


async def main_async(args):
    api_base = _api_base_from_ws_url(args.url)
    print(f"Listing uploaded voices from {api_base}/v1/audio/voices ...")
    names = await list_uploaded_voices(api_base)

    if not names:
        print("No uploaded/cloned voices found.")
        return

    print(f"Found {len(names)} clone(s): {', '.join(names)}")
    print(f"Deleting via {args.url} ...")

    deleted = 0
    for name in names:
        if await delete_voice(args.url, name):
            print(f"  Deleted: {name}")
            deleted += 1

    print(f"Done. Deleted {deleted}/{len(names)} voice(s).")


def main():
    parser = argparse.ArgumentParser(description="Clear all uploaded TTS voice clones")
    parser.add_argument(
        "--url",
        default="ws://localhost:8091/v1/audio/speech/stream",
        help="WebSocket speech stream URL (used to derive API base and send voice.delete)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
