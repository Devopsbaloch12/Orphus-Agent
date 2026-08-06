#!/usr/bin/env python3
"""Orphus Voice Agent — reference client + concurrent load test.

Usage:
    # Single call: stream a WAV in, save the agent's reply
    python orphus_client.py --wav caller_audio.wav --out reply.wav

    # 20 concurrent calls from the same WAV, print per-call latency
    python orphus_client.py --wav caller_audio.wav --concurrency 20

Requires: pip install websockets aiohttp
Audio in : 16 kHz mono PCM16 (16-bit signed little-endian), raw binary frames
Audio out: 24 kHz mono PCM16 binary frames
ViciDial/Asterisk note: Asterisk SLIN is 8 kHz — upsample x2 toward Orphus,
downsample the 24 kHz replies x3 toward the channel.
"""

import argparse
import asyncio
import struct
import sys
import time
import wave

import aiohttp
import websockets

import os

# Point these at your deployment. Never commit a real key.
BASE_URL = os.environ.get("ORPHUS_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("ORPHUS_KEY", "")

FRAME_MS = 20                      # send 20 ms frames, like a live mic
IN_RATE = 16000
FRAME_BYTES = IN_RATE * 2 * FRAME_MS // 1000   # 640 bytes per frame


def load_wav_as_pcm16_16k(path: str) -> bytes:
    """Read a WAV file and return raw 16 kHz mono PCM16 (naive resample)."""
    with wave.open(path, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit(f"{path}: need 16-bit PCM, got {8 * width}-bit")
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    if ch == 2:
        samples = samples[::2]
    if rate != IN_RATE:
        ratio = rate / IN_RATE
        samples = [samples[int(i * ratio)] for i in range(int(len(samples) / ratio))]
    return struct.pack(f"<{len(samples)}h", *samples)


async def run_call(call_id: int, pcm: bytes, out_path: str | None, voice: str):
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as http:
        resp = await http.post(
            f"{BASE_URL}/v1/sessions",
            json={"voice": voice},
            headers={"x-api-key": API_KEY},
        )
        if resp.status != 201:
            print(f"[{call_id}] session failed: {resp.status} {await resp.text()}")
            return
        session_id = (await resp.json())["session_id"]

    reply = bytearray()
    first_audio_at = None
    last_audio_at = None
    ws_url = BASE_URL.replace("https", "wss", 1) + f"/v1/ws/{session_id}"
    async with websockets.connect(
        ws_url, additional_headers={"x-api-key": API_KEY}, max_size=None
    ) as ws:

        async def sender():
            for off in range(0, len(pcm), FRAME_BYTES):
                await ws.send(pcm[off : off + FRAME_BYTES])
                await asyncio.sleep(FRAME_MS / 1000)   # realtime pacing
            # keep sending silence so VAD can close the turn
            silence = b"\x00" * FRAME_BYTES
            for _ in range(400):                        # up to 8 s of silence
                await ws.send(silence)
                await asyncio.sleep(FRAME_MS / 1000)

        send_task = asyncio.create_task(sender())
        speech_ends_at = t0 + len(pcm) / (IN_RATE * 2)
        try:
            while True:
                # Wait longer for the first frame (ASR + LLM + TTS cold path),
                # then only long enough to notice the reply has finished.
                timeout = 5 if first_audio_at is not None else 90
                frame = await asyncio.wait_for(ws.recv(), timeout=timeout)
                if isinstance(frame, bytes):
                    now = time.monotonic()
                    if first_audio_at is None:
                        first_audio_at = now
                    last_audio_at = now
                    reply.extend(frame)
        except (TimeoutError, websockets.ConnectionClosed):
            pass
        finally:
            send_task.cancel()

    # Release the slot: sessions otherwise hold their seat until the 300s idle
    # timeout, and the next run starts against a full house.
    try:
        async with aiohttp.ClientSession() as http:
            await http.delete(
                f"{BASE_URL}/v1/sessions/{session_id}",
                headers={"x-api-key": API_KEY},
            )
    except Exception:
        pass

    latency = (first_audio_at - speech_ends_at) if first_audio_at else None
    print(
        f"[{call_id}] reply={len(reply) / 48000:.1f}s audio, "
        f"response latency={f'{latency:.2f}s' if latency is not None else 'NO REPLY'}"
    )
    if out_path and reply:
        with wave.open(out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(bytes(reply))
        print(f"[{call_id}] saved {out_path}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="caller audio (WAV)")
    ap.add_argument("--out", default=None, help="save reply WAV (single-call mode)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--voice", default="tara",
                    help="tara|leah|jess|mia|zoe|leo|dan|zac")
    args = ap.parse_args()

    pcm = load_wav_as_pcm16_16k(args.wav)
    print(f"caller audio: {len(pcm) / (IN_RATE * 2):.1f}s, "
          f"{args.concurrency} concurrent call(s), voice={args.voice}")
    await asyncio.gather(*[
        run_call(i, pcm, args.out if args.concurrency == 1 else None, args.voice)
        for i in range(args.concurrency)
    ])


if __name__ == "__main__":
    asyncio.run(main())
