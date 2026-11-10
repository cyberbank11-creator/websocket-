# websocket-
live api endpoint for video streaming 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini Multimodal Live API — Raw WebSocket video streaming client
=================================================================
- WebSocket connection to generativelanguage.googleapis.com
- Captures video frames from camera via OpenCV
- Sends base64 JPEG frames as realtimeInput (JSON, not raw binary frames!)
- Handles server responses (setupComplete, serverContent, turnComplete)

Requirements:
    pip install websockets opencv-python

Environment:
    export GEMINI_API_KEY="your-api-key-here"

Run:
    python gemini_live_video.py
"""

import asyncio
import base64
import json
import os
import sys
from datetime import datetime

import cv2
import websockets

# ---------------------------------------------------------------------------
# Configuration (from official docs: ai.google.dev/api/live)
# ---------------------------------------------------------------------------

API_KEY_ENV = "GEMINI_API_KEY"

# Currently supported Live model (per docs example)
MODEL_NAME = "gemini-3.8-live"

WS_HOST = "generativelanguage.googleapis.com"
WS_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

# Video streaming rate: about 1 frame per second (docs recommend low fps)
FRAME_INTERVAL_SECONDS = 1.0
JPEG_QUALITY = 70  # lower quality => smaller payload

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that can see the user's live video feed. "
    "Answer in the language the user speaks."
)


def build_ws_url() -> str:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        sys.exit(
            f"ERROR: environment variable {API_KEY_ENV} is not set.\n"
            f"Run: export {API_KEY_ENV}='your-api-key'"
        )
    return f"wss://{WS_HOST}{WS_PATH}?key={api_key}"


# ---------------------------------------------------------------------------
# Client messages
# ---------------------------------------------------------------------------

def build_setup_message() -> dict:
    """First message over the socket MUST be BidiGenerateContentSetup."""
    return {
        "setup": {
            "model": f"models/{MODEL_NAME}",
            "responseModalities": ["AUDIO"],
            "systemInstruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}],
            },
        }
    }


def build_video_message(frame) -> dict:
    """Encode an OpenCV frame (BGR ndarray) to base64 JPEG and wrap it
    in a realtimeInput message. Frames go as base64 JPEG inside JSON —
    NOT as raw binary WebSocket frames."""
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
    return {
        "realtimeInput": {
            "video": {
                "data": b64,
                "mimeType": "image/jpeg",
            }
        }
    }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class GeminiLiveVideoClient:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.shutdown_event = asyncio.Event()
        self.setup_complete = asyncio.Event()

    # ---------------- sending loop (video frames) ----------------

    async def send_video_loop(self, ws):
        """Captures frames from the camera and sends them throttled ~1 fps."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[video] ERROR: could not open camera (index 0)")
            self.shutdown_event.set()
            return

        print("[video] camera opened, streaming frames...")
        try:
            while not self.shutdown_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    print("[video] WARNING: failed to read frame")
                    await asyncio.sleep(0.5)
                    continue

                msg = build_video_message(frame)
                await ws.send(json.dumps(msg))
                await asyncio.sleep(FRAME_INTERVAL_SECONDS)
        finally:
            cap.release()
            print("[video] camera released")

    # ---------------- receiving loop ----------------

    async def receive_loop(self, ws):
        """Reads server messages until shutdown or the socket closes."""
        try:
            async for raw in ws:
                try:
                    response = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"[recv] non-JSON message: {raw[:200]!r}")
                    continue

                if "setupComplete" in response:
                    print("[recv] setupComplete received — session ready")
                    self.setup_complete.set()

                elif "serverContent" in response:
                    sc = response["serverContent"]

                    # Model turn: may contain audio (inlineData) or text parts
                    if "modelTurn" in sc and "parts" in sc["modelTurn"]:
                        for part in sc["modelTurn"]["parts"]:
                            if "inlineData" in part:
                                mime = part["inlineData"].get("mimeType", "?")
                                data = part["inlineData"]["data"]
                                pcm = base64.b64decode(data)
                                ts = datetime.now().strftime("%H:%M:%S")
                                print(f"[recv] {ts} audio chunk ({mime}) {len(pcm)} bytes")
                                # NOTE: real playback (e.g. pyaudio) of the 24kHz
                                # little-endian PCM16 stream would go here.
                            elif "text" in part:
                                ts = datetime.now().strftime("%H:%M:%S")
                                print(f"[recv] {ts} text: {part['text']}")

                    # Transcriptions
                    if "inputTranscription" in sc:
                        print(f"[recv] User (transcribed): {sc['inputTranscription'].get('text', '')}")
                    if "outputTranscription" in sc:
                        print(f"[recv] Gemini (transcribed): {sc['outputTranscription'].get('text', '')}")

                    if sc.get("interrupted"):
                        print("[recv] generation interrupted")

                    if sc.get("turnComplete"):
                        print("[recv] turnComplete — model finished this turn")

                    if sc.get("generationComplete"):
                        print("[recv] generationComplete")

                elif "toolCall" in response:
                    print(f"[recv] toolCall: {response['toolCall']}")
                    # Respond if you configure functions:
                    # await ws.send(json.dumps({"toolResponse": {"functionResponses": [...]}}))

                elif "toolCallCancellation" in response:
                    print(f"[recv] toolCallCancellation: {response['toolCallCancellation']}")

                elif "goAway" in response:
                    print(f"[recv] goAway: {response['goAway']} — server will disconnect soon")
                    self.shutdown_event.set()

                elif "sessionResumptionUpdate" in response:
                    upd = response["sessionResumptionUpdate"]
                    if upd.get("resumable"):
                        print(f"[recv] sessionResumptionUpdate (resumable): {upd.get('newHandle', '')}")

                else:
                    print(f"[recv] other message: {json.dumps(response)[:300]}")
        except websockets.ConnectionClosed as e:
            print(f"[recv] connection closed: code={e.code} reason={e.reason}")
        finally:
            self.shutdown_event.set()

    # ---------------- orchestration ----------------

    async def run(self):
        print(f"[conn] connecting to {WS_HOST}{WS_PATH}")
        print(f"[conn] model: {MODEL_NAME}")

        async with websockets.connect(
            self.ws_url,
            max_size=None,          # video frames can be large
            ping_interval=30,
            ping_timeout=10,
        ) as ws:
            # 1) The very first message must be the setup message
            await ws.send(json.dumps(build_setup_message()))
            print("[conn] setup message sent, waiting for setupComplete...")

            # Wait for setupComplete before streaming frames
            try:
                await asyncio.wait_for(self.setup_complete.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                print("[conn] ERROR: no setupComplete within 15s — aborting")
                return

            print("[conn] session ready. Press Ctrl+C to stop.\n")

            # 2) Run sender and receiver concurrently
            sender = asyncio.create_task(self.send_video_loop(ws))
            receiver = asyncio.create_task(self.receive_loop(ws))
            done, pending = await asyncio.wait(
                {sender, receiver}, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc:
                    print(f"[conn] task error: {exc!r}")
                    self.shutdown_event.set()

        print("[conn] session closed cleanly")


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(GeminiLiveVideoClient(build_ws_url()).run())
    except KeyboardInterrupt:
        print("\n[main] stopped by user (Ctrl+C)")


if __name__ == "__main__":
    main()
