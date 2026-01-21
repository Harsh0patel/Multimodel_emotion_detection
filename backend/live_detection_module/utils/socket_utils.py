from fastapi import WebSocket
import numpy as np
import cv2
import time
import asyncio
from configs import config
from collections import deque
from utils import model_run, socket_utils

WINDOW_SECONDS = config.WINDOW_SECONDS
INFERENCE_INTERVAL = config.INFERENCE_INTERVAL
video_buffer = deque()   # (timestamp, frame)
audio_buffer = deque()   # (timestamp, audio_bytes)
buffer_lock = asyncio.Lock()

def decode_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def prune_buffer(buffer, now):
    while buffer and now - buffer[0][0] > WINDOW_SECONDS:
        buffer.popleft()

# Receiver (ASYNC)
async def receiver_loop(websocket: WebSocket):
    while True:
        msg = await websocket.receive()

        if msg["type"] == "websocket.disconnect":
            break

        if msg["bytes"] is None:
            continue

        data = msg["bytes"]
        header = data[:5]
        payload = data[5:]
        ts = time.time()

        async with buffer_lock:
            if header == b"FRAME":
                frame = socket_utils.decode_image(payload)
                video_buffer.append((ts, frame))

            elif header == b"AUDIO":
                audio_buffer.append((ts, payload))

            socket_utils.prune_buffer(video_buffer, ts)
            socket_utils.prune_buffer(audio_buffer, ts)

# Inference Scheduler (ASYNC wrapper)
async def inference_scheduler(websocket: WebSocket):
    while True:
        await asyncio.sleep(WINDOW_SECONDS)

        async with buffer_lock:
            frames = [f for _, f in video_buffer]
            audio = [a for _, a in audio_buffer]

        if not frames or not audio:
            continue

        # ---- RUN SYNC INFERENCE (NON-BLOCKING) ----
        result = await asyncio.to_thread(
            model_run.infrence_loop,
            frames,
            audio
        )

        await websocket.send_json({
            "type": "result",
            "emotion": result,
            "window_seconds": WINDOW_SECONDS,
            "timestamp": time.time()
        })

        # ---- WAIT UNTIL NEXT CYCLE ----
        await asyncio.sleep(
            max(0, INFERENCE_INTERVAL - WINDOW_SECONDS)
        )