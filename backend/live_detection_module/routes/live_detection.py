import asyncio
from fastapi import APIRouter, WebSocket
from utils import socket_utils

router = APIRouter() 

# WebSocket Endpoint
@router.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()

    receiver_task = asyncio.create_task(socket_utils.receiver_loop(websocket))
    inference_task = asyncio.create_task(socket_utils.inference_scheduler(websocket))

    done, pending = await asyncio.wait(
        [receiver_task, inference_task],
        return_when=asyncio.FIRST_EXCEPTION
    )

    for task in pending:
        task.cancel()