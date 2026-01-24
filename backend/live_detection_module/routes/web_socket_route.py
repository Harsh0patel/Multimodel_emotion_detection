from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.emotion_service import EmotionDetectionService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ Client connected")
    
    service = EmotionDetectionService(websocket)
    
    try:
        await service.start()
    except WebSocketDisconnect:
        logger.info("❌ Client disconnected")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
    finally:
        await service.cleanup()
        logger.info("🔄 Connection closed")