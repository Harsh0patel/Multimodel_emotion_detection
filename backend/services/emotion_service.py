from fastapi import WebSocket
import asyncio
import time
import logging
from collections import deque
from typing import List, Tuple
from utils.data_processor import DataProcessor
from utils.embedding_generator import EmbeddingGenerator
from utils.emotion_predictor import EmotionPredictor

logger = logging.getLogger(__name__)

class EmotionDetectionService:
    def __init__(self, websocket: WebSocket, window_seconds: float = 2.0):
        self.websocket = websocket
        self.window_seconds = window_seconds
        
        # Buffers
        self.video_buffer = deque()  # (timestamp, frame_bytes)
        self.audio_buffer = deque()  # (timestamp, audio_bytes)
        self.buffer_lock = asyncio.Lock()
        
        # Components
        self.data_processor = DataProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.emotion_predictor = EmotionPredictor()
        
        self.running = True
        
    async def start(self):
        """Start receiver and processor tasks"""
        receiver_task = asyncio.create_task(self._receiver_loop())
        processor_task = asyncio.create_task(self._processor_loop())
        
        # Wait for either to complete/error
        done, pending = await asyncio.wait(
            [receiver_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def _receiver_loop(self):
        """Receive and buffer data from frontend"""
        logger.info("📥 Receiver started")
        
        while self.running:
            try:
                msg = await self.websocket.receive()
                
                if msg["type"] == "websocket.disconnect":
                    logger.info("Client disconnected")
                    self.running = False
                    break
                
                if msg.get("bytes") is None:
                    continue
                
                data = msg["bytes"]
                if len(data) < 5:
                    continue
                
                header = data[:5]
                payload = data[5:]
                ts = time.time()
                
                async with self.buffer_lock:
                    if header == b"FRAME":
                        self.video_buffer.append((ts, payload))
                        logger.debug(f"📹 Buffered frame (total: {len(self.video_buffer)})")
                        
                    elif header == b"AUDIO":
                        self.audio_buffer.append((ts, payload))
                        logger.debug(f"🎵 Buffered audio (total: {len(self.audio_buffer)})")
                    
                    # Prune old data
                    self._prune_buffers(ts)
                    
            except Exception as e:
                logger.error(f"Receiver error: {e}")
                self.running = False
                break
    
    async def _processor_loop(self):
        """Process buffered data every window_seconds"""
        logger.info("⚙️ Processor started")
        
        while self.running:
            try:
                # Wait for window duration
                await asyncio.sleep(self.window_seconds)
                
                # Get buffered data
                async with self.buffer_lock:
                    video_data = list(self.video_buffer)
                    audio_data = list(self.audio_buffer)
                
                if not video_data or not audio_data:
                    logger.warning("⚠️ Insufficient data, skipping inference")
                    continue
                
                logger.info(f"🔄 Processing: {len(video_data)} frames, {len(audio_data)} audio chunks")
                
                # Process data in parallel
                result = await self._run_inference(video_data, audio_data)
                
                # Send result
                await self.websocket.send_json({
                    "type": "result",
                    "emotion": result,
                    "frames_processed": len(video_data),
                    "audio_chunks": len(audio_data),
                    "timestamp": time.time()
                })
                
                logger.info(f"✅ Emotion detected: {result}")
                
            except Exception as e:
                logger.error(f"Processor error: {e}", exc_info=True)
                await self.websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": time.time()
                })
    
    async def _run_inference(self, video_data: List[Tuple], audio_data: List[Tuple]):
        """Run parallel embedding generation and prediction"""
        
        # Step 1: Decode data in parallel
        frames_task = asyncio.to_thread(
            self.data_processor.decode_frames, 
            [data for _, data in video_data]
        )
        audio_task = asyncio.to_thread(
            self.data_processor.decode_audio,
            [data for _, data in audio_data]
        )
        
        frames, audio_array = await asyncio.gather(frames_task, audio_task)
        
        logger.info(f"📊 Decoded: {len(frames)} frames, audio shape: {audio_array.shape}")
        
        # Step 2: Generate embeddings IN PARALLEL
        video_emb_task = asyncio.to_thread(
            self.embedding_generator.get_face_embeddings,
            frames
        )
        audio_emb_task = asyncio.to_thread(
            self.embedding_generator.get_audio_embeddings,
            audio_array
        )
        text_emb_task = asyncio.to_thread(
            self.embedding_generator.get_text_embeddings,
            audio_array
        )
        
        video_emb, audio_emb, text_emb = await asyncio.gather(
            video_emb_task,
            audio_emb_task,
            text_emb_task
        )
        
        logger.info("✅ All embeddings generated")
        
        # Step 3: Predict emotion
        emotion = await asyncio.to_thread(
            self.emotion_predictor.predict,
            video_emb,
            audio_emb,
            text_emb
        )
        
        return emotion
    
    def _prune_buffers(self, current_time: float):
        """Remove data older than window_seconds"""
        cutoff = current_time - self.window_seconds
        
        while self.video_buffer and self.video_buffer[0][0] < cutoff:
            self.video_buffer.popleft()
        
        while self.audio_buffer and self.audio_buffer[0][0] < cutoff:
            self.audio_buffer.popleft()
    
    async def cleanup(self):
        """Cleanup resources"""
        self.running = False
        logger.info("🧹 Cleanup complete")