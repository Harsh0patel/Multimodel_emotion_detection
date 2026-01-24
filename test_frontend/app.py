from ultralytics.models.yolo import YOLO
import streamlit as st
import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
import pyaudio
import time
import threading
from collections import deque

# Page config
st.set_page_config(page_title="Face Detection", layout="centered")

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False

# Load YOLO model
@st.cache_resource
def load_yolo():
    return YOLO('C:/Users/hp333/Desktop/Multimodel_emotion_detection/backend/live_detection_module/models/yolov8n-face.pt')

# WebSocket config
WS_URL = st.text_input("WebSocket URL", "ws://localhost:8000/ws/stream")

# Audio config - 2 seconds
CHUNK = 4096  # Larger chunks for better performance
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
BUFFER_DURATION = 2  # 2 seconds

class SyncCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.model = load_yolo()
        self.audio_buffer = deque(maxlen=int(RATE * BUFFER_DURATION / CHUNK) + 1)
        
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_buffer.append(in_data)
        return (None, pyaudio.paContinue)
    
    def get_audio_chunk(self):
        """Get 2 seconds of buffered audio"""
        if len(self.audio_buffer) > 0:
            return b''.join(list(self.audio_buffer))
        return b''
    
    def capture_frame(self):
        """Capture single frame"""
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def detect_face(self, frame):
        """Detect and return first face"""
        results = self.model(frame, conf=0.5, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Add padding
                h, w = frame.shape[:2]
                pad = 20
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    return face, (x1, y1, x2, y2)
        return None, None
    
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.cap.release()

async def process_stream(placeholder, status_text, ws_url):
    ws = None
    capture = None
    
    try:
        # Connect to WebSocket with longer timeout
        ws = await websockets.connect(
            ws_url, 
            ping_interval=None,  # Disable ping/pong
            close_timeout=10,
            max_size=10_000_000  # 10MB max message size
        )
        status_text.success("🟢 Connected to WebSocket")
        
        capture = SyncCapture()
        frame_count = 0
        last_emotion = "Waiting..."
        
        # Create task to receive responses
        async def receive_results():
            nonlocal last_emotion
            try:
                while st.session_state.running:
                    response = await ws.recv()
                    data = json.loads(response)
                    if data.get("type") == "result":
                        last_emotion = data.get("emotion", "Unknown")
            except:
                pass
        
        receive_task = asyncio.create_task(receive_results())
        
        while st.session_state.running:
            # Capture frame
            frame = capture.capture_frame()
            
            if frame is None:
                status_text.error("❌ Camera error")
                break
            
            frame_count += 1
            
            # Detect face
            face, bbox = capture.detect_face(frame)
            
            # Send face frame (every frame)
            if face is not None:
                try:
                    # Encode face as JPEG bytes
                    _, face_buffer = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    face_bytes = face_buffer.tobytes()
                    
                    # Send with FRAME header
                    await asyncio.wait_for(ws.send(b"FRAME" + face_bytes), timeout=5.0)
                except Exception as e:
                    status_text.error(f"❌ Frame send error: {str(e)}")
                    break
            
            # Send audio every 2 seconds
            if frame_count % 60 == 0:  # ~2 seconds at 30fps
                audio_chunk = capture.get_audio_chunk()
                
                if len(audio_chunk) > 0:
                    try:
                        # Send with AUDIO header
                        await asyncio.wait_for(ws.send(b"AUDIO" + audio_chunk), timeout=5.0)
                        status_text.success(f"✅ Sent data | Emotion: {last_emotion}")
                    except asyncio.TimeoutError:
                        status_text.error("❌ Send timeout")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        status_text.error("❌ Connection closed")
                        break
                    except Exception as e:
                        status_text.error(f"❌ Audio send error: {str(e)}")
                        break
                else:
                    status_text.warning("⚠️ No audio buffered")
            
            # Draw bbox on frame
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face | Emotion: {last_emotion}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb_frame, channels="RGB", width='stretch')
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        receive_task.cancel()
            
    except websockets.exceptions.InvalidURI:
        status_text.error("❌ Invalid WebSocket URL")
    except websockets.exceptions.WebSocketException as e:
        status_text.error(f"❌ WebSocket error: {str(e)}")
    except Exception as e:
        status_text.error(f"❌ Error: {str(e)}")
    finally:
        if capture:
            capture.close()
        if ws:
            try:
                await asyncio.wait_for(ws.close(), timeout=2.0)
            except:
                pass
        status_text.info("⚪ Disconnected")

# UI
st.title("🎥 Face Detection Client")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start", disabled=st.session_state.running, width='stretch'):
        st.session_state.running = True
        st.rerun()

with col2:
    if st.button("⏹️ Stop", disabled=not st.session_state.running, width='stretch'):
        st.session_state.running = False
        st.rerun()

# Video placeholder
video_placeholder = st.empty()

# Status
status_text = st.empty()

if st.session_state.running:
    try:
        asyncio.run(process_stream(video_placeholder, status_text, WS_URL))
    except KeyboardInterrupt:
        st.session_state.running = False
    except Exception as e:
        st.error(f"Fatal Error: {str(e)}")
        st.session_state.running = False
else:
    status_text.info("⚪ Stopped - Press Start to begin")