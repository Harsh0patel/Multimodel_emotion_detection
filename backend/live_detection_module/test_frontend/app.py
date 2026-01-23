import streamlit as st
import cv2
import time
import threading
import websocket
import json
import numpy as np
import sounddevice as sd
from ultralytics import YOLO

# ================= CONFIG =================
WS_URL = "ws://localhost:8000/ws/stream"
SEND_INTERVAL = 4.0
FACE_SIZE = (224, 224)
AUDIO_RATE = 16000
AUDIO_DURATION = 2  # seconds
# ==========================================

st.set_page_config(layout="wide")
st.title("Live Face Emotion Test (Streamlit)")

# ================= STATE ==================
if "running" not in st.session_state:
    st.session_state.running = False
if "ws" not in st.session_state:
    st.session_state.ws = None
if "last_send" not in st.session_state:
    st.session_state.last_send = 0
if "emotion" not in st.session_state:
    st.session_state.emotion = "N/A"
# ==========================================

# ================= YOLO ===================
@st.cache_resource
def load_yolo():
    return YOLO("C:/Users/hp333/Desktop/Multimodel_emotion_detection/backend/live_detection_module/models/yolov8n-face.pt").to("cpu")
yolo = load_yolo()
# ==========================================

# ================= AUDIO ==================
def record_audio():
    audio = sd.rec(
        int(AUDIO_DURATION * AUDIO_RATE),
        samplerate=AUDIO_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    return audio.tobytes()
# ==========================================

# ================= WEBSOCKET ==============
def on_message(ws, message):
    try:
        msg = json.loads(message)
        if msg.get("type") == "result":
            st.session_state.emotion = msg["data"]["emotion"]
    except:
        pass

def ws_thread():
    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message
    )
    st.session_state.ws = ws
    ws.run_forever()
# ==========================================

# ================= BUTTONS =================
col1, col2 = st.columns(2)

with col1:
    start = st.button("▶ Start")

with col2:
    stop = st.button("⏹ Stop")

frame_placeholder = st.empty()
# ==========================================

# ================= START ==================
if start and not st.session_state.running:
    st.session_state.running = True
    st.session_state.last_send = 0
    threading.Thread(target=ws_thread, daemon=True).start()
    time.sleep(1)
# ==========================================

# ================= STOP ===================
if stop:
    st.session_state.running = False
    if st.session_state.ws:
        st.session_state.ws.close()
    st.session_state.ws = None
    st.stop()
# ==========================================

# ================= VIDEO LOOP ==============
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    while st.session_state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = yolo(frame, verbose=False)

        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append((x1, y1, x2, y2))

        if boxes:
            # select largest face
            x1, y1, x2, y2 = max(
                boxes,
                key=lambda b: (b[2]-b[0]) * (b[3]-b[1])
            )

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                face = cv2.resize(face, FACE_SIZE)

                # draw
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"Emotion: {st.session_state.emotion}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

                # send every 4 seconds
                now = time.time()
                if now - st.session_state.last_send >= SEND_INTERVAL:
                    st.session_state.last_send = now

                    _, img_buf = cv2.imencode(".jpg", face)
                    audio_bytes = record_audio()

                    if st.session_state.ws and st.session_state.ws.sock:
                        st.session_state.ws.send(
                            b"FRAME" + img_buf.tobytes(),
                            opcode=websocket.ABNF.OPCODE_BINARY
                        )
                        st.session_state.ws.send(
                            b"AUDIO" + audio_bytes,
                            opcode=websocket.ABNF.OPCODE_BINARY
                        )

        frame_placeholder.image(frame, channels="BGR")
        time.sleep(0.03)

    cap.release()
