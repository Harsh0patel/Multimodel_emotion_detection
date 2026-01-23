import sys
import os
import base64
import json
import time

# Force Transformers to use PyTorch only and reduce log noise
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import io
import cv2
from pydantic import BaseModel
from typing import Optional, List, Dict
from PIL import Image
import tempfile

# Add parent directory to sys.path to allow imports from Model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Model.infrence import InferenceModel

app = FastAPI(title="Video Emotion Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Model
model_path = os.path.join(parent_dir, "Model", "checkpoints", "model1.pt")
try:
    inference_model = InferenceModel(model_path=model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    inference_model = InferenceModel()

@app.get("/")
async def root():
    return {"message": "Video Emotion Detection API is running"}

# --- AUTHENTICATION (OAUTH SCAFFOLDING) ---

@app.post("/auth/login/google")
async def google_login():
    # Placeholder for Google OAuth Logic
    return {"message": "Google Login initiated", "url": "https://accounts.google.com/o/oauth2/v2/auth..."}

@app.post("/auth/login/github")
async def github_login():
    # Placeholder for GitHub OAuth Logic
    return {"message": "GitHub Login initiated", "url": "https://github.com/login/oauth/authorize..."}

# --- VIDEO STREAMING (WEBSOCKET) ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def rel_connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.rel_connect(websocket)
    try:
        while True:
            # Expecting base64 image data
            data = await websocket.receive_text()
            try:
                # Decode base64 frame
                header, encoded = data.split(",", 1) if "," in data else (None, data)
                image_data = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_np = np.array(image)
                
                # Predict emotion
                result = inference_model.predict(image_frame=image_np)
                
                # Send back the result
                await websocket.send_json(result)
            except Exception as e:
                await websocket.send_json({"error": f"Frame processing error: {str(e)}"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        manager.disconnect(websocket)

# --- VIDEO UPLOAD & BATCH PROCESSING ---

@app.post("/predict/video-upload")
async def predict_video_upload(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary location
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Process video using OpenCV
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # We sample frames to avoid overloading (e.g., 1 frame per second)
        # Assuming we want a summary of the video
        sampled_results = []
        count = 0
        success, frame = cap.read()
        
        while success:
            # Only process one frame per second of video
            if count % int(max(1, frame_rate)) == 0:
                # Convert BGR (OpenCV) to RGB (PIL/Inference)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = inference_model.predict(image_frame=frame_rgb)
                sampled_results.append({
                    "second": count // int(max(1, frame_rate)),
                    "dominant": res["dominant"],
                    "probs": res["fused"]
                })
            
            success, frame = cap.read()
            count += 1
            # Limit to 60 seconds or 60 detections for now
            if len(sampled_results) >= 60:
                break

        cap.release()
        os.unlink(tmp_path) # Cleanup

        # Aggregate results
        if not sampled_results:
            return {"error": "No frames processed"}

        # Calculate average probabilities
        avg_probs = {emotion: 0.0 for emotion in inference_model.emotions}
        for r in sampled_results:
            for emotion, prob in r["probs"].items():
                avg_probs[emotion] += prob
        
        avg_probs = {k: v / len(sampled_results) for k, v in avg_probs.items()}
        dominant = max(avg_probs, key=avg_probs.get)

        return {
            "summary": {
                "dominant": dominant,
                "average_probs": avg_probs,
                "duration_seconds": len(sampled_results)
            },
            "timeline": sampled_results
        }

    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
