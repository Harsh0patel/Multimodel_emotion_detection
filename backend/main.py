import sys
import os

# Force Transformers to use PyTorch only and reduce log noise
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import numpy as np
import io
import cv2
from pydantic import BaseModel
from typing import Optional, Dict
from PIL import Image

# Add parent directory to sys.path to allow imports from Model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Model.infrence import InferenceModel

app = FastAPI(title="Multimodal Emotion Detection API")

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

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Multimodal Emotion Detection API is running"}

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    try:
        result = inference_model.predict(text=request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        audio_content = await file.read()
        audio_data, sr = librosa.load(io.BytesIO(audio_content), sr=16000)
        result = inference_model.predict(audio_values=audio_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/vision")
async def predict_vision(file: UploadFile = File(...)):
    try:
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content)).convert("RGB")
        image_np = np.array(image)
        result = inference_model.predict(image_frame=image_np)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/multimodal")
async def predict_multimodal(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    try:
        audio_data = None
        if audio:
            audio_content = await audio.read()
            audio_data, sr = librosa.load(io.BytesIO(audio_content), sr=16000)
        
        vision_data = None
        if video:
            video_content = await video.read()
            # If it's a single frame/image
            image = Image.open(io.BytesIO(video_content)).convert("RGB")
            vision_data = np.array(image)

        result = inference_model.predict(text=text, audio_values=audio_data, image_frame=vision_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
