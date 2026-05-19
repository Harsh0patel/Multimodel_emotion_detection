# 🎭 Emotion Detection System

Real-time 8-emotion detection with live webcam (WebSocket) + file upload (REST API).

## Architecture

```
Frontend (HTML/JS)
    │
    ├── WebSocket /ws/live  ──► Live Camera (base64 frames)
    └── REST POST /api/v1/  ──► File Upload (image/video)
            │
            ▼
        FastAPI Backend
            │
            ├── Face Detection: YOLOv8n-face
            └── Emotion Classification: EfficientNet-B0 (HSEmotion)
```

## 8 Emotions
`Anger · Contempt · Disgust · Fear · Happiness · Neutral · Sadness · Surprise`

Trained on **AffectNet** (8 classes) + fine-tuned on **AFEW** — one of the strongest publicly available CPU-viable emotion models.

## Model Performance
| Model | Dataset | Accuracy | CPU Speed |
|---|---|---|---|
| EfficientNet-B0 (HSEmotion) | AffectNet-8 | ~60% | ~30ms/face |
| YOLOv8n-face | WiderFace | mAP 0.72 | ~15ms/frame |

---

## Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Download model weights
```bash
python download_models.py
```

This downloads:
- `models/yolov8n-face.pt` — YOLOv8n face detector (~6MB)
- `models/enet_b0_8_best_afew.pt` — EfficientNet-B0 emotion model (~16MB)

> **Manual downloads if script fails:**
> - YOLOv8n-face: https://github.com/akanametov/yolov8-face/releases
> - HSEmotion: https://github.com/av-savchenko/face-emotion-recognition/tree/main/models/affectnet_emotions

### 3. Start the backend
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the frontend
Open `frontend/index.html` in your browser (or serve it with `python -m http.server 3000`).

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/api/v1/emotions` | List of 8 emotion labels |
| POST | `/api/v1/detect/image` | Upload image → annotated JPEG (download) |
| POST | `/api/v1/detect/image/json` | Upload image → JSON emotion results |
| POST | `/api/v1/detect/video` | Upload video → annotated MP4 |

#### Example: Image detection
```bash
curl -X POST http://localhost:8000/api/v1/detect/image/json \
  -F "file=@photo.jpg" | jq .
```

Response:
```json
{
  "face_count": 2,
  "faces": [
    {
      "face_index": 0,
      "bbox": { "x1": 120, "y1": 80, "x2": 260, "y2": 240 },
      "emotion": "Happiness",
      "emotion_confidence": 0.921,
      "scores": {
        "Anger": 0.003,
        "Contempt": 0.002,
        "Disgust": 0.001,
        "Fear": 0.004,
        "Happiness": 0.921,
        "Neutral": 0.058,
        "Sadness": 0.007,
        "Surprise": 0.004
      }
    }
  ]
}
```

### WebSocket

**URL:** `ws://localhost:8000/ws/live`

**Client → Server:**
```json
{ "frame": "<base64-encoded JPEG>" }
```

**Server → Client:**
```json
{
  "face_count": 1,
  "processing_ms": 45.2,
  "faces": [
    {
      "face_index": 0,
      "bbox": { "x1": 100, "y1": 80, "x2": 250, "y2": 240 },
      "emotion": "Happiness",
      "emotion_confidence": 0.89,
      "scores": { ... }
    }
  ]
}
```

---

## Project Structure

```
emotion-detection/
├── backend/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── requirements.txt
│   ├── download_models.py       # One-time model downloader
│   ├── models/                  # Downloaded weights go here
│   ├── api/
│   │   ├── routes.py            # REST endpoints
│   │   └── websocket.py         # WebSocket live endpoint
│   ├── core/
│   │   ├── face_detector.py     # YOLOv8 wrapper
│   │   └── emotion_model.py     # EfficientNet-B0 wrapper
│   └── services/
│       └── pipeline.py          # Combined detection pipeline
└── frontend/
    └── index.html               # Test UI (Live + Upload)
```

---

## Performance Tips

- **Reduce frame rate** to 5–10 FPS on the client side to reduce CPU load
- **Skip frames** in the WebSocket loop (`processing` flag in JS prevents queue buildup)
- **Use `skip_frames=2`** on video upload to process every 3rd frame
- For GPU: change `torch.device("cpu")` to `torch.device("cuda")` in `emotion_model.py`

## Alternative Models

If you want even better accuracy and have more RAM:
- **EfficientNet-B2** (HSEmotion variant) — higher accuracy, ~2x slower
- **HSEmotionNet** with ResNet-50 backbone
- **DeepFace** with `enforce_detection=False` — 7 emotions, slower but very robust

All available at: https://github.com/av-savchenko/face-emotion-recognition
