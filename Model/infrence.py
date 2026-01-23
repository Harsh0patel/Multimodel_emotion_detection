import os
import sys

# Force Transformers to use PyTorch only and reduce log noise
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import numpy as np
import gc
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# Add the parent directory to sys.path to allow imports from Model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Model.Model import TripleFusionClassifier
from Model.utils import compute_text_embeddings, compute_audio_embeddings

class InferenceModel:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.audio_model_name = "facebook/wav2vec2-base-960h"
        
        print(f"Starting staggered model loading on {self.device}...")
        
        # 1. Load Text Encoder
        print("Loading Text Encoder...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_encoder = AutoModel.from_pretrained(self.text_model_name, low_cpu_mem_usage=True).to(self.device)
        gc.collect()
        
        # 2. Load Vision Encoder (Smaller than Audio, load first)
        print("Loading Vision Encoder...")
        self.vision_encoder = models.resnet18(pretrained=True)
        self.vision_encoder.fc = nn.Identity() 
        self.vision_encoder = self.vision_encoder.to(self.device)
        gc.collect()
        
        # 3. Load Audio Encoder (The largest one)
        print("Loading Audio Encoder (this may take a moment)...")
        self.audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(self.audio_model_name, low_cpu_mem_usage=True).to(self.device)
        gc.collect()
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        self.vision_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load Fusion Model (MiniLM has 384 dim, DistilBERT had 768)
        self.model = TripleFusionClassifier(text_dim=384).to(self.device)
        
        if model_path:
             if os.path.exists(model_path):
                print(f"Loading checkpoint from {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    # Handle state dict mismatch if checkpoint was for dual fusion
                    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                    self.model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"Warning: Could not load checkpoint fully: {e}")
             else:
                 print(f"Warning: Model path {model_path} does not exist. Using random weights.")
        
        self.model.eval()
        self.emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']

    def predict(self, text=None, audio_values=None, image_frame=None):
        text_emb = None
        audio_emb = None
        vision_emb = None
        
        with torch.no_grad():
            # Text Processing
            if text:
                inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
                text_emb = compute_text_embeddings(inputs["input_ids"], inputs["attention_mask"], self.text_encoder, self.device)
            else:
                text_emb = torch.zeros((1, 384)).to(self.device)

            # Audio Processing
            if audio_values is not None:
                inputs = self.audio_extractor(audio_values, sampling_rate=16000, return_tensors="pt", padding=True)
                audio_emb = compute_audio_embeddings(inputs.input_values, self.audio_encoder, self.device)
            else:
                audio_emb = torch.zeros((1, 768)).to(self.device)

            # Vision Processing
            if image_frame is not None:
                # Expect image_frame to be a PIL Image or numpy array
                if isinstance(image_frame, np.ndarray):
                    image_frame = Image.fromarray(image_frame)
                
                img_t = self.vision_transform(image_frame).unsqueeze(0).to(self.device)
                vision_emb = self.vision_encoder(img_t)
            else:
                vision_emb = torch.zeros((1, 512)).to(self.device)

            # Triple Fusion Forward
            logits = self.model(text_emb, audio_emb, vision_emb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Map to emotions
            emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.emotions, probs)}
            dominant_emotion = self.emotions[np.argmax(probs)]
            
            # Confidence calculation (Approximation)
            # Note: MiniLM output is 384, we project it in the model
            text_conf = float(torch.max(torch.softmax(self.model.fc(torch.cat([torch.relu(self.model.text_proj(text_emb)), torch.zeros((1, 512)).to(self.device), torch.zeros((1, 512)).to(self.device)], dim=1)), dim=1)).item())
            audio_conf = float(torch.max(torch.softmax(self.model.fc(torch.cat([torch.zeros((1, 512)).to(self.device), torch.relu(self.model.audio_proj(audio_emb)), torch.zeros((1, 512)).to(self.device)], dim=1)), dim=1)).item())
            vision_conf = float(torch.max(torch.softmax(self.model.fc(torch.cat([torch.zeros((1, 512)).to(self.device), torch.zeros((1, 512)).to(self.device), torch.relu(self.model.vision_proj(vision_emb))], dim=1)), dim=1)).item())

            return {
                "fused": emotion_probs,
                "dominant": dominant_emotion,
                "text_confidence": text_conf,
                "audio_confidence": audio_conf,
                "vision_confidence": vision_conf
            }

