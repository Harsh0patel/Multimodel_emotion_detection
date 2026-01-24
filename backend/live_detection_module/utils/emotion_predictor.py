import torch
import numpy as np
from configs import config
from models import Model
EMOTION_LABELS = {
    0: "Anger",      
    1: "Disgust",
    2: "Fear",
    3: "Joy",       
    4: "Neutral",
    5: "Sad",    
    6: "Surprise"
}

class EmotionPredictor:
    """Final emotion prediction from embeddings"""
    
    def __init__(self):
        self.model = Model.FusionClassifier().to(device=config.DEVICE)
        checkpoint = torch.load(config.FUSION_MODEL, map_location=config.DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
    
    def predict(self, video_emb: np.ndarray, audio_emb: np.ndarray, text_emb: np.ndarray):
        """
        Predict emotion from all embeddings
        
        Args:
            video_emb: Video embeddings
            audio_emb: Audio embeddings
            text_emb: Text embeddings
        
        Returns:
            emotion: Predicted emotion label and probabilities
        """
        print("model running.")
        face = torch.from_numpy(video_emb).unsqueeze(0).to(config.DEVICE)
        audio = torch.from_numpy(audio_emb).unsqueeze(0).to(config.DEVICE)
        text = torch.from_numpy(text_emb).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():

            outputs = self.model(text, audio, face)
            probs = torch.softmax(outputs, dim = 1)
            preds_idx = int(torch.argmax(probs, dim = 1).item())
            emotion_name = EMOTION_LABELS[preds_idx]
            print("getting outputs done.")
            
            return {
                "emotion": emotion_name,
                "confidence": torch.softmax(outputs, dim=-1)[0][preds_idx].item(),
                "emotion_id": preds_idx
            }