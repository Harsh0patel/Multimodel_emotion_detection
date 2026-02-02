import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from configs import config

transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

class EmbeddingGenerator:
    """Generate embeddings from raw data"""

    def __init__(self):
        self.K = config.K
        self.VIDEO_MODEL = config.VIDEO_MODEL
        self.DEVICE = config.DEVICE
        self.AUDIO_EXTRACTOR = config.AUDIO_EXTRACTOR
        self.AUDIO_ENCODER_BASE = config.AUDIO_ENCODER_BASE
        self.AUDIO_ENCODER_CTC = config.AUDIO_ENCODER_CTC
        self.AUDIO_PROCESSER = config.AUDIO_PROCESSER
        self.MAX_AUDIO_LEN = config.MAX_AUDIO_LEN
        self.MAX_TEXT_LEN = config.MAX_TEXT_LEN
        self.TEXT_TOKENIZER = config.TEXT_TOKENIZER
        self.TEXT_ENCODER = config.TEXT_ENCODER
        pass

    def pad_or_truncate_np(self, audio_array, target_len):
        """Ensure audio array is exactly target_len long."""
        if len(audio_array) > target_len:
            return audio_array[:target_len]
        elif len(audio_array) < target_len:
            return np.pad(audio_array, (0, target_len - len(audio_array)))
        return audio_array
    
    def get_face_embeddings(self, images):
        print(f"total images recive: {len(images)}")
        if len(images) == 0:
            return np.zeros(2048, dtype = np.float32)
        
        if len(images) > self.K:
            indices = np.linspace(0, len(images) - 1, self.K).astype(int)
            sampled_frames = [images[i] for i in indices]
        else:
            sampled_frames = images
        
        temp = []
        with torch.no_grad():
            for frame in sampled_frames:
                img = Image.fromarray(frame).convert("RGB")
                img = transform(img).unsqueeze(0).to(self.DEVICE)
                feture = self.VIDEO_MODEL(img)
                feture = feture.squeeze(0).cpu().numpy().astype(np.float32)
                temp.append(feture)
            print("made face embeddings.")

        if len(temp) == 0:
            return np.zeros(2048, dtype = np.float32)
        else:
            return np.mean(temp, axis = 0)
    
    def get_audio_embeddings(self, audio_array):
    
        if len(audio_array) == 0:
            return np.zeros(768, dtype = np.float32)
        print(f"audio array recived: {len(audio_array)}")
        audio_array = self.pad_or_truncate_np(audio_array, self.MAX_AUDIO_LEN)
        audio_inputs = self.AUDIO_EXTRACTOR([audio_array], sampling_rate=16000, return_tensors="pt", padding=True)
        audio_inputs = {
            k: v.to(self.DEVICE) for k, v in audio_inputs.items()
        }

        with torch.no_grad():
            outputs = self.AUDIO_ENCODER_BASE(**audio_inputs)
            feture = outputs.last_hidden_state.mean(dim = 1)
            print("made audio embeddings.")
        
        return feture.squeeze(0).cpu().numpy().astype(np.float32)
    
    def get_text_embeddings(self, audio_array):

        if len(audio_array) == 0:
            return np.zeros(768, dtype= np.float32)

        with torch.no_grad():
            inputs = self.AUDIO_EXTRACTOR(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.DEVICE)
            logits = self.AUDIO_ENCODER_CTC(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.AUDIO_PROCESSER.batch_decode(predicted_ids)[0].strip()
            print(f"translated text : {transcription}")

        if transcription == "":
            return np.zeros(768, dtype = np.float32)

        text_inputs = self.TEXT_TOKENIZER(
            transcription,
            truncation=True,
            padding="max_length",
            max_length=self.MAX_TEXT_LEN,
            return_tensors="pt"
        )

        text_inputs = {k: v.to(self.DEVICE) for k, v in text_inputs.items()}

        with torch.no_grad():
            output = self.TEXT_ENCODER(**text_inputs)
            output = output.last_hidden_state[:, 0, :]
            print("made text embeddings.")
        
        return output.squeeze(0).cpu().numpy().astype(np.float32)