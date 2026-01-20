import torch
from torchvision import transforms
import numpy as np
from PIL import Image

transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

def pad_or_truncate_np(audio_array, target_len):
    """Ensure audio array is exactly target_len long."""
    if len(audio_array) > target_len:
        return audio_array[:target_len]
    elif len(audio_array) < target_len:
        return np.pad(audio_array, (0, target_len - len(audio_array)))
    return audio_array

def get_face_embeddings(K, model, images, DEVICE):

    if len(images) == 0:
        return np.zeros(2048, dtype = np.float32)
    
    if len(images) > K:
        indices = np.linspace(0, len(images) - 1, K).astype(int)
        sampled_frames = [images[i] for i in indices]
    else:
        sampled_frames = images
    
    temp = []
    with torch.no_grad():
        for frame in sampled_frames:
            img = Image.fromarray(frame).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)
            feture = model(img)
            feture = feture.squeeze(0).cpu().numpy().astype(np.float32)
            temp.append(feture)

    if len(temp) == 0:
        return np.zeros(2048, dtype = np.float32)
    else:
        return np.mean(temp, axis = 0)

def get_audio_embeddings(DEVICE, audio_array, audio_extractor, audio_encoder, MAX_AUDIO_LENGTH):
    
    if len(audio_array) == 0:
        return np.zeros(768, dtype = np.float32)
    
    audio_array = pad_or_truncate_np(audio_array, MAX_AUDIO_LENGTH)
    audio_inputs = audio_extractor([audio_array], sampling_rate=16000, return_tensors="pt", padding=True)
    audio_inputs = {
        k: v.to(DEVICE) for k, v in audio_inputs.items()
    }

    with torch.no_grad():
        outputs = audio_encoder(**audio_inputs)
        feture = outputs.last_hidden_state.mean(dim = 1)
    
    return feture.squeeze(0).cpu().numpy().astype(np.float32)

def get_text_embeddings(DEVICE, audio_array, preprocesser, audio_model, text_tokenizer, text_encoder, MAX_TEXT_LEN):

    if len(audio_array) == 0:
        return np.zeros(768, dtype= np.float32)

    with torch.no_grad():
        inputs = preprocesser(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(DEVICE)
        logits = audio_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = preprocesser.batch_decode(predicted_ids)[0].strip().lower()

    if transcription == "":
        return np.zeros(768, dtype = np.float32)

    text_inputs = text_tokenizer(
        transcription,
        truncation=True,
        padding="max_length",
        max_length= MAX_TEXT_LEN,
        return_tensors="pt"
    )

    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

    with torch.no_grad():
        output = text_encoder(**text_inputs)
        output = output.last_hidden_state[:, 0, :]
    
    return output.squeeze(0).cpu().numpy().astype(np.float32)
