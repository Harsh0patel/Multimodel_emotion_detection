import torch
import numpy as np
from utils.make_embeddings import get_face_embeddings, get_audio_embeddings, get_text_embeddings
from configs import config
from Model import Model

model = Model.FusionClassifier().to(device=config.DEVICE)
model.load_state_dict(torch.load(config.FUSION_MODEL, map_location=config.DEVICE))
model.eval()

def infrence_loop(images, audio_array):

    face_embeddings = get_face_embeddings(
        K=config.K,
        model = config.VIDEO_MODEL,
        images = images,
        DEVICE= config.DEVICE)
    audio_embeddings = get_audio_embeddings(
        DEVICE= config.DEVICE,
        audio_array=audio_array,
        audio_encoder=config.AUDIO_ENCODER,
        audio_extractor=config.AUDIO_EXTRACTOR,
        MAX_AUDIO_LENGTH=config.MAX_AUDIO_LEN)
    text_embeddings = get_text_embeddings(
        DEVICE=config.DEVICE,
        audio_array=audio_array,
        preprocesser = config.AUDIO_ENCODER,
        audio_model= config.AUDIO_EXTRACTOR,
        text_tokenizer= config.TEXT_TOKENIZER,
        text_encoder=config.TEXT_TOKENIZER,
        MAX_TEXT_LEN= config.MAX_TEXT_LEN)
    
    face = torch.from_numpy(face_embeddings).unsqueeze(0).to(config.DEVICE)
    audio = torch.from_numpy(audio_embeddings).unsqueeze(0).to(config.DEVICE)
    text = torch.from_numpy(text_embeddings).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():

        outputs = model(face, audio, text)
        probs = torch.softmax(outputs, dim = 1)
        preds = torch.argmax(probs, dim = 1)
        
        return preds.item(), probs.squeeze(0).cpu().tolist()
