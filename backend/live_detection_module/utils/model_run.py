import torch
import numpy as np
from utils.make_embeddings import get_face_embeddings, get_audio_embeddings, get_text_embeddings
from configs import config

def infrence_loop(images, audio_array):

    face_embeddings = get_face_embeddings(K=config.K, model = config.VIDEO_MODEL, images = images, DEVICE= config.DEVICE)
    audio_embeddings = get_audio_embeddings(DEVICE= config.DEVICE, audio_array=audio_array, audio_encoder=config.AUDIO_ENCODER, audio_extractor=config.AUDIO_EXTRACTOR, MAX_AUDIO_LENGTH=config.MAX_AUDIO_LEN)
    text_embeddings = get_text_embeddings(DEVICE=config.DEVICE, audio_array=audio_array, preprocesser = config.AUDIO_ENCODER, audio_model= config.AUDIO_EXTRACTOR, text_tokenizer= config.TEXT_TOKENIZER, text_encoder=config.TEXT_TOKENIZER, MAX_TEXT_LEN= config.MAX_TEXT_LEN)

    model = torch.load(config.FUSION_MODEL, map_location=config.DEVICE)