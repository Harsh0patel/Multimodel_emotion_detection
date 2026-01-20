import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_VIDEO_LEN = 512
MAX_AUDIO_LEN = 16000 * 3
MAX_TEXT_LEN = 64
K = 16

AUDIO_MODEL_NAME = "facebook/wav2vec2-base-960h"
TEXT_MODEL_NAME = "distilbert-base-uncased"

TEXT_TOKENIZER = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
TEXT_ENCODER = AutoModel.from_pretrained(TEXT_MODEL_NAME)
TEXT_ENCODER = TEXT_ENCODER.to(DEVICE)
AUDIO_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
AUDIO_ENCODER = Wav2Vec2Model.from_pretrained(AUDIO_MODEL_NAME)
AUDIO_ENCODER = AUDIO_ENCODER.to(DEVICE)
VIDEO_MODEL = models.resnet50(pretrained = True)
VIDEO_MODEL.fc = nn.Identity()  # type: ignore
for parms in VIDEO_MODEL.parameters():
    parms.requires_grad = False
VIDEO_MODEL = VIDEO_MODEL.to(DEVICE)
VIDEO_MODEL.eval()
FUSION_MODEL = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/backend/live_detection_module/models/fusion_model/Model_v2.pt"
