import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from Model import dataloader, utils, Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_AUDIO_LEN = 16000 * 5  # 5 sec clips
MAX_TEXT_LEN = 64
BATCH_SIZE = 8
EPOCHS = 15
LR = 2e-5

text_model_name = "distilbert-base-uncased"
audio_model_name = "facebook/wav2vec2-base-960h"

text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained(text_model_name).to(DEVICE)

audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name).to(DEVICE)

train_csv = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/data/MELD.Raw/train/train_sent_emo.csv"
dev_csv = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/data/MELD.Raw/dev/dev_sent_emo.csv"

le = LabelEncoder()
train_df = pd.read_csv(train_csv)
le.fit(train_df["Emotion"])

train_set = dataloader.MELDFusionDataset(train_csv, "C:/Users/hp333/Desktop/Multimodel_emotion_detection/data/meld_audio/train", text_tokenizer, audio_extractor, le)
dev_set = dataloader.MELDFusionDataset(dev_csv, "C:/Users/hp333/Desktop/Multimodel_emotion_detection/data/meld_audio/dev", text_tokenizer, audio_extractor, le)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = Model.FusionClassifier().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
SAVE_DIR = "C:/Users/hp333/Desktop/Multimodel_emotion_detection/Model/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
checkpoint_path = os.path.join(SAVE_DIR, "fusion_checkpoint.pt")

start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_dev_f1 = checkpoint["best_dev_f1"]
    print(f"✅ Resuming training from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    y_true, y_pred = [], []
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        optimizer.zero_grad()
        text_emb = utils.compute_text_embeddings(batch["input_ids"], batch["attention_mask"], text_encoder, DEVICE)
        audio_emb = utils.compute_audio_embeddings(batch["audio_values"], audio_encoder, DEVICE)
        logits = model(text_emb, audio_emb)
        loss = criterion(logits, batch["labels"].to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(batch["labels"].cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    dev_loss, dev_acc, dev_f1 = utils.evaluate(model, dev_loader, epoch, EPOCHS, DEVICE, text_encoder, audio_encoder)

    checkpoint_path = os.path.join(SAVE_DIR, "fusion_checkpoint.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dev_f1": dev_f1
    }, checkpoint_path)

    print("💾 Checkpoint saved (can resume training)")
    print(
        f"Epoch {epoch+1}: "
        f"Train Loss={total_loss/ len(train_loader):.4f}, Acc={acc:.4f}, F1={f1:.4f} | "
        f"Dev Loss={dev_loss:.4f}, Acc={dev_acc:.4f}, F1={dev_f1:.4f}"
    )