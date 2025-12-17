import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def compute_text_embeddings(input_ids, attention_mask, text_encoder, DEVICE):
    with torch.no_grad():
        outputs = text_encoder(input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE))
        return outputs.last_hidden_state[:, 0, :]  # CLS token


def compute_audio_embeddings(audio_values, audio_encoder, DEVICE):
    with torch.no_grad():
        outputs = audio_encoder(audio_values.to(DEVICE))
        return outputs.last_hidden_state.mean(dim=1)

def pad_or_truncate_np(audio_array, target_len):
    """Ensure audio array is exactly target_len long."""
    if len(audio_array) > target_len:
        return audio_array[:target_len]
    elif len(audio_array) < target_len:
        return np.pad(audio_array, (0, target_len - len(audio_array)))
    return audio_array

def evaluate(model, data_loader, epoch, EPOCHS, DEVICE, text_encoder, audio_encoder):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader,desc=f"Epoch {epoch+1}/{EPOCHS} [Dev]"):
            # move batch to device
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            text_emb = compute_text_embeddings(
                batch["input_ids"], batch["attention_mask"], text_encoder, DEVICE
            )
            audio_emb = compute_audio_embeddings(
                batch["audio_values"], audio_encoder, DEVICE
            )

            logits = model(text_emb, audio_emb)
            loss = criterion(logits, batch["labels"])

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return total_loss / len(data_loader), acc, f1