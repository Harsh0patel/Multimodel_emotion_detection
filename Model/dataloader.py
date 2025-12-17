import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa
import os
from Model import utils

MAX_AUDIO_LEN = 16000 * 5  

class MELDFusionDataset(Dataset):
    def __init__(self, csv_path, audio_dir, tokenizer, feature_extractor, label_encoder, max_text_len=64):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self.max_text_len = max_text_len
        self.df = self.df[self.df['Utterance'].notnull()]  # clean missing text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        utt = row["Utterance"]
        label = self.label_encoder.transform([row["Emotion"]])[0]
        dia, utt_id = row["Dialogue_ID"], row["Utterance_ID"]
        audio_path = os.path.join(self.audio_dir, f"dia{dia}_utt{utt_id}.wav")

        # --- TEXT ---
        text_inputs = self.tokenizer(
            utt,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt"
        )

        # --- AUDIO ---
        if os.path.exists(audio_path):
            try:
                audio, _ = librosa.load(audio_path, sr=16000)
                audio = utils.pad_or_truncate_np(audio, MAX_AUDIO_LEN)
            except Exception:
                audio = np.zeros(MAX_AUDIO_LEN)
        else:
            audio = np.zeros(MAX_AUDIO_LEN)

        # ✅ Explicitly wrap in list to force feature extractor padding uniformity
        audio_inputs = self.feature_extractor(
            [audio], sampling_rate=16000, return_tensors="pt", padding=True
        )

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "audio_values": audio_inputs["input_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }