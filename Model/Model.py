import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, hidden_dim=512, num_classes=7):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text_emb, audio_emb):
        t = torch.relu(self.text_proj(text_emb))
        a = torch.relu(self.audio_proj(audio_emb))
        fused = torch.cat([t, a], dim=1)
        fused = self.dropout(fused)
        return self.fc(fused)