import torch
import torch.nn as nn


class TripleFusionClassifier(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, vision_dim=512, hidden_dim=512, num_classes=7):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 3, num_classes)

    def forward(self, text_emb, audio_emb, vision_emb):
        t = torch.relu(self.text_proj(text_emb))
        a = torch.relu(self.audio_proj(audio_emb))
        v = torch.relu(self.vision_proj(vision_emb))
        fused = torch.cat([t, a, v], dim=1)
        fused = self.dropout(fused)
        return self.fc(fused)