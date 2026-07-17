"""
model.py
=========
EmotionClassifier: Wav2Vec2 encoder + Statistical Pooling + MLP head.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class EmotionClassifier(nn.Module):
    """
    wav2vec2 encoder (frozen)
        ↓
    Statistical Pooling: concat(mean, std)  [B, 2H]
        ↓
    MLP Emotion Head  →  logits [B, num_classes]
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float = 0.3,
        freeze: bool = True,
    ):
        super().__init__()
        self.freeze = freeze
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        H = self.encoder.config.hidden_size  # 768 (base)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Emotion Head — 3 layers: 2H → H → H/2 → C
        self.head = nn.Sequential(
            nn.LayerNorm(H * 2),
            nn.Linear(H * 2, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(H // 2, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                h = self.encoder(x).last_hidden_state  # [B, T', H]
        else:
            h = self.encoder(x).last_hidden_state

        mean = h.mean(dim=1)
        std = h.std(dim=1, unbiased=False)
        feat = torch.cat([mean, std], dim=1)  # [B, 2H]
        return self.head(feat)

    def unfreeze_encoder(self, last_n_layers: int = 4):
        """
        Unfreeze n transformer layer cuối để fine-tune tiếp
        (sau khi emotion head đã converge).
        """
        for p in self.encoder.parameters():
            p.requires_grad = False

        layers = self.encoder.encoder.layers
        for layer in layers[-last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.freeze = False
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[✓] Unfroze last {last_n_layers} transformer layers — trainable: {n:,}")
