"""
dataset.py
===========
Dataset và collate function cho Emotion Recognition.
"""

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
import pandas as pd


class EmotionDataset(Dataset):
    """Load audio từ metadata và trả về (input_values, label)."""

    def __init__(
        self,
        df: pd.DataFrame,
        processor: Wav2Vec2Processor,
        sr: int,
        max_sec: float,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.sr = sr
        self.max_len = int(sr * max_sec)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> torch.Tensor:
        wav, orig_sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if orig_sr != self.sr:
            wav = torchaudio.transforms.Resample(orig_sr, self.sr)(wav)
        return wav.squeeze(0)

    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        # Time shift ±10%
        if torch.rand(1) > 0.5:
            shift = int(len(wav) * 0.10 * (torch.rand(1) * 2 - 1).item())
            wav = torch.roll(wav, shift)
        # Gaussian noise
        if torch.rand(1) > 0.5:
            wav = wav + torch.randn_like(wav) * 0.004
        # Volume jitter ×[0.85, 1.15]
        if torch.rand(1) > 0.5:
            wav = wav * (0.85 + torch.rand(1) * 0.30)
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            wav = self._load(row["path"])
        except Exception as e:
            print(f"[WARN] Cannot load {row['path']}: {e}")
            wav = torch.zeros(self.sr)

        if self.augment:
            wav = self._augment(wav)

        wav = wav[: self.max_len]

        inp = self.proc(
            wav.numpy(),
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=False,
        ).input_values.squeeze(0)

        return inp, torch.tensor(row["label"], dtype=torch.long)


def pad_collate(batch):
    """Dynamic padding cho variable-length audio."""
    inputs, labels = zip(*batch)
    max_T = max(x.shape[0] for x in inputs)
    padded = torch.zeros(len(inputs), max_T)
    for i, x in enumerate(inputs):
        padded[i, : x.shape[0]] = x
    return padded, torch.stack(labels)
