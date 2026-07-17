"""
inference.py
=============
Load checkpoint và dự đoán cảm xúc từ file audio.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Processor

from config import CFG, NUM_CLASSES, ID2EMO, EMO2VI
from model import EmotionClassifier


def load_emotion_model(
    checkpoint_path: str | None = None,
    device: str | None = None,
):
    """
    Load EmotionClassifier + processor từ checkpoint.

    Returns
    -------
    model, processor, id2emo, emo2vi, device
    """
    device = device or CFG.device
    ckpt_path = Path(checkpoint_path or f"{CFG.save_dir}/best_model.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {ckpt_path}\n"
            "Chạy `python train_emotion.py` trước."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", CFG)
    model_name = getattr(cfg, "model_name", CFG.model_name)
    dropout = getattr(cfg, "dropout", CFG.dropout)

    model = EmotionClassifier(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        dropout=dropout,
        freeze=True,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    id2emo = ckpt.get("id2emo", ID2EMO)
    emo2vi = ckpt.get("emo2vi", EMO2VI)
    return model, processor, id2emo, emo2vi, device


def _load_wav(path: str, sr: int, max_sec: float) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
    wav = wav.squeeze(0)
    max_len = int(sr * max_sec)
    return wav[:max_len]


@torch.no_grad()
def predict_emotion(
    audio_path: str,
    model: EmotionClassifier,
    processor: Wav2Vec2Processor,
    device: str,
    sample_rate: int | None = None,
    max_sec: float | None = None,
    id2emo: dict | None = None,
    emo2vi: dict | None = None,
) -> dict:
    """
    Dự đoán cảm xúc từ 1 file audio.

    Returns
    -------
    dict với keys: label_id, label_en, label_vi, probs, scores_str
    """
    sample_rate = sample_rate or CFG.sample_rate
    max_sec = max_sec or CFG.max_sec
    id2emo = id2emo or ID2EMO
    emo2vi = emo2vi or EMO2VI

    wav = _load_wav(audio_path, sample_rate, max_sec)
    inp = processor(
        wav.numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=False,
    ).input_values.to(device)

    logits = model(inp)
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
    label_id = int(probs.argmax().item())

    scores_str = "\n".join(
        f"{emo2vi[i]} ({id2emo[i]}): {probs[i].item():.1%}"
        for i in range(NUM_CLASSES)
    )

    return {
        "label_id": label_id,
        "label_en": id2emo[label_id],
        "label_vi": emo2vi[label_id],
        "probs": probs.tolist(),
        "scores_str": scores_str,
    }
