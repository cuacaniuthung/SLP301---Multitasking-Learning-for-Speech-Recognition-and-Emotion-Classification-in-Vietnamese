"""
config.py
==========
Cấu hình huấn luyện và emotion mapping (Russell's Circumplex Model).
"""

from dataclasses import dataclass

import torch


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    """Cấu hình huấn luyện Emotion Recognition."""

    # Dataset
    csv_path: str = "./data/metadata.csv"
    # Cột bắt buộc trong CSV: path, valence, arousal

    # Audio
    sample_rate: int = 16_000
    max_sec: float = 8.0

    # Ngưỡng phân nhóm cảm xúc (0.0 = chuẩn circumplex)
    v_thr: float = 0.0
    a_thr: float = 0.0

    # Wav2Vec2 backbone
    # Gợi ý tiếng Việt: "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
    model_name: str = "facebook/wav2vec2-base"
    freeze: bool = True
    dropout: float = 0.3

    # Training
    batch_size: int = 8
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    num_workers: int = 0  # 0 ổn định trên Windows; Colab có thể đặt 2

    # Random seed
    seed: int = 42

    # Output
    save_dir: str = "./checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# ============================================================================
# EMOTION MAPPING — Russell's Circumplex Model
# ============================================================================
#
#              HIGH AROUSAL
#                   │
#    Anxious (0)    │    Happy (1)
#    Lo lắng        │    Vui vẻ
#   ────────────────┼────────────────  VALENCE (+)
#    Sad (2)        │    Calm (3)
#    Buồn bã        │    Bình tĩnh
#                   │
#              LOW AROUSAL

NUM_CLASSES = 4

ID2EMO = {
    0: "anxious",
    1: "happy",
    2: "sad",
    3: "calm",
}

EMO2VI = {
    0: "Lo lắng",
    1: "Vui vẻ",
    2: "Buồn bã",
    3: "Bình tĩnh",
}

EMO2ID = {v: k for k, v in ID2EMO.items()}


def to_label(
    valence: float,
    arousal: float,
    v_thr: float = 0.0,
    a_thr: float = 0.0,
) -> int:
    """
    Ánh xạ (valence, arousal) → class id.

    Returns
    -------
    0 : anxious
    1 : happy
    2 : sad
    3 : calm
    """
    if valence < v_thr and arousal >= a_thr:
        return 0
    if valence >= v_thr and arousal >= a_thr:
        return 1
    if valence < v_thr and arousal < a_thr:
        return 2
    return 3
