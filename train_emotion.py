"""
train_emotion.py
=================
Huấn luyện EmotionClassifier trên metadata (valence, arousal → 4 classes).

Cài đặt:
    pip install -r requirements.txt

Chạy:
    python train_emotion.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split

from config import CFG, Config, NUM_CLASSES, ID2EMO, EMO2VI, to_label
from dataset import EmotionDataset, pad_collate
from model import EmotionClassifier
from utils import compute_class_weights, run_epoch, print_report

warnings.filterwarnings("ignore")


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # ── 1. Load & map labels ───────────────────────────────────────────────
    df = pd.read_csv(cfg.csv_path)
    df["label"] = df.apply(
        lambda r: to_label(r["valence"], r["arousal"], cfg.v_thr, cfg.a_thr),
        axis=1,
    )

    print("\n── Phân phối nhãn ────────────────────────────────────────")
    for i, name in ID2EMO.items():
        n = (df["label"] == i).sum()
        bar = "█" * (n // 5)
        print(f"  [{i}] {EMO2VI[i]:<10} ({name:<8}): {n:>3}  {bar}")
    print(f"  Total: {len(df)}\n")

    # ── 2. Train / Val split ──────────────────────────────────────────────
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.val_ratio,
        stratify=df["label"],
        random_state=cfg.seed,
    )
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}\n")

    # ── 3. Processor & Dataloaders ────────────────────────────────────────
    proc = Wav2Vec2Processor.from_pretrained(cfg.model_name)
    tr_set = EmotionDataset(
        train_df, proc, cfg.sample_rate, cfg.max_sec, augment=True
    )
    va_set = EmotionDataset(
        val_df, proc, cfg.sample_rate, cfg.max_sec, augment=False
    )

    cw = compute_class_weights(train_df["label"].tolist())
    sw = cw[train_df["label"].values]
    sampler = WeightedRandomSampler(sw, num_samples=len(train_df), replacement=True)

    tr_loader = DataLoader(
        tr_set,
        cfg.batch_size,
        sampler=sampler,
        collate_fn=pad_collate,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    va_loader = DataLoader(
        va_set,
        cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ── 4. Model, Loss, Optimizer ─────────────────────────────────────────
    model = EmotionClassifier(
        cfg.model_name, NUM_CLASSES, cfg.dropout, cfg.freeze
    ).to(cfg.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Backbone : {cfg.model_name}")
    print(f"  Params   : {trainable:,} trainable / {total_p:,} total")
    print(f"  Device   : {cfg.device}\n")

    criterion = nn.CrossEntropyLoss(weight=cw.to(cfg.device))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    # ── 5. Training loop ──────────────────────────────────────────────────
    best_acc = 0.0
    print(f"{'Ep':>4} │ {'Tr Loss':>8} │ {'Tr Acc':>7} │ {'Va Loss':>8} │ {'Va Acc':>7}")
    print("─" * 52)

    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, tr_loader, optimizer, criterion, cfg.device, is_train=True
        )
        va_loss, va_acc = run_epoch(
            model, va_loader, optimizer, criterion, cfg.device, is_train=False
        )
        scheduler.step()

        tag = "  ◀ best" if va_acc > best_acc else ""
        print(
            f"{ep:>4} │ {tr_loss:>8.4f} │ {tr_acc:>7.4f} │ "
            f"{va_loss:>8.4f} │ {va_acc:>7.4f}{tag}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "epoch": ep,
                    "val_acc": va_acc,
                    "model": model.state_dict(),
                    "id2emo": ID2EMO,
                    "emo2vi": EMO2VI,
                    "config": cfg,
                },
                f"{cfg.save_dir}/best_model.pt",
            )

        if ep % 20 == 0 or ep == cfg.epochs:
            print(f"\n── Classification Report (Epoch {ep}) ──────────")
            print_report(model, va_loader, cfg.device)
            print()

    print(f"\n[✓] Training xong.  Best Val Acc: {best_acc:.4f}")
    print(f"    Checkpoint: {cfg.save_dir}/best_model.pt")
    print(
        """
─────────────────────────────────────────────────────────
Bước tiếp theo (sau khi val acc ổn định):
  1. Unfreeze encoder và fine-tune với lr nhỏ hơn:
       model.unfreeze_encoder(last_n_layers=4)
       optimizer = AdamW(filter(lambda p: p.requires_grad,
                                model.parameters()), lr=1e-5)
  2. Chạy demo: python app.py
─────────────────────────────────────────────────────────
"""
    )


if __name__ == "__main__":
    main(CFG)
