"""
utils.py
=========
Helpers cho training: class weights, epoch loop, classification report.
"""

from collections import Counter

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from config import NUM_CLASSES, EMO2VI


def compute_class_weights(labels: list) -> torch.Tensor:
    cnt = Counter(labels)
    n = sum(cnt.values())
    return torch.tensor(
        [n / (NUM_CLASSES * cnt.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float,
    )


def run_epoch(model, loader, optimizer, criterion, device, is_train: bool):
    model.train() if is_train else model.eval()
    # Khi encoder frozen: giữ encoder ở eval mode (BN/LN dùng running stats)
    if is_train and model.freeze:
        model.encoder.eval()

    total_loss = correct = total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                1.0,
            )
            optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(-1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def print_report(model, loader, device):
    all_p, all_y = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).argmax(-1)
            all_p += p.cpu().tolist()
            all_y += y.tolist()
    names = [EMO2VI[i] for i in range(NUM_CLASSES)]
    print(classification_report(all_y, all_p, target_names=names, zero_division=0))
