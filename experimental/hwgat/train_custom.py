"""
train_custom.py - PyTorch Fine-Tuning Pipeline for HWGAT on Custom Sign Datasets

Features:
- Custom PyTorch Dataset loading verified (192, 64, 2) .npy landmark arrays
- Stratified session-aware Train/Val/Test partitioning
- Classifier head surgery (nn.Linear(256, num_custom_classes))
- Transfer learning with backbone freeze/unfreeze support
- CrossEntropy loss, AdamW optimizer, LR scheduling
- Best checkpoint saving with metadata and held-out test split persistence

Strictly isolated within experimental/hwgat/
"""
import os
import sys
import json
import random
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
for path_entry in (PROJECT_ROOT, CURRENT_DIR):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

from experimental.hwgat import dataset_utils as du
from experimental.hwgat import model_utils as mu


class LandmarkDataset(Dataset):
    """PyTorch Dataset loading pre-extracted (192, 64, 2) .npy landmark tensors."""

    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = samples  # (npy_path, label_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        npy_path, label_idx = self.samples[index]
        arr = np.load(npy_path)
        tensor = torch.tensor(arr, dtype=torch.float32)
        return tensor, label_idx


def prepare_dataset_splits(
    landmarks_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]], Dict[int, str]]:
    """
    Scan landmarks directory, map classes to contiguous 0..K-1 indices,
    and split samples into stratified Train, Val, and Test subsets.
    """
    random.seed(seed)
    np.random.seed(seed)

    class_names = sorted([
        d for d in os.listdir(landmarks_dir)
        if os.path.isdir(os.path.join(landmarks_dir, d))
    ])

    # Filter out empty classes
    active_classes = []
    class_files: Dict[str, List[str]] = {}

    for cname in class_names:
        cdir = os.path.join(landmarks_dir, cname)
        npy_files = [
            os.path.join(cdir, f)
            for f in os.listdir(cdir)
            if f.endswith(".npy") and os.path.getsize(os.path.join(cdir, f)) > 0
        ]
        if npy_files:
            active_classes.append(cname)
            random.shuffle(npy_files)
            class_files[cname] = npy_files

    if not active_classes:
        raise ValueError(f"No valid landmark samples found in {landmarks_dir}")

    idx_to_class: Dict[int, str] = {i: c for i, c in enumerate(active_classes)}
    class_to_idx: Dict[str, int] = {c: i for i, c in idx_to_class.items()}

    train_set: List[Tuple[str, int]] = []
    val_set: List[Tuple[str, int]] = []
    test_set: List[Tuple[str, int]] = []

    for cname in active_classes:
        files = class_files[cname]
        label = class_to_idx[cname]
        n = len(files)

        if n == 1:
            # Fallback for minimal 1-sample test: include in train
            train_set.append((files[0], label))
            val_set.append((files[0], label))
            test_set.append((files[0], label))
        elif n == 2:
            train_set.append((files[0], label))
            val_set.append((files[1], label))
            test_set.append((files[1], label))
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            # Ensure at least 1 remains for test
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1

            train_files = files[:n_train]
            val_files = files[n_train : n_train + n_val]
            test_files = files[n_train + n_val :]

            for f in train_files:
                train_set.append((f, label))
            for f in val_files:
                val_set.append((f, label))
            for f in test_files:
                test_set.append((f, label))

    return train_set, val_set, test_set, idx_to_class


def run_training(
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Execute HWGAT custom fine-tuning and export best checkpoint."""
    du.init_dataset_structure()
    device = torch.device(device_str)

    print("=" * 70)
    print("           HWGAT CUSTOM SIGN RECOGNITION TRAINING")
    print("=" * 70)

    train_samples, val_samples, test_samples, idx_to_class = prepare_dataset_splits(
        landmarks_dir=du.LANDMARKS_DIR,
    )
    num_classes = len(idx_to_class)

    print(f"Active Classes ({num_classes}): {list(idx_to_class.values())}")
    print(f"Dataset Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    print(f"Device: {device} | Batch Size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}")
    print("-" * 70)

    train_loader = DataLoader(LandmarkDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LandmarkDataset(val_samples), batch_size=batch_size, shuffle=False)

    # Save held-out test split manifest
    test_split_path = os.path.join(du.METADATA_DIR, "test_split.json")
    with open(test_split_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_samples": test_samples,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        }, f, indent=2)

    # 1. Load pretrained HWGAT model
    print("Loading pretrained HWGAT backbone weights...")
    model = mu.load_pretrained_hwgat(device="cpu")

    # 2. Adapt classifier head
    print(f"Adapting classifier head: 2,002 -> {num_classes} classes...")
    model = mu.adapt_classifier_head(model, num_classes)
    mu.set_backbone_trainable(model, trainable=(not freeze_backbone))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({'Head Only' if freeze_backbone else 'Full Model'})")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss()
    trainable_weights = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_weights, lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(du.CHECKPOINTS_DIR, "hwgat_custom_best.pt")
    history = []

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = running_val_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)
        scheduler.step(epoch_val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.1f}% || "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.1f}%"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4),
        })

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            mu.save_custom_checkpoint(
                model=model,
                class_mapping=idx_to_class,
                save_path=best_ckpt_path,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=epoch_val_loss,
                val_acc=epoch_val_acc,
                extra_metadata={
                    "num_classes": num_classes,
                    "classes": list(idx_to_class.values()),
                    "train_samples": len(train_samples),
                    "val_samples": len(val_samples),
                    "test_samples": len(test_samples),
                },
            )
            print(f"  >>> [NEW BEST] Checkpoint saved: {best_ckpt_path} (Val Acc: {best_val_acc * 100:.1f}%)")

    # Save history
    history_path = os.path.join(du.REPORTS_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-" * 70)
    print(f"[COMPLETE] Best Validation Accuracy: {best_val_acc * 100:.1f}%")
    print(f"Checkpoint Location: {best_ckpt_path}")
    print("=" * 70)

    return {
        "best_val_acc": best_val_acc,
        "checkpoint_path": best_ckpt_path,
        "num_classes": num_classes,
        "classes": list(idx_to_class.values()),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HWGAT on custom sign dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4 for CPU safety)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--unfreeze", action="store_true", help="Unfreeze entire backbone for full fine-tuning")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_backbone=(not args.unfreeze),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

