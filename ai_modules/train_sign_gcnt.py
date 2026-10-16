"""
Indian Sign Language (ISL) GCN-Transformer Training Pipeline
============================================================
MCA Major Project: InclusiveMatch AI

Enhanced Scientific Implementation:
- 42-node dual-hand MediaPipe anatomical skeleton graph
- 24-frame temporal sequence with [x, y, dx, dy] features
- Strict 3-way split (Train / Validation / Test) using nested StratifiedGroupKFold
- Zero video-identity / session leakage across all splits
- Realistic landmark data augmentation (scale, coordinate jitter, planar rotation)
- Class-balanced weighted CrossEntropyLoss + Label Smoothing
- AdamW optimizer with Cosine Annealing learning rate schedule
- Model checkpoint selection strictly via Validation Macro F1 and Accuracy (no test set leakage)
- Honest, rigorous evaluation on held-out test split
- Saves:
    models/sign_gcnt_best.pt
    models/sign_gcnt_best_classes.json
    reports/sign_classification_report.csv
    reports/sign_confusion_matrix.png
    reports/sign_accuracy_curve.png
    reports/sign_loss_curve.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from ai_modules.sign_model_gcnt import (
    GCNTransformerClassifier,
    NUM_NODES,
    NUM_NODES_PER_HAND,
    preprocess_landmark_sequence,
)


# ---------------------------------------------------------------------------
# Video Grouping Helper
# ---------------------------------------------------------------------------

def extract_source_video_id(stem: str) -> str:
    """
    Extracts underlying video identity from file stems across datasets
    (CISLR, ISL500, INCLUDE, ISLRTC) to ensure complete split independence.
    """
    parts = stem.split("__")
    if len(parts) >= 4 and parts[1] == "CISLR":
        # CISLR format: class__CISLR__clip_id__youtube_id(_segment)
        vid = parts[3]
        if "_" in vid and vid.rsplit("_", 1)[1].isdigit():
            vid = vid.rsplit("_", 1)[0]
        return f"CISLR__{vid}"
    elif len(parts) >= 5 and parts[1] == "ISL500":
        # ISL500 format: class__ISL500__clip_id__word__session_id__clip_num
        return f"ISL500__{parts[4]}"
    elif len(parts) >= 4 and parts[1] == "INCLUDE":
        # INCLUDE format: class__INCLUDE__clip_id__MVI_xxxx
        return f"INCLUDE__{parts[3]}"
    elif len(parts) >= 4 and parts[1] == "ISLRTC":
        return f"ISLRTC__{parts[3]}"
    return stem


# ---------------------------------------------------------------------------
# Data Augmentation (Applied strictly to Training Data only)
# ---------------------------------------------------------------------------

def augment_sample(arr: np.ndarray) -> np.ndarray:
    """
    Applies realistic kinematic augmentations to a landmark sequence:
    1. Small coordinate jitter (Gaussian noise) on active joints.
    2. Random uniform scaling (0.95 - 1.05).
    3. Small planar rotation (+/- 8 degrees) around hand center of mass.
    4. Recomputes clean dynamic velocities [dx, dy] and relative wrist offsets.
    """
    coords = arr[:, :, :2].copy()
    T, J, C = arr.shape

    # 1. Coordinate Gaussian jitter on active joints
    active_mask = (coords != 0).any(axis=-1, keepdims=True)
    if random.random() < 0.65:
        noise = np.random.normal(0, 0.005, size=coords.shape).astype(np.float32)
        coords = coords + (noise * active_mask)

    # 2. Scale variation (0.95 to 1.05)
    if random.random() < 0.50:
        scale = random.uniform(0.95, 1.05)
        coords = coords * scale

    # 3. Planar rotation (+/- 8 degrees) per hand around hand center of mass
    if random.random() < 0.40:
        angle_deg = random.uniform(-8.0, 8.0)
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

        for offset in (0, NUM_NODES_PER_HAND):
            hand_pts = coords[:, offset : offset + NUM_NODES_PER_HAND]  # (T, 21, 2)
            act_mask = (hand_pts != 0).any(axis=-1)  # (T, 21)
            if np.any(act_mask):
                active_coords = hand_pts[act_mask]
                center = active_coords.mean(axis=0, keepdims=True)
                diff = hand_pts - center
                rot_diff = np.matmul(diff, rot_mat.T)
                rotated = center + rot_diff
                coords[:, offset : offset + NUM_NODES_PER_HAND] = np.where(
                    act_mask[:, :, None], rotated, 0.0
                )

    # 4. Clip to valid image bounds [0.0, 1.0] for active points
    coords = np.where(active_mask, np.clip(coords, 0.0, 1.0), 0.0)

    # 5. Recompute velocities for active joints cleanly
    vel = np.zeros_like(coords)
    h1_active = [bool(np.any(coords[t, :21] != 0)) for t in range(T)]
    h2_active = [bool(np.any(coords[t, 21:] != 0)) for t in range(T)]

    for t in range(1, T):
        if h1_active[t] and h1_active[t - 1]:
            vel[t, :21] = coords[t, :21] - coords[t - 1, :21]
        if h2_active[t] and h2_active[t - 1]:
            vel[t, 21:] = coords[t, 21:] - coords[t - 1, 21:]

    if C == 6:
        rel = np.zeros_like(coords)
        for t in range(T):
            if h1_active[t]:
                rel[t, :NUM_NODES_PER_HAND] = coords[t, :NUM_NODES_PER_HAND] - coords[t, 0:1]
            if h2_active[t]:
                rel[t, NUM_NODES_PER_HAND:] = coords[t, NUM_NODES_PER_HAND:] - coords[t, NUM_NODES_PER_HAND : NUM_NODES_PER_HAND + 1]
        return np.concatenate([coords, rel, vel], axis=-1).astype(np.float32)
    else:
        return np.concatenate([coords, vel], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SignLandmarkDataset(Dataset):
    def __init__(
        self,
        features: List[np.ndarray],
        labels: List[int],
        augment: bool = False,
    ):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[idx]
        if self.augment and random.random() < 0.75:
            feat = augment_sample(feat)
        return torch.from_numpy(feat).float(), torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Learning Rate Cosine Schedule with Warmup
# ---------------------------------------------------------------------------

def cosine_lr_schedule(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    lr_min_ratio: float = 0.01,
) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / max(warmup_epochs, 1)
    progress = float(epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main Training & Evaluation Pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    data_dir: str = r"C:\Users\ACER\isl-40words\processed_landmarks",
    output_path: str = "models/sign_gcnt_best.pt",
    min_samples: int = 15,
    seq_len: int = 24,
    node_dim: int = 6,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 18,
    seed: int = 42,
    gcn_dim: int = 64,
    tf_heads: int = 4,
    tf_layers: int = 2,
    dropout: float = 0.20,
    label_smoothing: float = 0.04,
    use_temp_conv: bool = True,
    use_multiscale_pool: bool = True,
) -> Dict[str, Any]:
    # Set seeds for strict reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("INCLUSIVE MATCH AI - INDIAN SIGN LANGUAGE (ISL) GCN-TRANSFORMER TRAINING")
    print("=" * 75)
    print(f"[Device] Hardware execution target: {device}")

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    # 1. Discover classes and samples
    classes = []
    file_list = []
    label_list = []
    group_list = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        valid_files = sorted(folder.glob("*.npy"))
        if len(valid_files) < min_samples:
            continue
        classes.append(folder.name)

        for f in valid_files:
            file_list.append(f)
            label_list.append(folder.name)
            group_list.append(extract_source_video_id(f.stem))

    print(f"[Dataset] Discovered {len(file_list)} total samples across {len(classes)} classes (>= {min_samples} samples/class).")
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    y_indices = np.array([class_to_idx[name] for name in label_list], dtype=np.int64)
    groups = np.array(group_list)

    # 2. Preprocess raw sequences into (seq_len, 42, node_dim)
    print(f"[Preprocessing] Preprocessing landmark sequences into uniform shape ({seq_len}, 42, {node_dim})...")
    preprocessed_data = []
    for f in file_list:
        raw_arr = np.load(f)
        feat = preprocess_landmark_sequence(raw_arr, seq_len=seq_len, node_dim=node_dim)
        preprocessed_data.append(feat)

    # 3. Leak-free 3-way Split (Train / Validation / Test)
    # Outer split: 80% Train+Val, 20% Held-out Test
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_outer.split(preprocessed_data, y_indices, groups))

    # Inner split: 80% Train, 20% Validation from Train+Val
    sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_sub, val_sub = next(sgkf_inner.split(
        [preprocessed_data[i] for i in train_val_idx],
        y_indices[train_val_idx],
        groups[train_val_idx],
    ))
    train_idx = train_val_idx[train_sub]
    val_idx = train_val_idx[val_sub]

    # Verification of zero leakage across video identities
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert len(train_groups & test_groups) == 0, "Data leakage between Train and Test!"
    assert len(val_groups & test_groups) == 0, "Data leakage between Val and Test!"
    assert len(train_groups & val_groups) == 0, "Data leakage between Train and Val!"

    print(f"[Split] Train samples: {len(train_idx)} | Validation samples: {len(val_idx)} | Test samples: {len(test_idx)}")
    print(f"[Split] Video groups in Train: {len(train_groups)} | Val: {len(val_groups)} | Test: {len(test_groups)}")
    print("[Split] Zero video leakage verified across all three splits.")

    # 4. Data Loaders
    train_features = [preprocessed_data[i] for i in train_idx]
    train_labels = [int(y_indices[i]) for i in train_idx]
    val_features = [preprocessed_data[i] for i in val_idx]
    val_labels = [int(y_indices[i]) for i in val_idx]
    test_features = [preprocessed_data[i] for i in test_idx]
    test_labels = [int(y_indices[i]) for i in test_idx]

    train_ds = SignLandmarkDataset(train_features, train_labels, augment=True)
    val_ds = SignLandmarkDataset(val_features, val_labels, augment=False)
    test_ds = SignLandmarkDataset(test_features, test_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5. Model Initialization
    model = GCNTransformerClassifier(
        num_classes=len(classes),
        seq_len=seq_len,
        node_dim=node_dim,
        gcn_dim=gcn_dim,
        tf_heads=tf_heads,
        tf_layers=tf_layers,
        dropout=dropout,
        use_temp_conv=use_temp_conv,
        use_multiscale_pool=use_multiscale_pool,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Initialized GCNTransformerClassifier: {total_params:,} parameters (Node Dim: {node_dim}, TempConv: {use_temp_conv}, MultiScale: {use_multiscale_pool}).")

    # Class-balanced loss weights
    class_counts = np.bincount(train_labels, minlength=len(classes)).astype(np.float32)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = (weights / weights.mean()).clip(0.3, 3.5)
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: cosine_lr_schedule(ep, epochs, warmup_epochs),
    )

    # 6. Training Loop with Metric History & Validation Early Stopping
    best_macro_f1 = -1.0
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    print("\n" + "-" * 75)
    print("BEGINNING TRAINING EPOCHS")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_preds, train_targets = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * len(yb)
            train_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            train_targets.extend(yb.cpu().tolist())

        scheduler.step()
        train_loss = running_train_loss / len(train_idx)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation Pass
        model.eval()
        running_val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                running_val_loss += loss.item() * len(yb)
                val_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                val_targets.extend(yb.cpu().tolist())

        val_loss = running_val_loss / len(val_idx)
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        # Monitor validation macro F1 as primary metric
        if val_macro_f1 > best_macro_f1 or (val_macro_f1 == best_macro_f1 and val_acc > best_acc):
            best_macro_f1 = val_macro_f1
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            mark = " * [BEST]"
        else:
            patience_counter += 1
            mark = ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.1f}%) | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Macro F1: {val_macro_f1:.4f} | "
            f"LR: {curr_lr:.2e}{mark}"
        )

        if patience_counter >= patience:
            print(f"[Training] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    # 7. Evaluate on Held-out Independent Test Split
    print("\n" + "=" * 75)
    print(f"[Evaluation] Restoring best checkpoint from Epoch {best_epoch} for Test Evaluation...")
    print("=" * 75)
    model.load_state_dict(best_state)
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            test_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            test_targets.extend(yb.tolist())

    y_true = np.array(test_targets)
    y_pred = np.array(test_preds)

    test_acc = float(accuracy_score(y_true, y_pred))
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    test_macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    test_macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    test_weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    test_weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    correct_count = int(np.sum(y_true == y_pred))
    total_test = len(y_true)

    print("\nHONEST TEST PERFORMANCE METRICS (Held-out Test Split):")
    print("-" * 75)
    print(f"Top-1 Test Accuracy     : {test_acc * 100:.2f}% ({correct_count} / {total_test} correct)")
    print(f"Macro F1-Score          : {test_macro_f1:.4f}")
    print(f"Weighted F1-Score       : {test_weighted_f1:.4f}")
    print(f"Macro Precision         : {test_macro_prec:.4f}")
    print(f"Macro Recall            : {test_macro_rec:.4f}")
    print(f"Weighted Precision      : {test_weighted_prec:.4f}")
    print(f"Weighted Recall         : {test_weighted_rec:.4f}")
    print(f"Classes Evaluated       : {len(classes)}")

    # Classification Report
    rep_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    rep_str = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )
    print("\nCLASSIFICATION REPORT (Test Split):\n")
    print(rep_str)

    # 8. Save Checkpoint and Artifacts
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        "arch": "gcn_transformer_2hand",
        "state_dict": best_state,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": len(classes),
        "seq_len": seq_len,
        "num_nodes": NUM_NODES,
        "node_dim": node_dim,
        "gcn_dim": gcn_dim,
        "tf_heads": tf_heads,
        "tf_layers": tf_layers,
        "dropout": dropout,
        "use_temp_conv": use_temp_conv,
        "use_multiscale_pool": use_multiscale_pool,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_acc),
        "best_val_macro_f1": float(best_macro_f1),
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "train_indices": [int(i) for i in train_idx],
        "val_indices": [int(i) for i in val_idx],
        "test_indices": [int(i) for i in test_idx],
        "test_file_stems": [file_list[i].stem for i in test_idx],
        "seed": seed,
    }
    torch.save(checkpoint_payload, out_file)
    print(f"[Artifact] Saved model checkpoint: {out_file.resolve()}")

    mapping_file = out_file.parent / f"{out_file.stem}_classes.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": classes,
                "class_to_idx": class_to_idx,
                "best_epoch": best_epoch,
                "best_val_accuracy": float(best_acc),
                "best_val_macro_f1": float(best_macro_f1),
                "test_accuracy": test_acc,
                "test_macro_f1": test_macro_f1,
                "test_weighted_f1": test_weighted_f1,
            },
            f,
            indent=2,
        )
    print(f"[Artifact] Saved class metadata: {mapping_file.resolve()}")

    # 9. Save Classification Report CSV
    rep_df = pd.DataFrame(rep_dict).transpose()
    rep_csv_path = reports_dir / "sign_classification_report.csv"
    rep_df.to_csv(rep_csv_path)
    print(f"[Artifact] Saved classification report CSV: {rep_csv_path.resolve()}")

    # 10. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"ISL Confusion Matrix (Top-1 Test Acc: {test_acc*100:.1f}%)", fontsize=14, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60, ha="right", fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)
    plt.xlabel("Predicted Sign Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Sign Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = reports_dir / "sign_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"[Artifact] Saved confusion matrix plot: {cm_plot_path.resolve()}")

    # 11. Learning Curves (Accuracy & Loss)
    epochs_range = list(range(1, len(history["train_loss"]) + 1))

    # Accuracy Curve
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#2563eb", linewidth=2)
    plt.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#16a34a", linewidth=2)
    plt.axvline(best_epoch, color="#dc2626", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Ep {best_epoch})")
    plt.title("ISL Model Training & Validation Accuracy Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    acc_curve_path = reports_dir / "sign_accuracy_curve.png"
    plt.savefig(acc_curve_path, dpi=200)
    plt.close()
    print(f"[Artifact] Saved accuracy curve: {acc_curve_path.resolve()}")

    # Loss Curve
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", color="#ea580c", linewidth=2)
    plt.axvline(best_epoch, color="#dc2626", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Ep {best_epoch})")
    plt.title("ISL Model Training & Validation Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (CrossEntropy)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    loss_curve_path = reports_dir / "sign_loss_curve.png"
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()
    print(f"[Artifact] Saved loss curve: {loss_curve_path.resolve()}")

    return {
        "best_epoch": best_epoch,
        "val_accuracy": float(best_acc),
        "val_macro_f1": float(best_macro_f1),
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "checkpoint_path": str(out_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Train 2-Hand GCN-Transformer ISL Classifier")
    parser.add_argument("--data-dir", default=r"C:\Users\ACER\isl-40words\processed_landmarks")
    parser.add_argument("--output", default="models/sign_gcnt_best.pt")
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--node-dim", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gcn-dim", type=int, default=64)
    parser.add_argument("--tf-heads", type=int, default=4)
    parser.add_argument("--tf-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--label-smoothing", type=float, default=0.04)

    args = parser.parse_args()
    train_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        min_samples=args.min_samples,
        seq_len=args.seq_len,
        node_dim=args.node_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        gcn_dim=args.gcn_dim,
        tf_heads=args.tf_heads,
        tf_layers=args.tf_layers,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()
