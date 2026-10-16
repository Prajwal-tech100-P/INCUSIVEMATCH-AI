import sys
sys.path.insert(0, ".")
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from ai_modules.sign_model_gcnt import (
    GCNTransformerClassifier,
    build_two_hand_adjacency,
    PositionalEncoding,
    GCNBlock,
    NUM_NODES,
    NUM_NODES_PER_HAND,
    preprocess_landmark_sequence,
)
from ai_modules.train_sign_gcnt import (
    extract_source_video_id,
    cosine_lr_schedule,
    augment_sample,
)

# Load metadata from checkpoint to guarantee identical splits
ckpt = torch.load("models/sign_gcnt_best.pt", map_location="cpu", weights_only=False)
classes = ckpt["classes"]
class_to_idx = ckpt["class_to_idx"]
train_indices = ckpt["train_indices"]
val_indices = ckpt["val_indices"]

data_root = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")

file_list, label_list = [], []
for folder in sorted(data_root.iterdir()):
    if not folder.is_dir() or folder.name not in class_to_idx:
        continue
    valid_files = sorted(folder.glob("*.npy"))
    if len(valid_files) < 15:
        continue
    for f in valid_files:
        file_list.append(f)
        label_list.append(folder.name)

y_indices = np.array([class_to_idx[l] for l in label_list], dtype=np.int64)

# Load raw sequences
raw_sequences = [np.load(f) for f in file_list]

def preprocess_6dim_sequence(raw_arr: np.ndarray, seq_len: int = 24) -> np.ndarray:
    """
    Computes 6 features per joint:
    [x, y, rel_wrist_x, rel_wrist_y, dx, dy]
    rel_wrist is relative to joint 0 for hand 1, and joint 21 for hand 2.
    """
    raw = np.asarray(raw_arr, dtype=np.float32)
    if raw.ndim == 2:
        raw = raw.reshape(-1, NUM_NODES, 2)
    num_frames = len(raw)
    h1 = raw[:, :21]
    h2 = raw[:, 21:]
    h1_active = [bool(np.any(h1[t] != 0)) for t in range(num_frames)]
    h2_active = [bool(np.any(h2[t] != 0)) for t in range(num_frames)]
    frame_active = [h1_active[t] or h2_active[t] for t in range(num_frames)]

    if any(frame_active):
        first = frame_active.index(True)
        last = num_frames - 1 - frame_active[::-1].index(True)
        idx = np.linspace(first, last, seq_len).round().astype(int)
        sampled = raw[idx].copy()
        s_h1_act = [h1_active[i] for i in idx]
        s_h2_act = [h2_active[i] for i in idx]
    else:
        sampled = np.zeros((seq_len, NUM_NODES, 2), dtype=np.float32)
        s_h1_act = [False] * seq_len
        s_h2_act = [False] * seq_len

    # Velocity
    vel = np.zeros_like(sampled)
    for t in range(1, seq_len):
        if s_h1_act[t] and s_h1_act[t - 1]:
            vel[t, :21] = sampled[t, :21] - sampled[t - 1, :21]
        if s_h2_act[t] and s_h2_act[t - 1]:
            vel[t, 21:] = sampled[t, 21:] - sampled[t - 1, 21:]

    # Relative to wrist
    rel = np.zeros_like(sampled)
    for t in range(seq_len):
        if s_h1_act[t]:
            wrist1 = sampled[t, 0:1]  # (1, 2)
            rel[t, :21] = sampled[t, :21] - wrist1
        else:
            sampled[t, :21] = 0.0
            vel[t, :21] = 0.0

        if s_h2_act[t]:
            wrist2 = sampled[t, 21:22]  # (1, 2)
            rel[t, 21:] = sampled[t, 21:] - wrist2
        else:
            sampled[t, 21:] = 0.0
            vel[t, 21:] = 0.0

    # 6 dims: [x, y, rel_x, rel_y, dx, dy]
    return np.concatenate([sampled, rel, vel], axis=-1)

print("Preprocessing 4-dim baseline features...")
feats_4dim = [preprocess_landmark_sequence(arr, seq_len=24) for arr in raw_sequences]

print("Preprocessing 6-dim features (with wrist-relative handshape)...")
feats_6dim = [preprocess_6dim_sequence(arr, seq_len=24) for arr in raw_sequences]

class ExpDataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels, augment=False, is_6dim=False):
        self.feats = feats
        self.labels = labels
        self.augment = augment
        self.is_6dim = is_6dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        f = self.feats[idx]
        if self.augment and random.random() < 0.70:
            if not self.is_6dim:
                f = augment_sample(f)
            else:
                # Augment base coords and recompute rel and vel
                coords = f[:, :, :2].copy()
                T = coords.shape[0]
                act_mask = (coords != 0).any(axis=-1, keepdims=True)
                if random.random() < 0.65:
                    noise = np.random.normal(0, 0.005, size=coords.shape).astype(np.float32)
                    coords = coords + (noise * act_mask)
                if random.random() < 0.50:
                    scale = random.uniform(0.95, 1.05)
                    coords = coords * scale
                coords = np.where(act_mask, np.clip(coords, 0.0, 1.0), 0.0)

                # Recompute rel
                rel = np.zeros_like(coords)
                vel = np.zeros_like(coords)
                h1_act = [bool(np.any(coords[t, :21] != 0)) for t in range(T)]
                h2_act = [bool(np.any(coords[t, 21:] != 0)) for t in range(T)]
                for t in range(T):
                    if h1_act[t]:
                        rel[t, :21] = coords[t, :21] - coords[t, 0:1]
                    if h2_act[t]:
                        rel[t, 21:] = coords[t, 21:] - coords[t, 21:22]
                for t in range(1, T):
                    if h1_act[t] and h1_act[t - 1]:
                        vel[t, :21] = coords[t, :21] - coords[t - 1, :21]
                    if h2_act[t] and h2_act[t - 1]:
                        vel[t, 21:] = coords[t, 21:] - coords[t - 1, 21:]
                f = np.concatenate([coords, rel, vel], axis=-1).astype(np.float32)

        return torch.from_numpy(f).float(), torch.tensor(self.labels[idx], dtype=torch.long)

def run_val_experiment(features, is_6dim, name, node_dim=4, gcn_dim=64, tf_heads=4, tf_layers=2, dropout=0.25, epochs=50, lr=1e-3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_x = [features[i] for i in train_indices]
    tr_y = y_indices[train_indices].tolist()
    val_x = [features[i] for i in val_indices]
    val_y = y_indices[val_indices].tolist()

    tr_ds = ExpDataset(tr_x, tr_y, augment=True, is_6dim=is_6dim)
    val_ds = ExpDataset(val_x, val_y, augment=False, is_6dim=is_6dim)

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    model = GCNTransformerClassifier(
        num_classes=len(classes),
        seq_len=24,
        node_dim=node_dim,
        gcn_dim=gcn_dim,
        tf_heads=tf_heads,
        tf_layers=tf_layers,
        dropout=dropout,
    ).to(device)

    class_counts = np.bincount(tr_y, minlength=len(classes)).astype(np.float32)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = (weights / weights.mean()).clip(0.3, 3.5)
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: cosine_lr_schedule(ep, epochs, 5))

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_ep = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                v_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                v_targets.extend(yb.tolist())

        val_acc = accuracy_score(v_targets, v_preds)
        val_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_ep = ep

    print(f"[{name}] Best Val Acc: {best_val_acc*100:.2f}% | Macro F1: {best_val_f1:.4f} (Epoch {best_ep})")
    return best_val_acc, best_val_f1

print("\n--- Running Validation Feature Experiments ---")
# 1. 4-dim baseline
run_val_experiment(feats_4dim, is_6dim=False, name="Baseline 4-dim [x, y, dx, dy]", node_dim=4)

# 2. 6-dim with wrist-relative coordinates
run_val_experiment(feats_6dim, is_6dim=True, name="Proposed 6-dim [x, y, rel_x, rel_y, dx, dy]", node_dim=6)
