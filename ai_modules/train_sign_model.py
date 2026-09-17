"""Train a GCN + Transformer sign-language classifier from MediaPipe sequences.

CSV format
----------
Required columns : x0,y0, x1,y1, …, x20,y20  (42 feature cols), label
Optional column  : sequence_id  (groups consecutive rows into one gesture clip)

Without sequence_id each row is treated as a static sign and tiled into a
constant-motion sequence (useful for letter signs, not dynamic gestures).

Usage
-----
  python ai_modules/train_sign_model.py data/sign_sequences.csv

Options
-------
  --output       models/sign_gcnt.pt   checkpoint path
  --seq-len      24                    frames per clip
  --epochs       80
  --batch-size   32
  --lr           3e-4
  --patience     15                    early-stop patience (epochs)
  --gcn-dim      64                    GCN output channels
  --tf-layers    2                     Transformer encoder depth
  --tf-heads     4                     attention heads (must divide gcn-dim)
  --dropout      0.30
  --warmup       10                    LR warmup epochs (cosine schedule)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------------------------------------------------------
# Hand skeleton
# ---------------------------------------------------------------------------
_HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),
    (0, 5),  (5, 6),  (6, 7),  (7, 8),
    (0, 9),  (9, 10), (10, 11),(11, 12),
    (0, 13), (13, 14),(14, 15),(15, 16),
    (0, 17), (17, 18),(18, 19),(19, 20),
    (5, 9),  (9, 13), (13, 17),
]
_NUM_JOINTS = 21


def build_adjacency() -> np.ndarray:
    A = np.zeros((_NUM_JOINTS, _NUM_JOINTS), dtype=np.float32)
    for i, j in _HAND_EDGES:
        A[i, j] = A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)
    D = A.sum(axis=1, keepdims=True)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    return (D_inv_sqrt * A * D_inv_sqrt.T).astype(np.float32)


_ADJ = build_adjacency()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adj: np.ndarray):
        super().__init__()
        self.register_buffer("adj", torch.from_numpy(adj))
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn     = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = self.linear(x)                                          # (BT, J, out)
        h  = torch.bmm(self.adj.unsqueeze(0).expand(h.size(0), -1, -1), h)
        bt, j, d = h.shape
        h  = self.bn(h.reshape(bt * j, d)).reshape(bt, j, d)
        return F.relu(h)


class GCNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adj: np.ndarray):
        super().__init__()
        self.gcn1 = GraphConv(in_dim, out_dim, adj)
        self.gcn2 = GraphConv(out_dim, out_dim, adj)
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.gcn2(self.gcn1(x)) + self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GCNTransformerNetwork(nn.Module):
    """
    Spatial GCN (per frame) → Temporal Transformer → classifier.

    Input : (B, T, J, node_dim)
    Output: (B, num_classes)
    """
    def __init__(
        self,
        num_classes: int,
        seq_len:     int   = 24,
        node_dim:    int   = 4,
        gcn_dim:     int   = 64,
        tf_heads:    int   = 4,
        tf_layers:   int   = 2,
        dropout:     float = 0.30,
    ):
        super().__init__()
        self.gcn        = GCNBlock(node_dim, gcn_dim, _ADJ)
        self.joint_pool = nn.Linear(gcn_dim * _NUM_JOINTS, gcn_dim)
        self.pos_enc    = PositionalEncoding(gcn_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=gcn_dim, nhead=tf_heads,
            dim_feedforward=gcn_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=tf_layers, enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(gcn_dim),
            nn.Dropout(dropout),
            nn.Linear(gcn_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, C = x.shape
        h = x.reshape(B * T, J, C)
        h = self.gcn(h)                               # (BT, J, gcn_dim)
        h = h.reshape(B, T, J * h.shape[-1])
        h = self.joint_pool(h)                        # (B, T, gcn_dim)
        h = self.pos_enc(h)
        h = self.transformer(h)
        h = h.mean(dim=1)                             # global avg pool
        return self.head(h)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_window(frames: np.ndarray, seq_len: int) -> np.ndarray:
    """Resample `frames` (N, 42) to (seq_len, 21, 4) with velocity."""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) >= seq_len:
        idx    = np.linspace(0, len(frames) - 1, seq_len).round().astype(int)
        frames = frames[idx]
    else:
        pad    = np.repeat(frames[-1:], seq_len - len(frames), axis=0)
        frames = np.vstack([frames, pad])
    # (seq_len, 21, 2)
    joints = frames.reshape(seq_len, _NUM_JOINTS, 2)
    vel    = np.zeros_like(joints)
    vel[1:] = joints[1:] - joints[:-1]
    return np.concatenate([joints, vel], axis=-1)   # (seq_len, 21, 4)


class SignDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, T, J, C)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, J, 4) — mutates a numpy copy, returns tensor.
    Augmentations:
      - Gaussian joint jitter
      - Random temporal speed (stretch/compress within window)
      - Random horizontal flip (mirror hand)
      - Random joint masking (simulate occluded fingers)
    """
    arr = x.detach().cpu().numpy().copy()   # (B, T, J, 4)
    B, T, J, C = arr.shape

    for b in range(B):
        # Jitter
        arr[b, :, :, :2] += np.random.normal(0, 0.008, (T, J, 2)).astype(np.float32)

        # Temporal speed jitter: resample T frames from a shorter/longer sub-range
        if random.random() < 0.5:
            speed = random.uniform(0.7, 1.3)
            new_T = max(2, min(T, int(T * speed)))
            idx   = np.linspace(0, T - 1, new_T).round().astype(int)
            tmp   = arr[b, idx, :, :2]          # (new_T, J, 2)
            # Pad/crop back to T
            if new_T < T:
                pad = np.repeat(tmp[-1:], T - new_T, axis=0)
                tmp = np.vstack([tmp, pad])
            else:
                tmp = tmp[:T]
            arr[b, :, :, :2] = tmp
            # Recompute velocities
            vel = np.zeros((T, J, 2), dtype=np.float32)
            vel[1:] = arr[b, 1:, :, :2] - arr[b, :-1, :, :2]
            arr[b, :, :, 2:] = vel

        # Horizontal flip (mirror x)
        if random.random() < 0.4:
            arr[b, :, :, 0]  = -arr[b, :, :, 0]
            arr[b, :, :, 2]  = -arr[b, :, :, 2]   # dx too

        # Joint masking (zero out random finger joints)
        if random.random() < 0.3:
            mask_joints = random.sample(range(1, J), k=random.randint(1, 4))
            arr[b, :, mask_joints, :] = 0.0

    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# LR schedule: cosine with warmup
# ---------------------------------------------------------------------------

def cosine_lr(epoch: int, total: int, warmup: int, lr_min: float = 1e-6) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (1.0 - lr_min) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train GCN+Transformer sign classifier")
    ap.add_argument("csv",                             help="CSV with landmarks + label + optional sequence_id")
    ap.add_argument("--output",      default="models/sign_gcnt.pt")
    ap.add_argument("--seq-len",     type=int,   default=24)
    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--batch-size",  type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--patience",    type=int,   default=15)
    ap.add_argument("--gcn-dim",     type=int,   default=64)
    ap.add_argument("--tf-layers",   type=int,   default=2)
    ap.add_argument("--tf-heads",    type=int,   default=4)
    ap.add_argument("--dropout",     type=float, default=0.30)
    ap.add_argument("--warmup",      type=int,   default=10)
    args = ap.parse_args()

    # ---- Load CSV ----
    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise SystemExit("CSV must contain a 'label' column.")

    feat_cols = [f"{ax}{i}" for i in range(21) for ax in ("x", "y")]
    if not all(c in df.columns for c in feat_cols):
        others = [c for c in df.columns if c not in ("label", "sequence_id")]
        if len(others) != 42:
            raise SystemExit("CSV needs x0,y0,…,x20,y20 (42 feature cols), label, optional sequence_id.")
        feat_cols = others

    labels      = sorted(df["label"].astype(str).unique())
    if len(labels) < 2:
        raise SystemExit("At least 2 sign classes required.")
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    print(f"Classes ({len(labels)}): {labels}")

    # ---- Build sequences ----
    sequences, targets = [], []
    if "sequence_id" in df.columns:
        for _, grp in df.groupby("sequence_id", sort=False):
            lab = str(grp["label"].iloc[0])
            arr = grp[feat_cols].to_numpy(dtype=np.float32)
            if len(arr) >= 2:
                sequences.append(make_window(arr, args.seq_len))
                targets.append(label_to_id[lab])
    else:
        for _, row in df.iterrows():
            arr = row[feat_cols].to_numpy(dtype=np.float32).reshape(1, 42)
            sequences.append(make_window(arr, args.seq_len))
            targets.append(label_to_id[str(row["label"])])

    X = np.stack(sequences).astype(np.float32)   # (N, T, J, 4)
    y = np.asarray(targets, dtype=np.int64)
    print(f"Dataset: {len(X)} sequences, shape {X.shape}")

    min_samples = max(20, len(labels) * 5)
    if len(X) < min_samples:
        raise SystemExit(f"Too few sequences ({len(X)} < {min_samples}). Collect more data.")

    # ---- Train / val split ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    train_ds = SignDataset(X_tr, y_tr)
    val_ds   = SignDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = GCNTransformerNetwork(
        num_classes=len(labels),
        seq_len=args.seq_len,
        node_dim=4,
        gcn_dim=args.gcn_dim,
        tf_heads=args.tf_heads,
        tf_layers=args.tf_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Class-weighted loss to handle imbalanced datasets
    counts  = np.bincount(y_tr, minlength=len(labels)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = (weights / weights.mean()).clip(0.2, 5.0)   # cap extreme weights
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.05,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: cosine_lr(ep, args.epochs, args.warmup),
    )

    # ---- Training loop ----
    best_acc, best_state, patience_cnt = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Augment ~70 % of batches
            if random.random() < 0.70:
                xb = augment_batch(xb).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # ---- Validation ----
        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device))
                preds.extend(out.argmax(1).cpu().numpy().tolist())
                truth.extend(yb.numpy().tolist())

        acc = accuracy_score(truth, preds)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"loss={train_loss/len(train_loader):.4f}  "
            f"val_acc={acc*100:.2f}%  lr={lr_now:.2e}"
        )

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    # ---- Save ----
    model.load_state_dict(best_state)
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb.to(device))
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            truth.extend(yb.numpy().tolist())

    print(f"\n{'='*60}")
    print(f"Best validation accuracy : {best_acc * 100:.2f}%")
    print(classification_report(truth, preds, target_names=labels, zero_division=0))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch":        "gcn_transformer",
            "state_dict":  model.state_dict(),
            "labels":      labels,
            "seq_len":     args.seq_len,
            "node_dim":    4,
            "gcn_dim":     args.gcn_dim,
            "tf_heads":    args.tf_heads,
            "tf_layers":   args.tf_layers,
            "dropout":     args.dropout,
            "val_accuracy": best_acc,
        },
        out_path,
    )
    print(f"\nSaved GCN-Transformer model → {out_path.resolve()}")
    labels_path = out_path.with_suffix(".labels.json")
    labels_path.write_text(json.dumps(labels, indent=2))
    print(f"Saved label map        → {labels_path.resolve()}")


if __name__ == "__main__":
    main()
