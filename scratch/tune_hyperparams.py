import sys
sys.path.insert(0, ".")
import math
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from ai_modules.sign_model_gcnt import (
    build_two_hand_adjacency,
    PositionalEncoding,
    GCNBlock,
    NUM_NODES,
)
from ai_modules.train_sign_gcnt import cosine_lr_schedule

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
raw_sequences = [np.load(f) for f in file_list]

def preprocess_6dim_sequence(raw_arr: np.ndarray, seq_len: int = 24) -> np.ndarray:
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

    vel = np.zeros_like(sampled)
    for t in range(1, seq_len):
        if s_h1_act[t] and s_h1_act[t - 1]:
            vel[t, :21] = sampled[t, :21] - sampled[t - 1, :21]
        if s_h2_act[t] and s_h2_act[t - 1]:
            vel[t, 21:] = sampled[t, 21:] - sampled[t - 1, 21:]

    rel = np.zeros_like(sampled)
    for t in range(seq_len):
        if s_h1_act[t]:
            rel[t, :21] = sampled[t, :21] - sampled[t, 0:1]
        else:
            sampled[t, :21] = 0.0
            vel[t, :21] = 0.0

        if s_h2_act[t]:
            rel[t, 21:] = sampled[t, 21:] - sampled[t, 21:22]
        else:
            sampled[t, 21:] = 0.0
            vel[t, 21:] = 0.0

    return np.concatenate([sampled, rel, vel], axis=-1)

feats_6dim = [preprocess_6dim_sequence(arr, seq_len=24) for arr in raw_sequences]

class TemporalGCNClassifier(nn.Module):
    def __init__(self, num_classes=29, seq_len=24, node_dim=6, gcn_dim=64, tf_heads=4, tf_layers=2, dropout=0.20):
        super().__init__()
        self.seq_len = seq_len
        self.gcn_dim = gcn_dim

        adj = build_two_hand_adjacency()
        self.gcn = GCNBlock(node_dim, gcn_dim, adj)
        self.joint_pool = nn.Sequential(
            nn.Linear(gcn_dim * NUM_NODES, gcn_dim),
            nn.LayerNorm(gcn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temp_conv = nn.Sequential(
            nn.Conv1d(gcn_dim, gcn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(gcn_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.pos_enc = PositionalEncoding(gcn_dim, max_len=seq_len + 64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=gcn_dim,
            nhead=tf_heads,
            dim_feedforward=gcn_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tf_layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(gcn_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(gcn_dim * 2, gcn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_dim, num_classes),
        )

    def forward(self, x):
        B, T, J, C = x.shape
        h = x.reshape(B * T, J, C)
        h = self.gcn(h)
        h = h.reshape(B, T, J * self.gcn_dim)
        h = self.joint_pool(h)

        h_conv = self.temp_conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + h_conv

        h = self.pos_enc(h)
        h = self.transformer(h)

        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h_pool = torch.cat([h_mean, h_max], dim=-1)
        return self.head(h_pool)

class Dataset6D(torch.utils.data.Dataset):
    def __init__(self, feats, labels, augment=False):
        self.feats = feats
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        f = self.feats[idx]
        if self.augment and random.random() < 0.70:
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

def eval_hparams(lr=1e-3, label_smoothing=0.04, dropout=0.20, weight_decay=1e-4, epochs=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_x = [feats_6dim[i] for i in train_indices]
    tr_y = y_indices[train_indices].tolist()
    val_x = [feats_6dim[i] for i in val_indices]
    val_y = y_indices[val_indices].tolist()

    tr_ds = Dataset6D(tr_x, tr_y, augment=True)
    val_ds = Dataset6D(val_x, val_y, augment=False)

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    model = TemporalGCNClassifier(dropout=dropout).to(device)

    class_counts = np.bincount(tr_y, minlength=len(classes)).astype(np.float32)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = (weights / weights.mean()).clip(0.3, 3.5)
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: cosine_lr_schedule(ep, epochs, 5))

    best_val_acc, best_val_f1, best_ep = 0.0, 0.0, 0

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

    print(f"HParams [lr={lr}, sm={label_smoothing}, drop={dropout}, wd={weight_decay}] -> Val Acc: {best_val_acc*100:.2f}% | Val Macro F1: {best_val_f1:.4f} (Epoch {best_ep})")
    return best_val_acc, best_val_f1

print("\n--- Tuning Hyperparameters on Validation Set ---")
eval_hparams(lr=1e-3, label_smoothing=0.04, dropout=0.20, weight_decay=1e-4)
eval_hparams(lr=8e-4, label_smoothing=0.03, dropout=0.18, weight_decay=5e-5)
eval_hparams(lr=1.2e-3, label_smoothing=0.05, dropout=0.22, weight_decay=1e-4)
