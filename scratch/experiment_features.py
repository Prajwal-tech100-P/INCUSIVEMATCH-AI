import sys
sys.path.insert(0, ".")
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from ai_modules.train_sign_gcnt import (
    extract_source_video_id,
    SignLandmarkDataset,
    cosine_lr_schedule,
)
from ai_modules.sign_model_gcnt import (
    GCNTransformerClassifier,
    build_two_hand_adjacency,
    NUM_NODES,
    NUM_NODES_PER_HAND,
    preprocess_landmark_sequence,
)

# Load split from models/sign_gcnt_best.pt
ckpt = torch.load("models/sign_gcnt_best.pt", map_location="cpu", weights_only=False)
classes = ckpt["classes"]
class_to_idx = ckpt["class_to_idx"]
train_indices = ckpt["train_indices"]
val_indices = ckpt["val_indices"]

data_root = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")

# Collect files in exact same order as training pipeline
file_list = []
label_list = []
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

print(f"Loaded {len(file_list)} files. Train: {len(train_indices)}, Val: {len(val_indices)}")

# ---------------------------------------------------------------------------
# Feature Extraction Variant 1: Enhanced Hand-Centric + Global Trajectory
# ---------------------------------------------------------------------------
def preprocess_enhanced_landmarks(raw: np.ndarray, seq_len: int = 24) -> np.ndarray:
    """
    Produces (seq_len, 42, 6) feature representation:
    - [x_rel, y_rel]: hand-centric normalized relative to wrist
    - [dx, dy]: frame-to-frame velocity
    - [x_wrist, y_wrist]: global wrist trajectory
    """
    arr = np.asarray(raw, dtype=np.float32).reshape(-1, 42, 2)
    num_frames = len(arr)

    h1 = arr[:, :21]
    h2 = arr[:, 21:]
    h1_act = [bool(np.any(h1[t] != 0)) for t in range(num_frames)]
    h2_act = [bool(np.any(h2[t] != 0)) for t in range(num_frames)]
    frame_act = [h1_act[t] or h2_act[t] for t in range(num_frames)]

    if any(frame_act):
        first = frame_act.index(True)
        last = num_frames - 1 - frame_act[::-1].index(True)
        idx = np.linspace(first, last, seq_len).round().astype(int)
        sampled = arr[idx].copy()
        s_h1 = [h1_act[i] for i in idx]
        s_h2 = [h2_act[i] for i in idx]
    else:
        sampled = np.zeros((seq_len, 42, 2), dtype=np.float32)
        s_h1 = [False] * seq_len
        s_h2 = [False] * seq_len

    # Relative coordinates + wrist global trajectory
    rel = sampled.copy()
    wrist_global = np.zeros((seq_len, 42, 2), dtype=np.float32)

    for t in range(seq_len):
        if s_h1[t]:
            w1 = sampled[t, 0].copy() # Hand 1 wrist
            # Palm scale (wrist to middle MCP joint 9)
            scale1 = max(np.linalg.norm(sampled[t, 9] - w1), 0.03)
            rel[t, :21] = (sampled[t, :21] - w1) / scale1
            wrist_global[t, :21] = w1
        else:
            rel[t, :21] = 0.0

        if s_h2[t]:
            w2 = sampled[t, 21].copy() # Hand 2 wrist
            scale2 = max(np.linalg.norm(sampled[t, 30] - w2), 0.03)
            rel[t, 21:] = (sampled[t, 21:] - w2) / scale2
            wrist_global[t, 21:] = w2
        else:
            rel[t, 21:] = 0.0

    # Velocities of raw landmarks
    vel = np.zeros_like(sampled)
    for t in range(1, seq_len):
        if s_h1[t] and s_h1[t - 1]:
            vel[t, :21] = sampled[t, :21] - sampled[t - 1, :21]
        if s_h2[t] and s_h2[t - 1]:
            vel[t, 21:] = sampled[t, 21:] - sampled[t - 1, 21:]

    # Concatenate [rel_x, rel_y, vel_x, vel_y, wrist_x, wrist_y] -> (seq_len, 42, 6)
    out = np.concatenate([rel, vel, wrist_global], axis=-1)
    return out.astype(np.float32)

# Preprocess both baseline (4-dim) and enhanced (6-dim)
print("Extracting baseline 4-dim features...")
base_feats = [preprocess_landmark_sequence(np.load(f), seq_len=24) for f in file_list]

print("Extracting enhanced 6-dim hand-centric features...")
enh_feats = [preprocess_enhanced_landmarks(np.load(f), seq_len=24) for f in file_list]

# ---------------------------------------------------------------------------
# Candidate Model 1: Enhanced Architecture with Hand Separation & Multi-scale Pooling
# ---------------------------------------------------------------------------
class MultiScaleGCNTransformer(nn.Module):
    def __init__(self, num_classes=29, seq_len=24, node_dim=6, gcn_dim=64, tf_heads=4, tf_layers=2, dropout=0.25):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.gcn_dim = gcn_dim

        adj = build_two_hand_adjacency()
        from ai_modules.sign_model_gcnt import GCNBlock, PositionalEncoding
        self.gcn = GCNBlock(node_dim, gcn_dim, adj)

        # Hand-separated pooling: Hand 1 (joints 0..20) and Hand 2 (joints 21..41)
        self.h1_pool = nn.Sequential(
            nn.Linear(gcn_dim * 21, gcn_dim // 2),
            nn.LayerNorm(gcn_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.h2_pool = nn.Sequential(
            nn.Linear(gcn_dim * 21, gcn_dim // 2),
            nn.LayerNorm(gcn_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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

        # Multi-scale temporal pooling: Mean + Max pooling (dim: gcn_dim * 2 = 128)
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
        h = self.gcn(h)  # (BT, 42, gcn_dim)
        h1 = h[:, :21].reshape(B * T, 21 * self.gcn_dim)
        h2 = h[:, 21:].reshape(B * T, 21 * self.gcn_dim)
        p1 = self.h1_pool(h1)  # (BT, 32)
        p2 = self.h2_pool(h2)  # (BT, 32)
        h = torch.cat([p1, p2], dim=-1).reshape(B, T, self.gcn_dim)  # (B, T, 64)

        h = self.pos_enc(h)
        h = self.transformer(h)  # (B, T, 64)

        # Multi-scale temporal pooling: mean + max
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h_pool = torch.cat([h_mean, h_max], dim=-1)  # (B, 128)
        return self.head(h_pool)

# ---------------------------------------------------------------------------
# Training & Validation Runner
# ---------------------------------------------------------------------------
def run_val_experiment(features, model_builder, name="Experiment", epochs=45, lr=1e-3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_x = [features[i] for i in train_indices]
    tr_y = y_indices[train_indices].tolist()
    val_x = [features[i] for i in val_indices]
    val_y = y_indices[val_indices].tolist()

    tr_ds = SignLandmarkDataset(tr_x, tr_y, augment=True)
    val_ds = SignLandmarkDataset(val_x, val_y, augment=False)

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    model = model_builder().to(device)
    class_counts = np.bincount(tr_y, minlength=len(classes)).astype(np.float32)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = (weights / weights.mean()).clip(0.3, 3.5)
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.08)
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

    print(f"[{name}] Best Validation Acc: {best_val_acc*100:.2f}% | Macro F1: {best_val_f1:.4f} (Epoch {best_ep})")
    return best_val_acc, best_val_f1

print("\n--- Running Validation Comparison Experiments ---")
# 1. Baseline 4-dim features + standard GCNTransformerClassifier
run_val_experiment(
    base_feats,
    lambda: GCNTransformerClassifier(num_classes=29, seq_len=24, node_dim=4),
    name="Baseline 4-dim (Global)",
)

# 2. Enhanced 6-dim hand-centric features + standard GCNTransformerClassifier
run_val_experiment(
    enh_feats,
    lambda: GCNTransformerClassifier(num_classes=29, seq_len=24, node_dim=6),
    name="Enhanced 6-dim (Hand-Centric + Global)",
)

# 3. Enhanced 6-dim hand-centric features + MultiScaleGCNTransformer (Separated Hands + Mean/Max Pooling)
run_val_experiment(
    enh_feats,
    lambda: MultiScaleGCNTransformer(num_classes=29, seq_len=24, node_dim=6),
    name="Enhanced 6-dim + MultiScale Model",
)

