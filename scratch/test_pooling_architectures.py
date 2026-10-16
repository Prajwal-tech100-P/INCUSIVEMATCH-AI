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
)

# Load metadata and indices from existing checkpoint
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

# Load 4-dim baseline preprocessed features
print(f"Loading {len(file_list)} preprocessed sequences (24, 42, 4)...")
features = [preprocess_landmark_sequence(np.load(f), seq_len=24) for f in file_list]

# ---------------------------------------------------------------------------
# Enhanced Kinematic Data Augmentation
# ---------------------------------------------------------------------------
def augment_4dim_sample(arr: np.ndarray) -> np.ndarray:
    """
    Kinematic augmentations on (24, 42, 4) landmark array:
    1. Gaussian coordinate jitter on active landmarks.
    2. Random scaling (0.90 to 1.10).
    3. Random planar rotation (+/- 10 deg) around hand centers.
    4. Random frame dropout (zeroing 1-2 frames to simulate brief occlusion).
    5. Recomputed velocities [dx, dy].
    """
    coords = arr[:, :, :2].copy()
    T, J, _ = coords.shape

    active_mask = (coords != 0).any(axis=-1, keepdims=True)

    # 1. Jitter
    if random.random() < 0.70:
        noise = np.random.normal(0, 0.006, size=coords.shape).astype(np.float32)
        coords = coords + (noise * active_mask)

    # 2. Scaling
    if random.random() < 0.60:
        scale = random.uniform(0.90, 1.10)
        coords = coords * scale

    # 3. Planar rotation
    if random.random() < 0.50:
        angle_deg = random.uniform(-10.0, 10.0)
        theta = math.radians(angle_deg)
        rot_mat = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
        for offset in (0, 21):
            hand_pts = coords[:, offset : offset + 21]
            act = (hand_pts != 0).any(axis=-1)
            if np.any(act):
                center = hand_pts[act].mean(axis=0, keepdims=True)
                diff = hand_pts - center
                coords[:, offset : offset + 21] = np.where(act[:, :, None], center + np.matmul(diff, rot_mat.T), 0.0)

    # 4. Random frame dropout (simulate frame drop / MediaPipe miss)
    if random.random() < 0.35:
        drop_idx = random.randint(1, T - 2)
        coords[drop_idx] = 0.0

    coords = np.where(active_mask, np.clip(coords, 0.0, 1.0), 0.0)

    # 5. Recompute velocities
    vel = np.zeros_like(coords)
    h1_act = [bool(np.any(coords[t, :21] != 0)) for t in range(T)]
    h2_act = [bool(np.any(coords[t, 21:] != 0)) for t in range(T)]
    for t in range(1, T):
        if h1_act[t] and h1_act[t - 1]:
            vel[t, :21] = coords[t, :21] - coords[t - 1, :21]
        if h2_act[t] and h2_act[t - 1]:
            vel[t, 21:] = coords[t, 21:] - coords[t - 1, 21:]

    return np.concatenate([coords, vel], axis=-1).astype(np.float32)

class CustomSignDataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels, augment=False):
        self.feats = feats
        self.labels = labels
        self.augment = augment
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        f = self.feats[idx]
        if self.augment and random.random() < 0.80:
            f = augment_4dim_sample(f)
        return torch.from_numpy(f).float(), torch.tensor(self.labels[idx], dtype=torch.long)

# ---------------------------------------------------------------------------
# Architecture Variant A: Symmetrical Dual-Hand Pooling + Mean/Max Temporal
# ---------------------------------------------------------------------------
class DualHandMultiScaleGCNT(nn.Module):
    def __init__(self, num_classes=29, seq_len=24, node_dim=4, gcn_dim=64, tf_heads=4, tf_layers=2, dropout=0.25):
        super().__init__()
        self.seq_len = seq_len
        self.gcn_dim = gcn_dim

        adj = build_two_hand_adjacency()
        self.gcn = GCNBlock(node_dim, gcn_dim, adj)

        # Symmetrical hand pooling: extracts 32 dims per hand
        self.hand_pool = nn.Sequential(
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

        # Dual temporal pooling: Mean + Max pooling -> gcn_dim * 2 = 128
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
        p1 = self.hand_pool(h1)  # (BT, 32)
        p2 = self.hand_pool(h2)  # (BT, 32)
        h = torch.cat([p1, p2], dim=-1).reshape(B, T, self.gcn_dim)  # (B, T, 64)

        h = self.pos_enc(h)
        h = self.transformer(h)  # (B, T, 64)

        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h_pool = torch.cat([h_mean, h_max], dim=-1)  # (B, 128)
        return self.head(h_pool)

# ---------------------------------------------------------------------------
# Architecture Variant B: Attentive Temporal Pooling GCNT
# ---------------------------------------------------------------------------
class AttentiveTemporalGCNT(nn.Module):
    def __init__(self, num_classes=29, seq_len=24, node_dim=4, gcn_dim=64, tf_heads=4, tf_layers=2, dropout=0.25):
        super().__init__()
        self.seq_len = seq_len
        self.gcn_dim = gcn_dim

        adj = build_two_hand_adjacency()
        self.gcn = GCNBlock(node_dim, gcn_dim, adj)

        # Shared hand pool
        self.hand_pool = nn.Sequential(
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

        # Attention weights over temporal frames
        self.temp_attn = nn.Sequential(
            nn.Linear(gcn_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # Head combines mean and attentive peak representation
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

        h1 = h[:, :21].reshape(B * T, 21 * self.gcn_dim)
        h2 = h[:, 21:].reshape(B * T, 21 * self.gcn_dim)
        p1 = self.hand_pool(h1)
        p2 = self.hand_pool(h2)
        h = torch.cat([p1, p2], dim=-1).reshape(B, T, self.gcn_dim)

        h = self.pos_enc(h)
        h = self.transformer(h)  # (B, T, gcn_dim)

        # Attentive frame weighting
        weights = F.softmax(self.temp_attn(h), dim=1)  # (B, T, 1)
        h_attn = (weights * h).sum(dim=1)              # (B, gcn_dim)
        h_mean = h.mean(dim=1)                         # (B, gcn_dim)

        h_pool = torch.cat([h_mean, h_attn], dim=-1)   # (B, gcn_dim * 2)
        return self.head(h_pool)

# ---------------------------------------------------------------------------
# Training Experiment Runner
# ---------------------------------------------------------------------------
def evaluate_architecture_on_val(model_class, name, epochs=45, lr=1e-3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_x = [features[i] for i in train_indices]
    tr_y = y_indices[train_indices].tolist()
    val_x = [features[i] for i in val_indices]
    val_y = y_indices[val_indices].tolist()

    tr_ds = CustomSignDataset(tr_x, tr_y, augment=True)
    val_ds = CustomSignDataset(val_x, val_y, augment=False)

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    model = model_class().to(device)
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

print("\n--- Running Validation Architectural Comparisons ---")
# 1. Baseline
evaluate_architecture_on_val(
    lambda: GCNTransformerClassifier(num_classes=29, seq_len=24, node_dim=4),
    name="Baseline GCN-T (Flattened Joint Pool + Mean Pool)",
)

# 2. Dual-Hand Multi-Scale (Separated Hands + Mean/Max Temporal)
evaluate_architecture_on_val(
    lambda: DualHandMultiScaleGCNT(num_classes=29, seq_len=24, node_dim=4),
    name="Variant A: Dual-Hand Multi-Scale (Shared Hand Pool + Mean/Max)",
)

# 3. Attentive Temporal Pooling
evaluate_architecture_on_val(
    lambda: AttentiveTemporalGCNT(num_classes=29, seq_len=24, node_dim=4),
    name="Variant B: Attentive Temporal Pooling (Shared Hand Pool + Attn)",
)
