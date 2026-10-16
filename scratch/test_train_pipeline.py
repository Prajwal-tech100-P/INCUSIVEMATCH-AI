import sys
sys.path.insert(0, ".")
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from ai_modules.sign_model_gcnt import (
    GCNTransformerClassifier,
    preprocess_landmark_sequence,
    NUM_NODES,
)
from ai_modules.train_sign_gcnt import (
    extract_source_video_id,
    SignLandmarkDataset,
    cosine_lr_schedule,
)

DATA_ROOT = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")
MIN_SAMPLES = 15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

classes = []
file_list = []
label_list = []
group_list = []

for folder in sorted(DATA_ROOT.iterdir()):
    if not folder.is_dir():
        continue
    valid_files = sorted(folder.glob("*.npy"))
    if len(valid_files) < MIN_SAMPLES:
        continue
    classes.append(folder.name)
    for f in valid_files:
        file_list.append(f)
        label_list.append(folder.name)
        group_list.append(extract_source_video_id(f.stem))

class_to_idx = {name: i for i, name in enumerate(classes)}
y = np.array([class_to_idx[l] for l in label_list], dtype=np.int64)
groups = np.array(group_list)

print(f"Loaded {len(file_list)} files across {len(classes)} classes.")

# 1. Preprocess all raw landmark sequences to (24, 42, 4)
seq_len = 24
preprocessed = []
for f in file_list:
    arr = np.load(f)
    feat = preprocess_landmark_sequence(arr, seq_len=seq_len)
    preprocessed.append(feat)

# 2. Outer 5-fold split to isolate a 20% test set
sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_val_idx, test_idx = next(sgkf_outer.split(preprocessed, y, groups))

# Inner 5-fold split on train_val to get train and val sets
sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_inner, val_inner = next(sgkf_inner.split(
    [preprocessed[i] for i in train_val_idx],
    y[train_val_idx],
    groups[train_val_idx]
))
train_idx = train_val_idx[train_inner]
val_idx = train_val_idx[val_inner]

# Verify zero leakage
print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, "Leakage Train-Test!"
assert len(set(groups[val_idx]) & set(groups[test_idx])) == 0, "Leakage Val-Test!"
assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0, "Leakage Train-Val!"
print("Zero leakage verified across all 3 splits!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_ds = SignLandmarkDataset([preprocessed[i] for i in train_idx], y[train_idx].tolist(), augment=True)
val_ds = SignLandmarkDataset([preprocessed[i] for i in val_idx], y[val_idx].tolist(), augment=False)
test_ds = SignLandmarkDataset([preprocessed[i] for i in test_idx], y[test_idx].tolist(), augment=False)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

model = GCNTransformerClassifier(
    num_classes=len(classes),
    seq_len=seq_len,
    node_dim=4,
    gcn_dim=64,
    tf_heads=4,
    tf_layers=2,
    dropout=0.25,
).to(device)

class_counts = np.bincount(y[train_idx], minlength=len(classes)).astype(np.float32)
weights = class_counts.sum() / np.maximum(class_counts, 1.0)
weights = (weights / weights.mean()).clip(0.3, 3.5)
class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 35
warmup = 5
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: cosine_lr_schedule(ep, epochs, warmup))

best_val_acc = 0.0
best_state = None

print("\nStarting Training...")
for ep in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * len(yb)
    
    scheduler.step()
    train_loss = running_loss / len(train_idx)

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            val_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            val_targets.extend(yb.tolist())
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        saved_mark = " * [BEST]"
    else:
        saved_mark = ""

    if ep % 5 == 0 or ep == epochs or saved_mark:
        print(f"Epoch {ep:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}{saved_mark}")

# Evaluate on Held-out Test Set
print("\n" + "="*70)
print(f"Evaluating Best Checkpoint on Dedicated Independent Test Set ({len(test_idx)} samples)...")
model.load_state_dict(best_state)
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        test_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        test_targets.extend(yb.tolist())

test_acc = accuracy_score(test_targets, test_preds)
test_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
test_wf1 = f1_score(test_targets, test_preds, average="weighted", zero_division=0)

print(f"Independent Test Accuracy : {test_acc*100:.2f}% ({np.sum(np.array(test_targets)==np.array(test_preds))}/{len(test_targets)})")
print(f"Macro F1                  : {test_f1:.4f}")
print(f"Weighted F1               : {test_wf1:.4f}")
