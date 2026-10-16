import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

new_ckpt = torch.load("models/sign_gcnt_best.pt", map_location="cpu", weights_only=False)
test_stems = set(new_ckpt["test_file_stems"])
classes = new_ckpt["classes"]
class_to_idx = new_ckpt["class_to_idx"]

DATA_ROOT = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")

# Load old model
old_ckpt = torch.load("models/sign_gcnt_legacy_backup.pt", map_location="cpu", weights_only=False)
mean = np.array(old_ckpt["normalization_mean"], dtype=np.float32).reshape(1, 1, 2)
std = np.array(old_ckpt["normalization_std"], dtype=np.float32).reshape(1, 1, 2)

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x, adjacency):
        x = torch.matmul(adjacency, x)
        return self.linear(x)

def build_adj_chain():
    A = torch.eye(42)
    for offset in (0, 21):
        for i in range(offset, offset + 20):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
    degree = A.sum(dim=1).clamp(min=1)
    D = torch.diag(degree.pow(-0.5))
    return D @ A @ D

class OldGCNTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.register_buffer("adjacency", build_adj_chain())
        self.gcn1 = GraphConv(2, 64)
        self.gcn2 = GraphConv(64, 64)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.2, batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.classifier = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.activation(self.gcn1(x, self.adjacency))
        x = self.dropout(self.activation(self.gcn2(x, self.adjacency)))
        x = x.mean(dim=2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

old_model = OldGCNTransformer(len(classes))
old_model.load_state_dict(old_ckpt["model_state_dict"])
old_model.eval()

# Collect exact test files
test_samples, test_targets = [], []
for folder in sorted(DATA_ROOT.iterdir()):
    if not folder.is_dir() or folder.name not in class_to_idx:
        continue
    c_idx = class_to_idx[folder.name]
    for f in sorted(folder.glob("*.npy")):
        if f.stem in test_stems:
            raw = np.load(f).astype(np.float32)
            # Old model took (60, 42, 2) normalized by (X - mean) / std
            arr = raw.reshape(60, 42, 2)
            arr = (arr - mean) / std
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            test_samples.append(arr)
            test_targets.append(c_idx)

X_test = torch.tensor(np.stack(test_samples), dtype=torch.float32)
y_test = np.array(test_targets)

with torch.no_grad():
    logits = old_model(X_test)
    preds = logits.argmax(dim=-1).numpy()

old_acc = float(accuracy_score(y_test, preds))
old_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
old_wf1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
old_prec = float(precision_score(y_test, preds, average="macro", zero_division=0))
old_rec = float(recall_score(y_test, preds, average="macro", zero_division=0))

print("=" * 60)
print("OLD MODEL ON IDENTICAL HELD-OUT TEST SET (127 SAMPLES):")
print("=" * 60)
print(f"Old Top-1 Accuracy: {old_acc * 100:.2f}% ({np.sum(y_test == preds)} / {len(y_test)})")
print(f"Old Macro F1      : {old_f1:.4f}")
print(f"Old Weighted F1   : {old_wf1:.4f}")
print(f"Old Macro Prec    : {old_prec:.4f}")
print(f"Old Macro Recall  : {old_rec:.4f}")
