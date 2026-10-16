from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------- CONFIGURATION ----------------
DATA_ROOT = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")
OUTPUT_DIR = Path("sign_model_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_SAMPLES = 15
SEQ_LEN = 60
NUM_NODES = 42
FEATURES_PER_NODE = 2
HIDDEN = 64
HEADS = 4
LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 60
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

# ---------------- LOAD DATA ----------------
samples = []
labels = []

for folder in sorted(DATA_ROOT.iterdir()):
    if not folder.is_dir():
        continue

    files = sorted(folder.glob("*.npy"))
    if len(files) < MIN_SAMPLES:
        continue

    for file in files:
        arr = np.load(file).astype(np.float32)
        if arr.shape != (SEQ_LEN, 84) or not np.isfinite(arr).all():
            print("Skipping invalid file:", file)
            continue

        samples.append(arr.reshape(SEQ_LEN, NUM_NODES, FEATURES_PER_NODE))
        labels.append(folder.name)

classes = sorted(set(labels))
class_to_idx = {name: i for i, name in enumerate(classes)}
y = np.array([class_to_idx[name] for name in labels])
X = np.stack(samples)

print("Samples:", len(X))
print("Classes:", len(classes))
print("Device:", DEVICE)

# Stratified preliminary split.
# Verify source-video independence before using this as final test accuracy.
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=SEED, stratify=y
)

# Compute normalization using training data only.
mean = X[train_idx].mean(axis=(0, 1, 2), keepdims=True)
std = X[train_idx].std(axis=(0, 1, 2), keepdims=True)
std = np.maximum(std, 1e-5)

X = (X - mean) / std
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

class SignDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(y[idx], dtype=torch.long)
        )

train_loader = DataLoader(
    SignDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    SignDataset(test_idx), batch_size=BATCH_SIZE, shuffle=False
)

# ---------------- GRAPH CONVOLUTION ----------------
# Two hand graphs, each with 21 landmark nodes.
# Each graph uses a simple chain adjacency as a baseline.
def build_adjacency():
    A = torch.eye(NUM_NODES)
    for offset in (0, 21):
        for i in range(offset, offset + 20):
            A[i, i + 1] = 1
            A[i + 1, i] = 1

    # Normalize adjacency.
    degree = A.sum(dim=1).clamp(min=1)
    D = torch.diag(degree.pow(-0.5))
    return D @ A @ D

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        # x: batch x frames x nodes x features
        x = torch.matmul(adjacency, x)
        return self.linear(x)

class GCNTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.register_buffer("adjacency", build_adjacency())
        self.gcn1 = GraphConv(2, HIDDEN)
        self.gcn2 = GraphConv(HIDDEN, HIDDEN)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=HEADS,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=LAYERS
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, num_classes)
        )

    def forward(self, x):
        # x: batch x frames x 42 nodes x 2 coordinates
        x = self.activation(self.gcn1(x, self.adjacency))
        x = self.dropout(self.activation(self.gcn2(x, self.adjacency)))

        # Pool landmarks into one representation per frame.
        x = x.mean(dim=2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

model = GCNTransformer(len(classes)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# ---------------- TRAIN ----------------
best_acc = -1.0
best_path = OUTPUT_DIR / "sign_gcn_transformer.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * len(yb)

    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(yb.tolist())

    accuracy = accuracy_score(targets, predictions)
    avg_loss = running_loss / len(train_idx)
    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train loss: {avg_loss:.4f} | "
        f"Test accuracy: {accuracy * 100:.2f}%"
    )

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": len(classes),
            "classes": classes,
            "input_shape": [SEQ_LEN, NUM_NODES, FEATURES_PER_NODE],
            "hidden_dim": HIDDEN,
            "attention_heads": HEADS,
            "transformer_layers": LAYERS,
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
            "best_test_accuracy": best_acc,
            "seed": SEED
        }, best_path)

# ---------------- FINAL REPORT ----------------
checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

predictions, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        predictions.extend(model(xb).argmax(1).tolist())
        targets.extend(yb.tolist())

print("\nBest test accuracy: {:.2f}%".format(best_acc * 100))
print(classification_report(
    targets, predictions, labels=list(range(len(classes))),
    target_names=classes, zero_division=0
))

with open(OUTPUT_DIR / "classes.json", "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

print("\nSaved checkpoint:", best_path.resolve())
print("Saved class mapping:", (OUTPUT_DIR / "classes.json").resolve())
print("WARNING: Treat this test result as preliminary until source-video")
print("independence is verified. Do not integrate until evaluated.")
