import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_ROOT = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")
MODEL_PATH = Path("models/sign_gcnt.pt")
SEQ_LEN = 60
NUM_NODES = 42
FEATURES_PER_NODE = 2
HIDDEN = 64
HEADS = 4
LAYERS = 2
MIN_SAMPLES = 15
SEED = 42

def build_adjacency():
    A = torch.eye(NUM_NODES)
    for offset in (0, 21):
        for i in range(offset, offset + 20):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
    degree = A.sum(dim=1).clamp(min=1)
    D = torch.diag(degree.pow(-0.5))
    return D @ A @ D

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x, adjacency):
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
            d_model=HIDDEN, nhead=HEADS, dim_feedforward=128,
            dropout=0.2, batch_first=True, norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=LAYERS)
        self.classifier = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, num_classes))

    def forward(self, x):
        x = self.activation(self.gcn1(x, self.adjacency))
        x = self.dropout(self.activation(self.gcn2(x, self.adjacency)))
        x = x.mean(dim=2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def evaluate():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    
    samples, labels = [], []
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir() or folder.name not in class_to_idx:
            continue
        files = sorted(folder.glob("*.npy"))
        for file in files:
            arr = np.load(file).astype(np.float32)
            if arr.shape != (SEQ_LEN, 84) or not np.isfinite(arr).all():
                continue
            samples.append(arr.reshape(SEQ_LEN, NUM_NODES, FEATURES_PER_NODE))
            labels.append(folder.name)

    y = np.array([class_to_idx[name] for name in labels])
    X = np.stack(samples)

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=y)

    mean = np.array(checkpoint["normalization_mean"])
    std = np.array(checkpoint["normalization_std"])
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    model = GCNTransformer(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = y[test_idx]

    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(1).numpy()

    acc = accuracy_score(y_test, preds)
    print(f"EVALUATION RESULT:")
    print(f"Overall Test Accuracy: {acc * 100:.2f}% ({np.sum(y_test == preds)} / {len(y_test)} correct)")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, labels=list(range(len(classes))), target_names=classes, zero_division=0))

if __name__ == "__main__":
    evaluate()
