"""Train a GRU sign-language classifier from MediaPipe landmark sequences.

CSV format:
- x0,y0,...,x20,y20,label plus optional sequence_id
- If sequence_id exists, consecutive rows form one gesture sequence.
- Without sequence_id, each row is treated as a static sign and repeated over
  the sequence window; this is useful for static signs but does not teach motion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class GRUNetwork(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_layers=2, num_classes=2, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(nn.Linear(hidden_size * 2, 96), nn.ReLU(), nn.Dropout(dropout), nn.Linear(96, num_classes))
    def forward(self, x):
        y, _ = self.gru(x)
        y = self.norm(y[:, -1, :])
        return self.head(y)


def make_window(frames, seq_len):
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) >= seq_len:
        idx = np.linspace(0, len(frames) - 1, seq_len).round().astype(int)
        frames = frames[idx]
    else:
        pad = np.repeat(frames[-1][None, :], seq_len - len(frames), axis=0)
        frames = np.vstack([frames, pad])
    velocity = np.vstack([np.zeros((1, 42), dtype=np.float32), np.diff(frames, axis=0)])
    return np.concatenate([frames, velocity], axis=1)


def augment(x):
    # Small landmark noise improves tolerance to camera jitter without changing labels.
    noise = np.random.normal(0, 0.008, x[:, :, :42].shape).astype(np.float32)
    y = x.copy()
    y[:, :, :42] += noise
    y[:, 1:, 42:] = y[:, 1:, :42] - y[:, :-1, :42]
    y[:, 0, 42:] = 0
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--output", default="models/sign_gru.pt")
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise SystemExit("CSV must contain a 'label' column.")
    feature_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]
    if not all(c in df.columns for c in feature_cols):
        # Also accept exactly 42 unnamed/numeric feature columns before label.
        other = [c for c in df.columns if c != "label" and c != "sequence_id"]
        if len(other) != 42:
            raise SystemExit("CSV needs x0,y0,...,x20,y20 (42 features), label, and optional sequence_id.")
        feature_cols = other

    labels = sorted(df["label"].astype(str).unique())
    if len(labels) < 2:
        raise SystemExit("At least 2 sign classes are required.")
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    sequences, targets = [], []
    if "sequence_id" in df.columns:
        for _, group in df.groupby("sequence_id", sort=False):
            lab = str(group["label"].iloc[0])
            arr = group[feature_cols].to_numpy(dtype=np.float32)
            if len(arr) >= 2:
                sequences.append(make_window(arr, args.seq_len)); targets.append(label_to_id[lab])
    else:
        # Static-sign compatibility mode. Each row becomes a constant sequence.
        for _, row in df.iterrows():
            arr = row[feature_cols].to_numpy(dtype=np.float32).reshape(1, 42)
            sequences.append(make_window(arr, args.seq_len)); targets.append(label_to_id[str(row["label"])])

    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(targets, dtype=np.int64)
    if len(X) < max(20, len(labels) * 5):
        raise SystemExit("Too few training sequences. Collect more examples per sign.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUNetwork(num_classes=len(labels)).to(device)
    counts = np.bincount(y_train, minlength=len(labels)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_acc, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if np.random.rand() < 0.7:
                xb = torch.from_numpy(augment(xb.detach().cpu().numpy())).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval(); preds=[]; truth=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device))
                preds.extend(out.argmax(1).cpu().numpy().tolist()); truth.extend(yb.numpy().tolist())
        acc = accuracy_score(truth, preds)
        scheduler.step(acc)
        print(f"Epoch {epoch:02d}/{args.epochs} - validation accuracy: {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f"Best held-out validation accuracy: {best_acc*100:.2f}%")
    print(classification_report(truth, preds, labels=list(range(len(labels))), target_names=labels, zero_division=0))

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "labels": labels, "seq_len": args.seq_len,
        "input_size": 84, "hidden_size": 128, "num_layers": 2, "dropout": 0.25,
        "validation_accuracy": best_acc,
    }, output)
    print(f"Saved GRU model: {output.resolve()}")
    Path(str(output) + ".labels.json").write_text(json.dumps(labels, indent=2))


if __name__ == "__main__":
    main()
