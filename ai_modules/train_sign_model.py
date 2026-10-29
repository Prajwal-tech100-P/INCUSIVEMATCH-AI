"""
Indian Sign Language (ISL) GCN + Transformer Training Pipeline
=============================================================
MCA Major Project: InclusiveMatch AI

Primary Research Architecture:
- Dual-Hand 42-Node MediaPipe Graph Skeleton (Graph Convolutional Network)
- Multi-Head Temporal Transformer Encoder (Self-Attention over frames)
- Replaces legacy GRU/LSTM with genuine GCN + Transformer Encoder

Default Dataset:
- datasets/isl_custom/<class_name>/<sample_id>.npy
- Fallback: C:\\Users\\ACER\\isl-40words\\processed_landmarks

Usage:
  python ai_modules/train_sign_model.py
  python ai_modules/train_sign_model.py --data-dir datasets/isl_custom --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai_modules.train_sign_gcnt import train_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train Indian Sign Language GCN + Transformer Encoder Classifier"
    )
    # Default to datasets/isl_custom if populated, else external benchmark
    default_dir = "datasets/isl_custom"
    custom_path = Path(default_dir)
    if not (custom_path.exists() and any(custom_path.glob("*/*.npy"))):
        default_dir = r"C:\Users\ACER\isl-40words\processed_landmarks"

    parser.add_argument("--data-dir", default=default_dir, help="Directory containing <class_name>/*.npy landmark files")
    parser.add_argument("--output", default="models/sign_gcnt_best.pt", help="Output checkpoint path")
    parser.add_argument("--min-samples", type=int, default=15, help="Minimum samples per class to include")
    parser.add_argument("--seq-len", type=int, default=24, help="Target sequence length in frames")
    parser.add_argument("--node-dim", type=int, default=6, help="Feature dimension per node (4: x,y,dx,dy or 6: +relative coords)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for AdamW")
    parser.add_argument("--patience", type=int, default=18, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gcn-dim", type=int, default=64, help="GCN channel dimension")
    parser.add_argument("--tf-heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--tf-layers", type=int, default=2, help="Transformer encoder depth")
    parser.add_argument("--dropout", type=float, default=0.20, help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0.04, help="Label smoothing regularization")

    args = parser.parse_args()

    print("[Architecture] Primary Model: Spatial Graph Convolutional Network (GCN) + Temporal Transformer Encoder.")
    print("[Architecture] Note: GRU and LSTM are explicitly deprecated and disabled in this pipeline.")

    train_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        min_samples=args.min_samples,
        seq_len=args.seq_len,
        node_dim=args.node_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        gcn_dim=args.gcn_dim,
        tf_heads=args.tf_heads,
        tf_layers=args.tf_layers,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()
