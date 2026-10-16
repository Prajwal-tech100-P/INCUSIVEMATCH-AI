"""
Indian Sign Language (ISL) 2-Hand GCN-Transformer (GCN-T) Core Architecture
==========================================================================
MCA Major Project: InclusiveMatch AI

Architecture Specifications:
- 42 Graph Nodes: Hand 1 (joints 0..20) + Hand 2 (joints 21..41)
- Anatomical MediaPipe hand skeleton topology per hand (no fabricated inter-hand edges)
- Symmetric normalized adjacency with self-loops
- Input features per node: 4 [x, y, dx, dy]
- GCN feature dimension: 64
- Temporal Transformer Encoder: 2 layers, 4 attention heads, d_model=64, dim_feedforward=256
- Sequence length: 24 frames
- Output: 29 class logits
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# MediaPipe Hand Skeleton Definition (21 landmarks per hand)
# ---------------------------------------------------------------------------
# Joint definitions:
# 0: Wrist
# 1..4: Thumb (CMC, MCP, IP, Tip)
# 5..8: Index finger (MCP, PIP, DIP, Tip)
# 9..12: Middle finger (MCP, PIP, DIP, Tip)
# 13..16: Ring finger (MCP, PIP, DIP, Tip)
# 17..20: Pinky finger (MCP, PIP, DIP, Tip)

MEDIAPIPE_HAND_EDGES: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles / transverse arch
    (5, 9), (9, 13), (13, 17),
]

NUM_NODES_PER_HAND = 21
NUM_HANDS = 2
NUM_NODES = NUM_NODES_PER_HAND * NUM_HANDS  # 42 nodes


def build_two_hand_adjacency() -> torch.Tensor:
    """
    Constructs the 42x42 block-diagonal normalized adjacency matrix.
    Hand 1 occupies indices 0..20.
    Hand 2 occupies indices 21..41.
    No artificial edges are placed between hands.
    Degree normalization: D^(-1/2) * (A + I) * D^(-1/2).
    """
    A = torch.zeros((NUM_NODES, NUM_NODES), dtype=torch.float32)

    # Hand 1 connections
    for u, v in MEDIAPIPE_HAND_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0

    # Hand 2 connections (shifted by 21)
    for u, v in MEDIAPIPE_HAND_EDGES:
        u2 = u + NUM_NODES_PER_HAND
        v2 = v + NUM_NODES_PER_HAND
        A[u2, v2] = 1.0
        A[v2, u2] = 1.0

    # Self-loops on all 42 joints
    A.fill_diagonal_(1.0)

    # Degree normalization
    degree = A.sum(dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_mat = torch.diag(d_inv_sqrt)

    norm_adj = D_mat @ A @ D_mat
    return norm_adj


# ---------------------------------------------------------------------------
# Preprocessing Pipeline (Shared between Training and Live Inference)
# ---------------------------------------------------------------------------

def preprocess_landmark_sequence(
    raw_sequence: np.ndarray,
    seq_len: int = 24,
    node_dim: int = 4,
) -> np.ndarray:
    """
    Consistent preprocessing for offline sequences (e.g. from dataset or video files).
    
    Parameters:
    -----------
    raw_sequence : np.ndarray
        Array of shape (T, 84) or (T, 42, 2) with landmark coordinates.
    seq_len : int
        Target number of frames (default 24).
    node_dim : int
        Feature dimensionality per node (4: [x, y, dx, dy] or 6: [x, y, rel_x, rel_y, dx, dy]).

    Returns:
    --------
    np.ndarray:
        Array of shape (seq_len, 42, node_dim).
    """
    raw = np.asarray(raw_sequence, dtype=np.float32)
    if raw.ndim == 2:
        if raw.shape[1] != 84:
            raise ValueError(f"Expected 84 features per frame, got {raw.shape[1]}")
        raw = raw.reshape(-1, NUM_NODES, 2)
    elif raw.ndim == 3:
        if raw.shape[1] != NUM_NODES or raw.shape[2] != 2:
            raise ValueError(f"Expected shape (T, 42, 2), got {raw.shape}")
    else:
        raise ValueError(f"Unexpected array ndim: {raw.ndim}")

    num_frames = len(raw)
    h1 = raw[:, :NUM_NODES_PER_HAND]
    h2 = raw[:, NUM_NODES_PER_HAND:]

    h1_active = [bool(np.any(h1[t] != 0)) for t in range(num_frames)]
    h2_active = [bool(np.any(h2[t] != 0)) for t in range(num_frames)]
    frame_active = [h1_active[t] or h2_active[t] for t in range(num_frames)]

    # 1. Temporal sampling
    if any(frame_active):
        first = frame_active.index(True)
        last = num_frames - 1 - frame_active[::-1].index(True)
        # Sample uniformly across active interval
        idx = np.linspace(first, last, seq_len).round().astype(int)
        sampled = raw[idx].copy()
        s_h1_act = [h1_active[i] for i in idx]
        s_h2_act = [h2_active[i] for i in idx]
    else:
        # If entire clip is missing landmarks
        sampled = np.zeros((seq_len, NUM_NODES, 2), dtype=np.float32)
        s_h1_act = [False] * seq_len
        s_h2_act = [False] * seq_len

    # 2. Velocity computation with missing-hand safety
    vel = np.zeros_like(sampled)
    for t in range(1, seq_len):
        # Velocity for Hand 1 only if active in BOTH t and t-1
        if s_h1_act[t] and s_h1_act[t - 1]:
            vel[t, :NUM_NODES_PER_HAND] = (
                sampled[t, :NUM_NODES_PER_HAND] - sampled[t - 1, :NUM_NODES_PER_HAND]
            )
        # Velocity for Hand 2 only if active in BOTH t and t-1
        if s_h2_act[t] and s_h2_act[t - 1]:
            vel[t, NUM_NODES_PER_HAND:] = (
                sampled[t, NUM_NODES_PER_HAND:] - sampled[t - 1, NUM_NODES_PER_HAND:]
            )

    # Ensure completely inactive joints are cleanly zeroed
    for t in range(seq_len):
        if not s_h1_act[t]:
            sampled[t, :NUM_NODES_PER_HAND] = 0.0
            vel[t, :NUM_NODES_PER_HAND] = 0.0
        if not s_h2_act[t]:
            sampled[t, NUM_NODES_PER_HAND:] = 0.0
            vel[t, NUM_NODES_PER_HAND:] = 0.0

    if node_dim == 6:
        # Relative coordinates to wrist of each respective hand
        rel = np.zeros_like(sampled)
        for t in range(seq_len):
            if s_h1_act[t]:
                rel[t, :NUM_NODES_PER_HAND] = (
                    sampled[t, :NUM_NODES_PER_HAND] - sampled[t, 0:1]
                )
            if s_h2_act[t]:
                rel[t, NUM_NODES_PER_HAND:] = (
                    sampled[t, NUM_NODES_PER_HAND:] - sampled[t, NUM_NODES_PER_HAND : NUM_NODES_PER_HAND + 1]
                )
        features = np.concatenate([sampled, rel, vel], axis=-1)  # (seq_len, 42, 6)
    else:
        features = np.concatenate([sampled, vel], axis=-1)  # (seq_len, 42, 4)

    return features


def preprocess_live_frame_buffer(
    frame_buffer: List[np.ndarray],
    seq_len: int = 24,
    node_dim: int = 4,
) -> np.ndarray:
    """
    Consistent preprocessing for live webcam frame buffers.
    Each item in frame_buffer is a (42, 2) or (84,) float32 array representing
    the 2-hand landmarks detected in a single video frame.
    
    If fewer than seq_len frames are available, the buffer is padded cleanly.
    """
    if len(frame_buffer) == 0:
        return np.zeros((seq_len, NUM_NODES, node_dim), dtype=np.float32)

    raw = np.stack(frame_buffer, axis=0).astype(np.float32)
    if raw.ndim == 2:
        raw = raw.reshape(-1, NUM_NODES, 2)

    # Pad or slice to seq_len
    if len(raw) < seq_len:
        pad_len = seq_len - len(raw)
        pad = np.repeat(raw[-1:], pad_len, axis=0)
        raw = np.vstack([raw, pad])
    else:
        raw = raw[-seq_len:]

    h1 = raw[:, :NUM_NODES_PER_HAND]
    h2 = raw[:, NUM_NODES_PER_HAND:]

    h1_active = [bool(np.any(h1[t] != 0)) for t in range(seq_len)]
    h2_active = [bool(np.any(h2[t] != 0)) for t in range(seq_len)]

    vel = np.zeros_like(raw)
    for t in range(1, seq_len):
        if h1_active[t] and h1_active[t - 1]:
            vel[t, :NUM_NODES_PER_HAND] = raw[t, :NUM_NODES_PER_HAND] - raw[t - 1, :NUM_NODES_PER_HAND]
        if h2_active[t] and h2_active[t - 1]:
            vel[t, NUM_NODES_PER_HAND:] = raw[t, NUM_NODES_PER_HAND:] - raw[t - 1, NUM_NODES_PER_HAND:]

    for t in range(seq_len):
        if not h1_active[t]:
            raw[t, :NUM_NODES_PER_HAND] = 0.0
            vel[t, :NUM_NODES_PER_HAND] = 0.0
        if not h2_active[t]:
            raw[t, NUM_NODES_PER_HAND:] = 0.0
            vel[t, NUM_NODES_PER_HAND:] = 0.0

    if node_dim == 6:
        rel = np.zeros_like(raw)
        for t in range(seq_len):
            if h1_active[t]:
                rel[t, :NUM_NODES_PER_HAND] = raw[t, :NUM_NODES_PER_HAND] - raw[t, 0:1]
            if h2_active[t]:
                rel[t, NUM_NODES_PER_HAND:] = raw[t, NUM_NODES_PER_HAND:] - raw[t, NUM_NODES_PER_HAND : NUM_NODES_PER_HAND + 1]
        features = np.concatenate([raw, rel, vel], axis=-1)  # (seq_len, 42, 6)
    else:
        features = np.concatenate([raw, vel], axis=-1)  # (seq_len, 42, 4)

    return features


# ---------------------------------------------------------------------------
# Neural Network Modules
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """
    Single Graph Convolution layer:
    H' = Norm(sigma(A_hat * H * W))
    """
    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", adj)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BT, nodes, in_features)
        h = self.linear(x)  # (BT, nodes, out_features)
        h = torch.bmm(self.adj.unsqueeze(0).expand(h.size(0), -1, -1), h)
        bt, j, d = h.shape
        h = self.bn(h.reshape(bt * j, d)).reshape(bt, j, d)
        return F.relu(h)


class GCNBlock(nn.Module):
    """
    Two stacked Graph Convolutions with residual connection.
    """
    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor):
        super().__init__()
        self.gcn1 = GraphConvolution(in_features, out_features, adj)
        self.gcn2 = GraphConvolution(out_features, out_features, adj)
        self.proj = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x)
        h = self.gcn1(x)
        h = self.gcn2(h)
        return h + res


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal Transformer sequence."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class GCNTransformerClassifier(nn.Module):
    """
    Complete GCN + Temporal Transformer Indian Sign Language Classifier.
    
    Spatial Graph Convolutions (42 joints) ->
    Joint Feature Pooling ->
    Optional 1D Temporal Convolutions (local motion dynamics) ->
    Temporal Transformer Encoder (24 frames) ->
    Global Temporal Pooling (Mean or Multi-Scale Mean+Max) ->
    Classification Head.
    """
    def __init__(
        self,
        num_classes: int,
        seq_len: int = 24,
        node_dim: int = 4,
        gcn_dim: int = 64,
        tf_heads: int = 4,
        tf_layers: int = 2,
        dropout: float = 0.20,
        use_temp_conv: bool = False,
        use_multiscale_pool: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.node_dim = node_dim
        self.gcn_dim = gcn_dim
        self.use_temp_conv = use_temp_conv
        self.use_multiscale_pool = use_multiscale_pool

        adj = build_two_hand_adjacency()
        self.gcn = GCNBlock(node_dim, gcn_dim, adj)
        
        # Spatial joint pool preserving hand joint identities
        self.joint_pool = nn.Sequential(
            nn.Linear(gcn_dim * NUM_NODES, gcn_dim),
            nn.LayerNorm(gcn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_temp_conv:
            self.temp_conv = nn.Sequential(
                nn.Conv1d(gcn_dim, gcn_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(gcn_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
        else:
            self.temp_conv = None

        self.pos_enc = PositionalEncoding(gcn_dim, max_len=seq_len + 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gcn_dim,
            nhead=tf_heads,
            dim_feedforward=gcn_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tf_layers,
            enable_nested_tensor=False,
        )

        head_in = gcn_dim * 2 if use_multiscale_pool else gcn_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, gcn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, seq_len=24, nodes=42, C)
        B, T, J, C = x.shape

        # Dynamic expansion from 4D to 6D if model was built with node_dim=6
        if C == 4 and self.node_dim == 6:
            coords = x[:, :, :, :2]
            vel = x[:, :, :, 2:4]
            rel1 = coords[:, :, :NUM_NODES_PER_HAND] - coords[:, :, 0:1]
            rel2 = coords[:, :, NUM_NODES_PER_HAND:] - coords[:, :, NUM_NODES_PER_HAND : NUM_NODES_PER_HAND + 1]
            h1_act = (coords[:, :, :NUM_NODES_PER_HAND] != 0).any(dim=-1, keepdim=True)
            h2_act = (coords[:, :, NUM_NODES_PER_HAND:] != 0).any(dim=-1, keepdim=True)
            rel1 = torch.where(h1_act, rel1, torch.zeros_like(rel1))
            rel2 = torch.where(h2_act, rel2, torch.zeros_like(rel2))
            rel = torch.cat([rel1, rel2], dim=2)
            x = torch.cat([coords, rel, vel], dim=-1)
            C = 6

        # Spatial GCN per frame
        h = x.reshape(B * T, J, C)
        h = self.gcn(h)                         # (BT, J, gcn_dim)
        h = h.reshape(B, T, J * self.gcn_dim)   # (B, T, J * gcn_dim)
        h = self.joint_pool(h)                  # (B, T, gcn_dim)

        if self.temp_conv is not None:
            h_conv = self.temp_conv(h.transpose(1, 2)).transpose(1, 2)
            h = h + h_conv

        # Temporal Transformer
        h = self.pos_enc(h)
        h = self.transformer(h)                 # (B, T, gcn_dim)

        # Temporal pooling & classification
        if self.use_multiscale_pool:
            h_mean = h.mean(dim=1)
            h_max = h.max(dim=1).values
            h_pool = torch.cat([h_mean, h_max], dim=-1)
        else:
            h_pool = h.mean(dim=1)

        logits = self.head(h_pool)              # (B, num_classes)
        return logits
