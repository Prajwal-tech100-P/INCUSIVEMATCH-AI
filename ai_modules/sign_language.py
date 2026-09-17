"""Real-time sign recognition — GCN + Transformer (GCN-T) architecture.

Pipeline
--------
1. MediaPipe extracts 21 hand landmarks (x, y) per frame.
2. A short sliding window of frames is kept per user session.
3. The GCN-T model processes the window:
   - Spatial GCN block encodes each frame as a hand-skeleton graph.
   - Temporal Transformer encoder attends across all frames.
   - A classification head outputs sign probabilities.

Backward compatibility
----------------------
* If a GRU checkpoint (arch != 'gcn_transformer') is found, the old BiGRU
  runtime is used automatically.
* An SVM joblib model is still supported as a last-resort fallback.

Model paths (searched in order)
--------------------------------
  models/sign_gcnt.pt       ← new GCN-T checkpoint
  models/sign_gru.pt        ← legacy BiGRU checkpoint (auto-detected)
  models/sign_classifier.joblib ← legacy SVM

Training
--------
  python ai_modules/train_sign_model.py data/sign_sequences.csv
"""

from __future__ import annotations

import math
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hand skeleton adjacency (MediaPipe 21-joint layout)
# ---------------------------------------------------------------------------
_HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),   # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),   # index
    (0, 9),  (9, 10), (10, 11),(11, 12), # middle
    (0, 13), (13, 14),(14, 15),(15, 16), # ring
    (0, 17), (17, 18),(18, 19),(19, 20), # pinky
    (5, 9),  (9, 13), (13, 17),          # palm
]
_NUM_JOINTS = 21


def _build_adjacency() -> "np.ndarray":
    """Symmetric, self-looped, degree-normalised adjacency matrix (21×21)."""
    A = np.zeros((_NUM_JOINTS, _NUM_JOINTS), dtype=np.float32)
    for i, j in _HAND_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)          # self-loops
    D = A.sum(axis=1, keepdims=True)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    return D_inv_sqrt * A * D_inv_sqrt.T   # D^{-1/2} A D^{-1/2}


_ADJ: np.ndarray = _build_adjacency()


# ---------------------------------------------------------------------------
# Lazy model builders (import torch only when needed)
# ---------------------------------------------------------------------------

def _build_gcn_transformer_class():
    """Build the GCNTransformerNetwork nn.Module class on first use."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GraphConv(nn.Module):
        """Single GCN layer: H' = σ(A_hat · H · W)."""
        def __init__(self, in_dim: int, out_dim: int, adj: np.ndarray):
            super().__init__()
            self.register_buffer("adj", torch.from_numpy(adj))
            self.linear = nn.Linear(in_dim, out_dim, bias=False)
            self.bn = nn.BatchNorm1d(out_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B*T, J, in_dim)
            h = self.linear(x)                      # (B*T, J, out_dim)
            h = torch.bmm(self.adj.unsqueeze(0).expand(h.size(0), -1, -1), h)
            bt, j, d = h.shape
            h = self.bn(h.reshape(bt * j, d)).reshape(bt, j, d)
            return F.relu(h)

    class GCNBlock(nn.Module):
        """Two stacked GCN layers with residual projection."""
        def __init__(self, in_dim: int, out_dim: int, adj: np.ndarray):
            super().__init__()
            self.gcn1 = GraphConv(in_dim, out_dim, adj)
            self.gcn2 = GraphConv(out_dim, out_dim, adj)
            self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        def forward(self, x):
            res = self.proj(x)
            return self.gcn2(self.gcn1(x)) + res

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class GCNTransformerNetwork(nn.Module):
        """
        Spatial GCN (per-frame skeleton graph) +
        Temporal Transformer (across frames) classifier.

        Input : (B, T, J, C)  where J=21, C=node_dim (default 4: x,y,dx,dy)
        Output: (B, num_classes)
        """
        def __init__(
            self,
            num_classes: int,
            seq_len: int = 24,
            node_dim: int = 4,
            gcn_dim: int = 64,
            tf_heads: int = 4,
            tf_layers: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()
            adj = _build_adjacency()
            self.gcn = GCNBlock(node_dim, gcn_dim, adj)
            # After GCN we pool joints → (B, T, gcn_dim)
            self.joint_pool = nn.Linear(gcn_dim * _NUM_JOINTS, gcn_dim)
            self.pos_enc = PositionalEncoding(gcn_dim, max_len=512)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=gcn_dim, nhead=tf_heads, dim_feedforward=gcn_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=tf_layers, enable_nested_tensor=False,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(gcn_dim),
                nn.Dropout(dropout),
                nn.Linear(gcn_dim, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, J, C)
            B, T, J, C = x.shape
            # --- Spatial GCN (applied to every frame independently) ---
            h = x.reshape(B * T, J, C)           # (B*T, J, C)
            h = self.gcn(h)                       # (B*T, J, gcn_dim)
            h = h.reshape(B, T, J * h.shape[-1]) # (B, T, J*gcn_dim)
            h = self.joint_pool(h)                # (B, T, gcn_dim)
            # --- Temporal Transformer ---
            h = self.pos_enc(h)
            h = self.transformer(h)               # (B, T, gcn_dim)
            h = h.mean(dim=1)                     # global avg pool → (B, gcn_dim)
            return self.head(h)

    return GCNTransformerNetwork


def _build_gru_class():
    """Legacy BiGRU class — kept for backward compatibility."""
    import torch.nn as nn

    class GRUNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes,
                     dropout=0.25, bidirectional=True):
            super().__init__()
            self.gru = nn.GRU(
                input_size, hidden_size, num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            out_size = hidden_size * (2 if bidirectional else 1)
            self.norm = nn.LayerNorm(out_size)
            self.head = nn.Sequential(
                nn.Linear(out_size, 96), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(96, num_classes),
            )

        def forward(self, x):
            y, _ = self.gru(x)
            y = self.norm(y[:, -1, :])
            return self.head(y)

    return GRUNetwork


# Lazy class caches
_GCNT_CLASS = None
_GRU_CLASS = None


def _make_gcnt(**kw):
    global _GCNT_CLASS
    if _GCNT_CLASS is None:
        _GCNT_CLASS = _build_gcn_transformer_class()
    return _GCNT_CLASS(**kw)


def _make_gru(**kw):
    global _GRU_CLASS
    if _GRU_CLASS is None:
        _GRU_CLASS = _build_gru_class()
    return _GRU_CLASS(**kw)


# ---------------------------------------------------------------------------
# Main recogniser class
# ---------------------------------------------------------------------------

class SignLanguageRecognizer:
    """Real-time sign recogniser backed by GCN-T (or legacy BiGRU/SVM)."""

    GCNT_MODEL_PATH = Path("models/sign_gcnt.pt")
    GRU_MODEL_PATH  = Path("models/sign_gru.pt")
    SVM_MODEL_PATH  = Path("models/sign_classifier.joblib")

    SEQ_LEN         = 24
    MIN_CONFIDENCE  = 0.72
    SMOOTHING_WINDOW = 5

    # GCN-T hyper-params (must match training)
    NODE_DIM  = 4   # x, y, dx, dy per joint
    GCN_DIM   = 64
    TF_HEADS  = 4
    TF_LAYERS = 2
    DROPOUT   = 0.30

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[object]  = None   # GCN-T or GRU nn.Module
        self.svm:   Optional[object]  = None
        self.hands: Optional[object]  = None
        self.mp_hands: Optional[object] = None
        self.error: Optional[str]     = None
        self.arch: str                = "none"
        self.labels: List[str]        = []
        self.seq_len: int             = self.SEQ_LEN
        self.lock                     = threading.Lock()
        self.sequences: Dict[str, deque] = {}
        self.recent_predictions: Dict[str, deque] = {}

        # Resolve model path: explicit > gcnt default > gru legacy
        if model_path:
            candidate = Path(model_path)
        elif self.GCNT_MODEL_PATH.exists():
            candidate = self.GCNT_MODEL_PATH
        else:
            candidate = self.GRU_MODEL_PATH
        self.model_path = candidate

        self._load_model()
        if self.model is None:
            self._load_svm()
        self._load_hand_tracker()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if not self.model_path.exists():
            self.error = (
                f"No trained model found at {self.model_path}. "
                "Run: python ai_modules/train_sign_model.py data/sign_sequences.csv"
            )
            return
        try:
            import torch
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.labels  = [str(x) for x in ckpt["labels"]]
            self.seq_len = int(ckpt.get("seq_len", self.SEQ_LEN))
            arch = ckpt.get("arch", "gru")

            if arch == "gcn_transformer":
                self._load_gcnt(ckpt)
            else:
                self._load_gru_ckpt(ckpt)
        except Exception as exc:
            self.model = None
            self.error = f"Could not load model ({self.model_path.name}): {exc}"

    def _load_gcnt(self, ckpt: dict):
        net = _make_gcnt(
            num_classes=len(self.labels),
            seq_len=self.seq_len,
            node_dim=int(ckpt.get("node_dim", self.NODE_DIM)),
            gcn_dim=int(ckpt.get("gcn_dim", self.GCN_DIM)),
            tf_heads=int(ckpt.get("tf_heads", self.TF_HEADS)),
            tf_layers=int(ckpt.get("tf_layers", self.TF_LAYERS)),
            dropout=float(ckpt.get("dropout", self.DROPOUT)),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self.model = net
        self.arch  = "gcn_transformer"

    def _load_gru_ckpt(self, ckpt: dict):
        net = _make_gru(
            input_size=int(ckpt.get("input_size", 84)),
            hidden_size=int(ckpt.get("hidden_size", 128)),
            num_layers=int(ckpt.get("num_layers", 2)),
            num_classes=len(self.labels),
            dropout=float(ckpt.get("dropout", 0.25)),
            bidirectional=True,
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self.model = net
        self.arch  = "gru"

    def _load_svm(self):
        if not self.SVM_MODEL_PATH.exists():
            return
        try:
            import joblib
            self.svm  = joblib.load(self.SVM_MODEL_PATH)
            self.arch = "svm"
        except Exception:
            self.svm = None

    def _load_hand_tracker(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.60,
                min_tracking_confidence=0.60,
            )
        except Exception as exc:
            self.error = self.error or f"MediaPipe is unavailable: {exc}"

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_joints(hand_landmarks) -> Optional[np.ndarray]:
        """Return (21, 2) float32 array — wrist-centred, scale-normalised."""
        pts = np.array(
            [(p.x, p.y) for p in hand_landmarks.landmark], dtype=np.float32
        )
        pts -= pts[0]                               # wrist at origin
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale <= 1e-6:
            return None
        return (pts / scale).astype(np.float32)    # (21, 2)

    @staticmethod
    def _flat_features(joints: np.ndarray) -> np.ndarray:
        """Flatten (21,2) → (42,) for legacy GRU / SVM."""
        return joints.reshape(-1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_gcnt(self, sequence: List[np.ndarray]) -> Tuple[str, float]:
        """
        sequence: list of (21, 2) arrays, length == seq_len.
        Returns (label, confidence).
        """
        import torch
        arr = np.stack(sequence, axis=0).astype(np.float32)   # (T, 21, 2)
        # Compute velocities per joint
        vel = np.zeros_like(arr)                               # (T, 21, 2)
        vel[1:] = arr[1:] - arr[:-1]
        node_feat = np.concatenate([arr, vel], axis=-1)        # (T, 21, 4)
        x = torch.from_numpy(node_feat).unsqueeze(0)           # (1, T, 21, 4)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)[0].numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def _predict_gru(self, sequence: List[np.ndarray]) -> Tuple[str, float]:
        """sequence: list of (42,) flat arrays."""
        import torch
        arr = np.stack(sequence, axis=0).astype(np.float32)   # (T, 42)
        vel = np.vstack([np.zeros((1, 42), dtype=np.float32), np.diff(arr, axis=0)])
        feat = np.concatenate([arr, vel], axis=1)              # (T, 84)
        x = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)[0].numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def _predict_svm(self, feature: np.ndarray) -> Tuple[str, float]:
        pred = self.svm.predict(feature.reshape(1, -1))[0]
        conf = (
            float(np.max(self.svm.predict_proba(feature.reshape(1, -1))[0]))
            if hasattr(self.svm, "predict_proba") else 0.0
        )
        return str(pred), conf

    # ------------------------------------------------------------------
    # Smoothing helper
    # ------------------------------------------------------------------

    def _smooth(self, key: str, label: str, conf: float) -> Tuple[str, float]:
        with self.lock:
            buf = self.recent_predictions.setdefault(key, deque(maxlen=self.SMOOTHING_WINDOW))
            buf.append((label, conf))
            votes: Dict[str, float] = {}
            for lab, c in buf:
                votes[lab] = votes.get(lab, 0.0) + c
            best = max(votes, key=votes.get)
            best_conf = max(c for l, c in buf if l == best)
        return best, best_conf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_user(self, user_id: str):
        key = str(user_id)
        with self.lock:
            self.sequences.pop(key, None)
            self.recent_predictions.pop(key, None)

    def recognize_frame(self, frame_data: bytes, user_id: str = "anonymous") -> dict:
        """
        Decode one JPEG/PNG frame, run hand detection + sign classification.

        Returns
        -------
        dict with keys: text, confidence, message, ready (optional), arch
        """
        # --- Decode frame ---
        arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if frame is None:
            return _resp("", 0.0, "Invalid image frame.")

        # --- Guard: model available? ---
        if self.model is None and self.svm is None:
            return _resp("", 0.0, self.error or "Train a sign model first.")
        if self.hands is None:
            return _resp("", 0.0, self.error or "MediaPipe is unavailable.")

        # --- Hand detection ---
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        key    = str(user_id)

        if not result.multi_hand_landmarks:
            with self.lock:
                self.sequences.pop(key, None)
            return _resp("", 0.0, "No hand detected.")

        joints = self._extract_joints(result.multi_hand_landmarks[0])
        if joints is None:
            return _resp("", 0.0, "Could not normalise hand landmarks.")

        # --- GCN-T path ---
        if self.arch == "gcn_transformer":
            with self.lock:
                q = self.sequences.setdefault(key, deque(maxlen=self.seq_len))
                q.append(joints)      # each element: (21, 2)
                seq = list(q)

            if len(seq) < self.seq_len:
                return _resp(
                    "", 0.0,
                    f"Buffering frames ({len(seq)}/{self.seq_len})…",
                    ready=False,
                )
            try:
                label, conf = self._predict_gcnt(seq)
                label, conf = self._smooth(key, label, conf)
                if conf < self.MIN_CONFIDENCE:
                    return _resp(
                        "UNCERTAIN", round(conf * 100, 2),
                        "Low confidence — please repeat the sign.",
                        ready=True, arch="GCN-Transformer",
                    )
                return _resp(
                    label, round(conf * 100, 2),
                    "GCN-Transformer prediction.",
                    ready=True, arch="GCN-Transformer",
                )
            except Exception as exc:
                return _resp("", 0.0, f"GCN-T inference error: {exc}")

        # --- Legacy GRU path ---
        if self.arch == "gru":
            flat = self._flat_features(joints)
            with self.lock:
                q = self.sequences.setdefault(key, deque(maxlen=self.seq_len))
                q.append(flat)
                seq = list(q)
            if len(seq) < self.seq_len:
                return _resp(
                    "", 0.0,
                    f"Collecting sign frames ({len(seq)}/{self.seq_len})…",
                    ready=False,
                )
            try:
                label, conf = self._predict_gru(seq)
                label, conf = self._smooth(key, label, conf)
                if conf < self.MIN_CONFIDENCE:
                    return _resp(
                        "UNCERTAIN", round(conf * 100, 2),
                        "Low confidence — please repeat the sign.",
                        ready=True, arch="BiGRU",
                    )
                return _resp(label, round(conf * 100, 2), "BiGRU sequence prediction.", ready=True, arch="BiGRU")
            except Exception as exc:
                return _resp("", 0.0, f"GRU inference error: {exc}")

        # --- SVM fallback ---
        if self.svm is not None:
            try:
                flat  = self._flat_features(joints)
                label, conf = self._predict_svm(flat)
                if conf < 0.65:
                    label = "UNCERTAIN"
                return _resp(label, round(conf * 100, 2), "SVM compatibility prediction.", arch="SVM")
            except Exception as exc:
                return _resp("", 0.0, f"SVM prediction failed: {exc}")

        return _resp("", 0.0, self.error or "No model available.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resp(
    text: str,
    confidence: float,
    message: str,
    *,
    ready: Optional[bool] = None,
    arch: str = "",
) -> dict:
    out: dict = {"text": text, "confidence": confidence, "message": message}
    if ready is not None:
        out["ready"] = ready
    if arch:
        out["arch"] = arch
    return out
