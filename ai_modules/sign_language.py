"""Real-time sign recognition — Indian Sign Language (ISL) GCN + Transformer (GCN-T) architecture.

Pipeline
--------
1. MediaPipe HandLandmarker extracts 42 hand landmarks (21 for Hand 1 + 21 for Hand 2, [x, y]) per frame.
2. A sliding window of frames is maintained per user session (target 60 frames).
3. The GCN-T model processes the window:
   - Spatial Graph Convolutions (42-joint skeleton graph) encode joint spatial relationships.
   - Temporal Transformer Encoder attends across temporal frames.
   - Classification head outputs probabilities over 29 Indian Sign Language classes.
4. Real-time dynamic resampling allows instant predictions as soon as >= 6 frames are captured,
   sliding smoothly to 60 full frames.

Model paths (searched in order)
--------------------------------
  sign_model_output/sign_gcn_transformer.pth  ← trained ISL 29-class GCN-T checkpoint
  models/sign_gcnt.pt                        ← workspace copy
  models/isl_gcn_transformer_best.pt         ← ISL 40-word checkpoint fallback
  models/sign_gru.pt                         ← legacy BiGRU checkpoint
  models/sign_classifier.joblib              ← legacy SVM
"""

from __future__ import annotations

import math
import threading
from collections import deque
from enum import Enum
import math
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe 42-joint hand topology (Hand 1: 0..20, Hand 2: 21..41)
# ---------------------------------------------------------------------------
_NUM_NODES = 42
_LANDMARKS_PER_HAND = 21

_HAND_EDGES_SINGLE: List[Tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # index
    (0, 9),  (9, 10), (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9),  (9, 13), (13, 17),           # palm
]


def _build_adjacency_42() -> "np.ndarray":
    """Construct normalized 42x42 block-diagonal adjacency matrix."""
    import torch
    A = torch.eye(_NUM_NODES)
    for offset in (0, 21):
        for i in range(offset, offset + 20):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
    degree = A.sum(dim=1).clamp(min=1)
    D = torch.diag(degree.pow(-0.5))
    norm_adj = D @ A @ D
    return norm_adj.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Model definitions (imported dynamically when torch is ready)
# ---------------------------------------------------------------------------

def _build_gcn_transformer_class():
    """Build the GCNTransformer nn.Module matching sign_gcn_transformer.pth."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def build_adj(num_nodes=42):
        A = torch.eye(num_nodes)
        for offset in (0, 21):
            for i in range(offset, offset + 20):
                A[i, i + 1] = 1.0
                A[i + 1, i] = 1.0
        degree = A.sum(dim=1).clamp(min=1)
        D = torch.diag(degree.pow(-0.5))
        return D @ A @ D

    class GraphConv(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
            # x: batch x frames x nodes x features
            x = torch.matmul(adjacency, x)
            return self.linear(x)

    class GCNTransformer(nn.Module):
        def __init__(
            self,
            num_classes: int,
            hidden_dim: int = 64,
            heads: int = 4,
            layers: int = 2,
            in_features: int = 2,
        ):
            super().__init__()
            self.register_buffer("adjacency", build_adj(num_nodes=_NUM_NODES))
            self.gcn1 = GraphConv(in_features, hidden_dim)
            self.gcn2 = GraphConv(hidden_dim, hidden_dim)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=128,
                dropout=0.2,
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: batch x frames x 42 nodes x 2 coordinates
            x = self.activation(self.gcn1(x, self.adjacency))
            x = self.dropout(self.activation(self.gcn2(x, self.adjacency)))
            # Pool landmarks into one representation per frame
            x = x.mean(dim=2)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    return GCNTransformer


def _build_gru_class():
    """Legacy BiGRU class for backward compatibility."""
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
# Automatic Signing Activity Detection & Debounced State Machine
# ---------------------------------------------------------------------------

class SigningState(str, Enum):
    IDLE = "IDLE"
    POSSIBLE_SIGNING = "POSSIBLE_SIGNING"
    SIGNING = "SIGNING"
    POSSIBLE_TURN_END = "POSSIBLE_TURN_END"


class SigningActivityDetector:
    """
    Automatic signing activity detector and turn-completion estimator.

    Uses a debounced 4-state machine:
      IDLE -> POSSIBLE_SIGNING -> SIGNING -> POSSIBLE_TURN_END -> IDLE

    Distinguishes intentional signing from non-signing movements:
      - Kinematic bounds: articulatory velocity must fall between idle threshold
        and erratic movement threshold (ignores resting hands and sudden camera jerks).
      - Debounced transitions: requires multiple consecutive evidence frames to enter SIGNING.
      - Pause tolerance: brief inter-sign pauses do NOT end the turn;
        inactivity timeout (default 1.6s) must expire before completing turn.
      - Duplicate filtering: prevents repeated emissions of the same sign
        within duplicate_suppress_window, while allowing legitimate repetitions after pause.
    """

    def __init__(
        self,
        min_active_frames: int = 3,
        inactivity_timeout: float = 1.6,
        velocity_idle_threshold: float = 0.0035,
        velocity_erratic_threshold: float = 0.22,
        duplicate_suppress_window: float = 1.0,
    ):
        self.min_active_frames = min_active_frames
        self.inactivity_timeout = inactivity_timeout
        self.velocity_idle_threshold = velocity_idle_threshold
        self.velocity_erratic_threshold = velocity_erratic_threshold
        self.duplicate_suppress_window = duplicate_suppress_window

        self.state: SigningState = SigningState.IDLE
        self.active_frame_count: int = 0
        self.pause_start_time: Optional[float] = None
        self.last_active_time: float = time.time()
        self.last_landmarks: Optional[np.ndarray] = None

        # Turn & phrase tracking
        self.accumulated_phrase: List[str] = []
        self.last_emitted_sign: str = ""
        self.last_emitted_time: float = 0.0

    def update(
        self,
        landmarks: Optional[np.ndarray],
        prediction: str = "",
        confidence: float = 0.0,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        if now is None:
            now = time.time()

        turn_completed = False
        finalized_phrase = ""
        accepted_sign = ""

        # 1. Kinematic evaluation
        has_hands = landmarks is not None and bool(np.any(landmarks != 0.0))
        velocity = 0.0

        if has_hands and self.last_landmarks is not None and bool(np.any(self.last_landmarks != 0.0)):
            active_mask = (landmarks[:, 0] != 0.0) & (self.last_landmarks[:, 0] != 0.0)
            if np.any(active_mask):
                diffs = landmarks[active_mask] - self.last_landmarks[active_mask]
                dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
                velocity = float(np.mean(dists))

        self.last_landmarks = landmarks.copy() if landmarks is not None else None

        # Kinematic articulatory motion check
        is_motion_active = (
            has_hands and
            (self.velocity_idle_threshold <= velocity <= self.velocity_erratic_threshold)
        )

        # Model confidence support
        has_confident_sign = bool(prediction and confidence >= 0.15 and prediction != "UNCERTAIN")

        # Frame evidence: articulatory motion or recognized sign
        frame_evidence = is_motion_active or (has_hands and has_confident_sign)

        # 2. Debounced state machine
        if self.state == SigningState.IDLE:
            if frame_evidence:
                self.active_frame_count += 1
                if self.active_frame_count >= 2 or has_confident_sign:
                    self.state = SigningState.POSSIBLE_SIGNING
                    self.last_active_time = now
            else:
                self.active_frame_count = max(0, self.active_frame_count - 1)

        elif self.state == SigningState.POSSIBLE_SIGNING:
            if frame_evidence:
                self.active_frame_count += 1
                self.last_active_time = now
                if self.active_frame_count >= self.min_active_frames or has_confident_sign:
                    self.state = SigningState.SIGNING
                    self.pause_start_time = None
            else:
                self.active_frame_count = max(0, self.active_frame_count - 1)
                if self.active_frame_count == 0:
                    self.state = SigningState.IDLE

        elif self.state == SigningState.SIGNING:
            if frame_evidence:
                self.last_active_time = now
                self.pause_start_time = None
            else:
                # Inter-sign pause or hand temporarily lowered
                self.state = SigningState.POSSIBLE_TURN_END
                self.pause_start_time = now

        elif self.state == SigningState.POSSIBLE_TURN_END:
            if frame_evidence:
                # Resumed signing within pause window: turn continues seamlessly!
                self.state = SigningState.SIGNING
                self.last_active_time = now
                self.pause_start_time = None
            else:
                pause_duration = now - (self.pause_start_time or now)
                if pause_duration >= self.inactivity_timeout:
                    # Inactivity timeout expired: turn is concluded
                    turn_completed = True
                    finalized_phrase = " ".join(self.accumulated_phrase).strip()
                    self.accumulated_phrase = []
                    self.last_emitted_sign = ""
                    self.state = SigningState.IDLE
                    self.active_frame_count = 0
                    self.pause_start_time = None

        # 3. Sign stabilization & phrase accumulation
        if self.state in (SigningState.SIGNING, SigningState.POSSIBLE_TURN_END) and has_confident_sign:
            is_duplicate = (
                prediction == self.last_emitted_sign and
                (now - self.last_emitted_time) < self.duplicate_suppress_window
            )
            if not is_duplicate:
                accepted_sign = prediction
                self.last_emitted_sign = prediction
                self.last_emitted_time = now
                self.accumulated_phrase.append(prediction)

        is_signing_active = self.state in (
            SigningState.POSSIBLE_SIGNING,
            SigningState.SIGNING,
            SigningState.POSSIBLE_TURN_END,
        )

        return {
            "state": self.state.value,
            "is_signing": is_signing_active,
            "turn_completed": turn_completed,
            "accepted_sign": accepted_sign,
            "phrase": " ".join(self.accumulated_phrase).strip(),
            "finalized_phrase": finalized_phrase,
            "velocity": round(velocity, 4),
        }

    def reset(self):
        self.state = SigningState.IDLE
        self.active_frame_count = 0
        self.pause_start_time = None
        self.last_active_time = time.time()
        self.last_landmarks = None
        self.accumulated_phrase = []
        self.last_emitted_sign = ""
        self.last_emitted_time = 0.0


# ---------------------------------------------------------------------------
# Main Recognizer Class
# ---------------------------------------------------------------------------

class SignLanguageRecognizer:
    """Real-time Indian Sign Language Recognizer backed by GCN-Transformer."""

    DEFAULT_PATHS = [
        Path("models/sign_gcnt_best.pt"),
        Path("sign_model_output/sign_gcn_transformer.pth"),
        Path("models/sign_gcnt.pt"),
        Path("models/isl_gcn_transformer_best.pt"),
        Path("models/sign_gru.pt"),
    ]
    SVM_MODEL_PATH = Path("models/sign_classifier.joblib")
    TASK_MODEL_PATHS = [
        Path("models/hand_landmarker.task"),
        Path(r"C:\Users\ACER\isl-40words\models\hand_landmarker.task"),
    ]

    SEQ_LEN = 60
    MIN_WARMUP_FRAMES = 6
    MIN_CONFIDENCE = 0.07  # Minimum top-1 confidence threshold for 29 classes
    SMOOTHING_WINDOW = 5

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[object] = None
        self.svm: Optional[object] = None
        self.landmarker: Optional[object] = None
        self.legacy_hands: Optional[object] = None
        self.error: Optional[str] = None
        self.arch: str = "none"
        self.is_2hand: bool = False
        self.labels: List[str] = []
        self.seq_len: int = self.SEQ_LEN
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.sequences: Dict[str, deque] = {}
        self.empty_streak: Dict[str, int] = {}
        self.recent_predictions: Dict[str, deque] = {}
        self.activity_detectors: Dict[str, SigningActivityDetector] = {}

        # Resolve model path
        self.model_path = None
        if model_path:
            p = Path(model_path)
            if p.exists():
                self.model_path = p
        if not self.model_path:
            for candidate in self.DEFAULT_PATHS:
                if candidate.exists():
                    self.model_path = candidate
                    break

        self._load_model()
        if self.model is None:
            self._load_svm()
        self._load_hand_tracker()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if not self.model_path or not self.model_path.exists():
            self.error = "No trained sign model checkpoint found."
            return

        try:
            import torch
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)

            # Check for enhanced 2-hand GCN-Transformer format (sign_gcnt_best.pt)
            if ckpt.get("arch") == "gcn_transformer_2hand":
                from ai_modules.sign_model_gcnt import GCNTransformerClassifier
                self.labels = [str(x) for x in ckpt["classes"]]
                self.seq_len = int(ckpt.get("seq_len", 24))
                gcn_dim = int(ckpt.get("gcn_dim", 64))
                heads = int(ckpt.get("tf_heads", 4))
                layers = int(ckpt.get("tf_layers", 2))
                dropout = float(ckpt.get("dropout", 0.25))
                node_dim = int(ckpt.get("node_dim", 4))
                use_temp_conv = bool(ckpt.get("use_temp_conv", False))
                use_multiscale_pool = bool(ckpt.get("use_multiscale_pool", False))

                net = GCNTransformerClassifier(
                    num_classes=len(self.labels),
                    seq_len=self.seq_len,
                    node_dim=node_dim,
                    gcn_dim=gcn_dim,
                    tf_heads=heads,
                    tf_layers=layers,
                    dropout=dropout,
                    use_temp_conv=use_temp_conv,
                    use_multiscale_pool=use_multiscale_pool,
                )
                net.load_state_dict(ckpt["state_dict"])
                net.eval()
                self.model = net
                self.arch = "gcn_transformer"
                self.is_2hand = True
                self.error = None
                return

            # Check for GCNTransformer checkpoint format
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                # Determine classes
                if "classes" in ckpt:
                    self.labels = [str(x) for x in ckpt["classes"]]
                elif "words" in ckpt:
                    self.labels = [str(x) for x in ckpt["words"]]
                elif "num_classes" in ckpt:
                    self.labels = [f"sign_{i}" for i in range(int(ckpt["num_classes"]))]

                # Sequence length & shapes
                if "input_shape" in ckpt and isinstance(ckpt["input_shape"], (list, tuple)):
                    self.seq_len = int(ckpt["input_shape"][0])
                elif "sequence_length" in ckpt:
                    self.seq_len = int(ckpt["sequence_length"])

                # Normalization parameters
                if "normalization_mean" in ckpt and "normalization_std" in ckpt:
                    self.mean = np.array(ckpt["normalization_mean"], dtype=np.float32).reshape(1, 1, -1)
                    self.std = np.array(ckpt["normalization_std"], dtype=np.float32).reshape(1, 1, -1)
                    self.std = np.maximum(self.std, 1e-5)

                hidden_dim = int(ckpt.get("hidden_dim", 64))
                heads = int(ckpt.get("attention_heads", 4))
                layers = int(ckpt.get("transformer_layers", 2))
                in_feat = 2

                # Detect in_features from gcn1.linear.weight
                if "gcn1.linear.weight" in state_dict:
                    in_feat = state_dict["gcn1.linear.weight"].shape[1]
                    hidden_dim = state_dict["gcn1.linear.weight"].shape[0]

                net = _make_gcnt(
                    num_classes=len(self.labels),
                    hidden_dim=hidden_dim,
                    heads=heads,
                    layers=layers,
                    in_features=in_feat,
                )
                net.load_state_dict(state_dict)
                net.eval()
                self.model = net
                self.arch = "gcn_transformer"
                self.error = None
                return

            # Legacy BiGRU format
            if "labels" in ckpt:
                self.labels = [str(x) for x in ckpt["labels"]]
                self.seq_len = int(ckpt.get("seq_len", self.SEQ_LEN))
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
                self.arch = "gru"
                self.error = None
        except Exception as exc:
            self.model = None
            self.error = f"Could not load sign model ({self.model_path.name}): {exc}"

    def _load_svm(self):
        if not self.SVM_MODEL_PATH.exists():
            return
        try:
            import joblib
            self.svm = joblib.load(self.SVM_MODEL_PATH)
            self.arch = "svm"
        except Exception:
            self.svm = None

    def _load_hand_tracker(self):
        """Initialize MediaPipe HandLandmarker with task model."""
        task_file = None
        for candidate in self.TASK_MODEL_PATHS:
            if candidate.exists():
                task_file = candidate
                break

        if task_file:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=str(task_file.resolve()))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                )
                self.landmarker = vision.HandLandmarker.create_from_options(options)
                return
            except Exception as exc:
                self.landmarker = None

        # Fallback to legacy mediapipe solutions if available
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.legacy_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
        except Exception:
            self.legacy_hands = None

        if self.landmarker is None and self.legacy_hands is None:
            self.error = self.error or "MediaPipe hand landmarker is unavailable."

    # ------------------------------------------------------------------
    # Landmark Extraction
    # ------------------------------------------------------------------

    def _extract_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract (42, 2) float32 landmark array from BGR image:
        Hand 1 at nodes 0..20, Hand 2 at nodes 21..41.
        Returns None if no hand detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. New MediaPipe Tasks API
        if self.landmarker is not None:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with self.lock:
                result = self.landmarker.detect(mp_image)

            if not result or not result.hand_landmarks:
                return None

            joints = np.zeros((_NUM_NODES, 2), dtype=np.float32)
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[:2]):
                offset = hand_idx * _LANDMARKS_PER_HAND
                for lm_idx, lm in enumerate(hand_landmarks[:_LANDMARKS_PER_HAND]):
                    joints[offset + lm_idx, 0] = float(lm.x)
                    joints[offset + lm_idx, 1] = float(lm.y)
            return joints

        # 2. Legacy MediaPipe solutions API
        if self.legacy_hands is not None:
            with self.lock:
                result = self.legacy_hands.process(rgb)

            if not result or not result.multi_hand_landmarks:
                return None

            joints = np.zeros((_NUM_NODES, 2), dtype=np.float32)
            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks[:2]):
                offset = hand_idx * _LANDMARKS_PER_HAND
                for lm_idx, lm in enumerate(hand_landmarks.landmark[:_LANDMARKS_PER_HAND]):
                    joints[offset + lm_idx, 0] = float(lm.x)
                    joints[offset + lm_idx, 1] = float(lm.y)
            return joints

        return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_gcnt(self, sequence: List[np.ndarray]) -> Tuple[str, float]:
        """
        sequence: list of (42, 2) arrays.
        Resamples / normalizes and runs model.
        """
        import torch

        # Enhanced 2-hand path with velocity features [x, y, dx, dy]
        if getattr(self, "is_2hand", False):
            from ai_modules.sign_model_gcnt import preprocess_live_frame_buffer
            node_dim = getattr(self.model, "node_dim", 4)
            arr = preprocess_live_frame_buffer(sequence, seq_len=self.seq_len, node_dim=node_dim)
            x = torch.from_numpy(arr).unsqueeze(0).float()
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            label = self.labels[idx] if idx < len(self.labels) else f"sign_{idx}"
            return label, float(probs[idx])

        # Legacy 2-feature uniform temporal resampling to target seq_len (60)
        num_frames = len(sequence)
        if num_frames == self.seq_len:
            arr = np.stack(sequence, axis=0).astype(np.float32)  # (60, 42, 2)
        else:
            indices = np.linspace(0, num_frames - 1, self.seq_len).round().astype(int)
            arr = np.array(sequence, dtype=np.float32)[indices]

        # Apply dataset normalization
        if self.mean is not None and self.std is not None:
            arr = (arr - self.mean) / self.std
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(arr).unsqueeze(0).float()  # (1, 60, 42, 2)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else f"sign_{idx}"
        return label, float(probs[idx])

    def _predict_gru(self, sequence: List[np.ndarray]) -> Tuple[str, float]:
        import torch
        flat_seq = [j.reshape(-1) for j in sequence[-self.seq_len:]]
        arr = np.stack(flat_seq, axis=0).astype(np.float32)
        x = torch.from_numpy(arr).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def _predict_svm(self, joints: np.ndarray) -> Tuple[str, float]:
        feat = joints.reshape(1, -1)
        pred = self.svm.predict(feat)[0]
        conf = float(np.max(self.svm.predict_proba(feat)[0])) if hasattr(self.svm, "predict_proba") else 0.5
        return str(pred), conf

    # ------------------------------------------------------------------
    # Temporal Smoothing Helper
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

    def _get_activity_detector(self, user_id: str) -> SigningActivityDetector:
        with self.lock:
            if user_id not in self.activity_detectors:
                self.activity_detectors[user_id] = SigningActivityDetector()
            return self.activity_detectors[user_id]

    def reset_user(self, user_id: str):
        key = str(user_id)
        with self.lock:
            self.sequences.pop(key, None)
            self.empty_streak.pop(key, None)
            self.recent_predictions.pop(key, None)
            self.activity_detectors.pop(key, None)

    def recognize_frame(self, frame_data: bytes, user_id: str = "anonymous") -> dict:
        """
        Decode one image frame, extract 42 hand landmarks, run ISL GCN-T inference.
        Also drives automatic signing-activity detection and turn estimation.

        Returns
        -------
        dict with keys:
          - text: stabilized recognized sign (accepted token)
          - confidence: confidence percentage (float)
          - message: human-readable status
          - ready: boolean
          - arch: model architecture string
          - signing_state: IDLE | POSSIBLE_SIGNING | SIGNING | POSSIBLE_TURN_END
          - is_signing: boolean
          - turn_completed: boolean
          - phrase: current accumulated phrase during this turn
          - finalized_phrase: final phrase string when turn completes
        """
        arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if frame is None:
            return _resp("", 0.0, "Invalid image frame.", ready=False)

        key = str(user_id)
        detector = self._get_activity_detector(key)

        if self.model is None and self.svm is None:
            activity_info = detector.update(None, prediction="", confidence=0.0)
            return _resp(
                "", 0.0, self.error or "Train or download a sign model first.",
                ready=False,
                signing_state=activity_info["state"],
                is_signing=activity_info["is_signing"],
                turn_completed=activity_info["turn_completed"],
                phrase=activity_info["phrase"],
                finalized_phrase=activity_info["finalized_phrase"],
            )

        if self.landmarker is None and self.legacy_hands is None:
            activity_info = detector.update(None, prediction="", confidence=0.0)
            return _resp(
                "", 0.0, self.error or "MediaPipe landmarker unavailable.",
                ready=False,
                signing_state=activity_info["state"],
                is_signing=activity_info["is_signing"],
                turn_completed=activity_info["turn_completed"],
                phrase=activity_info["phrase"],
                finalized_phrase=activity_info["finalized_phrase"],
            )

        joints = self._extract_landmarks(frame)

        # Track consecutive empty frames before resetting buffer
        if joints is None:
            activity_info = detector.update(None, prediction="", confidence=0.0)
            with self.lock:
                streak = self.empty_streak.get(key, 0) + 1
                self.empty_streak[key] = streak
                if streak >= 6:  # Allow brief gaps without flushing queue
                    self.sequences.pop(key, None)

            msg = "No hand detected. Position your hands in front of camera."
            if activity_info["turn_completed"] and activity_info["finalized_phrase"]:
                msg = f"Signing turn completed: {activity_info['finalized_phrase']}"

            return _resp(
                "", 0.0, msg,
                ready=False,
                signing_state=activity_info["state"],
                is_signing=activity_info["is_signing"],
                turn_completed=activity_info["turn_completed"],
                phrase=activity_info["phrase"],
                finalized_phrase=activity_info["finalized_phrase"],
            )

        # Hand detected: reset empty streak and store frame landmarks
        with self.lock:
            self.empty_streak[key] = 0
            q = self.sequences.setdefault(key, deque(maxlen=self.seq_len))
            q.append(joints)
            seq = list(q)

        # GCN-Transformer Prediction Path
        if self.arch == "gcn_transformer":
            if len(seq) < self.MIN_WARMUP_FRAMES:
                activity_info = detector.update(joints, prediction="", confidence=0.0)
                return _resp(
                    "", 0.0,
                    f"Capturing sign movement ({len(seq)}/{self.MIN_WARMUP_FRAMES})…",
                    ready=False,
                    arch="GCN-Transformer",
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                )

            try:
                raw_label, conf = self._predict_gcnt(seq)
                smoothed_label, smoothed_conf = self._smooth(key, raw_label, conf)
                display_text = smoothed_label.replace("_", " ").title()

                if smoothed_conf < self.MIN_CONFIDENCE:
                    activity_info = detector.update(joints, prediction="UNCERTAIN", confidence=smoothed_conf)
                    return _resp(
                        "UNCERTAIN",
                        round(smoothed_conf * 100, 1),
                        "Low confidence — continue holding or repeating the sign.",
                        ready=True,
                        arch="GCN-Transformer",
                        signing_state=activity_info["state"],
                        is_signing=activity_info["is_signing"],
                        turn_completed=activity_info["turn_completed"],
                        phrase=activity_info["phrase"],
                        finalized_phrase=activity_info["finalized_phrase"],
                    )

                activity_info = detector.update(joints, prediction=display_text, confidence=smoothed_conf)
                accepted = activity_info["accepted_sign"] or display_text

                return _resp(
                    accepted,
                    round(smoothed_conf * 100, 1),
                    f"Recognized: {display_text}",
                    ready=True,
                    arch="GCN-Transformer",
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                )
            except Exception as exc:
                activity_info = detector.update(joints, prediction="", confidence=0.0)
                return _resp(
                    "", 0.0, f"GCN-T inference error: {exc}",
                    ready=False,
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                )

        # Legacy BiGRU path
        if self.arch == "gru":
            if len(seq) < self.MIN_WARMUP_FRAMES:
                activity_info = detector.update(joints, prediction="", confidence=0.0)
                return _resp(
                    "", 0.0, f"Buffering ({len(seq)}/{self.MIN_WARMUP_FRAMES})…",
                    ready=False,
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                )
            try:
                raw_label, conf = self._predict_gru(seq)
                smoothed_label, smoothed_conf = self._smooth(key, raw_label, conf)
                display_text = smoothed_label.replace("_", " ").title()
                activity_info = detector.update(joints, prediction=display_text, confidence=smoothed_conf)
                accepted = activity_info["accepted_sign"] or display_text
                return _resp(
                    accepted, round(smoothed_conf * 100, 1),
                    f"Recognized: {display_text}", ready=True, arch="BiGRU",
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                )
            except Exception as exc:
                activity_info = detector.update(joints, prediction="", confidence=0.0)
                return _resp(
                    "", 0.0, f"GRU inference error: {exc}", ready=False,
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                )

        # SVM fallback
        if self.svm is not None:
            try:
                label, conf = self._predict_svm(joints)
                display_text = label.replace("_", " ").title()
                activity_info = detector.update(joints, prediction=display_text, confidence=conf)
                accepted = activity_info["accepted_sign"] or display_text
                return _resp(
                    accepted, round(conf * 100, 1),
                    f"Recognized: {display_text}", ready=True, arch="SVM",
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                )
            except Exception as exc:
                activity_info = detector.update(joints, prediction="", confidence=0.0)
                return _resp(
                    "", 0.0, f"SVM inference error: {exc}", ready=False,
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                )

        activity_info = detector.update(None, prediction="", confidence=0.0)
        return _resp(
            "", 0.0, self.error or "No model available.", ready=False,
            signing_state=activity_info["state"],
            is_signing=activity_info["is_signing"],
            turn_completed=activity_info["turn_completed"],
            phrase=activity_info["phrase"],
            finalized_phrase=activity_info["finalized_phrase"],
        )


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
    signing_state: str = "IDLE",
    is_signing: bool = False,
    turn_completed: bool = False,
    phrase: str = "",
    finalized_phrase: str = "",
) -> dict:
    out: dict = {
        "text": text,
        "confidence": confidence,
        "message": message,
        "signing_state": signing_state,
        "is_signing": is_signing,
        "turn_completed": turn_completed,
        "phrase": phrase,
        "finalized_phrase": finalized_phrase,
    }
    if ready is not None:
        out["ready"] = ready
    if arch:
        out["arch"] = arch
    return out
