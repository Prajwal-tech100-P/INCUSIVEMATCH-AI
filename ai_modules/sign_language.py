"""Real-time sign recognition — Indian Sign Language (ISL) HWGAT Architecture.

HWGAT: Hierarchical Weighted Graph Attention Network
Pretrained on 2,002 Indian Sign Language classes (FDMSE Dataset, RKMVERI Coimbatore).

Pipeline
--------
1. MediaPipe HandLandmarker extracts joint landmarks ([x, y]) per frame.
2. A sliding window buffer of frames is maintained per user session (target 192 frames).
3. The HWGAT model processes the window:
   - Spatial Graph Convolutions across 4 skeletal windows (64 keypoints) encode joint topologies.
   - Hierarchical Graph Attention attends across temporal and spatial windows.
   - Classification head outputs probabilities over 2,002 Indian Sign Language vocabulary classes.
4. Real-time dynamic temporal resampling allows instant predictions as soon as >= 6 frames are captured,
   interpolating smoothly to 192 frames.
5. Kinematic hand-velocity motion detection debounces intentional signs from resting/idle hands.

Model paths
--------------------------------
  models/hwgat_best.pt                        ← optimized 84 MB weights checkpoint
  models/class_map_hwgat.csv                  ← 2,002-class ISL vocabulary mapping
"""

from __future__ import annotations

import csv
import gc
import logging
import math
import os
import threading
from collections import deque
from enum import Enum
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch

logger = logging.getLogger("ai_modules.sign_language")


# ---------------------------------------------------------------------------
# MediaPipe Hand Joint Constants (Hand 1: 0..20, Hand 2: 21..41)
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


# ---------------------------------------------------------------------------
# Response formatting helper
# ---------------------------------------------------------------------------

def _resp(
    text: str = "",
    conf: float = 0.0,
    msg: str = "",
    ready: bool = True,
    arch: str = "HWGAT",
    signing_state: str = "IDLE",
    is_signing: bool = False,
    turn_completed: bool = False,
    phrase: str = "",
    finalized_phrase: str = "",
    new_token: str = "",
    vocabulary_size: int = 2002,
    confidence: Optional[float] = None,
    message: Optional[str] = None,
    **kwargs,
) -> dict:
    if confidence is not None:
        conf = confidence
    if message is not None:
        msg = message
    return {
        "text": text,
        "new_token": new_token or text,
        "confidence": conf,
        "message": msg,
        "ready": ready,
        "arch": arch,
        "signing_state": signing_state,
        "is_signing": is_signing,
        "turn_completed": turn_completed,
        "phrase": phrase,
        "finalized_phrase": finalized_phrase,
        "vocabulary_size": vocabulary_size,
    }


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
        idle_velocity_threshold: Optional[float] = None,
        erratic_velocity_threshold: Optional[float] = None,
        min_signing_evidence_frames: Optional[int] = None,
        turn_end_silence_seconds: Optional[float] = None,
        **kwargs,
    ):
        if idle_velocity_threshold is not None:
            velocity_idle_threshold = idle_velocity_threshold
        if erratic_velocity_threshold is not None:
            velocity_erratic_threshold = erratic_velocity_threshold
        if min_signing_evidence_frames is not None:
            min_active_frames = min_signing_evidence_frames
        if turn_end_silence_seconds is not None:
            inactivity_timeout = turn_end_silence_seconds

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

    def compute_articulatory_velocity(self, landmarks: Optional[np.ndarray]) -> float:
        """Compute mean Euclidean displacement of active hand joints between frames."""
        if landmarks is None or self.last_landmarks is None:
            return 0.0
        active_mask = (landmarks[:, 0] != 0.0) & (self.last_landmarks[:, 0] != 0.0)
        if np.any(active_mask):
            diffs = landmarks[active_mask] - self.last_landmarks[active_mask]
            dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
            return float(np.mean(dists))
        return 0.0

    def update(
        self,
        landmarks: Optional[np.ndarray],
        prediction: str = "",
        confidence: float = 0.0,
        now: Optional[float] = None,
        **kwargs,
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
        """Reset internal detector state and accumulated history."""
        self.state = SigningState.IDLE
        self.active_frame_count = 0
        self.pause_start_time = None
        self.last_landmarks = None
        self.last_valid_landmarks = None
        self.evidence_counter = 0
        self.silence_counter = 0
        self.accumulated_phrase = []
        self.last_emitted_sign = ""
        self.last_emitted_time = 0.0


# ---------------------------------------------------------------------------
# Main HWGAT Recognizer Class
# ---------------------------------------------------------------------------

class SignLanguageRecognizer:
    """Exclusive Indian Sign Language Recognizer powered by HWGAT (2,002 Classes)."""

    DEFAULT_PATHS = [
        Path("models/hwgat_best.pt"),
    ]
    CLASS_MAP_PATHS = [
        Path("models/class_map_hwgat.csv"),
    ]
    TASK_MODEL_PATHS = [
        Path("models/hand_landmarker.task"),
    ]

    SEQ_LEN = 192
    MIN_WARMUP_FRAMES = 6
    MIN_CONFIDENCE = 0.05  # Minimum top-1 confidence threshold for 2,002 classes
    SMOOTHING_WINDOW = 5

    # HWGAT Hand Landmark Indices (10 selected from 21 MediaPipe hand points)
    HWGAT_HAND_INDICES = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

    def __init__(self, model_path: Optional[str] = None):
        self.model: Any = None
        self.landmarker: Any = None
        self.legacy_hands: Any = None
        self.error: Optional[str] = None
        self.arch: str = "none"
        self.is_2hand: bool = True
        self.labels: List[str] = []
        self.seq_len: int = self.SEQ_LEN
        self.lock = threading.Lock()
        self.debug = os.environ.get("SIGN_AI_DEBUG", "0").lower() in ("1", "true", "yes")
        self.sequences: Dict[str, deque] = {}
        self.empty_streak: Dict[str, int] = {}
        self.recent_predictions: Dict[str, deque] = {}
        self.activity_detectors: Dict[str, SigningActivityDetector] = {}

        # Resolve model checkpoint
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

        self._load_class_map()
        self._load_model()
        self._load_hand_tracker()

    def _load_class_map(self):
        """Load 2,002 class vocabulary labels from CSV."""
        for path in self.CLASS_MAP_PATHS:
            if path.exists():
                try:
                    with open(path, mode="r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        labels = []
                        for row in reader:
                            if row and len(row) >= 2:
                                labels.append(row[1].strip())
                        if labels:
                            self.labels = labels
                            return
                except Exception as exc:
                    logger.warning("Could not read class map %s: %s", path, exc)

        # Fallback default class names if CSV is missing
        if not self.labels:
            self.labels = [f"Sign_{i}" for i in range(2002)]

    def _load_model(self):
        """Instantiate and load the pretrained HWGAT model architecture."""
        if not self.model_path or not self.model_path.exists():
            self.error = "Could not load sign model: No HWGAT model checkpoint found."
            return

        try:
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

            # Load checkpoint state dictionary onto CPU
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)

            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

            # Allow custom class override if stored in checkpoint
            if isinstance(ckpt, dict) and "classes" in ckpt:
                self.labels = [str(x) for x in ckpt["classes"]]

            num_classes = len(self.labels) or 2002

            # Import HWGAT architecture
            try:
                from ai_modules.hwgat.model_params import HWGATEParams
                from ai_modules.hwgat.HWGATE import Model as HWGATModel

                params = HWGATEParams({"num_class": num_classes, "src_len": self.seq_len}, 2, "cpu").get_model_params()
                model = HWGATModel(*params)
                model.load_state_dict(state_dict, strict=False)
                model.eval()

                self.model = model
                self.arch = "HWGAT"
                self.is_2hand = True
                self.error = None

            except Exception as e:
                # If custom mock model or custom test checkpoint is supplied
                if hasattr(ckpt, "__call__"):
                    self.model = ckpt
                    self.arch = "HWGAT"
                    self.error = None
                else:
                    raise e

            # Free temporary load references to keep memory under 150 MB
            del ckpt, state_dict
            gc.collect()

        except Exception as exc:
            self.model = None
            filename = self.model_path.name if self.model_path else "unknown"
            self.error = f"Could not load sign model (HWGAT {filename}): {exc}"
            logger.error("HWGAT model loading failure: %s", exc)

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
            except Exception:
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

    def _extract_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract 42 hand landmarks ([x, y]) from a BGR image frame."""
        if frame_bgr is None or not hasattr(frame_bgr, "size") or frame_bgr.size == 0 or frame_bgr.ndim != 3:
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. MediaPipe Tasks API
        if self.landmarker is not None:
            try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                with self.lock:
                    result = self.landmarker.detect(mp_image)

                if not result or not result.hand_landmarks:
                    return None

                joints = np.zeros((_NUM_NODES, 2), dtype=np.float32)
                num_hands = len(result.hand_landmarks)

                if num_hands == 1:
                    for lm_idx, lm in enumerate(result.hand_landmarks[0][:_LANDMARKS_PER_HAND]):
                        joints[lm_idx, 0] = float(lm.x)
                        joints[lm_idx, 1] = float(lm.y)
                else:
                    hands_with_meta = []
                    for h_idx in range(min(num_hands, 2)):
                        lms = result.hand_landmarks[h_idx]
                        cat = ""
                        if result.handedness and h_idx < len(result.handedness) and result.handedness[h_idx]:
                            cat = getattr(result.handedness[h_idx][0], "category_name", "")
                        hands_with_meta.append((cat, lms))

                    if len(hands_with_meta) == 2 and hands_with_meta[0][0] == "Left" and hands_with_meta[1][0] == "Right":
                        hands_with_meta.reverse()

                    for h_idx, (_, lms) in enumerate(hands_with_meta[:2]):
                        offset = h_idx * _LANDMARKS_PER_HAND
                        for lm_idx, lm in enumerate(lms[:_LANDMARKS_PER_HAND]):
                            joints[offset + lm_idx, 0] = float(lm.x)
                            joints[offset + lm_idx, 1] = float(lm.y)

                return joints
            except Exception:
                return None

        # 2. Legacy MediaPipe solutions API
        if self.legacy_hands is not None:
            try:
                with self.lock:
                    result = self.legacy_hands.process(rgb)

                if not result or not result.multi_hand_landmarks:
                    return None

                joints = np.zeros((_NUM_NODES, 2), dtype=np.float32)
                num_hands = len(result.multi_hand_landmarks)

                if num_hands == 1:
                    for lm_idx, lm in enumerate(result.multi_hand_landmarks[0].landmark[:_LANDMARKS_PER_HAND]):
                        joints[lm_idx, 0] = float(lm.x)
                        joints[lm_idx, 1] = float(lm.y)
                else:
                    hands_with_meta = []
                    for h_idx in range(min(num_hands, 2)):
                        lms = result.multi_hand_landmarks[h_idx]
                        cat = ""
                        if result.multi_handedness and h_idx < len(result.multi_handedness):
                            cat = result.multi_handedness[h_idx].classification[0].label
                        hands_with_meta.append((cat, lms))

                    if len(hands_with_meta) == 2 and hands_with_meta[0][0] == "Left" and hands_with_meta[1][0] == "Right":
                        hands_with_meta.reverse()

                    for h_idx, (_, lms) in enumerate(hands_with_meta[:2]):
                        offset = h_idx * _LANDMARKS_PER_HAND
                        for lm_idx, lm in enumerate(lms[:_LANDMARKS_PER_HAND]):
                            joints[offset + lm_idx, 0] = float(lm.x)
                            joints[offset + lm_idx, 1] = float(lm.y)

                return joints
            except Exception:
                return None

        return None

    def _build_hwgat_tensor(self, sequence: List[np.ndarray], target_len: int = 192) -> torch.Tensor:
        """
        Transform a variable-length sequence of landmark frames into the
        tensor expected by HWGAT: (1, 192, 64, 2).
        """
        n_frames = len(sequence)
        stacked_64 = np.zeros((n_frames, 64, 2), dtype=np.float32)

        # Reference normalized head and shoulder anchor keypoints
        head_arm_anchors = np.array([
            [0.50, 0.20], [0.48, 0.18], [0.52, 0.18],  # Head (nose, eyes)
            [0.40, 0.35], [0.35, 0.50], [0.30, 0.65],  # Left arm
        ], dtype=np.float32)

        for i, frame in enumerate(sequence):
            arr = np.array(frame, dtype=np.float32)
            if arr.shape == (64, 2):
                stacked_64[i] = arr
                continue

            if arr.ndim == 2 and arr.shape[0] >= 42:
                # 42 hand joints: Hand 1 (Right) at 0..20, Hand 2 (Left) at 21..41
                h1 = arr[:21]
                h2 = arr[21:42]
                h1_sub = h1[self.HWGAT_HAND_INDICES]  # 10 keypoints
                h2_sub = h2[self.HWGAT_HAND_INDICES]  # 10 keypoints

                # 4 windows of 16 points:
                # Window 0: Head + L_arm + L_hand
                w0 = np.vstack([head_arm_anchors, h2_sub])
                # Window 1: Head + R_arm + R_hand
                w1 = np.vstack([head_arm_anchors, h1_sub])
                # Window 2: Head + L_arm + R_hand
                w2 = np.vstack([head_arm_anchors, h1_sub])
                # Window 3: Head + R_arm + L_hand
                w3 = np.vstack([head_arm_anchors, h2_sub])

                stacked_64[i, 0:16] = w0
                stacked_64[i, 16:32] = w1
                stacked_64[i, 32:48] = w2
                stacked_64[i, 48:64] = w3

            elif arr.ndim == 2 and arr.shape[0] == 543:
                # 543 Holistic joints: select 29 points and create 4 windows
                # Window layout matches HWGAT specification
                selected_29 = arr[[0, 2, 5, 11, 12, 13, 14, 15, 16] +
                                  [501, 505, 506, 509, 510, 513, 514, 517, 518, 521] +
                                  [522, 526, 527, 530, 531, 534, 535, 538, 539, 542], :2]
                nose = selected_29[0:1]
                l_sh = selected_29[3]
                r_sh = selected_29[4]
                span = max(float(np.linalg.norm(l_sh - r_sh)), 0.05)
                norm_29 = (selected_29 - nose) / span

                # Window mapping
                w0_idx = [0, 1, 2, 3, 5, 7] + list(range(9, 19))
                w1_idx = [0, 1, 2, 4, 6, 8] + list(range(19, 29))
                w2_idx = [0, 1, 2, 3, 5, 7] + list(range(19, 29))
                w3_idx = [0, 1, 2, 4, 6, 8] + list(range(9, 19))

                stacked_64[i, 0:16] = norm_29[w0_idx]
                stacked_64[i, 16:32] = norm_29[w1_idx]
                stacked_64[i, 32:48] = norm_29[w2_idx]
                stacked_64[i, 48:64] = norm_29[w3_idx]

        # Uniform temporal linear interpolation to 192 frames
        if n_frames == target_len:
            resampled = stacked_64
        else:
            indices = np.linspace(0, max(n_frames - 1, 1), target_len)
            resampled = np.zeros((target_len, 64, 2), dtype=np.float32)
            xp = np.arange(n_frames)
            for k in range(64):
                for d in range(2):
                    resampled[:, k, d] = np.interp(indices, xp, stacked_64[:, k, d])

        tensor = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)
        return tensor

    def _predict_hwgat(self, sequence: List[np.ndarray]) -> Tuple[str, float]:
        """Perform forward inference on sequence of landmark frames using HWGAT."""
        if not sequence or self.model is None:
            return "unknown", 0.0

        tensor = self._build_hwgat_tensor(sequence, target_len=self.seq_len)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx] if idx < len(self.labels) else f"sign_{idx}"
        return label, conf

    # Backwards-compatible alias for existing test calls
    _predict_gcnt = _predict_hwgat

    def _smooth(self, key: str, label: str, conf: float) -> Tuple[str, float]:
        """Stabilize streaming predictions via temporal voting window."""
        with self.lock:
            buf = self.recent_predictions.setdefault(key, deque(maxlen=self.SMOOTHING_WINDOW))
            buf.append((label, conf))
            votes: Dict[str, float] = {}
            for lab, c in buf:
                votes[lab] = votes.get(lab, 0.0) + c
            best = max(votes, key=lambda k: votes[k])
            best_conf = max((c for l, c in buf if l == best), default=conf)
        return best, best_conf

    def _get_activity_detector(self, user_id: str) -> SigningActivityDetector:
        with self.lock:
            if user_id not in self.activity_detectors:
                self.activity_detectors[user_id] = SigningActivityDetector()
            return self.activity_detectors[user_id]

    def reset_user(self, user_id: str):
        key = user_id
        with self.lock:
            self.sequences.pop(key, None)
            self.empty_streak.pop(key, None)
            self.recent_predictions.pop(key, None)
            self.activity_detectors.pop(key, None)

    def recognize_frame(self, frame_data: bytes, user_id: str = "anonymous") -> dict:
        """
        Process incoming video frame through HWGAT inference pipeline.

        Returns
        -------
        dict with keys:
          - text: stabilized recognized sign
          - confidence: confidence percentage (float)
          - message: status description
          - ready: boolean
          - arch: 'HWGAT'
          - signing_state: IDLE | POSSIBLE_SIGNING | SIGNING | POSSIBLE_TURN_END
          - is_signing: boolean
          - turn_completed: boolean
          - phrase: current accumulated phrase during this turn
          - finalized_phrase: final phrase string when turn completes
          - vocabulary_size: total vocabulary classes (2,002)
        """
        arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if frame is None:
            return _resp("", 0.0, "Invalid image frame.", ready=False, vocabulary_size=len(self.labels))

        key = user_id
        detector = self._get_activity_detector(key)

        if self.model is None:
            activity_info = detector.update(None, prediction="", confidence=0.0)
            return _resp(
                "", 0.0,
                self.error or "HWGAT sign model not loaded.",
                ready=False,
                arch="HWGAT",
                signing_state=activity_info["state"],
                is_signing=activity_info["is_signing"],
                turn_completed=activity_info["turn_completed"],
                phrase=activity_info["phrase"],
                finalized_phrase=activity_info["finalized_phrase"],
                vocabulary_size=len(self.labels),
            )

        joints = self._extract_landmarks(frame)

        if joints is None:
            with self.lock:
                streak = self.empty_streak.get(key, 0) + 1
                self.empty_streak[key] = streak
                if streak > 15:
                    self.sequences.pop(key, None)

            activity_info = detector.update(None, prediction="", confidence=0.0)
            return _resp(
                "", 0.0, "No hand detected.",
                ready=False,
                arch="HWGAT",
                signing_state=activity_info["state"],
                is_signing=activity_info["is_signing"],
                turn_completed=activity_info["turn_completed"],
                phrase=activity_info["phrase"],
                finalized_phrase=activity_info["finalized_phrase"],
                vocabulary_size=len(self.labels),
            )

        with self.lock:
            self.empty_streak[key] = 0
            q = self.sequences.setdefault(key, deque(maxlen=self.seq_len))
            q.append(joints)
            seq = list(q)

        # Buffer warmup check (minimum 6 frames)
        if len(seq) < self.MIN_WARMUP_FRAMES:
            activity_info = detector.update(joints, prediction="", confidence=0.0)
            return _resp(
                "", 0.0,
                f"Capturing sign movement ({len(seq)}/{self.MIN_WARMUP_FRAMES})…",
                ready=False,
                arch="HWGAT",
                signing_state=activity_info["state"],
                is_signing=activity_info["is_signing"],
                turn_completed=activity_info["turn_completed"],
                phrase=activity_info["phrase"],
                finalized_phrase=activity_info["finalized_phrase"],
                vocabulary_size=len(self.labels),
            )

        # Execute HWGAT Inference
        try:
            raw_label, conf = self._predict_hwgat(seq)
            smoothed_label, smoothed_conf = self._smooth(key, raw_label, conf)
            display_text = smoothed_label.replace("_", " ").title()

            if smoothed_conf < self.MIN_CONFIDENCE:
                activity_info = detector.update(joints, prediction="UNCERTAIN", confidence=smoothed_conf)
                return _resp(
                    "UNCERTAIN",
                    round(smoothed_conf * 100, 1),
                    "Low confidence — continue holding or repeating the sign.",
                    ready=True,
                    arch="HWGAT",
                    signing_state=activity_info["state"],
                    is_signing=activity_info["is_signing"],
                    turn_completed=activity_info["turn_completed"],
                    phrase=activity_info["phrase"],
                    finalized_phrase=activity_info["finalized_phrase"],
                    vocabulary_size=len(self.labels),
                )

            activity_info = detector.update(joints, prediction=display_text, confidence=smoothed_conf)
            new_token = activity_info.get("accepted_sign", "")

            return _resp(
                display_text,
                round(smoothed_conf * 100, 1),
                f"Recognized: {display_text}",
                ready=True,
                arch="HWGAT",
                signing_state=activity_info["state"],
                is_signing=activity_info["is_signing"],
                turn_completed=activity_info["turn_completed"],
                phrase=activity_info["phrase"],
                finalized_phrase=activity_info["finalized_phrase"],
                new_token=new_token,
                vocabulary_size=len(self.labels),
            )

        except Exception as exc:
            activity_info = detector.update(joints, prediction="", confidence=0.0)
            return _resp(
                "", 0.0, f"HWGAT inference error: {exc}",
                ready=False,
                arch="HWGAT",
                signing_state=activity_info["state"],
                is_signing=activity_info["is_signing"],
                turn_completed=activity_info["turn_completed"],
                phrase=activity_info["phrase"],
                finalized_phrase=activity_info["finalized_phrase"],
                vocabulary_size=len(self.labels),
            )
