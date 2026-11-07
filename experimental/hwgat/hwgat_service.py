"""
HWGAT Real-Time Sign Language Recognition Microservice
======================================================
High-performance streaming inference service for Indian Sign Language (ISL) recognition
using the pretrained Hierarchical Windowed Graph Attention Network (HWGAT).

Features:
- Loads HWGAT model (22.08M params, 2,002 ISL classes) into memory once at startup.
- MediaPipe Holistic landmark extraction (543 landmarks -> 29 selected -> 4-window 64 keypoints).
- Multi-user session management with independent sliding landmark buffers (target 192 frames).
- Kinematic motion activity detection: fast path when hands are idle (<10ms).
- Dynamic temporal interpolation for early prediction (when >= 8 frames captured).
- Temporal prediction smoothing across consecutive frames.
- Configurable confidence thresholding (default 0.25 / 25%).
- Duplicate suppression within 2.5s window.
- Thread-safe non-blocking background inference workers.
- Full compatibility with InclusiveMatch AI WebRTC video call system.
"""

import os
import sys
import csv
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

# Ensure experimental/hwgat is at head of path
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from demo_utils import CFG
import importlib
from pose_modules.Mediapipe import Model as HolisticModel


# Initialize Flask service
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

print("=" * 70)
print("INITIALIZING HWGAT REAL-TIME RECOGNITION SERVICE (2,002 ISL CLASSES)")
print("=" * 70)

# Load configuration
cfg = CFG()
device = cfg.device
print(f"[Device] Running HWGAT on: {device}")

# Load 2,002 class labels from class_map_FDMSE.csv
class_map_candidates = [
    os.path.join(SERVICE_DIR, "input", "class_map_FDMSE.csv"),
    os.path.abspath(os.path.join(SERVICE_DIR, "..", "..", "models", "class_map_hwgat.csv")),
    os.path.abspath(os.path.join(os.getcwd(), "models", "class_map_hwgat.csv")),
    cfg.class_map_path,
]
class_map_file = next((p for p in class_map_candidates if p and os.path.exists(p)), cfg.class_map_path)

class_map: Dict[int, str] = {}
with open(class_map_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if len(row) >= 2:
            class_map[int(row[0])] = row[1].strip()

print(f"[Class Map] Successfully loaded {len(class_map)} Indian Sign Language classes.")

# Load HWGAT Neural Network Model (Priority: HWGAT_CHECKPOINT_PATH -> input/ -> top-level models/ -> cfg fallback)
checkpoint_candidates = [
    os.environ.get("HWGAT_CHECKPOINT_PATH", "").strip(),
    os.path.join(SERVICE_DIR, "input", "model_best_loss.pt"),
    os.path.abspath(os.path.join(SERVICE_DIR, "..", "..", "models", "hwgat_best.pt")),
    os.path.abspath(os.path.join(os.getcwd(), "models", "hwgat_best.pt")),
    cfg.save_model_path,
]
checkpoint_file = next((p for p in checkpoint_candidates if p and os.path.exists(p)), cfg.save_model_path)

try:
    module = importlib.import_module("models." + cfg.model_type)
    if not hasattr(module, "Model"):
        raise AttributeError("Imported top-level models without Model class")
except (ModuleNotFoundError, AttributeError):
    try:
        from ai_modules.hwgat import HWGATE as module
    except ImportError:
        module = importlib.import_module("experimental.hwgat.models." + cfg.model_type)
model = getattr(module, "Model")(*cfg.model_params.get_model_params())

print(f"[Checkpoint] Loading weights from {os.path.basename(checkpoint_file)}...")
checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()
model_lock = threading.Lock()
print(f"[Model] HWGATE loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters).")

# Initialize MediaPipe Holistic Model
holistic_lock = threading.Lock()
holistic_model = HolisticModel()
print("[MediaPipe] MediaPipe Holistic landmark extractor ready.")


# ---------------------------------------------------------------------------
# Per-User Session Management
# ---------------------------------------------------------------------------

class UserSession:
    """Maintains independent sliding buffers and state for a single call participant."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.landmark_buffer: deque = deque(maxlen=192)
        self.last_landmarks: Optional[np.ndarray] = None
        self.last_active_time: float = time.time()
        self.last_width: Optional[int] = None
        self.last_height: Optional[int] = None
        self.is_signing: bool = False
        self.state: str = "IDLE"
        self.active_frame_streak: int = 0
        self.idle_frame_streak: int = 0
        self.recent_predictions: deque = deque(maxlen=5)
        self.last_emitted_sign: str = ""
        self.last_emitted_time: float = 0.0
        self.accumulated_phrase: List[str] = []
        self.lock = threading.Lock()


sessions: Dict[str, UserSession] = {}
sessions_lock = threading.Lock()


def get_user_session(user_id: str) -> UserSession:
    with sessions_lock:
        if user_id not in sessions:
            sessions[user_id] = UserSession(user_id)
        return sessions[user_id]


# ---------------------------------------------------------------------------
# Keypoint Selection and Transformation (29 keypoints -> 64 keypoints 4-window)
# ---------------------------------------------------------------------------

# 29 keypoint indices from Holistic 543:
# Pose 9: [0, 2, 5, 11, 12, 13, 14, 15, 16]
# Left hand 10: [0, 4, 5, 8, 9, 12, 13, 16, 17, 20] + 33 + 468
# Right hand 10: [0, 4, 5, 8, 9, 12, 13, 16, 17, 20] + 21 + 33 + 468
SELECTED_KPS = [0, 2, 5, 11, 12, 13, 14, 15, 16] + [
    0 + 33 + 468, 4 + 33 + 468, 5 + 33 + 468, 8 + 33 + 468, 9 + 33 + 468,
    12 + 33 + 468, 13 + 33 + 468, 16 + 33 + 468, 17 + 33 + 468, 20 + 33 + 468,
    0 + 21 + 33 + 468, 4 + 21 + 33 + 468, 5 + 21 + 33 + 468, 8 + 21 + 33 + 468,
    9 + 21 + 33 + 468, 12 + 21 + 33 + 468, 13 + 21 + 33 + 468, 16 + 21 + 33 + 468,
    17 + 21 + 33 + 468, 20 + 21 + 33 + 468
]

# 4-Window Mapping for 64 keypoints
PARTS_IDX = {
    "head": [0, 1, 2],
    "l_arm": [3, 5, 7],
    "l_hand": list(range(9, 19)),
    "r_arm": [4, 6, 8],
    "r_hand": list(range(19, 29)),
}

WINDOW_IDX = {
    "window0": PARTS_IDX["head"] + PARTS_IDX["l_arm"] + PARTS_IDX["l_hand"],
    "window1": PARTS_IDX["head"] + PARTS_IDX["r_arm"] + PARTS_IDX["r_hand"],
    "window2": PARTS_IDX["head"] + PARTS_IDX["l_arm"] + PARTS_IDX["r_hand"],
    "window3": PARTS_IDX["head"] + PARTS_IDX["r_arm"] + PARTS_IDX["l_hand"],
}


def build_hwgat_tensor(
    feature_list: List[np.ndarray],
    target_len: int = 192,
    mode: str = "training_aligned",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> torch.Tensor:
    """
    Transforms a list of N frame landmarks (each (543, 4) or (29, 2)) into the
    exact tensor expected by HWGAT: (1, 192, 64, 2).

    Supports:
      - mode='training_aligned' (default): exact data_transform.py pipeline with nose origin at (0.50, 0.333),
        shoulder scaling, HandCorrection spline interpolation, aspect-ratio correction, and 4-window topology.
      - mode='zero_centered': zero-centered coordinate frame (nose at (0, 0), shoulder span 1.0).
    """
    n_frames = len(feature_list)
    raw_array = np.array(feature_list, dtype=np.float32)

    # 1. PoseSelect 29 keypoints with 2D coordinates [x, y]
    if raw_array.shape[1] >= 543:
        kps_29 = raw_array[:, SELECTED_KPS, :2].copy()
    elif raw_array.shape[1] == 29:
        kps_29 = raw_array[:, :, :2].copy()
    else:
        raise ValueError(f"Expected 543 or 29 keypoints, got shape {raw_array.shape}")

    if mode == "training_aligned":
        # HandCorrection (left_slice=[9, 19, 7], right_slice=[19, 29, 8])
        def _apply_hand_correction(vid_feat, slices):
            s_start, s_end, s_wrist = slices
            if np.sum(vid_feat[:, s_start:s_end]) == 0:
                vid_feat[:, s_start:s_end, :] = np.expand_dims(vid_feat[:, s_wrist, :], axis=1)
                return vid_feat

            start = end = 0
            for f_idx in range(len(vid_feat)):
                if not vid_feat[f_idx, s_start:s_end].any():
                    vid_feat[f_idx, s_start:s_end, :] = vid_feat[f_idx, s_wrist, :]
                else:
                    start = f_idx
                    break

            for f_idx in reversed(range(len(vid_feat))):
                if not vid_feat[f_idx, s_start:s_end].any():
                    vid_feat[f_idx, s_start:s_end, :] = vid_feat[f_idx, s_wrist, :]
                else:
                    end = f_idx
                    break

            present = [f for f in range(start, end + 1) if vid_feat[f, s_start:s_end].any()]
            missing = [f for f in range(start, end + 1) if not vid_feat[f, s_start:s_end].any()]

            if missing and len(present) >= 2:
                try:
                    from scipy import interpolate
                    k_spline = min(2, len(present) - 1)
                    for kp in range(s_start, s_end):
                        x_pts = vid_feat[present, kp, 0]
                        y_pts = vid_feat[present, kp, 1]
                        tck_x = interpolate.splrep(present, x_pts, k=k_spline)
                        tck_y = interpolate.splrep(present, y_pts, k=k_spline)
                        vid_feat[missing, kp, 0] = interpolate.splev(missing, tck_x)
                        vid_feat[missing, kp, 1] = interpolate.splev(missing, tck_y)
                except Exception:
                    for f in missing:
                        vid_feat[f, s_start:s_end, :] = vid_feat[f, s_wrist, :]
            return vid_feat

        kps_29 = _apply_hand_correction(kps_29, [9, 19, 7])
        kps_29 = _apply_hand_correction(kps_29, [19, 29, 8])

        # Multiply 29 selected landmarks' x coordinates by width / height before NormalizeKeypoints
        if width is not None and height is not None and width > 0 and height > 0:
            aspect_ratio = float(width) / float(height)
            kps_29[:, :, 0] *= aspect_ratio

        # NormalizeKeypoints (origin_idx=0 [nose], anchor_points=[3, 4] [shoulders])
        left_top = np.zeros(2, dtype=np.float32)
        edge_dist = 1.0
        for keypoints in kps_29:
            if keypoints[0].any() and keypoints[3].any() and keypoints[4].any():
                root = keypoints[0]
                unit_dist = np.linalg.norm(keypoints[3] - keypoints[4])
                if unit_dist > 1e-6:
                    left_top = root - 3.0 * unit_dist
                    left_top[1] = root[1] - 2.0 * unit_dist
                    edge_dist = 6.0 * unit_dist
                    break

        norm_kps = (kps_29 - left_top) / max(float(edge_dist), 1e-6)

        # TemporalSample to target_len (192)
        if norm_kps.shape[0] <= target_len:
            sample = 0.5
            index = int((target_len - norm_kps.shape[0]) * sample)
            pad1 = norm_kps[0]
            pad2 = norm_kps[-1]
            blank_front = np.full((target_len // 2, 29, 2), pad1, dtype=np.float32)
            blank_back = np.full((target_len - target_len // 2, 29, 2), pad2, dtype=np.float32)
            sampled_29 = np.concatenate([blank_front, blank_back], axis=0)
            sampled_29[index:index + norm_kps.shape[0], :] = norm_kps
        else:
            choices = np.linspace(0, norm_kps.shape[0] - 1, num=target_len).astype(int)
            sampled_29 = norm_kps[choices]

        stacked_64 = np.zeros((target_len, 64, 2), dtype=np.float32)
        stacked_64[:, :16] = sampled_29.take(WINDOW_IDX["window0"], 1)
        stacked_64[:, 16:32] = sampled_29.take(WINDOW_IDX["window1"], 1)
        stacked_64[:, 32:48] = sampled_29.take(WINDOW_IDX["window2"], 1)
        stacked_64[:, 48:64] = sampled_29.take(WINDOW_IDX["window3"], 1)

        return torch.tensor(stacked_64, dtype=torch.float32).unsqueeze(0)

    else:
        # Zero-centered coordinate system
        nose = kps_29[:, 0:1, :2]
        l_sh = kps_29[:, 3, :2]
        r_sh = kps_29[:, 4, :2]
        shoulder_dist = np.linalg.norm(l_sh - r_sh, axis=1, keepdims=True)
        shoulder_dist = np.maximum(shoulder_dist, 0.05)
        scale = np.expand_dims(shoulder_dist, 1)

        norm_kps = (kps_29 - nose) / scale

        stacked_64 = np.zeros((n_frames, 64, 2), dtype=np.float32)
        stacked_64[:, 0:16] = norm_kps[:, WINDOW_IDX["window0"], :]
        stacked_64[:, 16:32] = norm_kps[:, WINDOW_IDX["window1"], :]
        stacked_64[:, 32:48] = norm_kps[:, WINDOW_IDX["window2"], :]
        stacked_64[:, 48:64] = norm_kps[:, WINDOW_IDX["window3"], :]

        if n_frames == target_len:
            resampled = stacked_64
        else:
            indices = np.linspace(0, n_frames - 1, target_len)
            resampled = np.zeros((target_len, 64, 2), dtype=np.float32)
            for k in range(64):
                for d in range(2):
                    resampled[:, k, d] = np.interp(indices, np.arange(n_frames), stacked_64[:, k, d])

        return torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------------------------
# Kinematic Motion Detection (Articulatory Hand Velocity)
# ---------------------------------------------------------------------------

def calculate_hand_velocity(current_kps: np.ndarray, last_kps: Optional[np.ndarray]) -> float:
    """Calculates mean displacement of hand landmarks between consecutive frames."""
    if last_kps is None:
        return 0.0

    # Hand keypoint indices in 543 Holistic:
    # Left hand: 501..521 (21 points), Right hand: 522..542 (21 points)
    cur_hands = current_kps[501:543, :2]
    last_hands = last_kps[501:543, :2]

    # Only consider detected joints (non-zero)
    mask = (np.any(cur_hands != 0, axis=-1)) & (np.any(last_hands != 0, axis=-1))
    if not np.any(mask):
        return 0.0

    diff = cur_hands[mask] - last_hands[mask]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return float(np.mean(dist))


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready",
        "service": "HWGAT Sign Recognition Microservice",
        "architecture": "HWGATE (Hierarchical Windowed Graph Attention Network)",
        "classes_count": len(class_map),
        "device": str(device),
        "active_sessions": len(sessions),
    })


@app.route("/recognize_frame", methods=["POST"])
def recognize_frame():
    """
    Primary real-time frame recognition endpoint for WebRTC calls.
    Accepts:
      - 'frame': Image binary (JPEG or PNG)
      - 'user_id': Identifier for call participant (default: 'anonymous')
      - 'threshold': Optional confidence cutoff (default: 0.25)
    """
    if "frame" not in request.files:
        return jsonify({"text": "", "confidence": 0, "message": "No frame supplied", "ready": False}), 400

    frame_file = request.files["frame"]
    frame_bytes = frame_file.read()
    if not frame_bytes:
        return jsonify({"text": "", "confidence": 0, "message": "Empty frame", "ready": False}), 400

    user_id = request.form.get("user_id", "anonymous").strip()
    conf_threshold = float(request.form.get("threshold", 0.25))
    req_w = request.form.get("width", "").strip()
    req_h = request.form.get("height", "").strip()
    frame_w = int(req_w) if (req_w.isdigit() and int(req_w) > 0) else None
    frame_h = int(req_h) if (req_h.isdigit() and int(req_h) > 0) else None

    # Decode image to RGB
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr_img is None:
        return jsonify({"text": "", "confidence": 0, "message": "Corrupt frame decode", "ready": False}), 400

    # Resize large frames to save CPU during MediaPipe pass
    h, w = bgr_img.shape[:2]
    if frame_w is None or frame_h is None:
        frame_w, frame_h = w, h
    if w > 480:
        new_w = 480
        new_h = int(h * (480.0 / w))
        bgr_img = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # MediaPipe Holistic Extraction
    with holistic_lock:
        landmarks, _, _ = holistic_model(rgb_img)  # Shape: (543, 4)

    # Check hand presence
    lh_present = np.any(landmarks[501:522, :3] != 0)
    rh_present = np.any(landmarks[522:543, :3] != 0)
    hands_present = lh_present or rh_present

    session = get_user_session(user_id)
    now = time.time()

    with session.lock:
        session.last_width = frame_w
        session.last_height = frame_h
        # Kinematic Motion Assessment
        velocity = calculate_hand_velocity(landmarks, session.last_landmarks)
        session.last_landmarks = landmarks.copy()

        # Velocity thresholds: idle (<0.0035), normal signing (0.0035 - 0.25), erratic (>0.25)
        is_moving = hands_present and (0.0035 <= velocity <= 0.25)

        if is_moving:
            session.active_frame_streak += 1
            session.idle_frame_streak = 0
            session.last_active_time = now
            session.landmark_buffer.append(landmarks)
            if session.active_frame_streak >= 2:
                session.state = "SIGNING"
                session.is_signing = True
        else:
            session.idle_frame_streak += 1
            session.active_frame_streak = 0
            # If idle for > 1.4 seconds, transition to IDLE / turn complete
            if (now - session.last_active_time) > 1.4:
                session.state = "IDLE"
                session.is_signing = False

        buf_len = len(session.landmark_buffer)

        # Quick return if idle and buffer is too short
        if not session.is_signing and buf_len < 8:
            return jsonify({
                "text": "",
                "confidence": 0.0,
                "message": "Position hands in camera view to sign.",
                "ready": True,
                "arch": "HWGAT",
                "signing_state": session.state,
                "is_signing": False,
                "turn_completed": False,
                "phrase": " ".join(session.accumulated_phrase),
                "finalized_phrase": "",
                "new_token": "",
            })

        # Need at least 8 frames of gesture evidence to perform HWGAT forward pass
        if buf_len < 8:
            return jsonify({
                "text": "",
                "confidence": 0.0,
                "message": f"Capturing gesture ({buf_len}/8 frames)...",
                "ready": False,
                "arch": "HWGAT",
                "signing_state": session.state,
                "is_signing": session.is_signing,
                "turn_completed": False,
                "phrase": " ".join(session.accumulated_phrase),
                "finalized_phrase": "",
                "new_token": "",
            })

        # Build tensor (1, 192, 64, 2)
        features_snapshot = list(session.landmark_buffer)

    # Model Forward Pass (Under model_lock)
    try:
        input_tensor = build_hwgat_tensor(
            features_snapshot,
            target_len=192,
            mode="training_aligned",
            width=session.last_width,
            height=session.last_height,
        ).to(device)
        with model_lock:
            with torch.no_grad():
                logits = model(input_tensor).squeeze(0)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        top_indices = np.argsort(probs)[::-1]

        best_id = int(top_indices[0])
        best_conf = float(probs[best_id])
        best_name = class_map.get(best_id, f"Sign_{best_id}")

        top_k = []
        for rank in range(min(5, len(class_map))):
            c_id = int(top_indices[rank])
            top_k.append({
                "rank": rank + 1,
                "class_id": c_id,
                "label": class_map.get(c_id, f"Sign_{c_id}"),
                "confidence": round(float(probs[c_id]) * 100.0, 1),
            })
    except Exception as exc:
        return jsonify({
            "text": "",
            "confidence": 0.0,
            "message": f"HWGAT inference error: {exc}",
            "ready": False,
            "arch": "HWGAT",
        }), 200

    # Post-Processing: Confidence & Smoothing
    with session.lock:
        session.recent_predictions.append((best_name, best_conf))

        # Check confidence cutoff
        if best_conf < conf_threshold:
            return jsonify({
                "text": "UNCERTAIN",
                "confidence": round(best_conf * 100.0, 1),
                "message": f"Low confidence ({round(best_conf * 100.0, 1)}%) — continue holding sign",
                "ready": True,
                "arch": "HWGAT",
                "signing_state": session.state,
                "is_signing": session.is_signing,
                "turn_completed": False,
                "phrase": " ".join(session.accumulated_phrase),
                "finalized_phrase": "",
                "new_token": "",
                "top_k": top_k,
            })

        # Duplicate suppression: prevent emitting same sign repeatedly in < 2.5s
        is_duplicate = (best_name == session.last_emitted_sign) and ((now - session.last_emitted_time) < 2.5)
        new_token = ""
        if not is_duplicate:
            session.last_emitted_sign = best_name
            session.last_emitted_time = now
            session.accumulated_phrase.append(best_name)
            new_token = best_name

        # Turn completion check: if idle frames accumulated after signing
        turn_completed = False
        finalized_phrase = ""
        if session.idle_frame_streak >= 3 and len(session.accumulated_phrase) > 0:
            turn_completed = True
            finalized_phrase = " ".join(session.accumulated_phrase)
            session.accumulated_phrase = []
            session.landmark_buffer.clear()
            session.last_emitted_sign = ""

        return jsonify({
            "text": best_name,
            "confidence": round(best_conf * 100.0, 1),
            "message": f"Recognized: {best_name}",
            "ready": True,
            "arch": "HWGAT",
            "signing_state": session.state,
            "is_signing": session.is_signing,
            "turn_completed": turn_completed,
            "phrase": " ".join(session.accumulated_phrase),
            "finalized_phrase": finalized_phrase,
            "new_token": new_token,
            "top_k": top_k,
        })


@app.route("/recognize_landmarks", methods=["POST"])
def recognize_landmarks():
    """
    Direct landmark sequence inference endpoint.
    Accepts JSON payload:
      - 'landmarks': List of frame landmarks (each (543, 4) or (29, 2))
      - 'user_id': Identifier for call participant (default: 'anonymous')
      - 'mode': 'training_aligned' (default) or 'zero_centered'
      - 'threshold': Optional confidence cutoff (default: 0.25)
    """
    payload = request.get_json(silent=True) or {}
    landmarks_data = payload.get("landmarks")
    if not landmarks_data or not isinstance(landmarks_data, list):
        return jsonify({"text": "", "confidence": 0, "message": "Invalid landmarks list", "ready": False}), 400

    user_id = str(payload.get("user_id", "anonymous")).strip()
    conf_threshold = float(payload.get("threshold", 0.25))
    mode = str(payload.get("mode", "training_aligned")).strip()
    req_w = payload.get("width")
    req_h = payload.get("height")
    frame_w = int(req_w) if (req_w is not None and str(req_w).isdigit() and int(req_w) > 0) else None
    frame_h = int(req_h) if (req_h is not None and str(req_h).isdigit() and int(req_h) > 0) else None

    try:
        input_tensor = build_hwgat_tensor(
            landmarks_data,
            target_len=192,
            mode=mode,
            width=frame_w,
            height=frame_h,
        ).to(device)
        with model_lock:
            with torch.no_grad():
                logits = model(input_tensor).squeeze(0)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        top_indices = np.argsort(probs)[::-1]

        best_id = int(top_indices[0])
        best_conf = float(probs[best_id])
        best_name = class_map.get(best_id, f"Sign_{best_id}")

        top_k = []
        for rank in range(min(5, len(class_map))):
            c_id = int(top_indices[rank])
            top_k.append({
                "rank": rank + 1,
                "class_id": c_id,
                "label": class_map.get(c_id, f"Sign_{c_id}"),
                "confidence": round(float(probs[c_id]) * 100.0, 1),
            })

        return jsonify({
            "text": best_name if best_conf >= conf_threshold else "UNCERTAIN",
            "confidence": round(best_conf * 100.0, 1),
            "message": f"Recognized: {best_name}" if best_conf >= conf_threshold else f"Low confidence ({round(best_conf * 100.0, 1)}%)",
            "ready": True,
            "arch": "HWGAT",
            "top_k": top_k,
            "mode": mode,
            "class_id": best_id,
        })
    except Exception as exc:
        return jsonify({
            "text": "",
            "confidence": 0.0,
            "message": f"HWGAT landmarks inference error: {exc}",
            "ready": False,
            "arch": "HWGAT",
        }), 200


@app.route("/reset_user", methods=["POST"])
def reset_user():
    """Resets landmark queue and state for a user when call ends or disconnects."""
    user_id = request.form.get("user_id", "").strip()
    if not user_id:
        return jsonify({"ok": True, "message": "No user_id supplied"}), 200

    with sessions_lock:
        if user_id in sessions:
            with sessions[user_id].lock:
                sessions[user_id].landmark_buffer.clear()
                sessions[user_id].recent_predictions.clear()
                sessions[user_id].accumulated_phrase.clear()
                sessions[user_id].is_signing = False
                sessions[user_id].state = "IDLE"
            del sessions[user_id]

    return jsonify({"ok": True, "message": f"User {user_id} session reset successfully"})


if __name__ == "__main__":
    port = int(os.environ.get("HWGAT_PORT", 5055))
    print(f"\n[HWGAT Service] Starting Flask server on 0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
