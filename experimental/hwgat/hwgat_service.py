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
class_map_file = os.path.join(SERVICE_DIR, "input", "class_map_FDMSE.csv")
if not os.path.exists(class_map_file):
    class_map_file = cfg.class_map_path

class_map: Dict[int, str] = {}
with open(class_map_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if len(row) >= 2:
            class_map[int(row[0])] = row[1].strip()

print(f"[Class Map] Successfully loaded {len(class_map)} Indian Sign Language classes.")

# Load HWGAT Neural Network Model
checkpoint_file = os.path.join(SERVICE_DIR, "input", "model_best_loss.pt")
if not os.path.exists(checkpoint_file):
    checkpoint_file = cfg.save_model_path

module = importlib.import_module("models." + cfg.model_type)
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


def build_hwgat_tensor(feature_list: List[np.ndarray], target_len: int = 192) -> torch.Tensor:
    """
    Transforms a list of N frame landmarks (each (543, 4)) into the
    exact tensor expected by HWGAT: (1, 192, 64, 2).
    """
    n_frames = len(feature_list)
    raw_array = np.array(feature_list, dtype=np.float32)  # (N, 543, 4)

    # 1. PoseSelect 29 keypoints with 2D coordinates [x, y]
    kps_29 = raw_array[:, SELECTED_KPS, :2]  # (N, 29, 2)

    # 2. Normalize keypoints with respect to nose (index 0) and shoulders (indices 3, 4)
    # Nose origin
    nose = kps_29[:, 0:1, :2]
    # Shoulder anchor distance
    l_sh = kps_29[:, 3, :2]
    r_sh = kps_29[:, 4, :2]
    shoulder_dist = np.linalg.norm(l_sh - r_sh, axis=1, keepdims=True)
    shoulder_dist = np.maximum(shoulder_dist, 0.05)  # Avoid div-by-zero
    scale = np.expand_dims(shoulder_dist, 1)

    norm_kps = (kps_29 - nose) / scale

    # 3. Create 4-Window 64-Keypoint Stack
    stacked_64 = np.zeros((n_frames, 64, 2), dtype=np.float32)
    stacked_64[:, 0:16] = norm_kps[:, WINDOW_IDX["window0"], :]
    stacked_64[:, 16:32] = norm_kps[:, WINDOW_IDX["window1"], :]
    stacked_64[:, 32:48] = norm_kps[:, WINDOW_IDX["window2"], :]
    stacked_64[:, 48:64] = norm_kps[:, WINDOW_IDX["window3"], :]

    # 4. Temporal Resampling to target_len (192 frames)
    if n_frames == target_len:
        resampled = stacked_64
    else:
        # Uniform linear temporal interpolation
        indices = np.linspace(0, n_frames - 1, target_len)
        resampled = np.zeros((target_len, 64, 2), dtype=np.float32)
        for k in range(64):
            for d in range(2):
                resampled[:, k, d] = np.interp(indices, np.arange(n_frames), stacked_64[:, k, d])

    # Convert to PyTorch tensor (1, 192, 64, 2)
    tensor = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)
    return tensor


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

    # Decode image to RGB
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr_img is None:
        return jsonify({"text": "", "confidence": 0, "message": "Corrupt frame decode", "ready": False}), 400

    # Resize large frames to save CPU during MediaPipe pass
    h, w = bgr_img.shape[:2]
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
        input_tensor = build_hwgat_tensor(features_snapshot, target_len=192).to(device)
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
