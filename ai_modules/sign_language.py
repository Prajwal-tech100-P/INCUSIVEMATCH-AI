"""Real-time sign recognition using MediaPipe landmarks + a trained GRU.

The runtime keeps a short sequence of hand landmarks instead of classifying
one frame at a time. The model file is created by train_sign_model.py.
An old SVM model is supported as a compatibility fallback.
"""
from collections import deque
from pathlib import Path
import threading
import cv2
import numpy as np


class SignLanguageRecognizer:
    GRU_MODEL_PATH = Path("models/sign_gru.pt")
    SVM_MODEL_PATH = Path("models/sign_classifier.joblib")
    SEQ_LEN = 24
    MIN_CONFIDENCE = 0.72
    SMOOTHING_WINDOW = 5

    def __init__(self, model_path=None):
        self.gru = None
        self.svm = None
        self.hands = None
        self.mp_hands = None
        self.error = None
        self.lock = threading.Lock()
        self.sequences = {}
        self.recent_predictions = {}
        self.model_path = Path(model_path or self.GRU_MODEL_PATH)
        self._load_gru()
        if self.gru is None:
            self._load_svm()
        self._load_hand_tracker()

    def _load_gru(self):
        if not self.model_path.exists():
            self.error = f"Trained GRU model not found: {self.model_path}. Train it first."
            return
        try:
            import torch
            checkpoint = torch.load(self.model_path, map_location="cpu")
            labels = checkpoint["labels"]
            self.gru = _GRUNetwork(
                input_size=int(checkpoint.get("input_size", 84)),
                hidden_size=int(checkpoint.get("hidden_size", 128)),
                num_layers=int(checkpoint.get("num_layers", 2)),
                num_classes=len(labels),
                dropout=float(checkpoint.get("dropout", 0.25)),
                bidirectional=True,
            )
            self.gru.load_state_dict(checkpoint["state_dict"])
            self.gru.eval()
            self.labels = [str(x) for x in labels]
            self.seq_len = int(checkpoint.get("seq_len", self.SEQ_LEN))
        except Exception as exc:
            self.gru = None
            self.error = f"Could not load GRU model: {exc}"

    def _load_svm(self):
        if not self.SVM_MODEL_PATH.exists():
            return
        try:
            import joblib
            self.svm = joblib.load(self.SVM_MODEL_PATH)
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

    @staticmethod
    def _features(hand_landmarks):
        pts = np.array([(p.x, p.y) for p in hand_landmarks.landmark], dtype=np.float32)
        pts -= pts[0]
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale <= 1e-6:
            return None
        pts /= scale
        return pts.reshape(-1)

    def _predict_gru(self, sequence):
        import torch
        arr = np.asarray(sequence, dtype=np.float32)
        velocities = np.vstack([np.zeros((1, 42), dtype=np.float32), np.diff(arr, axis=0)])
        features = np.concatenate([arr, velocities], axis=1)
        x = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            logits = self.gru(x)
            probs = torch.softmax(logits, dim=1)[0].numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def _predict_svm(self, feature):
        prediction = self.svm.predict(feature.reshape(1, -1))[0]
        confidence = float(np.max(self.svm.predict_proba(feature.reshape(1, -1))[0])) if hasattr(self.svm, "predict_proba") else 0.0
        return str(prediction), confidence

    def reset_user(self, user_id):
        key = str(user_id)
        with self.lock:
            self.sequences.pop(key, None)
            self.recent_predictions.pop(key, None)

    def recognize_frame(self, frame_data, user_id="anonymous"):
        arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if frame is None:
            return {"text": "", "confidence": 0.0, "message": "Invalid image frame."}
        if self.gru is None and self.svm is None:
            return {"text": "", "confidence": 0.0, "message": self.error or "Train a sign model first."}
        if self.hands is None:
            return {"text": "", "confidence": 0.0, "message": self.error or "MediaPipe is unavailable."}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        key = str(user_id)
        if not result.multi_hand_landmarks:
            with self.lock:
                self.sequences.pop(key, None)
            return {"text": "", "confidence": 0.0, "message": "No hand detected."}

        feature = self._features(result.multi_hand_landmarks[0])
        if feature is None:
            return {"text": "", "confidence": 0.0, "message": "Could not normalize hand landmarks."}

        if self.gru is not None:
            with self.lock:
                q = self.sequences.setdefault(key, deque(maxlen=self.seq_len))
                q.append(feature)
                sequence = list(q)
            if len(sequence) < self.seq_len:
                return {"text": "", "confidence": 0.0, "message": f"Collecting sign frames ({len(sequence)}/{self.seq_len})...", "ready": False}
            try:
                label, confidence = self._predict_gru(sequence)
                with self.lock:
                    recent = self.recent_predictions.setdefault(key, deque(maxlen=self.SMOOTHING_WINDOW))
                    recent.append((label, confidence))
                    votes = {}
                    for lab, conf in recent:
                        votes[lab] = votes.get(lab, 0.0) + conf
                    stable_label = max(votes, key=votes.get)
                    stable_conf = max(conf for lab, conf in recent if lab == stable_label)
                if stable_conf < self.MIN_CONFIDENCE:
                    return {"text": "UNCERTAIN", "confidence": round(stable_conf * 100, 2), "message": "Low-confidence prediction; please repeat the sign.", "ready": True}
                return {"text": stable_label, "confidence": round(stable_conf * 100, 2), "message": "GRU sequence prediction.", "ready": True}
            except Exception as exc:
                return {"text": "", "confidence": 0.0, "message": f"GRU prediction failed: {exc}"}

        try:
            label, confidence = self._predict_svm(feature)
            if confidence < 0.65:
                label = "UNCERTAIN"
            return {"text": label, "confidence": round(confidence * 100, 2), "message": "SVM compatibility prediction."}
        except Exception as exc:
            return {"text": "", "confidence": 0.0, "message": f"Prediction failed: {exc}"}


class _GRUNetwork:
    pass


def _build_gru_class():
    import torch.nn as nn
    class GRUNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.25, bidirectional=True):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
            out_size = hidden_size * (2 if bidirectional else 1)
            self.norm = nn.LayerNorm(out_size)
            self.head = nn.Sequential(nn.Linear(out_size, 96), nn.ReLU(), nn.Dropout(dropout), nn.Linear(96, num_classes))
        def forward(self, x):
            y, _ = self.gru(x)
            y = self.norm(y[:, -1, :])
            return self.head(y)
    return GRUNetwork

_GRU_CLASS = None
def _GRUNetworkFactory(input_size, hidden_size, num_layers, num_classes, dropout, bidirectional):
    global _GRU_CLASS
    if _GRU_CLASS is None:
        _GRU_CLASS = _build_gru_class()
    return _GRU_CLASS(input_size, hidden_size, num_layers, num_classes, dropout, bidirectional)

# Make the runtime constructor work without importing torch when the module is imported.
OriginalGRUNetwork = _GRUNetwork
_GRUNetwork = _GRUNetworkFactory
