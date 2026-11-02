"""
inference.py - Standalone Lightweight Sign Language Recognition Engine

Self-contained inference module inside the deployment package.
"""
import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
import torch
import numpy as np

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from demo_utils import CFG
import importlib

CHECKPOINT_PATH = os.path.join(PACKAGE_DIR, "checkpoint.pt")
CLASS_MAPPING_PATH = os.path.join(PACKAGE_DIR, "class_mapping.json")


class SignRecognizer:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.cfg = CFG()

        # Load class mapping
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
            self.class_mapping = {int(k): v for k, v in raw_map.items()}
        self.num_classes = len(self.class_mapping)

        # Load architecture
        module = importlib.import_module("models." + self.cfg.model_type)
        self.model = getattr(module, "Model")(*self.cfg.model_params.get_model_params())

        # Load checkpoint
        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Adapt head if num_classes differs from 2002
        if self.num_classes != 2002 and hasattr(self.model, "head"):
            in_features = self.model.head.in_features
            self.model.head = torch.nn.Linear(in_features, self.num_classes)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_landmarks(self, landmarks_array: np.ndarray) -> Dict[str, Any]:
        """
        Predict sign from normalized (192, 64, 2) numpy array.
        Returns top prediction, confidence, and all class probabilities.
        """
        if landmarks_array.shape != (192, 64, 2):
            raise ValueError(f"Expected array of shape (192, 64, 2), got {landmarks_array.shape}")

        tensor = torch.tensor(landmarks_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            t0 = time.perf_counter()
            logits = self.model(tensor)
            t1 = time.perf_counter()
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = self.class_mapping.get(pred_idx, f"Class_{pred_idx}")
        confidence = float(probs[pred_idx])
        latency_ms = (t1 - t0) * 1000.0

        ranked_indices = np.argsort(probs)[::-1]
        top_k = [
            {"rank": i + 1, "class": self.class_mapping.get(idx, str(idx)), "confidence": round(float(probs[idx]), 4)}
            for i, idx in enumerate(ranked_indices[:min(5, self.num_classes)])
        ]

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "top_predictions": top_k,
        }


if __name__ == "__main__":
    print("Testing standalone SignRecognizer...")
    recognizer = SignRecognizer(device="cpu")
    dummy = np.random.randn(192, 64, 2).astype(np.float32)
    res = recognizer.predict_landmarks(dummy)
    print("Inference Test Result:")
    print(json.dumps(res, indent=2))
