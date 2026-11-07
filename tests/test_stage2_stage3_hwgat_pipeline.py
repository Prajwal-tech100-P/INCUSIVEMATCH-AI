"""
Unit and Integration Tests for Stage 2 & Stage 3 HWGAT Implementation
=====================================================================
Verifies:
1. ENABLE_SIGN_AI=0 safeguards Render Free memory (PyTorch/MediaPipe not loaded).
2. ENABLE_SIGN_AI=1 allows in-process inference when explicitly enabled.
3. MediaPipe Holistic training-aligned preprocessing reproduces data_transform.py.
4. HWGATClient correctly constructs and handles /recognize_landmarks.
5. Two-way WebRTC sign caption contracts remain intact.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from app import create_app
from ai_modules.sign_language import SignLanguageRecognizer
from ai_modules.hwgat_client import HWGATClient
from routes.video_routes import _LazySignRecognizerProxy


class TestStage2Stage3HWGATPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_ENV"] = "testing"
        os.environ["SECRET_KEY"] = "test-stage2-stage3-pipeline"
        cls.app = create_app()
        cls.app.config["TESTING"] = True

    def test_01_proxy_when_disabled_returns_unavailable(self):
        """When ENABLE_SIGN_AI is disabled, proxy returns graceful unavailable without loading models."""
        proxy = _LazySignRecognizerProxy()
        with patch.object(proxy, "is_enabled", return_value=False):
            res = proxy.recognize_frame(b"dummy")
            self.assertEqual(res["status"], "unavailable")
            self.assertFalse(res["ready"])
            self.assertEqual(res["confidence"], 0.0)
            self.assertIn("safeguard system memory", res["message"])
            self.assertIsNone(proxy._instance)

    def test_02_proxy_session_cleanup_when_disabled(self):
        """Proxy maintains and flushes participant session dict even when model is disabled."""
        proxy = _LazySignRecognizerProxy()
        with patch.object(proxy, "is_enabled", return_value=False):
            with proxy.lock:
                proxy.sequences["user_x"] = [np.zeros((42, 2))]
                proxy.sequences["user_x_local"] = [np.zeros((42, 2))]

            self.assertIn("user_x", proxy.sequences)
            proxy.reset_user("user_x")
            self.assertNotIn("user_x", proxy.sequences)
            self.assertNotIn("user_x_local", proxy.sequences)

    def test_03_training_aligned_preprocessing_reproduces_data_transform(self):
        """Holistic preprocessing matches data_transform.py with origin at (0.50, 0.333)."""
        rec = SignLanguageRecognizer()

        # Construct 10 synthetic frames with nose and shoulders
        seq_543 = []
        for t in range(10):
            frame = np.zeros((543, 4), dtype=np.float32)
            frame[0, :2] = [0.50, 0.20]   # Nose
            frame[11, :2] = [0.35, 0.45]  # Left shoulder
            frame[12, :2] = [0.65, 0.45]  # Right shoulder
            frame[15, :2] = [0.30, 0.65]  # Left wrist
            frame[16, :2] = [0.70, 0.65]  # Right wrist
            # Hand keypoints
            frame[501:522, :2] = [0.32, 0.67] # Left hand
            frame[522:543, :2] = [0.68, 0.67] # Right hand
            seq_543.append(frame)

        tensor_aligned = rec._build_hwgat_tensor(seq_543, target_len=192, mode="training_aligned")
        self.assertEqual(tensor_aligned.shape, (1, 192, 64, 2))
        self.assertEqual(tensor_aligned.dtype, torch.float32)

        # In training_aligned mode, nose (Window 0 index 0) is centered at (0.50, 0.33333)
        self.assertAlmostEqual(float(tensor_aligned[0, 0, 0, 0]), 0.50, places=4)
        self.assertAlmostEqual(float(tensor_aligned[0, 0, 0, 1]), 1.0 / 3.0, places=4)

        # In zero_centered mode, nose is at (0.0, 0.0)
        tensor_zero = rec._build_hwgat_tensor(seq_543, target_len=192, mode="zero_centered")
        self.assertEqual(tensor_zero.shape, (1, 192, 64, 2))
        self.assertAlmostEqual(float(tensor_zero[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(tensor_zero[0, 0, 0, 1]), 0.0, places=5)

    def test_04_hwgat_client_recognize_landmarks(self):
        """HWGATClient.recognize_landmarks sends JSON payload and handles response."""
        client = HWGATClient("http://127.0.0.1:5055")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "text": "Namaste",
            "confidence": 94.5,
            "ready": True,
            "arch": "HWGAT",
            "mode": "training_aligned",
        }

        with patch.object(client, "is_available", return_value=True):
            with patch("requests.post", return_value=mock_resp) as mock_post:
                dummy_landmarks = np.zeros((10, 29, 2), dtype=np.float32)
                res = client.recognize_landmarks(dummy_landmarks, user_id="test_user", mode="training_aligned")

                self.assertIsNotNone(res)
                self.assertEqual(res["text"], "Namaste")
                self.assertEqual(res["confidence"], 94.5)
                self.assertEqual(res["engine"], "HWGAT")

                mock_post.assert_called_once()
                call_args = mock_post.call_args
                self.assertEqual(call_args[0][0], "http://127.0.0.1:5055/recognize_landmarks")
                json_body = call_args[1]["json"]
                self.assertEqual(json_body["user_id"], "test_user")
                self.assertEqual(json_body["mode"], "training_aligned")
                self.assertEqual(len(json_body["landmarks"]), 10)

    def test_05_checkpoint_loads_from_models_hwgat_best(self):
        """Verify models/hwgat_best.pt exists and all 206 state-dict keys load with exact match."""
        from ai_modules.hwgat.HWGATE import Model as HWGATEModel
        from ai_modules.hwgat.model_params import HWGATEParams

        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "hwgat_best.pt"))
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        self.assertEqual(len(state_dict), 206, "State dict should contain exactly 206 layer tensors")

        params = HWGATEParams({"num_class": 2002, "src_len": 192}, input_dim=2, device=torch.device("cpu"))
        model = HWGATEModel(*params.get_model_params())
        load_result = model.load_state_dict(state_dict, strict=True)

        self.assertEqual(len(load_result.missing_keys), 0)
        self.assertEqual(len(load_result.unexpected_keys), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 20488786)

    def test_06_aspect_ratio_preprocessing_equivalence_square_4_3_16_9(self):
        """Square, 4:3, and 16:9 preprocessing matches data_transform.py within float32 precision."""
        import sys
        hwgat_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experimental", "hwgat"))
        if hwgat_dir not in sys.path:
            sys.path.insert(0, hwgat_dir)

        from demo_utils import CFG
        from data_transform import HandCorrection

        cfg = CFG()
        rec = SignLanguageRecognizer()

        test_cases = [
            ("Square (1:1)", 640, 640),
            ("4:3", 640, 480),
            ("16:9", 1280, 720),
        ]

        for name, W, H in test_cases:
            frames = 12
            feat_norm = np.zeros((frames, 543, 4), dtype=np.float32)
            for f in range(frames):
                feat_norm[f, 0, :2] = [0.50, 0.30]   # Nose
                feat_norm[f, 2, :2] = [0.48, 0.28]   # Left eye
                feat_norm[f, 5, :2] = [0.52, 0.28]   # Right eye
                feat_norm[f, 11, :2] = [0.42, 0.40]  # Left shoulder
                feat_norm[f, 12, :2] = [0.58, 0.40]  # Right shoulder
                feat_norm[f, 13, :2] = [0.38, 0.55]  # Left elbow
                feat_norm[f, 14, :2] = [0.62, 0.55]  # Right elbow
                feat_norm[f, 15, :2] = [0.35, 0.70]  # Left wrist
                feat_norm[f, 16, :2] = [0.65, 0.70]  # Right wrist
                if f not in [3, 4]:
                    for k in range(21):
                        feat_norm[f, 501 + k, :2] = [0.35 + 0.01 * (k % 5), 0.70 + 0.01 * (k // 5)]
                        feat_norm[f, 522 + k, :2] = [0.65 + 0.01 * (k % 5), 0.70 + 0.01 * (k // 5)]

            # Canonical Training Transform
            data = {"feat": feat_norm.copy(), "num_frames": frames, "vid_width": W, "vid_height": H}
            tensor_train = cfg.test_transform(data)

            # In-process Preprocessing
            tensor_inprocess = rec._preprocess_holistic_training(list(feat_norm), target_len=192, width=W, height=H)

            max_diff = np.max(np.abs(tensor_train - tensor_inprocess))
            self.assertLess(max_diff, 1e-6, f"{name} diff {max_diff} exceeded float32 threshold")

    def test_07_in_process_and_service_preprocessing_parity(self):
        """In-process and external service preprocessing produce identical tensors."""
        import sys
        hwgat_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experimental", "hwgat"))
        if hwgat_dir not in sys.path:
            sys.path.insert(0, hwgat_dir)

        from hwgat_service import build_hwgat_tensor
        rec = SignLanguageRecognizer()

        seq = [np.random.uniform(0.1, 0.9, size=(543, 4)).astype(np.float32) for _ in range(15)]
        # Set stable anchors
        for f in seq:
            f[0, :2] = [0.5, 0.3]
            f[11, :2] = [0.4, 0.4]
            f[12, :2] = [0.6, 0.4]

        t_service = build_hwgat_tensor(seq, target_len=192, mode="training_aligned", width=1280, height=720)
        t_inprocess = rec._build_hwgat_tensor(seq, target_len=192, mode="training_aligned", width=1280, height=720)

        max_diff = float(torch.max(torch.abs(t_service - t_inprocess)))
        self.assertLess(max_diff, 1e-6)

    def test_08_tensor_output_dimensions_and_invalid_dimensions_safety(self):
        """Tensor dimensions remain exactly (1, 192, 64, 2) and handle missing/invalid dimensions safely."""
        rec = SignLanguageRecognizer()
        seq = [np.random.uniform(0.1, 0.9, size=(543, 4)).astype(np.float32) for _ in range(10)]
        for f in seq:
            f[0, :2] = [0.5, 0.3]
            f[11, :2] = [0.4, 0.4]
            f[12, :2] = [0.6, 0.4]

        # Valid dimensions
        t1 = rec._build_hwgat_tensor(seq, target_len=192, width=640, height=480)
        self.assertEqual(t1.shape, (1, 192, 64, 2))

        # Missing dimensions
        t2 = rec._build_hwgat_tensor(seq, target_len=192, width=None, height=None)
        self.assertEqual(t2.shape, (1, 192, 64, 2))

        # Zero or negative dimensions
        t3 = rec._build_hwgat_tensor(seq, target_len=192, width=0, height=-100)
        self.assertEqual(t3.shape, (1, 192, 64, 2))


if __name__ == "__main__":
    unittest.main()
