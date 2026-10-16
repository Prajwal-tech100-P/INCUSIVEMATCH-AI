"""
Automated Test Suite for Improved ISL Recognition Checkpoint & Preprocessing Contract
======================================================================================
MCA Major Project: InclusiveMatch AI

Verifies:
1. Checkpoint loading integrity (models/sign_gcnt_best.pt)
2. Label mapping parity with models/sign_gcnt_best_classes.json
3. Input tensor dimensions (Batch, 24, 42, 4)
4. Preprocessing consistency (offline vs online streaming buffer)
5. Prediction output validity (softmax, finite logits, confidence bounds)
6. Invalid and missing landmark handling (all-zero, single-hand, NaN guard)
7. Model loading fallback behavior on invalid checkpoint paths
"""

import json
import unittest
from pathlib import Path
import numpy as np
import torch

from ai_modules.sign_language import SignLanguageRecognizer, _resp
from ai_modules.sign_model_gcnt import (
    GCNTransformerClassifier,
    preprocess_landmark_sequence,
    preprocess_live_frame_buffer,
    NUM_NODES,
)


class TestImprovedSignEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = Path("models/sign_gcnt_best.pt")
        cls.json_path = Path("models/sign_gcnt_best_classes.json")
        cls.recognizer = SignLanguageRecognizer(model_path=str(cls.ckpt_path))

    def test_checkpoint_exists_and_loads(self):
        """Verify models/sign_gcnt_best.pt exists and loads cleanly."""
        self.assertTrue(self.ckpt_path.exists(), "models/sign_gcnt_best.pt does not exist!")
        self.assertIsNotNone(self.recognizer.model)
        self.assertTrue(self.recognizer.is_2hand)
        self.assertIsNone(self.recognizer.error)

    def test_label_mapping_parity(self):
        """Verify label mapping in checkpoint matches sign_gcnt_best_classes.json."""
        self.assertTrue(self.json_path.exists(), "models/sign_gcnt_best_classes.json missing!")
        with open(self.json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        classes_from_json = meta["classes"]
        self.assertEqual(len(self.recognizer.labels), 29)
        self.assertEqual(self.recognizer.labels, classes_from_json)

    def test_input_dimensions_and_forward(self):
        """Verify GCNTransformerClassifier accepts (B, 24, 42, 4) and outputs (B, 29)."""
        model = self.recognizer.model
        model.eval()
        for b_size in [1, 4]:
            x = torch.randn(b_size, 24, 42, 4, dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out.shape, (b_size, 29))
            self.assertTrue(torch.isfinite(out).all().item())

    def test_preprocessing_consistency(self):
        """Verify parity between offline preprocess_landmark_sequence and online preprocess_live_frame_buffer."""
        # Create a deterministic mock 24-frame movement sequence
        raw_stream = []
        for t in range(24):
            frame_landmarks = np.zeros((42, 2), dtype=np.float32)
            # Move index finger tip (joint 8) linearly
            frame_landmarks[8, 0] = 0.2 + (t * 0.01)
            frame_landmarks[8, 1] = 0.5 - (t * 0.005)
            # Hand 2 wrist (joint 21) stationary
            frame_landmarks[21, 0] = 0.7
            frame_landmarks[21, 1] = 0.6
            raw_stream.append(frame_landmarks)

        raw_offline_arr = np.stack(raw_stream, axis=0)  # (24, 42, 2)
        offline_feat = preprocess_landmark_sequence(raw_offline_arr, seq_len=24)
        online_feat = preprocess_live_frame_buffer(raw_stream, seq_len=24)

        self.assertEqual(offline_feat.shape, (24, 42, 4))
        self.assertEqual(online_feat.shape, (24, 42, 4))
        # Coordinates and velocities must match
        self.assertTrue(np.allclose(offline_feat, online_feat, atol=1e-5))

    def test_prediction_output_and_confidence_bounds(self):
        """Verify _predict_gcnt returns a valid class label and confidence in [0, 1]."""
        mock_seq = [np.random.uniform(0.1, 0.9, size=(42, 2)).astype(np.float32) for _ in range(20)]
        label, conf = self.recognizer._predict_gcnt(mock_seq)
        self.assertIn(label, self.recognizer.labels)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_invalid_and_empty_landmarks(self):
        """Verify all-zero landmarks produce clean, zero-padded features without NaN or explosion."""
        all_zeros = [np.zeros((42, 2), dtype=np.float32) for _ in range(15)]
        feat = preprocess_live_frame_buffer(all_zeros, seq_len=24)
        self.assertEqual(feat.shape, (24, 42, 4))
        self.assertTrue(np.all(feat == 0.0))
        self.assertTrue(np.isfinite(feat).all())

        # Feed to model
        x = torch.from_numpy(feat).unsqueeze(0).float()
        with torch.no_grad():
            out = self.recognizer.model(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_model_loading_fallback(self):
        """Verify recognizer gracefully falls back to DEFAULT_PATHS when an invalid path is passed without raising exceptions."""
        rec = SignLanguageRecognizer(model_path="non_existent_model.pt")
        # Should gracefully fall back to the default available model checkpoint
        self.assertIsNotNone(rec.model)
        self.assertEqual(rec.arch, "gcn_transformer")
        self.assertIsNone(rec.error)

        # When explicitly pointing to an empty / corrupted dummy file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            rec_corrupt = SignLanguageRecognizer(model_path=tmp.name)
            self.assertIsNotNone(rec_corrupt.error)
            self.assertIn("Could not load sign model", rec_corrupt.error)
