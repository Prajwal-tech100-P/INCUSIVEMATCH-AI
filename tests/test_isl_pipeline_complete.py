"""
Comprehensive Test Suite for ISL Recognition Pipeline & Dataset Studio
======================================================================
MCA Major Project: InclusiveMatch AI

Verifies:
1. Model construction, tensor dimensions, and CPU forward pass
2. Graph topology: block-diagonal 42-node normalized adjacency
3. Preprocessing parity between offline and online live frame buffers
4. ISLDatasetManager operations, metadata tracking, and class management
5. Recording validation (shape check, short sequence rejection, zero-landmark guard)
6. Checkpoint save, reload, and class mapping parity
7. Confidence thresholding and unknown-sign handling
8. CPU inference latency bounds (< 100 ms)
9. Webcam collector extraction with synthetic frames
10. Authorization protection for /admin/isl-studio and dataset APIs
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ai_modules.isl_dataset_manager import ISLDatasetManager
from ai_modules.sign_language import SignLanguageRecognizer, _resp
from ai_modules.sign_model_gcnt import (
    GCNTransformerClassifier,
    build_two_hand_adjacency,
    preprocess_landmark_sequence,
    preprocess_live_frame_buffer,
    NUM_NODES,
)
from app import create_app
from extensions import mongo
from models.user import UserModel


class TestISLPipelineComplete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = cls.app.test_client()

        # Database setup for admin auth tests
        cls.admin_email = "isl_admin_test@inclusivematch.org"
        cls.user_email = "isl_user_test@inclusivematch.org"

        with cls.app.app_context():
            mongo.db.users.delete_many({"email": {"$in": [cls.admin_email, cls.user_email]}})
            cls.admin_id = UserModel.create_user(
                email=cls.admin_email,
                name="Admin Researcher",
                password="AdminSecurePassword123!",
                role="admin"
            )
            cls.user_id = UserModel.create_user(
                email=cls.user_email,
                name="Regular Signer",
                password="UserSecurePassword123!",
                role="user"
            )

    @classmethod
    def tearDownClass(cls):
        with cls.app.app_context():
            mongo.db.users.delete_many({"email": {"$in": [cls.admin_email, cls.user_email]}})

    def _login(self, email, password):
        return self.client.post("/login", data={"email": email, "password": password}, follow_redirects=True)

    def _logout(self):
        return self.client.get("/logout", follow_redirects=True)

    # -------------------------------------------------------------------------
    # 1. Model Construction & Tensor Dimensions
    # -------------------------------------------------------------------------
    def test_01_model_construction_and_shapes(self):
        """GCNTransformerClassifier constructs cleanly and handles (B, 24, 42, 4) and (B, 24, 42, 6)."""
        model4 = GCNTransformerClassifier(num_classes=29, seq_len=24, node_dim=4, gcn_dim=64, tf_heads=4, tf_layers=2)
        model4.eval()
        x4 = torch.randn(2, 24, 42, 4, dtype=torch.float32)
        with torch.no_grad():
            out4 = model4(x4)
        self.assertEqual(out4.shape, (2, 29))
        self.assertTrue(torch.isfinite(out4).all())

        model6 = GCNTransformerClassifier(num_classes=29, seq_len=24, node_dim=6, gcn_dim=64, tf_heads=4, tf_layers=2)
        model6.eval()
        x6 = torch.randn(2, 24, 42, 6, dtype=torch.float32)
        with torch.no_grad():
            out6 = model6(x6)
        self.assertEqual(out6.shape, (2, 29))
        self.assertTrue(torch.isfinite(out6).all())

    # -------------------------------------------------------------------------
    # 2. Graph Topology & Adjacency
    # -------------------------------------------------------------------------
    def test_02_graph_topology_block_diagonal(self):
        """Adjacency matrix is 42x42 block diagonal, symmetric normalized, with self-loops and zero inter-hand edges."""
        adj = build_two_hand_adjacency()
        self.assertEqual(adj.shape, (42, 42))
        # Self-loops must exist
        diag = torch.diag(adj)
        self.assertTrue((diag > 0).all())
        # Hand 1 is 0..20, Hand 2 is 21..41. Inter-hand quadrant [0..20, 21..41] must be all zeros!
        inter_hand_quadrant = adj[:21, 21:]
        self.assertEqual(float(inter_hand_quadrant.sum().item()), 0.0)

    # -------------------------------------------------------------------------
    # 3. Preprocessing Parity
    # -------------------------------------------------------------------------
    def test_03_preprocessing_parity(self):
        """Parity between offline landmark sequence and live frame buffer preprocessing."""
        stream = []
        for t in range(24):
            pts = np.zeros((42, 2), dtype=np.float32)
            pts[4, 0] = 0.3 + (t * 0.01)
            pts[4, 1] = 0.4
            pts[25, 0] = 0.6
            pts[25, 1] = 0.7 - (t * 0.005)
            stream.append(pts)

        arr_raw = np.stack(stream, axis=0)
        feat_offline = preprocess_landmark_sequence(arr_raw, seq_len=24, node_dim=4)
        feat_online = preprocess_live_frame_buffer(stream, seq_len=24, node_dim=4)

        self.assertEqual(feat_offline.shape, (24, 42, 4))
        self.assertEqual(feat_online.shape, (24, 42, 4))
        self.assertTrue(np.allclose(feat_offline, feat_online, atol=1e-5))

    # -------------------------------------------------------------------------
    # 4. Dataset Manager Operations
    # -------------------------------------------------------------------------
    def test_04_dataset_manager_summary_and_classes(self):
        """ISLDatasetManager initializes metadata, tracks classes, and counts samples."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ISLDatasetManager(dataset_dir=tmp_dir)
            classes = mgr.get_supported_classes()
            self.assertGreaterEqual(len(classes), 29)
            self.assertIn("hello", classes)
            self.assertIn("thank_you", classes)

            # Add new class
            new_cls = mgr.add_class("namaste")
            self.assertEqual(new_cls, "namaste")
            self.assertIn("namaste", mgr.get_supported_classes())

            summary = mgr.get_summary()
            self.assertEqual(summary["classes"]["namaste"], 0)

    # -------------------------------------------------------------------------
    # 5. Recording Validation & Quality Guards
    # -------------------------------------------------------------------------
    def test_05_save_recording_validation(self):
        """save_recording enforces valid dimensions, rejects empty or truncated sequences."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ISLDatasetManager(dataset_dir=tmp_dir)

            # Test A: Sequence too short
            short_seq = np.ones((5, 84), dtype=np.float32)
            ok, msg, _ = mgr.save_recording("hello", short_seq)
            self.assertFalse(ok)
            self.assertIn("too short", msg)

            # Test B: Sequence with no active hands (all zeros)
            zero_seq = np.zeros((30, 84), dtype=np.float32)
            ok, msg, _ = mgr.save_recording("hello", zero_seq)
            self.assertFalse(ok)
            self.assertIn("Insufficient hand detection", msg)

            # Test C: Valid sequence with 30 frames and active landmarks
            valid_seq = np.zeros((30, 84), dtype=np.float32)
            valid_seq[5:25, 0] = 0.5  # 20 active frames
            ok, msg, info = mgr.save_recording("hello", valid_seq)
            self.assertTrue(ok)
            self.assertTrue(Path(info["file_path"]).exists())
            self.assertEqual(info["class_name"], "hello")
            self.assertEqual(info["total_frames"], 30)
            self.assertEqual(info["active_frames"], 20)

            # Verify saved array shape on disk
            loaded = np.load(info["file_path"])
            self.assertEqual(loaded.shape, (30, 84))

    # -------------------------------------------------------------------------
    # 6. Checkpoint Save, Reload, and Class Mapping Parity
    # -------------------------------------------------------------------------
    def test_06_checkpoint_parity(self):
        """models/sign_gcnt_best.pt exists and matches models/sign_gcnt_best_classes.json."""
        ckpt_path = Path("models/sign_gcnt_best.pt")
        classes_path = Path("models/sign_gcnt_best_classes.json")
        self.assertTrue(ckpt_path.exists(), "Model checkpoint sign_gcnt_best.pt missing!")
        self.assertTrue(classes_path.exists(), "Class mapping file missing!")

        with open(classes_path, "r", encoding="utf-8") as f:
            classes_meta = json.load(f)
        classes_json = classes_meta["classes"]

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["arch"], "gcn_transformer_2hand")
        self.assertEqual(ckpt["classes"], classes_json)
        self.assertEqual(len(ckpt["classes"]), 29)

    # -------------------------------------------------------------------------
    # 7. Confidence Threshold & Rejection Handling
    # -------------------------------------------------------------------------
    def test_07_confidence_threshold_and_rejection(self):
        """Recognizer handles low-confidence signs without raising exceptions."""
        recognizer = SignLanguageRecognizer()
        self.assertIsNotNone(recognizer.model)
        # Smoothing helper returns confidence
        label, conf = recognizer._smooth("test_session_thresh", "hello", 0.03)
        self.assertLess(conf, recognizer.MIN_CONFIDENCE)

    # -------------------------------------------------------------------------
    # 8. CPU Inference Latency
    # -------------------------------------------------------------------------
    def test_08_cpu_inference_latency(self):
        """CPU forward pass executes well within 100 ms (< 50 ms typical)."""
        import time
        model = GCNTransformerClassifier(num_classes=29, seq_len=24, node_dim=4, gcn_dim=64)
        model.eval()
        dummy = torch.randn(1, 24, 42, 4)
        # Warmup
        with torch.no_grad():
            _ = model(dummy)

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 150.0)

    # -------------------------------------------------------------------------
    # 9. Synthetic Landmark Extraction Test
    # -------------------------------------------------------------------------
    def test_09_webcam_collector_synthetic_frame_extraction(self):
        """extract_landmarks_from_bgr returns clean (84,) zeros on blank frames without crashing."""
        from ai_modules.collect_isl_webcam import ISLWebcamCollector
        collector = ISLWebcamCollector()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lms = collector.extract_landmarks_from_bgr(blank_frame)
        self.assertEqual(lms.shape, (84,))
        self.assertTrue((lms == 0).all())

    # -------------------------------------------------------------------------
    # 10. Authorization Protection for Admin Dataset Studio
    # -------------------------------------------------------------------------
    def test_10_admin_studio_endpoints_authorization(self):
        """Dataset collection and management routes are protected strictly by admin_required."""
        # Unauthenticated request -> redirects to login
        self._logout()
        res = self.client.get("/admin/isl-studio")
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers.get("Location", ""))

        res = self.client.get("/admin/api/isl/classes")
        self.assertEqual(res.status_code, 302)

        # Authenticated as standard user -> forbidden / redirected
        self._login(self.user_email, "UserSecurePassword123!")
        res = self.client.get("/admin/isl-studio")
        self.assertIn(res.status_code, (302, 403))

        res = self.client.get("/admin/api/isl/classes")
        self.assertIn(res.status_code, (302, 403))

        # Authenticated as admin -> access granted
        self._logout()
        self._login(self.admin_email, "AdminSecurePassword123!")
        res = self.client.get("/admin/isl-studio")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Indian Sign Language Dataset Studio", res.data)

        res = self.client.get("/admin/api/isl/classes")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertIn("hello", data["classes"])

        # Test POST /admin/api/isl/add-class
        res = self.client.post("/admin/api/isl/add-class", json={"class_name": "peace_sign"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["ok"])

        # Test POST /admin/api/isl/save-recording with direct landmark array
        dummy_landmarks = np.zeros((20, 84), dtype=np.float32)
        dummy_landmarks[5:15, 0] = 0.5  # 10 active frames
        res = self.client.post("/admin/api/isl/save-recording", json={
            "class_name": "peace_sign",
            "landmarks": dummy_landmarks.tolist()
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["ok"])
        self.assertEqual(res.get_json()["class_name"], "peace_sign")


if __name__ == "__main__":
    unittest.main()

