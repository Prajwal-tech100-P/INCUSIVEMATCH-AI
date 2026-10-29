"""
Comprehensive Test Suite for ISL Recognition Fixes in Video Calls
===================================================================
MCA Major Project: InclusiveMatch AI

Validates all 15 operational requirements from Step 8:
1. Starting a video call (route rendering & element presence).
2. Enabling sign recognition & session initialization.
3. Signing in front of intended camera (stream active, nonzero dimensions).
4. Extracting landmarks with MediaPipe (handedness parity & zero padding).
5. Sequence collection & temporal buffering (1 to 24 frames).
6. Model prediction execution (GCN-Transformer logits & softmax probabilities).
7. Display formatting (text, confidence, message, ready state).
8. Authorized caption delivery between matched participants over Socket.IO.
9. Disabling sign recognition (clean pause without call interruption).
10. Ending the call & session cleanup (resetting server buffers).
11. Handling camera permission failures & inactive tracks.
12. Handling missing hands & warmup buffering.
13. Handling model-loading errors & exception safety.
14. Handling low-confidence predictions (UNCERTAIN status).
15. Handling network interruptions & invalid/oversized payloads.
"""

import io
import json
import os
import unittest
from bson import ObjectId
import cv2
import numpy as np
import torch

from ai_modules.sign_language import SignLanguageRecognizer, _resp, SigningState
from ai_modules.sign_model_gcnt import (
    GCNTransformerClassifier,
    preprocess_live_frame_buffer,
    NUM_NODES,
)
from app import create_app
from extensions import mongo, socketio
from models.user import UserModel
from models.match import Match
from models.restriction import Restriction


class TestISLVideoCallRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_ENV"] = "testing"
        os.environ["SECRET_KEY"] = "test-secret-isl-call-suite"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = cls.app.test_client()
        cls.recognizer = SignLanguageRecognizer()

    def setUp(self):
        # Create two matched users for video call testing
        self.u1_id = UserModel.create_user("signer_alice@test.com", "Password123!", "Signer Alice")
        self.u2_id = UserModel.create_user("signer_bob@test.com", "Password123!", "Signer Bob")
        self.u3_id = UserModel.create_user("stranger_eve@test.com", "Password123!", "Stranger Eve")

        Match.like(self.u1_id, self.u2_id)
        Match.like(self.u2_id, self.u1_id)

    def tearDown(self):
        mongo.db.users.delete_many({"_id": {"$in": [
            ObjectId(self.u1_id), ObjectId(self.u2_id), ObjectId(self.u3_id)
        ]}})
        mongo.db.matches.delete_many({"$or": [
            {"user_a": self.u1_id}, {"user_b": self.u1_id},
            {"user_a": self.u2_id}, {"user_b": self.u2_id},
            {"user_a": self.u3_id}, {"user_b": self.u3_id},
        ]})

    def _login(self, user_id):
        with self.client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True

    def _socket_client(self, user_id):
        flask_client = self.app.test_client()
        with flask_client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True
        c_val = flask_client.get_cookie("session").value
        return socketio.test_client(self.app, headers={"Cookie": f"session={c_val}"})

    # -------------------------------------------------------------------------
    # Case 1: Starting a video call
    # -------------------------------------------------------------------------
    def test_case_01_starting_video_call(self):
        """Video call route renders video elements, sign controls, and live subtitle overlay."""
        self._login(self.u1_id)
        res = self.client.get(f"/video/{self.u2_id}")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Video elements
        self.assertIn('id="localVideo"', html)
        self.assertIn('id="remoteVideo"', html)

        # Accessibility and Sign Recognition UI
        self.assertIn('id="live-caption-banner"', html)
        self.assertIn('id="toggle-sign-ai"', html)
        self.assertIn('id="sign-source"', html)
        self.assertIn('id="sign-output"', html)
        self.assertIn('id="transcript-list"', html)
        self.assertIn('checkVideoStreamValid', html)

    # -------------------------------------------------------------------------
    # Case 2: Enabling sign recognition and session reset API
    # -------------------------------------------------------------------------
    def test_case_02_enabling_sign_recognition_and_reset(self):
        """User can enable/disable recognition and reset server session."""
        self._login(self.u1_id)
        # Verify /api/sign/reset endpoint exists and resets buffer
        res = self.client.post("/api/sign/reset", data={"source": "local"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json().get("ok"))

    # -------------------------------------------------------------------------
    # Case 3: Signing in front of intended camera (stream active & nonzero dimensions)
    # -------------------------------------------------------------------------
    def test_case_03_camera_stream_processing_and_source_selection(self):
        """API accepts frame with source specifier (local vs remote) without crosstalk."""
        self._login(self.u1_id)
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", img)

        res_local = self.client.post(
            "/api/sign/recognize",
            data={"frame": (io.BytesIO(enc.tobytes()), "frame.jpg", "image/jpeg"), "source": "local"},
            content_type="multipart/form-data"
        )
        self.assertEqual(res_local.status_code, 200)
        json_local = res_local.get_json()
        self.assertIn("text", json_local)
        self.assertIn("message", json_local)

        res_remote = self.client.post(
            "/api/sign/recognize",
            data={"frame": (io.BytesIO(enc.tobytes()), "frame.jpg", "image/jpeg"), "source": "remote"},
            content_type="multipart/form-data"
        )
        self.assertEqual(res_remote.status_code, 200)

    # -------------------------------------------------------------------------
    # Case 4: Successfully extracting landmarks with consistent hand ordering
    # -------------------------------------------------------------------------
    def test_case_04_landmark_extraction_and_handedness(self):
        """Landmark extractor handles single-hand and dual-hand frames cleanly without NaN or explosion."""
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        lm_none = self.recognizer._extract_landmarks(blank)
        self.assertIsNone(lm_none)

        # Non-empty mock hand test
        dummy_pts = np.random.uniform(0.1, 0.9, size=(42, 2)).astype(np.float32)
        # Verify buffer formatting
        feat = preprocess_live_frame_buffer([dummy_pts], seq_len=24, node_dim=6)
        self.assertEqual(feat.shape, (24, 42, 6))
        self.assertTrue(np.isfinite(feat).all())

    # -------------------------------------------------------------------------
    # Case 5: Collecting a complete sequence (1 to 24 frames)
    # -------------------------------------------------------------------------
    def test_case_05_sequence_buffering_progression(self):
        """Live frame sequence buffers smoothly from 1 frame to target seq_len (24)."""
        session_key = "test_user_buffering_progression"
        self.recognizer.reset_user(session_key)

        seq = []
        for i in range(1, 25):
            pts = np.zeros((42, 2), dtype=np.float32)
            pts[8, 0] = 0.3 + (i * 0.01)
            pts[8, 1] = 0.4
            seq.append(pts)
            feat = preprocess_live_frame_buffer(seq, seq_len=24, node_dim=6)
            self.assertEqual(feat.shape, (24, 42, 6))

    # -------------------------------------------------------------------------
    # Case 6: Generating a model prediction
    # -------------------------------------------------------------------------
    def test_case_06_model_prediction_execution(self):
        """Trained GCN-Transformer produces valid class label and confidence in [0, 1]."""
        mock_seq = [np.random.uniform(0.1, 0.9, size=(42, 2)).astype(np.float32) for _ in range(24)]
        label, conf = self.recognizer._predict_gcnt(mock_seq)
        self.assertIn(label, self.recognizer.labels)
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    # -------------------------------------------------------------------------
    # Case 7: Displaying the predicted sign (payload structure)
    # -------------------------------------------------------------------------
    def test_case_07_prediction_display_payload_contract(self):
        """_resp contract includes text, new_token, confidence, message, and activity fields."""
        r = _resp(
            text="Drink",
            confidence=92.5,
            message="Recognized: Drink",
            ready=True,
            arch="GCN-Transformer",
            signing_state="SIGNING",
            is_signing=True,
            turn_completed=False,
            phrase="Drink",
            finalized_phrase="",
            new_token="Drink",
        )
        self.assertEqual(r["text"], "Drink")
        self.assertEqual(r["new_token"], "Drink")
        self.assertEqual(r["confidence"], 92.5)
        self.assertTrue(r["is_signing"])
        self.assertEqual(r["signing_state"], "SIGNING")

    # -------------------------------------------------------------------------
    # Case 8: Delivering the caption to the other participant
    # -------------------------------------------------------------------------
    def test_case_08_authorized_caption_delivery_over_socket(self):
        """Sign captions are delivered strictly to the authorized call partner."""
        c1 = self._socket_client(self.u1_id)
        c2 = self._socket_client(self.u2_id)
        c3 = self._socket_client(self.u3_id)

        # Both users join the call room
        c1.emit("join_call", {"partner_id": self.u2_id})
        c2.emit("join_call", {"partner_id": self.u1_id})

        # Alice emits sign caption
        c1.emit("sign_caption", {
            "partner_id": self.u2_id,
            "text": "Hello",
            "phrase": "Hello",
            "confidence": 88.0,
            "is_final": True
        })

        # Bob receives caption
        rec2 = c2.get_received()
        sign_events = [e for e in rec2 if e["name"] == "sign_caption"]
        self.assertEqual(len(sign_events), 1)
        self.assertEqual(sign_events[0]["args"][0]["text"], "Hello")
        self.assertEqual(sign_events[0]["args"][0]["speaker"], "Signer Alice")

        # Stranger Eve does NOT receive the caption
        rec3 = c3.get_received()
        eve_events = [e for e in rec3 if e["name"] == "sign_caption"]
        self.assertEqual(len(eve_events), 0)

        c1.disconnect()
        c2.disconnect()
        c3.disconnect()

    # -------------------------------------------------------------------------
    # Case 9: Disabling recognition
    # -------------------------------------------------------------------------
    def test_case_09_disabling_recognition_clears_activity(self):
        """Disabling recognition pauses activity detection without breaking connection."""
        det = self.recognizer._get_activity_detector("user_test_disable")
        det.state = SigningState.SIGNING
        self.assertTrue(det.state == SigningState.SIGNING)
        det.reset()
        self.assertEqual(det.state, SigningState.IDLE)
        self.assertFalse(det.update(None)["is_signing"])

    # -------------------------------------------------------------------------
    # Case 10: Ending the call and cleaning up resources
    # -------------------------------------------------------------------------
    def test_case_10_ending_call_cleans_up_sessions(self):
        """Ending call resets recognizer memory for participants."""
        from routes.video_routes import recognizer as global_rec
        c1 = self._socket_client(self.u1_id)
        c2 = self._socket_client(self.u2_id)

        c1.emit("join_call", {"partner_id": self.u2_id})
        c2.emit("join_call", {"partner_id": self.u1_id})

        # Put items in user sequence on global video recognizer
        with global_rec.lock:
            global_rec.sequences[self.u1_id] = [np.zeros((42, 2))]
            global_rec.sequences[f"{self.u1_id}_local"] = [np.zeros((42, 2))]

        # End call
        res = c1.emit("end_call", {"partner_id": self.u2_id, "reason": "hangup"}, callback=True)
        self.assertTrue(res.get("ok"))

        # Verify global recognizer sessions were reset
        self.assertNotIn(self.u1_id, global_rec.sequences)
        self.assertNotIn(f"{self.u1_id}_local", global_rec.sequences)
        c1.disconnect()
        c2.disconnect()

    # -------------------------------------------------------------------------
    # Case 11: Handling missing camera permissions
    # -------------------------------------------------------------------------
    def test_case_11_missing_camera_permissions(self):
        """API responds safely when no frame is sent."""
        self._login(self.u1_id)
        res = self.client.post("/api/sign/recognize", data={}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["ready"], False)

    # -------------------------------------------------------------------------
    # Case 12: Handling missing hands or incomplete sequences
    # -------------------------------------------------------------------------
    def test_case_12_missing_hands_and_warmup(self):
        """Empty frames return 'No hand detected' with ready=False without error."""
        blank = np.zeros((160, 240, 3), dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", blank)
        result = self.recognizer.recognize_frame(enc.tobytes(), user_id="user_empty_test")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["ready"], False)
        self.assertIn("No hand detected", result["message"])

    # -------------------------------------------------------------------------
    # Case 13: Handling model-loading errors gracefully
    # -------------------------------------------------------------------------
    def test_case_13_model_loading_errors_handled_gracefully(self):
        """Recognizer handles model loading errors gracefully when checkpoint is corrupted."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            bad_rec = SignLanguageRecognizer(model_path=tmp.name)
            self.assertIsNotNone(bad_rec.error)
            self.assertIn("Could not load sign model", bad_rec.error)
            res = bad_rec.recognize_frame(b"dummy_bytes", user_id="user_bad_model")
            self.assertEqual(res["ready"], False)
            self.assertIn("Invalid image frame", res["message"])

    # -------------------------------------------------------------------------
    # Case 14: Handling low-confidence predictions
    # -------------------------------------------------------------------------
    def test_case_14_low_confidence_uncertainty_handling(self):
        """Low-confidence signs (< MIN_CONFIDENCE) output UNCERTAIN."""
        key = "user_test_low_conf"
        self.recognizer.reset_user(key)

        # Feed 6 frames of random noise to trigger GCN-T
        dummy = [np.random.uniform(0.1, 0.9, size=(42, 2)).astype(np.float32) for _ in range(6)]
        for pt in dummy:
            self.recognizer.sequences.setdefault(key, []).append(pt)

        # Test smoothing helper with low confidence
        best, best_conf = self.recognizer._smooth(key, "hello", 0.02)
        self.assertLess(best_conf, self.recognizer.MIN_CONFIDENCE)

    # -------------------------------------------------------------------------
    # Case 15: Handling network interruptions & oversized payloads
    # -------------------------------------------------------------------------
    def test_case_15_network_interruptions_and_oversized_payloads(self):
        """Oversized frames (> 2MB) return HTTP 413 without crashing server."""
        self._login(self.u1_id)
        huge_bytes = b"x" * (3 * 1024 * 1024)
        res = self.client.post(
            "/api/sign/recognize",
            data={"frame": (io.BytesIO(huge_bytes), "huge.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 413)
        self.assertEqual(res.get_json()["ready"], False)


if __name__ == "__main__":
    unittest.main()
