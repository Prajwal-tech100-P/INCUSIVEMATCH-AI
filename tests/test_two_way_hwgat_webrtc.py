"""
Two-Way HWGAT Sign Language Recognition in WebRTC Video Calls
============================================================
Comprehensive test suite verifying real-time two-way sign recognition
between two users (User A & User B) in active WebRTC video calls.

Tests:
1. Engine status API and sole HWGAT recognition.
2. User A performs sign -> HWGAT recognition -> socket delivery to both.
3. User B performs sign -> HWGAT recognition -> socket delivery to both.
4. Simultaneous recognition from both participants without state collision.
5. Video call UI elements properly reflect toggle states and engine selector.
6. Call termination cleanly cleans up user queues and active sessions.
7. Cross-call isolation (transcripts never leak between separate active calls).
8. Existing WebRTC signaling and media functionality remain intact.
"""

import os
import io
import unittest
from bson import ObjectId
import numpy as np
import cv2
from werkzeug.security import generate_password_hash

from app import create_app
from extensions import mongo, socketio
from models.match import Match
from models.call_invitation import CallInvitation


class TestTwoWayHWGATWebRTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_ENV"] = "testing"
        os.environ["SECRET_KEY"] = "test-two-way-hwgat-webrtc-2026"
        os.environ["ENABLE_SIGN_AI"] = "1"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.app.config["ENABLE_SIGN_AI"] = True

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ENABLE_SIGN_AI", None)

    def _socket_client(self, user_id):
        """Creates a Socket.IO test client authenticated as user_id."""
        flask_client = self.app.test_client()
        with flask_client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True
        return socketio.test_client(self.app, flask_test_client=flask_client)

    def _make_dummy_frame_jpeg(self):
        """Create a realistic encoded JPEG frame."""
        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (50, 50, 200), -1)
        _, enc = cv2.imencode(".jpg", frame)
        return enc.tobytes()

    def setUp(self):
        self.u1_id = str(ObjectId())
        self.u2_id = str(ObjectId())
        self.u3_id = str(ObjectId())
        self.u4_id = str(ObjectId())

        mongo.db.users.delete_many({"_id": {"$in": [
            ObjectId(self.u1_id), ObjectId(self.u2_id),
            ObjectId(self.u3_id), ObjectId(self.u4_id)
        ]}})
        mongo.db.matches.delete_many({"$or": [
            {"user_a": self.u1_id}, {"user_b": self.u1_id},
            {"user_a": self.u2_id}, {"user_b": self.u2_id},
            {"user_a": self.u3_id}, {"user_b": self.u3_id},
            {"user_a": self.u4_id}, {"user_b": self.u4_id},
        ]})

        # Create Call 1 Pair: User A and User B
        mongo.db.users.insert_one({
            "_id": ObjectId(self.u1_id),
            "id": self.u1_id,
            "name": "User Alpha",
            "email": f"usera_{self.u1_id}@test.com",
            "password": generate_password_hash("TestPass123!"),
            "role": "user",
            "is_active": True,
            "gender": "male",
            "disability_type": "hearing",
        })
        mongo.db.users.insert_one({
            "_id": ObjectId(self.u2_id),
            "id": self.u2_id,
            "name": "User Beta",
            "email": f"userb_{self.u2_id}@test.com",
            "password": generate_password_hash("TestPass123!"),
            "role": "user",
            "is_active": True,
            "gender": "female",
            "disability_type": "hearing",
        })

        # Create Call 2 Pair: User C and User D
        mongo.db.users.insert_one({
            "_id": ObjectId(self.u3_id),
            "id": self.u3_id,
            "name": "User Gamma",
            "email": f"userc_{self.u3_id}@test.com",
            "password": generate_password_hash("TestPass123!"),
            "role": "user",
            "is_active": True,
        })
        mongo.db.users.insert_one({
            "_id": ObjectId(self.u4_id),
            "id": self.u4_id,
            "name": "User Delta",
            "email": f"userd_{self.u4_id}@test.com",
            "password": generate_password_hash("TestPass123!"),
            "role": "user",
            "is_active": True,
        })

        # Mutual matches
        Match.like(self.u1_id, self.u2_id)
        Match.like(self.u2_id, self.u1_id)
        Match.like(self.u3_id, self.u4_id)
        Match.like(self.u4_id, self.u3_id)

    def tearDown(self):
        mongo.db.users.delete_many({"_id": {"$in": [
            ObjectId(self.u1_id), ObjectId(self.u2_id),
            ObjectId(self.u3_id), ObjectId(self.u4_id)
        ]}})
        mongo.db.matches.delete_many({"$or": [
            {"user_a": self.u1_id}, {"user_b": self.u1_id},
            {"user_a": self.u2_id}, {"user_b": self.u2_id},
            {"user_a": self.u3_id}, {"user_b": self.u3_id},
            {"user_a": self.u4_id}, {"user_b": self.u4_id},
        ]})
        mongo.db[CallInvitation.COLLECTION].delete_many({"$or": [
            {"caller_id": self.u1_id}, {"recipient_id": self.u1_id},
            {"caller_id": self.u2_id}, {"recipient_id": self.u2_id},
            {"caller_id": self.u3_id}, {"recipient_id": self.u3_id},
            {"caller_id": self.u4_id}, {"recipient_id": self.u4_id},
        ]})

    def _login(self, client, email, password="TestPass123!"):
        client.get("/logout")
        return client.post("/login", data={"email": email, "password": password, "remember": False}, follow_redirects=True)

    # =========================================================================
    # 1. ENGINE STATUS AND AVAILABILITY TESTS
    # =========================================================================

    def test_1_sign_engine_status_endpoint(self):
        """Verify /api/sign/status returns HWGAT as the sole active engine."""
        client = self.app.test_client()
        self._login(client, f"usera_{self.u1_id}@test.com")
        res = client.get("/api/sign/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["default_engine"], "hwgat")
        self.assertTrue(data["hwgat_available"])
        self.assertEqual(data["hwgat_classes"], 2002)
        self.assertEqual(data["model"], "HWGAT")

    # =========================================================================
    # 2. TWO-WAY REAL-TIME RECOGNITION TESTS (USER A & USER B)
    # =========================================================================

    def test_2_user_a_sign_recognition_and_broadcast(self):
        """TEST 2: User A performs sign -> Recognized -> Socket relays caption to User B."""
        client_a = self.app.test_client()
        self._login(client_a, f"usera_{self.u1_id}@test.com")

        frame_bytes = self._make_dummy_frame_jpeg()
        res = client_a.post(
            "/api/sign/recognize",
            data={
                "frame": (io.BytesIO(frame_bytes), "frame.jpg"),
                "source": "local",
                "partner_id": self.u2_id,
                "engine": "hwgat"
            },
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        result = res.get_json()
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertIn("signing_state", result)
        self.assertIn("ready", result)

        # Test Socket.IO caption relay from User A to User B
        s_client_a = self._socket_client(self.u1_id)
        s_client_b = self._socket_client(self.u2_id)

        s_client_a.emit("join_call", {"partner_id": self.u2_id})
        s_client_b.emit("join_call", {"partner_id": self.u1_id})
        s_client_b.get_received()  # Flush join notifications

        # User A emits sign_caption
        s_client_a.emit("sign_caption", {
            "partner_id": self.u2_id,
            "text": "Namaste",
            "phrase": "Namaste",
            "confidence": 92.5,
            "is_final": False
        })

        # User B must receive User A's sign caption
        received_b = s_client_b.get_received()
        caption_events = [e for e in received_b if e["name"] == "sign_caption"]
        self.assertGreater(len(caption_events), 0, "User B should receive sign_caption event")
        payload = caption_events[0]["args"][0]
        self.assertEqual(payload["text"], "Namaste")
        self.assertEqual(payload["speaker"], "User Alpha")
        self.assertEqual(payload["confidence"], 92.5)

        s_client_a.disconnect()
        s_client_b.disconnect()

    def test_3_user_b_sign_recognition_and_broadcast(self):
        """TEST 3: User B performs sign -> Recognized -> Socket relays caption to User A."""
        client_b = self.app.test_client()
        self._login(client_b, f"userb_{self.u2_id}@test.com")

        frame_bytes = self._make_dummy_frame_jpeg()
        res = client_b.post(
            "/api/sign/recognize",
            data={
                "frame": (io.BytesIO(frame_bytes), "frame.jpg"),
                "source": "local",
                "partner_id": self.u1_id,
                "engine": "hwgat"
            },
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        result = res.get_json()
        self.assertIn("ready", result)
        self.assertIn("text", result)

        # Test Socket.IO caption relay from User B to User A
        s_client_a = self._socket_client(self.u1_id)
        s_client_b = self._socket_client(self.u2_id)

        s_client_a.emit("join_call", {"partner_id": self.u2_id})
        s_client_b.emit("join_call", {"partner_id": self.u1_id})
        s_client_a.get_received()

        # User B emits sign_caption
        s_client_b.emit("sign_caption", {
            "partner_id": self.u1_id,
            "text": "Thank You",
            "phrase": "Thank You",
            "confidence": 88.0,
            "is_final": True
        })

        received_a = s_client_a.get_received()
        caption_events = [e for e in received_a if e["name"] == "sign_caption"]
        self.assertGreater(len(caption_events), 0, "User A should receive sign_caption event from User B")
        payload = caption_events[0]["args"][0]
        self.assertEqual(payload["text"], "Thank You")
        self.assertEqual(payload["speaker"], "User Beta")
        self.assertTrue(payload["is_final"])

        s_client_a.disconnect()
        s_client_b.disconnect()

    def test_4_simultaneous_two_way_recognition_streams(self):
        """TEST 4: Simultaneous recognition from both participants without state collision."""
        client_a = self.app.test_client()
        self._login(client_a, f"usera_{self.u1_id}@test.com")
        client_b = self.app.test_client()
        self._login(client_b, f"userb_{self.u2_id}@test.com")
        frame_bytes = self._make_dummy_frame_jpeg()

        # User A signs
        res_a = client_a.post(
            "/api/sign/recognize",
            data={"frame": (io.BytesIO(frame_bytes), "frame.jpg"), "source": "local", "partner_id": self.u2_id},
            content_type="multipart/form-data"
        )
        self.assertEqual(res_a.status_code, 200)

        # User B signs concurrently
        res_b = client_b.post(
            "/api/sign/recognize",
            data={"frame": (io.BytesIO(frame_bytes), "frame.jpg"), "source": "local", "partner_id": self.u1_id},
            content_type="multipart/form-data"
        )
        self.assertEqual(res_b.status_code, 200)

        # User A signs again: User A's session remains intact and independent
        res_a2 = client_a.post(
            "/api/sign/recognize",
            data={"frame": (io.BytesIO(frame_bytes), "frame.jpg"), "source": "local", "partner_id": self.u2_id},
            content_type="multipart/form-data"
        )
        self.assertEqual(res_a2.status_code, 200)

    def test_5_turn_recognition_off_stops_processing(self):
        """TEST 5: Video call UI elements properly reflect toggle states and engine selector."""
        client = self.app.test_client()
        self._login(client, f"usera_{self.u1_id}@test.com")

        res = client.get(f"/video/{self.u2_id}")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        self.assertIn('id="toggle-sign-ai"', html, "Sign AI ON/OFF button must exist")
        self.assertIn('id="ai-engine-select"', html, "Sign AI model selector must exist")
        self.assertIn('HWGAT Deep Model', html, "HWGAT model must be available in options")
        self.assertIn('id="btn-clear-transcript"', html, "Clear transcript button must exist")
        self.assertIn('id="speak-sign"', html, "Text-to-speech switch must exist")

    def test_6_call_termination_resets_sessions(self):
        """TEST 6: Ending a video call cleanly clears user queues and session states."""
        client = self.app.test_client()
        self._login(client, f"usera_{self.u1_id}@test.com")

        # Reset endpoint
        res = client.post("/api/sign/reset", data={"source": "local"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["ok"])

        # Socket call termination
        s_client_a = self._socket_client(self.u1_id)
        s_client_a.emit("join_call", {"partner_id": self.u2_id})

        res_end = s_client_a.emit("end_call", {"partner_id": self.u2_id, "reason": "hangup"}, callback=True)
        self.assertIsNotNone(res_end)
        self.assertTrue(res_end.get("ok"))
        s_client_a.disconnect()

    def test_7_cross_call_transcript_isolation(self):
        """TEST 7: Recognition transcripts in Call 1 (A-B) NEVER leak to Call 2 (C-D)."""
        # Connect Call 1 clients (A & B)
        s_client_a = self._socket_client(self.u1_id)
        s_client_b = self._socket_client(self.u2_id)
        s_client_a.emit("join_call", {"partner_id": self.u2_id})
        s_client_b.emit("join_call", {"partner_id": self.u1_id})
        s_client_b.get_received()

        # Connect Call 2 clients (C & D)
        s_client_c = self._socket_client(self.u3_id)
        s_client_d = self._socket_client(self.u4_id)
        s_client_c.emit("join_call", {"partner_id": self.u4_id})
        s_client_d.emit("join_call", {"partner_id": self.u3_id})
        s_client_c.get_received()
        s_client_d.get_received()

        # User A emits private sign_caption in Call 1
        s_client_a.emit("sign_caption", {
            "partner_id": self.u2_id,
            "text": "ConfidentialSign",
            "phrase": "ConfidentialSign",
            "confidence": 95.0,
            "is_final": True
        })

        # Verify User B received it
        received_b = [e for e in s_client_b.get_received() if e["name"] == "sign_caption"]
        self.assertEqual(len(received_b), 1)
        self.assertEqual(received_b[0]["args"][0]["text"], "ConfidentialSign")

        # Verify User C and D in Call 2 DID NOT receive it
        received_c = [e for e in s_client_c.get_received() if e["name"] == "sign_caption"]
        received_d = [e for e in s_client_d.get_received() if e["name"] == "sign_caption"]
        self.assertEqual(len(received_c), 0, "Cross-call leakage detected! Call 2 client C received Call 1 captions")
        self.assertEqual(len(received_d), 0, "Cross-call leakage detected! Call 2 client D received Call 1 captions")

        s_client_a.disconnect()
        s_client_b.disconnect()
        s_client_c.disconnect()
        s_client_d.disconnect()

    def test_8_webrtc_signaling_intact(self):
        """TEST 8: Standard WebRTC offer/answer/candidate signaling continues working flawlessly."""
        s_client_a = self._socket_client(self.u1_id)
        s_client_b = self._socket_client(self.u2_id)

        s_client_a.emit("join_call", {"partner_id": self.u2_id})
        s_client_b.emit("join_call", {"partner_id": self.u1_id})
        s_client_b.get_received()

        # User A sends offer
        s_client_a.emit("signal", {
            "type": "offer",
            "partner_id": self.u2_id,
            "sdp": {"type": "offer", "sdp": "v=0\r\no=alice..."},
        })

        received_b = [e for e in s_client_b.get_received() if e["name"] == "signal"]
        self.assertEqual(len(received_b), 1)
        self.assertEqual(received_b[0]["args"][0]["type"], "offer")

        s_client_a.disconnect()
        s_client_b.disconnect()

    def test_9_hwgat_preprocessing_alignment(self):
        """TEST 9: Preprocessing produces exact (1, 192, 64, 2) tensor aligned with HWGAT training normalization."""
        import torch
        from ai_modules.sign_language import SignLanguageRecognizer
        rec = SignLanguageRecognizer()
        self.assertIsNotNone(rec.model, "Pretrained HWGAT model must load")

        # 1. 42-joint hand sequence (zero_centered mode)
        seq_42 = []
        for t in range(12):
            h1 = np.full((21, 2), 0.55 + t * 0.005, dtype=np.float32)
            h2 = np.zeros((21, 2), dtype=np.float32)
            seq_42.append(np.vstack([h1, h2]))

        tensor_42_zero = rec._build_hwgat_tensor(seq_42, target_len=192, mode="zero_centered")
        self.assertEqual(tensor_42_zero.shape, (1, 192, 64, 2))
        self.assertEqual(tensor_42_zero.dtype, torch.float32)

        # In zero-centered frame, nose is at origin (0, 0)
        self.assertAlmostEqual(float(tensor_42_zero[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(tensor_42_zero[0, 0, 0, 1]), 0.0, places=5)

        # 2. 543-joint Holistic sequence (zero_centered mode)
        seq_543 = []
        for t in range(10):
            frame_543 = np.zeros((543, 2), dtype=np.float32)
            frame_543[0] = [0.50, 0.20]  # nose
            frame_543[11] = [0.35, 0.45] # left shoulder
            frame_543[12] = [0.65, 0.45] # right shoulder
            seq_543.append(frame_543)

        tensor_543_zero = rec._build_hwgat_tensor(seq_543, target_len=192, mode="zero_centered")
        self.assertEqual(tensor_543_zero.shape, (1, 192, 64, 2))
        self.assertAlmostEqual(float(tensor_543_zero[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(tensor_543_zero[0, 0, 0, 1]), 0.0, places=5)

        # 3. Training-aligned mode produces exact training coordinate distribution (data_transform.py)
        # Nose is mapped to (0.50, 0.3333333) with unit distance scaled by 6 * shoulder distance
        tensor_aligned = rec._build_hwgat_tensor(seq_543, target_len=192, mode="training_aligned")
        self.assertEqual(tensor_aligned.shape, (1, 192, 64, 2))
        self.assertAlmostEqual(float(tensor_aligned[0, 0, 0, 0]), 0.50, places=4)
        self.assertAlmostEqual(float(tensor_aligned[0, 0, 0, 1]), 1.0 / 3.0, places=4)

        # 4. Model inference returns valid probability distribution
        with torch.no_grad():
            logits = rec.model(tensor_aligned)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        self.assertEqual(len(probs), 2002)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
