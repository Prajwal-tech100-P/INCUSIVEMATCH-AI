"""
Test Suite for Live Indian Sign Language Recognition Engine & API
==================================================================
Covers:
1. GCN-Transformer checkpoint loading & architecture verification
2. MediaPipe HandLandmarker initialization
3. Live sliding window sequence buffering & dynamic resampling
4. Temporal vote smoothing and session reset
5. Flask API endpoint `/api/sign/recognize` response contract
"""

import io
import unittest
import numpy as np
import cv2
import torch
from bson import ObjectId

from ai_modules.sign_language import SignLanguageRecognizer, _resp
from app import create_app
from models.user import UserModel


class TestLiveSignRecognizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = SignLanguageRecognizer()

    def test_recognizer_model_loading(self):
        """Verify model weights, architecture, class labels, and landmarker load cleanly."""
        self.assertIsNotNone(self.recognizer.model, "GCN-Transformer model failed to load.")
        self.assertEqual(self.recognizer.arch, "gcn_transformer")
        self.assertEqual(len(self.recognizer.labels), 29)
        self.assertIn("hello", self.recognizer.labels)
        self.assertIn("thank_you", self.recognizer.labels)
        self.assertIsNotNone(self.recognizer.landmarker, "MediaPipe HandLandmarker failed to load.")
        self.assertIsNone(self.recognizer.error)

    def test_predict_gcnt_with_landmark_sequence(self):
        """Verify _predict_gcnt accepts variable-length sequences (e.g. 10 frames) and resamples to 60."""
        mock_seq = [np.random.uniform(0.1, 0.9, size=(42, 2)).astype(np.float32) for _ in range(12)]
        label, conf = self.recognizer._predict_gcnt(mock_seq)
        self.assertIsInstance(label, str)
        self.assertIn(label, self.recognizer.labels)
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_temporal_smoothing_and_reset(self):
        """Verify smoothing stabilizes predictions and reset_user clears memory."""
        user_key = "user_test_smoothing"
        self.recognizer.reset_user(user_key)

        # Feed three 'hello' predictions and one outlier
        self.recognizer._smooth(user_key, "hello", 0.5)
        self.recognizer._smooth(user_key, "hello", 0.6)
        best, best_conf = self.recognizer._smooth(user_key, "tea", 0.2)
        self.assertEqual(best, "hello")
        self.assertEqual(best_conf, 0.6)

        self.recognizer.reset_user(user_key)
        self.assertNotIn(user_key, self.recognizer.recent_predictions)
        self.assertNotIn(user_key, self.recognizer.sequences)

    def test_recognize_frame_empty_image(self):
        """Verify blank image returns 'No hand detected' with ready=False."""
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", blank)
        result = self.recognizer.recognize_frame(encoded.tobytes(), user_id="user_blank")

        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertIn("message", result)
        self.assertIn("ready", result)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["ready"], False)
        self.assertIn("No hand detected", result["message"])

    def test_recognize_frame_invalid_bytes(self):
        """Verify garbage bytes return a handled error response."""
        result = self.recognizer.recognize_frame(b"not_an_image", user_id="user_err")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["ready"], False)
        self.assertEqual(result["message"], "Invalid image frame.")


class TestSignRecognizeAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.client = self.app.test_client()

        from extensions import mongo
        # Create authenticated test user
        self.test_user_id = str(ObjectId())
        mongo.db.users.delete_many({"email": "sign_api_tester@test.com"})
        mongo.db.users.insert_one({
            "_id": ObjectId(self.test_user_id),
            "id": self.test_user_id,
            "name": "Sign API Tester",
            "email": "sign_api_tester@test.com",
            "password": "hash123",
            "is_active": True,
            "gender": "male",
            "interests": ["Sign Language"],
        })

    def tearDown(self):
        from extensions import mongo
        mongo.db.users.delete_many({"email": "sign_api_tester@test.com"})

    def _login(self):
        with self.client.session_transaction() as sess:
            sess["_user_id"] = self.test_user_id
            sess["_fresh"] = True

    def test_api_sign_recognize_requires_login(self):
        """Verify /api/sign/recognize redirects unauthenticated requests."""
        res = self.client.post("/api/sign/recognize")
        self.assertEqual(res.status_code, 302)

    def test_api_sign_recognize_with_frame(self):
        """Verify /api/sign/recognize processes image frame upload."""
        self._login()
        img = np.zeros((160, 240, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)

        data = {
            "frame": (io.BytesIO(encoded.tobytes()), "frame.jpg", "image/jpeg")
        }
        res = self.client.post("/api/sign/recognize", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertIn("text", json_data)
        self.assertIn("confidence", json_data)
        self.assertIn("message", json_data)
        self.assertIn("ready", json_data)

    def test_api_sign_recognize_missing_frame(self):
        """Verify 400 error when request lacks 'frame' part."""
        self._login()
        res = self.client.post("/api/sign/recognize", data={}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)

    def test_video_call_template_has_sign_recognition_ui(self):
        """Verify video_call.html template includes on-screen captions, transcript log, and controls."""
        with open("templates/video_call.html", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('id="live-caption-banner"', content)
        self.assertIn('id="toggle-sign-ai"', content)
        self.assertIn('id="toggle-cc"', content)
        self.assertIn('id="sign-source"', content)
        self.assertIn('id="transcript-list"', content)
        self.assertIn('id="sign-output"', content)
        self.assertIn('sign_caption', content)


if __name__ == "__main__":
    unittest.main()

