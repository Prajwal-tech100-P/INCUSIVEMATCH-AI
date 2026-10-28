"""
Automated Test Suite for InclusiveMatch AI Accessible Video Call Communication System
=====================================================================================

Covers:
1. SigningActivityDetector state machine & debounced transitions
   (IDLE -> POSSIBLE_SIGNING -> SIGNING -> POSSIBLE_TURN_END -> IDLE).
2. Kinematic validation: stationary hands vs articulatory motion vs erratic movement.
3. Pause tolerance: inter-sign pauses do not prematurely kill turns; inactivity timeout finalizes turn.
4. Duplicate sign filtering and legitimate repeated sign acceptance.
5. SignLanguageRecognizer integration returning activity states, turn flags, and phrases.
6. Authorized Socket.IO caption delivery (sign_caption, speech_caption, signing_status).
7. Unauthorized caption injection rejection (unauthenticated, non-matched, restricted users).
8. Template accessibility inspection:
   - Movie-style subtitle overlay with speaker badges
   - Automatic remote signing indicator ("Your partner is signing...")
   - "Speak Recognized Text: OFF" default state
   - Absence of manual "Start Signing" or "Finish Signing" buttons
"""

import os
import unittest
import numpy as np
from bson import ObjectId

from ai_modules.sign_language import (
    SignLanguageRecognizer,
    SigningActivityDetector,
    SigningState,
)
from app import create_app
from extensions import mongo, socketio
from models.user import UserModel
from models.match import Match
from models.restriction import Restriction


class TestSigningActivityDetector(unittest.TestCase):
    """Test suite for automatic signing activity detection and turn finalization."""

    def setUp(self):
        self.detector = SigningActivityDetector(
            min_active_frames=3,
            inactivity_timeout=1.5,
            velocity_idle_threshold=0.0035,
            velocity_erratic_threshold=0.22,
            duplicate_suppress_window=1.0,
        )

    def test_initial_state_is_idle(self):
        """Initial state must be IDLE with is_signing=False."""
        self.assertEqual(self.detector.state, SigningState.IDLE)
        info = self.detector.update(None)
        self.assertEqual(info["state"], "IDLE")
        self.assertFalse(info["is_signing"])
        self.assertFalse(info["turn_completed"])

    def test_stationary_hands_do_not_trigger_signing(self):
        """Stationary hands (resting/scratching below threshold) must not trigger signing."""
        static_landmarks = np.ones((42, 2), dtype=np.float32) * 0.5
        for _ in range(5):
            info = self.detector.update(static_landmarks, prediction="", confidence=0.0)
            self.assertEqual(info["state"], "IDLE")
            self.assertFalse(info["is_signing"])

    def test_erratic_wild_motion_does_not_trigger_signing(self):
        """Violent camera shaking or erratic jerks above threshold must be rejected."""
        lm1 = np.zeros((42, 2), dtype=np.float32)
        lm2 = np.ones((42, 2), dtype=np.float32) * 0.9  # extreme jump > 0.22
        self.detector.update(lm1)
        info = self.detector.update(lm2)
        self.assertEqual(info["state"], "IDLE")
        self.assertFalse(info["is_signing"])

    def test_articulatory_motion_transitions_to_signing(self):
        """Structured articulatory motion transitions IDLE -> POSSIBLE_SIGNING -> SIGNING."""
        base = np.zeros((42, 2), dtype=np.float32)
        base[:21] = 0.4  # Hand 1 present

        # Frame 1: Initial baseline
        self.detector.update(base, now=10.0)
        self.assertEqual(self.detector.state, SigningState.IDLE)

        # Frame 2: Moderate articulatory movement
        f2 = base.copy()
        f2[:21] += 0.02
        self.detector.update(f2, now=10.2)

        # Frame 3: Second active motion frame -> enters POSSIBLE_SIGNING
        f3 = f2.copy()
        f3[:21] += 0.02
        info3 = self.detector.update(f3, now=10.4)
        self.assertEqual(info3["state"], "POSSIBLE_SIGNING")
        self.assertTrue(info3["is_signing"])

        # Frame 4: Third active frame -> reaches min_active_frames (3) -> enters SIGNING
        f4 = f3.copy()
        f4[:21] += 0.02
        info4 = self.detector.update(f4, now=10.6)
        self.assertEqual(info4["state"], "SIGNING")
        self.assertTrue(info4["is_signing"])

    def test_inter_sign_pause_tolerance_and_turn_completion(self):
        """Brief pause moves to POSSIBLE_TURN_END without ending turn; inactivity timeout concludes turn."""
        base = np.zeros((42, 2), dtype=np.float32)
        base[:21] = 0.4

        # Drive into SIGNING
        self.detector.update(base, now=10.0)
        f2 = base.copy()
        f2[:21] += 0.03
        self.detector.update(f2, prediction="Hello", confidence=0.85, now=10.2)
        f3 = f2.copy()
        f3[:21] += 0.03
        info_sign = self.detector.update(f3, prediction="Hello", confidence=0.85, now=10.4)
        self.assertEqual(info_sign["state"], "SIGNING")
        self.assertTrue(info_sign["is_signing"])
        self.assertEqual(info_sign["phrase"], "Hello")

        # Brief pause (hands stationary or lowered for 0.4 seconds)
        info_pause1 = self.detector.update(None, now=10.8)
        self.assertEqual(info_pause1["state"], "POSSIBLE_TURN_END")
        self.assertTrue(info_pause1["is_signing"])  # Still signing! Indicator must not flicker!
        self.assertFalse(info_pause1["turn_completed"])

        # Signing resumes within pause window (at 11.2s, pause = 0.4s < 1.5s timeout)
        f4 = f3.copy()
        f4[:21] += 0.03
        info_resume = self.detector.update(f4, prediction="Help", confidence=0.80, now=11.2)
        self.assertEqual(info_resume["state"], "SIGNING")
        self.assertTrue(info_resume["is_signing"])
        self.assertIn("Help", info_resume["phrase"])

        # Signer stops signing: pause begins at 11.4s
        self.detector.update(None, now=11.4)
        self.assertEqual(self.detector.state, SigningState.POSSIBLE_TURN_END)

        # Inactivity timeout reached at 13.0s (1.6s elapsed > 1.5s timeout)
        info_complete = self.detector.update(None, now=13.0)
        self.assertEqual(info_complete["state"], "IDLE")
        self.assertFalse(info_complete["is_signing"])
        self.assertTrue(info_complete["turn_completed"])
        self.assertEqual(info_complete["finalized_phrase"], "Hello Help")

    def test_duplicate_prediction_suppression(self):
        """Consecutive duplicate predictions within suppress window are not emitted repeatedly."""
        base = np.zeros((42, 2), dtype=np.float32)
        base[:21] = 0.5

        # Frame 1: Initial
        self.detector.update(base, now=1.0)
        # Frame 2: Enter possible signing
        f2 = base.copy()
        f2[:21] += 0.02
        self.detector.update(f2, prediction="Water", confidence=0.75, now=1.2)
        # Frame 3: Confirmed signing - accepts "Water"
        f3 = f2.copy()
        f3[:21] += 0.02
        res1 = self.detector.update(f3, prediction="Water", confidence=0.75, now=1.4)
        self.assertEqual(res1["accepted_sign"], "Water")

        # Frame 4: 200ms later, model still predicts "Water" -> must be suppressed!
        f4 = f3.copy()
        f4[:21] += 0.02
        res2 = self.detector.update(f4, prediction="Water", confidence=0.75, now=1.6)
        self.assertEqual(res2["accepted_sign"], "")  # Suppressed duplicate!
        self.assertEqual(res2["phrase"], "Water")  # But phrase maintains single token


class TestSignRecognizerActivityIntegration(unittest.TestCase):
    """Test SignLanguageRecognizer integration with automatic activity detection."""

    @classmethod
    def setUpClass(cls):
        cls.recognizer = SignLanguageRecognizer()

    def test_recognize_frame_returns_activity_and_turn_keys(self):
        """recognize_frame must return signing_state, is_signing, turn_completed, phrase, and finalized_phrase."""
        blank = np.zeros((160, 240, 3), dtype=np.uint8)
        import cv2
        _, encoded = cv2.imencode(".jpg", blank)
        result = self.recognizer.recognize_frame(encoded.tobytes(), user_id="test_activity_user")

        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertIn("message", result)
        self.assertIn("ready", result)
        self.assertIn("signing_state", result)
        self.assertIn("is_signing", result)
        self.assertIn("turn_completed", result)
        self.assertIn("phrase", result)
        self.assertIn("finalized_phrase", result)
        self.assertEqual(result["signing_state"], "IDLE")
        self.assertFalse(result["is_signing"])

    def test_reset_user_clears_activity_detector(self):
        """reset_user must clear detector state for the given user."""
        user_key = "user_activity_cleanup"
        det = self.recognizer._get_activity_detector(user_key)
        det.state = SigningState.SIGNING
        self.assertIn(user_key, self.recognizer.activity_detectors)

        self.recognizer.reset_user(user_key)
        self.assertNotIn(user_key, self.recognizer.activity_detectors)


class TestSocketAccessibilityCommunication(unittest.TestCase):
    """Test Socket.IO communication for sign captions, speech captions, and signing indicators."""

    @classmethod
    def setUpClass(cls):
        os.environ["APP_ENV"] = "testing"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def _socket_client(self, user_id):
        flask_client = self.app.test_client()
        with flask_client.session_transaction() as sess:
            sess["_user_id"] = user_id
            sess["_fresh"] = True
        return socketio.test_client(self.app, flask_test_client=flask_client)

    def setUp(self):
        self.u1_id = str(ObjectId())
        self.u2_id = str(ObjectId())
        self.stranger_id = str(ObjectId())

        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(self.u1_id), ObjectId(self.u2_id), ObjectId(self.stranger_id)]}})
        mongo.db.matches.delete_many({"$or": [{"user_a": self.u1_id}, {"user_b": self.u1_id}]})
        mongo.db.user_restrictions.delete_many({"$or": [{"restrictor_id": self.u1_id}, {"restricted_user_id": self.u1_id}]})

        mongo.db.users.insert_one({
            "_id": ObjectId(self.u1_id),
            "id": self.u1_id,
            "name": "Alice Signer",
            "email": f"alice_{self.u1_id}@test.com",
            "password": "pw",
            "role": "user",
            "is_active": True,
        })
        mongo.db.users.insert_one({
            "_id": ObjectId(self.u2_id),
            "id": self.u2_id,
            "name": "Bob Receiver",
            "email": f"bob_{self.u2_id}@test.com",
            "password": "pw",
            "role": "user",
            "is_active": True,
        })
        mongo.db.users.insert_one({
            "_id": ObjectId(self.stranger_id),
            "id": self.stranger_id,
            "name": "Eve Stranger",
            "email": f"eve_{self.stranger_id}@test.com",
            "password": "pw",
            "role": "user",
            "is_active": True,
        })

        # Mutual match between u1 and u2
        Match.like(self.u1_id, self.u2_id)
        Match.like(self.u2_id, self.u1_id)

    def tearDown(self):
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(self.u1_id), ObjectId(self.u2_id), ObjectId(self.stranger_id)]}})
        mongo.db.matches.delete_many({"$or": [{"user_a": self.u1_id}, {"user_b": self.u1_id}]})
        mongo.db.user_restrictions.delete_many({"$or": [{"restrictor_id": self.u1_id}, {"restricted_user_id": self.u1_id}]})

    def test_authorized_sign_caption_delivery(self):
        """Authorized user emits sign_caption and partner receives it in the call room."""
        client1 = self._socket_client(self.u1_id)
        client2 = self._socket_client(self.u2_id)

        client1.emit("join_call", {"partner_id": self.u2_id})
        client2.emit("join_call", {"partner_id": self.u1_id})

        # Flush initial join events
        client2.get_received()

        # Alice emits sign caption
        client1.emit("sign_caption", {
            "partner_id": self.u2_id,
            "text": "Hello",
            "phrase": "Hello",
            "confidence": 88.5,
            "is_final": True,
        })

        received = client2.get_received()
        sign_events = [ev for ev in received if ev["name"] == "sign_caption"]
        self.assertEqual(len(sign_events), 1)
        args = sign_events[0]["args"][0]
        self.assertEqual(args["speaker"], "Alice Signer")
        self.assertEqual(args["text"], "Hello")
        self.assertEqual(args["phrase"], "Hello")
        self.assertEqual(args["confidence"], 88.5)
        self.assertTrue(args["is_final"])

        client1.disconnect()
        client2.disconnect()

    def test_authorized_speech_caption_delivery(self):
        """Authorized participant emits speech_caption and partner receives it."""
        client1 = self._socket_client(self.u1_id)
        client2 = self._socket_client(self.u2_id)

        client1.emit("join_call", {"partner_id": self.u2_id})
        client2.emit("join_call", {"partner_id": self.u1_id})
        client2.get_received()

        # Alice speaks
        client1.emit("speech_caption", {
            "partner_id": self.u2_id,
            "text": "Can you hear me clearly?",
            "is_final": True,
        })

        received = client2.get_received()
        speech_events = [ev for ev in received if ev["name"] == "speech_caption"]
        self.assertEqual(len(speech_events), 1)
        args = speech_events[0]["args"][0]
        self.assertEqual(args["speaker"], "Alice Signer")
        self.assertEqual(args["text"], "Can you hear me clearly?")
        self.assertTrue(args["is_final"])

        client1.disconnect()
        client2.disconnect()

    def test_authorized_signing_status_delivery(self):
        """Automatic signing status change is broadcast to partner."""
        client1 = self._socket_client(self.u1_id)
        client2 = self._socket_client(self.u2_id)

        client1.emit("join_call", {"partner_id": self.u2_id})
        client2.emit("join_call", {"partner_id": self.u1_id})
        client2.get_received()

        # Alice starts signing
        client1.emit("signing_status", {
            "partner_id": self.u2_id,
            "state": "SIGNING",
            "is_signing": True,
        })

        received = client2.get_received()
        status_events = [ev for ev in received if ev["name"] == "signing_status"]
        self.assertEqual(len(status_events), 1)
        args = status_events[0]["args"][0]
        self.assertEqual(args["speaker"], "Alice Signer")
        self.assertEqual(args["state"], "SIGNING")
        self.assertTrue(args["is_signing"])

        client1.disconnect()
        client2.disconnect()

    def test_unauthorized_user_cannot_inject_captions(self):
        """Stranger who is not mutually matched cannot inject captions into another call."""
        client_eve = self._socket_client(self.stranger_id)
        client_bob = self._socket_client(self.u2_id)

        client_bob.emit("join_call", {"partner_id": self.u1_id})
        client_bob.get_received()

        # Eve attempts to send sign_caption and speech_caption to Bob
        client_eve.emit("sign_caption", {
            "partner_id": self.u2_id,
            "text": "Injected Sign",
            "confidence": 99.0,
        })
        client_eve.emit("speech_caption", {
            "partner_id": self.u2_id,
            "text": "Injected Speech",
            "is_final": True,
        })
        client_eve.emit("signing_status", {
            "partner_id": self.u2_id,
            "state": "SIGNING",
            "is_signing": True,
        })

        received = client_bob.get_received()
        captions = [ev for ev in received if ev["name"] in ("sign_caption", "speech_caption", "signing_status")]
        self.assertEqual(len(captions), 0, "Unauthorized caption was injected into call room!")

        client_eve.disconnect()
        client_bob.disconnect()

    def test_restricted_user_cannot_inject_captions(self):
        """Restricted users cannot inject captions into call room."""
        # Restrict u1 by u2
        Restriction.create(self.u2_id, self.u1_id)

        client1 = self._socket_client(self.u1_id)
        client2 = self._socket_client(self.u2_id)

        client1.emit("sign_caption", {
            "partner_id": self.u2_id,
            "text": "Blocked Sign",
            "confidence": 90.0,
        })

        received = client2.get_received()
        captions = [ev for ev in received if ev["name"] == "sign_caption"]
        self.assertEqual(len(captions), 0, "Restricted user was able to inject caption!")

        client1.disconnect()
        client2.disconnect()


class TestVideoCallTemplateAccessibility(unittest.TestCase):
    """Verify video_call.html contains required accessible elements and adheres to constraints."""

    def setUp(self):
        template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "video_call.html")
        with open(template_path, "r", encoding="utf-8") as f:
            self.content = f.read()

    def test_no_manual_start_finish_signing_buttons(self):
        """CRITICAL: Must NOT contain manual 'Start Signing' or 'Finish Signing' buttons."""
        lower_content = self.content.lower()
        self.assertNotIn("start signing", lower_content)
        self.assertNotIn("finish signing", lower_content)
        self.assertNotIn("stop signing", lower_content)

    def test_automatic_remote_signing_indicator_present(self):
        """Verify presence of remote signing indicator displaying '... is signing'."""
        self.assertIn('id="remote-signing-indicator"', self.content)
        self.assertIn('id="remote-signing-text"', self.content)
        self.assertIn("is signing", self.content)

    def test_movie_style_subtitle_overlay_present(self):
        """Verify on-screen movie-style subtitle banner and elements."""
        self.assertIn('id="live-caption-banner"', self.content)
        self.assertIn('id="caption-speaker"', self.content)
        self.assertIn('id="caption-text"', self.content)
        self.assertIn('id="caption-type"', self.content)

    def test_speak_recognized_text_toggle_and_default_off(self):
        """Verify 'Speak Recognized Text: OFF' button and switch, defaulting to OFF."""
        self.assertIn('id="toggle-tts"', self.content)
        self.assertIn("Speak Recognized Text: OFF", self.content)
        # Checkbox must not be checked by default
        self.assertIn('id="speak-sign"', self.content)
        self.assertNotIn('id="speak-sign" checked', self.content)
        self.assertIn("ttsEnabled = false", self.content)

    def test_speech_to_text_and_feedback_prevention_logic(self):
        """Verify SpeechRecognition integration and audio feedback prevention flag."""
        self.assertIn("SpeechRecognition", self.content)
        self.assertIn("speech_caption", self.content)
        self.assertIn("isSpeakingSynthesized", self.content)


if __name__ == "__main__":
    unittest.main()
