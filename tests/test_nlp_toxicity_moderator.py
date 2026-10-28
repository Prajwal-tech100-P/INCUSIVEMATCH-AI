"""Automated tests for Trained NLP Toxicity Classifier & Moderation Workflow.

Covers:
1. Model loading & resilience (valid artifact, missing file, corrupted structure, inference failure fallback, singleton caching).
2. Classification & inference (benign messages, threats, insults, harassment, hate speech, empty inputs, probability ranges, custom thresholds).
3. Chat integration & policy lifecycle (safe messages deliver cleanly, harmful messages generate warnings, 5th warning triggers 7-day restriction, restricted users blocked).
"""
import os
import io
import pytest
from datetime import datetime, timedelta
from bson import ObjectId
from unittest.mock import patch, MagicMock

from app import create_app
from extensions import mongo
from models.user import UserModel
from models.match import Match
from ai_modules.nlp_safety import NLPModerator, DEFAULT_MODEL_PATH, EXPECTED_CATEGORIES


# Top-level picklable dummy pipeline for testing corrupted artifact structures
class DummyCorruptPipeline:
    def predict(self, X): return []
    def predict_proba(self, X): return []


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-nlp-toxicity-key"
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app_context():
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


# ===========================================================================
# 1. MODEL LOADING & RESILIENCE TESTS
# ===========================================================================
class TestModelLoadingAndResilience:

    def test_default_model_loading_success(self):
        """Verify that the real models/toxicity_classifier.joblib loads cleanly with valid structure."""
        assert os.path.exists(DEFAULT_MODEL_PATH), f"Expected artifact at {DEFAULT_MODEL_PATH}"
        moderator = NLPModerator()

        assert moderator.model_loaded is True
        assert moderator.model_status == "ml_classifier"
        assert moderator.pipeline is not None
        assert hasattr(moderator.pipeline, "predict_proba")
        assert tuple(moderator.categories) == EXPECTED_CATEGORIES

    def test_missing_model_file_fallback(self, tmp_path):
        """Verify that a nonexistent model path fails gracefully to keyword fallback without crashing."""
        nonexistent_path = str(tmp_path / "missing_model.joblib")
        moderator = NLPModerator(model_path=nonexistent_path)

        assert moderator.model_loaded is False
        assert moderator.model_status == "fallback_keyword_only"
        assert moderator.pipeline is None

        # Verify fallback detection still functions on keywords
        res_safe = moderator.analyze_message("Hello, how are you?")
        assert res_safe["is_safe"] is True
        assert res_safe["model_status"] == "fallback_keyword_only"

        res_toxic = moderator.analyze_message("Fuck you, bastard")
        assert res_toxic["is_safe"] is False
        assert "fuck" in res_toxic["detected_terms"] or "bastard" in res_toxic["detected_terms"]
        assert res_toxic["model_status"] == "fallback_keyword_only"

    def test_invalid_artifact_structure_fallback(self, tmp_path):
        """Verify that an artifact with invalid structure or wrong categories safely falls back."""
        import joblib
        corrupt_path = str(tmp_path / "corrupt_artifact.joblib")
        # Save a non-dict object
        joblib.dump(["not", "a", "valid", "dictionary"], corrupt_path)

        moderator = NLPModerator(model_path=corrupt_path)
        assert moderator.model_loaded is False
        assert moderator.model_status == "fallback_keyword_only"

        # Save dict with wrong categories using picklable dummy object
        corrupt_cats_path = str(tmp_path / "wrong_cats.joblib")
        joblib.dump({"pipeline": DummyCorruptPipeline(), "categories": ["cat1", "cat2"]}, corrupt_cats_path)
        moderator2 = NLPModerator(model_path=corrupt_cats_path)
        assert moderator2.model_loaded is False
        assert moderator2.model_status == "fallback_keyword_only"

    def test_inference_exception_resilience(self):
        """Verify that runtime errors during predict_proba do not crash the app and fall back to keyword check."""
        moderator = NLPModerator()
        # Mock pipeline.predict_proba to simulate an unexpected runtime error
        with patch.object(moderator.pipeline, "predict_proba", side_effect=RuntimeError("GPU/CPU memory fault")):
            # Clear inference cache to force execution
            with moderator._cache_lock:
                moderator._inference_cache.clear()

            # Benign text without keywords: falls back to safe
            res_benign = moderator.analyze_message("A perfectly benign sentence during a crash test")
            assert res_benign["is_safe"] is True
            assert res_benign["model_status"] == "fallback_keyword_only"

            # Toxic text with keywords: falls back to keyword-based block
            res_toxic = moderator.analyze_message("I will shoot you dead")
            assert res_toxic["is_safe"] is False
            assert res_toxic["model_status"] == "fallback_keyword_only"

    def test_singleton_model_caching(self):
        """Verify that multiple instances of NLPModerator reuse the loaded pipeline rather than reloading."""
        mod1 = NLPModerator()
        mod2 = NLPModerator()
        assert mod1.pipeline is mod2.pipeline


# ===========================================================================
# 2. CLASSIFICATION & INFERENCE TESTS
# ===========================================================================
class TestClassificationAccuracy:

    @pytest.fixture(autouse=True)
    def setup_moderator(self):
        self.moderator = NLPModerator()

    def test_safe_conversational_messages(self):
        """Test that diverse benign messages are classified as safe without false positives."""
        safe_samples = [
            "Hello, how are you doing today?",
            "Thank you so much for your help!",
            "thank you",
            "Are you free for coffee or tea this weekend?",
            "I love hiking in the mountains, watching movies, and cooking.",
            "That is so funny haha!",
            "What kind of music do you like to listen to?",
            "Good morning! Hope you have a wonderful day ahead.",
            "This leg workout is killing me today haha!",  # Colloquial 'killing'
            "That was an absolute killer guitar solo!",      # Colloquial 'killer'
            "I hate getting stuck in traffic on Monday mornings.",  # Colloquial 'hate'
            "Let's shoot some hoops at the basketball court.",     # Colloquial 'shoot'
            "I am deaf and Indian Sign Language is my native language.",
            "Can we do a sign language video call tonight?",
        ]

        for text in safe_samples:
            res = self.moderator.analyze_message(text)
            assert res["is_safe"] is True, f"Expected safe for '{text}', got: {res}"
            assert res["reason"] is None
            assert res["flagged_categories"] == []
            assert res["model_status"] == "ml_classifier"

    def test_violent_threats_flagged(self):
        """Verify that direct physical threats and violent harm are reliably detected and flagged."""
        threat_samples = [
            "I will murder you in your sleep",
            "I will kill you and burn your house down",
            "I will find where you live and beat you until you bleed",
            "I will slit your throat",
            "watch your back because I am going to hunt you down",
        ]

        for text in threat_samples:
            res = self.moderator.analyze_message(text)
            assert res["is_safe"] is False, f"Expected threat to be flagged for '{text}', got: {res}"
            assert any(cat in res["flagged_categories"] for cat in ["threat", "severe_toxic", "toxic"])
            assert "Threatening or violent language" in res["reason"] or "Extremely harmful" in res["reason"] or "harmful" in res["reason"]

    def test_targeted_insults_flagged(self):
        """Verify that targeted personal abuse and insults are flagged."""
        insult_samples = [
            "You are an idiot and a pathetic loser",
            "Shut up you stupid bitch",
            "Fuck you, you arrogant prick",
            "you are a disgusting piece of shit",
            "you ugly moron, look in the mirror",
        ]

        for text in insult_samples:
            res = self.moderator.analyze_message(text)
            assert res["is_safe"] is False, f"Expected insult to be flagged for '{text}', got: {res}"
            assert "insult" in res["flagged_categories"] or "toxic" in res["flagged_categories"]

    def test_harassment_and_stalking_flagged(self):
        """Verify that harassment and stalking threats without direct murder words are caught by ML."""
        harassment_samples = [
            "I know where you live and I am watching you",
            "Send nudes or I will leak your photos",
            "you cannot escape from me, I will find you anywhere",
            "stop ignoring me or else I will ruin your life",
        ]

        for text in harassment_samples:
            res = self.moderator.analyze_message(text)
            assert res["is_safe"] is False, f"Expected harassment to be flagged for '{text}', got: {res}"
            assert any(cat in res["flagged_categories"] for cat in ["harassment", "threat", "toxic"])

    def test_hate_speech_flagged(self):
        """Verify that discriminatory hate speech and slurs are flagged."""
        hate_samples = [
            "All people of your race are filthy subhumans",
            "go back to your country you dirty immigrant",
            "all members of your religion are terrorists",
            "disabled people like you are subhuman trash",
        ]

        for text in hate_samples:
            res = self.moderator.analyze_message(text)
            assert res["is_safe"] is False, f"Expected hate speech to be flagged for '{text}', got: {res}"
            assert any(cat in res["flagged_categories"] for cat in ["hate", "severe_toxic", "insult"])

    def test_multiple_categories_predicted(self):
        """Verify that multi-label messages can trigger multiple distinct categories simultaneously."""
        # This phrase contains murder threat + severe harm + harassment
        res = self.moderator.analyze_message("I will kill you and burn your house down")
        assert res["is_safe"] is False
        assert len(res["flagged_categories"]) >= 2
        assert "threat" in res["flagged_categories"]
        assert "severe_toxic" in res["flagged_categories"]

    def test_probability_outputs_shape_and_range(self):
        """Verify that probability scores contain all 6 categories with valid floats in [0.0, 1.0]."""
        res = self.moderator.analyze_message("This is a test of probability scores")
        scores = res["scores"]
        assert set(scores.keys()) == set(EXPECTED_CATEGORIES)
        for cat, prob in scores.items():
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_empty_and_invalid_inputs(self):
        """Verify that None, empty strings, and non-string types are handled safely."""
        for val in [None, "", "   ", "\n\t", 12345, []]:
            res = self.moderator.analyze_message(val)
            assert res["is_safe"] is True
            assert res["reason"] is None
            assert res["flagged_categories"] == []

    def test_configurable_threshold_overrides(self):
        """Verify that custom threshold parameters passed to constructor are strictly respected."""
        # Custom ultra-strict moderator where threat threshold is set to 0.01
        strict_mod = NLPModerator(thresholds={"threat": 0.01})
        # Even benign phrases have small baseline threat probabilities ~0.15
        res_strict = strict_mod.analyze_message("That was a good movie")
        assert "threat" in res_strict["flagged_categories"]


# ===========================================================================
# 3. CHAT MODERATION WORKFLOW & WARNING RESTRICTION TESTS
# ===========================================================================
class TestChatModerationWorkflow:

    def _login_client(self, app, email, password):
        client = app.test_client()
        client.get("/logout", follow_redirects=True)
        resp = client.post("/login", data={"email": email, "password": password}, follow_redirects=True)
        assert resp.status_code == 200
        return client

    def _create_matched_users(self):
        email_a = f"sender_{ObjectId()}@test.com"
        email_b = f"receiver_{ObjectId()}@test.com"
        pwd = "TestPassword123!"

        uid_a = UserModel.create_user(email=email_a, password=pwd, name="Alice Sender")
        uid_b = UserModel.create_user(email=email_b, password=pwd, name="Bob Receiver")

        Match.like(uid_a, uid_b)
        Match.like(uid_b, uid_a)
        assert Match.are_matched(uid_a, uid_b)

        return uid_a, uid_b, email_a, email_b, pwd

    def test_safe_chat_message_sent_without_warnings(self, app):
        """Verify that sending a safe message succeeds (200 OK) and does NOT increment warnings."""
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()
        client = self._login_client(app, email_a, pwd)

        resp = client.post(
            "/chat/send",
            json={"receiver_id": uid_b, "text": "Hello Bob! How are you doing today?"}
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("status") == "sent"
        assert data.get("message_id")

        # Confirm sender warning_count remains 0
        u = UserModel.get_by_id(uid_a)
        assert u.warning_count == 0

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})
        mongo.db.messages.delete_many({"$or": [{"sender_id": uid_a}, {"sender_id": uid_b}]})

    def test_harmful_chat_message_increments_warning(self, app):
        """Verify that sending a toxic message blocks delivery (400) and increments warning count by 1."""
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()
        client = self._login_client(app, email_a, pwd)

        resp = client.post(
            "/chat/send",
            json={"receiver_id": uid_b, "text": "Shut up you stupid bitch"}
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "Warning 1/5" in data.get("error", "")

        # Verify database record updated
        u = UserModel.get_by_id(uid_a)
        assert u.warning_count == 1

        # Confirm no message was persisted in the messages collection
        saved_msg = mongo.db.messages.find_one({"sender_id": uid_a, "receiver_id": uid_b})
        assert saved_msg is None

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})

    def test_single_message_does_not_double_count_warning(self, app):
        """Verify that a single harmful message increments the warning counter exactly once."""
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()
        client = self._login_client(app, email_a, pwd)

        resp = client.post(
            "/chat/send",
            json={"receiver_id": uid_b, "text": "You are an idiot and a total loser"}
        )
        assert resp.status_code == 400

        u = UserModel.get_by_id(uid_a)
        assert u.warning_count == 1

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})

    def test_fifth_warning_triggers_7day_restriction(self, app):
        """Verify that the 5th warning triggers account deactivation and a 7-day restriction."""
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()
        client = self._login_client(app, email_a, pwd)

        # Send 4 harmful messages
        for i in range(1, 5):
            r = client.post(
                "/chat/send",
                json={"receiver_id": uid_b, "text": f"You are a pathetic loser #{i}"}
            )
            assert r.status_code == 400
            assert f"Warning {i}/5" in r.get_json().get("error", "")

        # The 5th message must trigger the 7-day temporary ban with 403 Forbidden
        resp_5 = client.post(
            "/chat/send",
            json={"receiver_id": uid_b, "text": "I am going to murder you in your sleep"}
        )
        assert resp_5.status_code == 403
        data_5 = resp_5.get_json()
        assert "temporarily restricted for 7 days" in data_5.get("error", "")

        # Check DB user status
        user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc.get("is_active") is False
        assert user_doc.get("warning_count") == 5

        expiry = user_doc.get("block_expiry")
        assert expiry is not None
        now = datetime.utcnow()
        diff = expiry - now
        assert timedelta(days=6, hours=23) <= diff <= timedelta(days=7, hours=1)

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})

    def test_restricted_user_cannot_bypass_messaging_controls(self, app):
        """Verify that a user with an active block_expiry cannot send messages."""
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()

        # Set 4 warnings
        mongo.db.users.update_one(
            {"_id": ObjectId(uid_a)},
            {"$set": {"warning_count": 4}}
        )

        client = self._login_client(app, email_a, pwd)

        # Trigger 5th warning
        r5 = client.post(
            "/chat/send",
            json={"receiver_id": uid_b, "text": "I will kill you and burn your house down"}
        )
        assert r5.status_code == 403

        # User is now restricted. Even an innocent message is blocked
        resp = client.post(
            "/chat/send",
            json={"receiver_id": uid_b, "text": "Hello, I apologize, how are you?"}
        )
        assert resp.status_code in (403, 302)

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})
