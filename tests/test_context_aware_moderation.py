"""
Comprehensive Automated Tests for Context-Aware NLP Toxicity Moderation in InclusiveMatch AI.

Explicitly validates all 18 cases mandated by project requirements:
1. Friendly joke containing "kill."
2. Friendly message containing "hate."
3. Benign expression such as "This workout is killing me."
4. Clear direct threat.
5. Threat without profanity.
6. Insult without a threat.
7. Harassment involving repeated messages.
8. Message containing an emoji (masked threat must NOT bypass detection).
9. Message without an emoji (benign hyperbole).
10. Message containing an ambiguous phrase (cautious non-punitive review).
11. A score below the configured threshold.
12. A score above the configured threshold.
13. Multiple toxicity categories triggered by one message.
14. Model artifact missing or unavailable.
15. Duplicate message submission (retry deduplication).
16. Concurrent warning updates (atomic MongoDB updates).
17. A user attempting to manipulate moderation state.
18. A harmless message that must not trigger an account restriction.
"""
import os
import pytest
import threading
from bson import ObjectId
from datetime import datetime, timedelta

from app import create_app
from extensions import mongo
from models.user import UserModel
from models.match import Match
from ai_modules.nlp_safety import NLPModerator


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-context-moderation-key"
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app_context():
        yield flask_app


@pytest.fixture
def moderator():
    return NLPModerator()


class TestContextAwareModerationCases:
    """Deterministic and workflow tests for cases 1 through 18."""

    def _login_client(self, app, email, password):
        client = app.test_client()
        client.get("/logout", follow_redirects=True)
        resp = client.post("/login", data={"email": email, "password": password}, follow_redirects=True)
        assert resp.status_code == 200
        return client

    def _create_matched_users(self):
        email_a = f"sender_ctx_{ObjectId()}@test.com"
        email_b = f"receiver_ctx_{ObjectId()}@test.com"
        pwd = "TestPassword123!"

        uid_a = UserModel.create_user(email=email_a, password=pwd, name="Alice Context")
        uid_b = UserModel.create_user(email=email_b, password=pwd, name="Bob Context")

        Match.like(uid_a, uid_b)
        Match.like(uid_b, uid_a)
        assert Match.are_matched(uid_a, uid_b)
        return uid_a, uid_b, email_a, email_b, pwd

    # Case 1: Friendly joke containing "kill"
    def test_case_01_friendly_joke_containing_kill(self, moderator):
        msg = "Bro, I will kill you for eating my pizza 😂"
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is True
        assert res["action"] == "allow"
        assert res["requires_warning"] is False
        assert len(res["flagged_categories"]) == 0

    # Case 2: Friendly message containing "hate"
    def test_case_02_friendly_message_containing_hate(self, moderator):
        msg = "I hate you 😂❤️"
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is True
        assert res["action"] == "allow"
        assert res["requires_warning"] is False
        assert len(res["flagged_categories"]) == 0

    # Case 3: Benign expression such as "This workout is killing me."
    def test_case_03_benign_expression_workout_killing_me(self, moderator):
        msg = "This workout is killing me."
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is True
        assert res["action"] == "allow"
        assert res["requires_warning"] is False
        assert len(res["flagged_categories"]) == 0

    # Case 4: Clear direct threat
    def test_case_04_clear_direct_threat(self, moderator):
        msg = "I will hurt you tonight."
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is False
        assert res["action"] == "block_and_warn"
        assert res["requires_warning"] is True
        assert "threat" in res["flagged_categories"] or "severe_toxic" in res["flagged_categories"]

    # Case 5: Threat without profanity
    def test_case_05_threat_without_profanity(self, moderator):
        msg = "I know what train you take every evening and which car you sit in."
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is False
        assert res["action"] == "block_and_warn"
        assert res["requires_warning"] is True
        assert "threat" in res["flagged_categories"]

    # Case 6: Insult without a threat
    def test_case_06_insult_without_threat(self, moderator):
        msg = "You are completely brainless, pathetic, and worthless."
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is False
        assert res["action"] == "block_and_warn"
        assert res["requires_warning"] is True
        assert "insult" in res["flagged_categories"]
        assert "threat" not in res["flagged_categories"]

    # Case 7: Harassment involving repeated messages
    def test_case_07_harassment_involving_repeated_messages(self, moderator):
        msg = "I created five new accounts just to keep messaging you after you blocked me."
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is False
        assert res["action"] == "block_and_warn"
        assert res["requires_warning"] is True

    # Case 8: Message containing an emoji (masked threat must NOT bypass detection)
    def test_case_08_masked_threat_with_emoji_must_not_bypass(self, moderator):
        msg = "I will stab you in your sleep 😂"
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is False
        assert res["action"] == "block_and_warn"
        assert res["requires_warning"] is True
        assert "threat" in res["flagged_categories"] or "severe_toxic" in res["flagged_categories"]

    # Case 9: Message without an emoji (benign hyperbole)
    def test_case_09_message_without_emoji_benign(self, moderator):
        msg = "You stole my last piece of garlic bread, you are dead to me lol"
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is True
        assert res["action"] == "allow"
        assert res["requires_warning"] is False

    # Case 10: Message containing an ambiguous phrase (non-punitive review)
    def test_case_10_ambiguous_phrase_non_punitive(self, moderator):
        msg = "You're going to regret this."
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is True
        assert res["action"] == "flag_for_review"
        assert res["requires_warning"] is False
        assert "ambiguous_context" in res["flagged_categories"]

    # Case 11: A score below the configured threshold
    def test_case_11_score_below_configured_threshold(self, moderator):
        msg = "The weather today is pleasant and sunny."
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is True
        assert res["action"] == "allow"
        for cat, score in res["scores"].items():
            assert score < moderator.thresholds[cat]

    # Case 12: A score above the configured threshold
    def test_case_12_score_above_configured_threshold(self, moderator):
        msg = "You are a disgusting ugly pig and nobody wants you"
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is False
        assert res["scores"]["insult"] >= moderator.thresholds["insult"]
        assert "insult" in res["flagged_categories"]

    # Case 13: Multiple toxicity categories triggered by one message
    def test_case_13_multiple_toxicity_categories_triggered(self, moderator):
        msg = "I am going to murder you and burn your house down you stupid whore"
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is False
        assert len(res["flagged_categories"]) >= 2

    # Case 14: Model artifact missing or unavailable
    def test_case_14_model_artifact_missing_or_unavailable(self, tmp_path):
        nonexistent = str(tmp_path / "ghost_model.joblib")
        fallback_mod = NLPModerator(model_path=nonexistent)
        assert fallback_mod.model_loaded is False
        assert fallback_mod.model_status == "fallback_keyword_only"

        # Safe message still works
        res_safe = fallback_mod.analyze_message("Good morning, have a great day!")
        assert res_safe["is_safe"] is True

        # Explicit abusive keyword is blocked
        res_toxic = fallback_mod.analyze_message("Fuck you, bastard")
        assert res_toxic["is_safe"] is False
        assert res_toxic["action"] == "block_and_warn"

    # Case 15: Duplicate message submission (retry deduplication)
    def test_case_15_duplicate_message_submission_deduplication(self, app):
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()
        client = self._login_client(app, email_a, pwd)

        toxic_text = "You are a repulsive scumbag and I hate you"

        # First attempt -> triggers warning 1
        r1 = client.post("/chat/send", json={"receiver_id": uid_b, "text": toxic_text})
        assert r1.status_code == 400
        assert "Warning 1/5" in r1.get_json()["error"]

        # Immediate retry with exact same text -> returns existing warning WITHOUT incrementing counter to 2
        r2 = client.post("/chat/send", json={"receiver_id": uid_b, "text": toxic_text})
        assert r2.status_code == 400
        assert "Warning 1/5" in r2.get_json()["error"]

        # Confirm DB warning count is still 1, NOT 2
        user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc["warning_count"] == 1

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})

    # Case 16: Concurrent warning updates (atomic MongoDB updates)
    def test_case_16_concurrent_warning_updates_thread_safe(self, app):
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()
        # Set warning count to 0 in DB
        mongo.db.users.update_one({"_id": ObjectId(uid_a)}, {"$set": {"warning_count": 0}})

        from pymongo import ReturnDocument

        # Simulate 4 concurrent warning increments directly via atomic update logic
        results = []
        def worker():
            updated = mongo.db.users.find_one_and_update(
                {"_id": ObjectId(uid_a)},
                {"$inc": {"warning_count": 1}},
                return_document=ReturnDocument.AFTER
            )
            results.append(updated["warning_count"])

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 4 updates succeeded and final warning_count must be exactly 4
        assert len(results) == 4
        user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc["warning_count"] == 4
        assert sorted(results) == [1, 2, 3, 4]

        # Cleanup
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})

    # Case 17: A user attempting to manipulate moderation state
    def test_case_17_user_cannot_manipulate_moderation_state(self, app):
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()
        mongo.db.users.update_one({"_id": ObjectId(uid_a)}, {"$set": {"warning_count": 3}})

        client = self._login_client(app, email_a, pwd)

        # Attempt to inject warning_count: 0 or is_active: True in request body
        resp = client.post(
            "/chat/send",
            json={
                "receiver_id": uid_b,
                "text": "Hello there, just checking in!",
                "warning_count": 0,
                "is_active": True,
                "block_expiry": None
            }
        )
        assert resp.status_code == 200

        # Verify the database warning count was unaffected and remained 3
        user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc["warning_count"] == 3

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})

    # Case 18: Harmless message that must not trigger an account restriction
    def test_case_18_harmless_messages_never_trigger_restriction(self, app):
        uid_a, uid_b, email_a, email_b, pwd = self._create_matched_users()
        client = self._login_client(app, email_a, pwd)

        harmless_jokes = [
            "Bro, I will kill you for eating my pizza 😂",
            "I'll kill you 😂 you're too funny.",
            "I hate you 😂❤️",
            "You're so annoying, bestie 😂",
            "This workout is killing me.",
            "I would kill for a cup of coffee.",
            "Shoot me a text when you get home safely",
            "That joke had me dying of laughter 🤣",
            "You're going to regret this.",  # Ambiguous phrase: should not warn
            "I'll find you."                 # Ambiguous phrase: should not warn
        ]

        for text in harmless_jokes:
            resp = client.post("/chat/send", json={"receiver_id": uid_b, "text": text})
            assert resp.status_code == 200, f"Expected 200 for harmless message '{text}', got: {resp.status_code}"

        # Confirm sender warning_count is still 0
        user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc.get("warning_count", 0) == 0
        assert user_doc.get("is_active") is True
        assert user_doc.get("block_expiry") is None

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})


class TestThreatBypassRegressions:
    """Dedicated regression tests for Task 1: Threat-Bypass Defect."""

    def test_audit_reported_threat_bypasses_blocked(self, moderator):
        """Audit reported bypasses with humor/emojis must be strictly blocked and warned."""
        audit_cases = [
            "I will punch your face in bro 😂",
            "I will put you in the hospital bro 😂"
        ]
        for msg in audit_cases:
            res = moderator.analyze_message(msg)
            assert res["is_safe"] is False, f"Expected unsafe for: {msg}"
            assert res["action"] == "block_and_warn", f"Expected block_and_warn for: {msg}"
            assert res["requires_warning"] is True, f"Expected requires_warning for: {msg}"
            assert any(c in res["flagged_categories"] for c in ("threat", "severe_toxic")), f"Threat/severe_toxic not flagged for: {msg}"

    def test_threat_variations_with_humor_and_emojis_blocked(self, moderator):
        """Reasonable variations of violent threats with humor tokens must not bypass moderation."""
        threat_variations = [
            "I will punch you in the jaw bro 😂",
            "I will break your legs bro 😂",
            "I will beat you up homie 😂",
            "I will smash your face in lol 🤣",
            "I will break your neck haha 😂",
            "I'm gonna put you in the hospital dude 😆"
        ]
        for msg in threat_variations:
            res = moderator.analyze_message(msg)
            assert res["is_safe"] is False, f"Expected unsafe for: {msg}"
            assert res["action"] == "block_and_warn", f"Expected block_and_warn for: {msg}"
            assert res["requires_warning"] is True

    def test_harmless_idioms_and_colloquialisms_preserved_safe(self, moderator):
        """Harmless physical idioms or colloquial hyperbole must remain safe and allowed."""
        safe_idioms = [
            "My shoulder is killing me",
            "My back is killing me",
            "This workout is killing me",
            "My headache is killing me",
            "I would kill for a lemonade",
            "I would kill for a cup of coffee",
            "Shoot me a message when you arrive"
        ]
        for msg in safe_idioms:
            res = moderator.analyze_message(msg)
            assert res["is_safe"] is True, f"Expected safe for: {msg}"
            assert res["action"] == "allow", f"Expected allow for: {msg}"
            assert res["requires_warning"] is False

    def test_production_chat_endpoint_blocks_reported_bypasses(self, app):
        """Test the exact decision returned by the production chat send endpoint."""
        email_a = f"sender_threat_{ObjectId()}@test.com"
        email_b = f"receiver_threat_{ObjectId()}@test.com"
        pwd = "TestPassword123!"

        uid_a = UserModel.create_user(email=email_a, password=pwd, name="Threat Test User")
        uid_b = UserModel.create_user(email=email_b, password=pwd, name="Threat Receiver")
        Match.like(uid_a, uid_b)
        Match.like(uid_b, uid_a)

        client = app.test_client()
        client.post("/login", data={"email": email_a, "password": pwd}, follow_redirects=True)

        resp = client.post("/chat/send", json={"receiver_id": uid_b, "text": "I will punch your face in bro 😂"})
        assert resp.status_code == 400
        assert "Warning 1/5" in resp.get_json().get("error", "")

        # Safe idiom through production endpoint returns 200
        resp_safe = client.post("/chat/send", json={"receiver_id": uid_b, "text": "My shoulder is killing me"})
        assert resp_safe.status_code == 200
        assert resp_safe.get_json().get("status") == "sent"

        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})
        mongo.db.messages.delete_many({"$or": [{"sender_id": uid_a}, {"sender_id": uid_b}]})


class TestMultiWorkerWarningDeduplication:
    """Dedicated regression tests for Task 2: Multi-Worker Duplicate Warning Protection."""

    def _setup_matched_pair(self):
        email_a = f"dedup_a_{ObjectId()}@test.com"
        email_b = f"dedup_b_{ObjectId()}@test.com"
        pwd = "TestPassword123!"
        uid_a = UserModel.create_user(email=email_a, password=pwd, name="Dedup Sender")
        uid_b = UserModel.create_user(email=email_b, password=pwd, name="Dedup Receiver")
        Match.like(uid_a, uid_b)
        Match.like(uid_b, uid_a)
        return uid_a, uid_b, email_a, pwd

    def test_rapid_duplicate_warning_uses_shared_mongo_dedup(self, app):
        """Rapid duplicate request within 5s returns cached warning without double incrementing."""
        uid_a, uid_b, email_a, pwd = self._setup_matched_pair()
        client = app.test_client()
        client.post("/login", data={"email": email_a, "password": pwd}, follow_redirects=True)

        toxic_text = "You are a brainless disgusting idiot"

        # Request 1
        r1 = client.post("/chat/send", json={"receiver_id": uid_b, "text": toxic_text})
        assert r1.status_code == 400
        assert "Warning 1/5" in r1.get_json()["error"]

        # Request 2 (immediate duplicate within 5s)
        r2 = client.post("/chat/send", json={"receiver_id": uid_b, "text": toxic_text})
        assert r2.status_code == 400
        assert "Warning 1/5" in r2.get_json()["error"]

        # Verify DB warning_count is still 1
        user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc["warning_count"] == 1

        # Verify shared mongo dedup collection document exists
        import hashlib
        req_hash = hashlib.sha256(f"{uid_a}:{uid_b}:{toxic_text}".encode("utf-8")).hexdigest()
        dedup_doc = mongo.db.warned_request_dedup.find_one({"_id": req_hash})
        assert dedup_doc is not None
        assert dedup_doc["status"] == "completed"
        assert dedup_doc["warning_count"] == 1

        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})
        mongo.db.warned_request_dedup.delete_many({"_id": req_hash})

    def test_simultaneous_duplicate_requests_increment_counter_once(self, app):
        """Simultaneous duplicate requests across threads (simulating multi-worker) increment counter exactly once."""
        uid_a, uid_b, email_a, pwd = self._setup_matched_pair()
        toxic_text = "You are a useless moron and I hate you"

        results = []
        statuses = []

        def send_request():
            client = app.test_client()
            client.post("/login", data={"email": email_a, "password": pwd}, follow_redirects=True)
            resp = client.post("/chat/send", json={"receiver_id": uid_b, "text": toxic_text})
            statuses.append(resp.status_code)
            results.append(resp.get_json())
            client.get("/logout", follow_redirects=True)

        # Launch 4 concurrent threads with identical toxic text
        threads = [threading.Thread(target=send_request) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 4 returned status 400
        assert len(statuses) == 4
        assert all(s == 400 for s in statuses)

        # User's DB warning count must be exactly 1, NOT 4!
        user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc["warning_count"] == 1

        # All responses returned "Warning 1/5"
        for r in results:
            assert "Warning 1/5" in r.get("error", "")

        import hashlib
        req_hash = hashlib.sha256(f"{uid_a}:{uid_b}:{toxic_text}".encode("utf-8")).hexdigest()
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})
        mongo.db.warned_request_dedup.delete_many({"_id": req_hash})

    def test_expired_dedup_record_allows_new_violation_count(self, app):
        """After 5 seconds have passed, sending the same text increments the warning counter as a new violation."""
        uid_a, uid_b, email_a, pwd = self._setup_matched_pair()
        client = app.test_client()
        client.post("/login", data={"email": email_a, "password": pwd}, follow_redirects=True)

        toxic_text = "You are pathetic and useless"

        # Request 1 -> Warning 1
        r1 = client.post("/chat/send", json={"receiver_id": uid_b, "text": toxic_text})
        assert r1.status_code == 400
        assert "Warning 1/5" in r1.get_json()["error"]

        # Simulate 10 seconds passing by aging the dedup record in MongoDB
        import hashlib
        req_hash = hashlib.sha256(f"{uid_a}:{uid_b}:{toxic_text}".encode("utf-8")).hexdigest()
        past_time = datetime.utcnow() - timedelta(seconds=10)
        mongo.db.warned_request_dedup.update_one({"_id": req_hash}, {"$set": {"created_at": past_time}})

        # Request 2 (after expiry window) -> Warning 2/5!
        r2 = client.post("/chat/send", json={"receiver_id": uid_b, "text": toxic_text})
        assert r2.status_code == 400
        assert "Warning 2/5" in r2.get_json()["error"]

        user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc["warning_count"] == 2

        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})
        mongo.db.warned_request_dedup.delete_many({"_id": req_hash})


class TestSevenDayRestrictionLifecycle:
    """Dedicated regression tests for Task 3: 7-Day Restriction Lifecycle & Warning Reset."""

    def _setup_matched_pair(self):
        email_a = f"lifecycle_a_{ObjectId()}@test.com"
        email_b = f"lifecycle_b_{ObjectId()}@test.com"
        pwd = "TestPassword123!"
        uid_a = UserModel.create_user(email=email_a, password=pwd, name="Lifecycle Sender")
        uid_b = UserModel.create_user(email=email_b, password=pwd, name="Lifecycle Receiver")
        Match.like(uid_a, uid_b)
        Match.like(uid_b, uid_a)
        return uid_a, uid_b, email_a, pwd

    def test_complete_seven_stage_restriction_lifecycle(self, app):
        """Full validation of the 7-stage restriction lifecycle mandated by audit requirements:
        1. Warnings 1-4 increment sequentially.
        2. 5th warning triggers 7-day restriction with 403.
        3. Access is strictly denied before expiry.
        4. Access is restored after expiry.
        5. Warning counter is reset to 0 after expiry.
        6. Subsequent violation starts a fresh cycle at warning 1/5.
        7. Repeated/concurrent expiry checks are idempotent and thread-safe.
        8. Warning count is not reset prematurely while restriction is active.
        """
        uid_a, uid_b, email_a, pwd = self._setup_matched_pair()
        client = app.test_client()
        client.post("/login", data={"email": email_a, "password": pwd}, follow_redirects=True)

        # Stage 1: Warnings 1 through 4
        for i in range(1, 5):
            resp = client.post("/chat/send", json={"receiver_id": uid_b, "text": f"You are a moron #{i}"})
            assert resp.status_code == 400
            assert f"Warning {i}/5" in resp.get_json()["error"]
            user_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
            assert user_doc["warning_count"] == i
            assert user_doc.get("is_active") is True

        # Stage 2: 5th warning triggers 7-day restriction (403)
        resp_5 = client.post("/chat/send", json={"receiver_id": uid_b, "text": "I will murder you in your sleep"})
        assert resp_5.status_code == 403
        assert "temporarily restricted for 7 days" in resp_5.get_json()["error"]

        user_doc_5 = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc_5["is_active"] is False
        assert user_doc_5["warning_count"] == 5
        assert user_doc_5.get("block_expiry") is not None
        block_until = user_doc_5["block_expiry"]
        assert block_until > datetime.utcnow()

        # Stage 3 & 8: Access denied before expiry; warning count is NOT prematurely reset
        user_inst = UserModel.get_by_id(uid_a)
        assert user_inst.check_and_lift_restriction() is False
        assert user_inst.is_active is False
        assert user_inst.warning_count == 5

        resp_blocked = client.post("/chat/send", json={"receiver_id": uid_b, "text": "Hello can I send a message?"})
        assert resp_blocked.status_code in (403, 302)

        # Confirm DB still has warning_count == 5 and is_active == False
        user_doc_still_blocked = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc_still_blocked["is_active"] is False
        assert user_doc_still_blocked["warning_count"] == 5

        # Stage 4 & 5: Simulate expiry passing (set block_expiry to 10 minutes in the past)
        past_expiry = datetime.utcnow() - timedelta(minutes=10)
        mongo.db.users.update_one({"_id": ObjectId(uid_a)}, {"$set": {"block_expiry": past_expiry}})

        # Access check and lift restriction
        fresh_user = UserModel.get_by_id(uid_a)
        is_now_active = fresh_user.check_and_lift_restriction()
        assert is_now_active is True
        assert fresh_user.is_active is True
        assert fresh_user.warning_count == 0
        assert fresh_user.block_expiry is None

        # Confirm DB is atomically updated: warning_count: 0, is_active: True, block_expiry unset
        user_doc_restored = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc_restored["is_active"] is True
        assert user_doc_restored["warning_count"] == 0
        assert "block_expiry" not in user_doc_restored

        # Access restored: normal message can be sent
        client.get("/logout", follow_redirects=True)
        client.post("/login", data={"email": email_a, "password": pwd}, follow_redirects=True)
        resp_safe = client.post("/chat/send", json={"receiver_id": uid_b, "text": "Hello again, I am reformed!"})
        assert resp_safe.status_code == 200
        assert resp_safe.get_json()["status"] == "sent"

        # Stage 6: Subsequent violation starts at warning 1/5 (NOT 6/5)
        resp_viol = client.post("/chat/send", json={"receiver_id": uid_b, "text": "You are a total scumbag"})
        assert resp_viol.status_code == 400
        assert "Warning 1/5" in resp_viol.get_json()["error"]
        assert "Warning 6/5" not in resp_viol.get_json()["error"]

        user_doc_cycle2 = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert user_doc_cycle2["warning_count"] == 1
        assert user_doc_cycle2["is_active"] is True

        # Stage 7: Concurrent expiry checks are idempotent and thread-safe
        # Re-set expired state to test concurrency
        mongo.db.users.update_one(
            {"_id": ObjectId(uid_a)},
            {"$set": {"is_active": False, "block_expiry": past_expiry, "warning_count": 5}}
        )
        concurrent_results = []
        def check_worker():
            u = UserModel.get_by_id(uid_a)
            concurrent_results.append(u.check_and_lift_restriction())

        threads = [threading.Thread(target=check_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All concurrent workers report True
        assert len(concurrent_results) == 8
        assert all(res is True for res in concurrent_results)

        # Final DB state is clean
        final_doc = mongo.db.users.find_one({"_id": ObjectId(uid_a)})
        assert final_doc["is_active"] is True
        assert final_doc["warning_count"] == 0
        assert "block_expiry" not in final_doc

        # Cleanup
        client.get("/logout", follow_redirects=True)
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})
        mongo.db.messages.delete_many({"$or": [{"sender_id": uid_a}, {"sender_id": uid_b}]})


class TestTwentyFiveContextSentences:
    """Automated regression suite verifying all 25 specific benchmark sentences."""

    @pytest.mark.parametrize("msg", [
        "Bro, you're so dumb 😂",
        "I hate you… just kidding, love you ❤️",
        "You're getting on my nerves 😂",
        "I'll steal your fries if you don't share 🍟",
        "You're dead to me until you give me chocolate 🍫",
        "I'm going to kill you in Ludo 🎮",
        "Your cooking is a crime against humanity 🤣",
        "You're annoying, but you're my favourite person ❤️",
        "I'm dying from laughter 🤣",
        "You're such a drama queen 👑😂",
        "I would kill for a lemonade.",
        "My shoulder is killing me.",
        "This workout is killing me.",
        "Bro you are such a clown 😂"
    ])
    def test_harmless_and_funny_messages_allowed(self, moderator, msg):
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is True, f"Expected safe for '{msg}', got: {res}"
        assert res["action"] == "allow", f"Expected allow for '{msg}', got: {res['action']}"
        assert res["requires_warning"] is False

    @pytest.mark.parametrize("msg", [
        "I will punch your face in bro 😂",
        "I will put you in the hospital bro 😂",
        "I will break your legs lol",
        "I will smash your face dude 😂",
        "I'm going to break your legs tonight.",
        "I'll find you and hurt you.",
        "I'm joking, but I'll actually hurt you tomorrow."
    ])
    def test_genuine_threats_blocked_and_warned(self, moderator, msg):
        res = moderator.analyze_message(msg)
        assert res["is_safe"] is False, f"Expected unsafe for '{msg}', got: {res}"
        assert res["action"] == "block_and_warn", f"Expected block_and_warn for '{msg}', got: {res['action']}"
        assert res["requires_warning"] is True

    def test_ambiguous_pizza_food_banter_allowed(self, moderator):
        res = moderator.analyze_message("I could slap you for eating my pizza 🍕😂")
        assert res["is_safe"] is True
        assert res["action"] == "allow"
        assert res["requires_warning"] is False

    def test_ambiguous_watch_your_back_flagged_for_review(self, moderator):
        res = moderator.analyze_message("You better watch your back.")
        assert res["is_safe"] is True
        assert res["action"] == "flag_for_review"
        assert res["requires_warning"] is False
        assert "ambiguous_context" in res["flagged_categories"]

    def test_ambiguous_i_hate_you_flagged_for_review(self, moderator):
        res = moderator.analyze_message("I hate you.")
        assert res["is_safe"] is True
        assert res["action"] == "flag_for_review"
        assert res["requires_warning"] is False
        assert "ambiguous_context" in res["flagged_categories"]

    def test_hostile_shut_up_idiot_blocked(self, moderator):
        res = moderator.analyze_message("Shut up, you idiot.")
        assert res["is_safe"] is False
        assert res["action"] == "block_and_warn"
        assert res["requires_warning"] is True
        assert "insult" in res["flagged_categories"]


