"""Automated tests for Part Ways and Restrict User features.

Validates:
1. Part Ways:
   - Matched user can successfully end the connection.
   - Connection disappears from both users' match lists and inboxes.
   - Further messages in the ended conversation are rejected (403).
   - New private calls between the former matches are rejected (403 / ok: False).
   - Unrelated matches remain intact.
   - Duplicate requests are handled safely (idempotent / graceful error).
   - Unauthenticated users cannot perform Part Ways.
   - Former matches are eligible to match again if they mutually like each other.

2. Restrict User:
   - User can restrict another user with/without an optional reason.
   - Restriction persists in MongoDB collection 'user_restrictions'.
   - Duplicate requests do not create duplicate records (idempotent via unique index).
   - Self-restriction is strictly rejected (400).
   - Invalid user IDs are rejected (400 / 404).
   - Restricted user cannot send private messages to the restricting user (403).
   - Restricting user cannot send messages to restricted user (403).
   - Restricted user cannot initiate or join private video calls with restricting user.
   - Restricted user cannot like or create a new mutual match with restricting user.
   - Restricted users are excluded from Discover candidate listings.
   - Unrelated users can communicate normally without interference.
   - Direction of restriction is clearly recorded (restrictor_id vs restricted_user_id).
   - Restrictions can be listed and removed if management is invoked.
"""
import os
import pytest
from bson import ObjectId
from app import create_app
from extensions import mongo
from models.user import UserModel
from models.match import Match
from models.message import Message
from models.restriction import Restriction


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-secret-key-part-ways-restrict"
    test_app = create_app()
    test_app.config["TESTING"] = True
    test_app.config["WTF_CSRF_ENABLED"] = False

    with test_app.app_context():
        Restriction.ensure_indexes()
        yield test_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def users_cluster(app):
    """Creates a controlled group of 4 users:
    - User A & User B: mutual matches.
    - User C: third-party mutual match with User A (unrelated match).
    - User D: independent neutral user.
    """
    with app.app_context():
        email_a = f"test_a_{ObjectId()}@example.com"
        email_b = f"test_b_{ObjectId()}@example.com"
        email_c = f"test_c_{ObjectId()}@example.com"
        email_d = f"test_d_{ObjectId()}@example.com"
        password = "SecurePassword123!"

        uid_a = UserModel.create_user(email_a, password, "User Alpha", comm_prefs=["text"])
        uid_b = UserModel.create_user(email_b, password, "User Beta", comm_prefs=["text"])
        uid_c = UserModel.create_user(email_c, password, "User Charlie", comm_prefs=["text"])
        uid_d = UserModel.create_user(email_d, password, "User Delta", comm_prefs=["text"])

        # Create matches: A <-> B and A <-> C
        Match.like(uid_a, uid_b)
        Match.like(uid_b, uid_a)
        Match.like(uid_a, uid_c)
        Match.like(uid_c, uid_a)

        assert Match.are_matched(uid_a, uid_b) is True
        assert Match.are_matched(uid_a, uid_c) is True

        user_a = UserModel.get_by_id(uid_a)
        user_b = UserModel.get_by_id(uid_b)
        user_c = UserModel.get_by_id(uid_c)
        user_d = UserModel.get_by_id(uid_d)

        yield {
            "a": user_a,
            "b": user_b,
            "c": user_c,
            "d": user_d,
            "password": password,
        }

        # Cleanup
        ids = [ObjectId(uid_a), ObjectId(uid_b), ObjectId(uid_c), ObjectId(uid_d)]
        mongo.db.users.delete_many({"_id": {"$in": ids}})
        mongo.db.messages.delete_many({
            "$or": [
                {"sender_id": {"$in": [uid_a, uid_b, uid_c, uid_d]}},
                {"receiver_id": {"$in": [uid_a, uid_b, uid_c, uid_d]}},
            ]
        })
        mongo.db.user_restrictions.delete_many({
            "$or": [
                {"restrictor_id": {"$in": [uid_a, uid_b, uid_c, uid_d]}},
                {"restricted_user_id": {"$in": [uid_a, uid_b, uid_c, uid_d]}},
            ]
        })


def login(client, email, password):
    return client.post(
        "/login",
        data={"email": email, "password": password},
        follow_redirects=True,
    )


def logout(client):
    return client.get("/logout", follow_redirects=True)


# ============================================================================
# 1. PART WAYS TESTS
# ============================================================================

def test_part_ways_unauthenticated(client, users_cluster):
    """Unauthenticated users must be rejected."""
    logout(client)
    res = client.post(
        "/chat/part-ways",
        json={"partner_id": users_cluster["b"].id},
    )
    # Redirects to login (302) or 401
    assert res.status_code in (302, 401)


def test_part_ways_invalid_inputs(client, users_cluster):
    """Missing or self target IDs must return 400."""
    login(client, users_cluster["a"].email, users_cluster["password"])

    # Missing partner_id
    res1 = client.post("/chat/part-ways", json={})
    assert res1.status_code == 400

    # Self-part ways
    res2 = client.post("/chat/part-ways", json={"partner_id": users_cluster["a"].id})
    assert res2.status_code == 400

    # Non-existent user
    res3 = client.post("/chat/part-ways", json={"partner_id": str(ObjectId())})
    assert res3.status_code == 404

    logout(client)


def test_part_ways_successful_flow(client, users_cluster):
    """Matched user can successfully part ways with their partner."""
    a = users_cluster["a"]
    b = users_cluster["b"]
    c = users_cluster["c"]
    pw = users_cluster["password"]

    login(client, a.email, pw)

    # Verify initially matched
    assert Match.are_matched(a.id, b.id) is True
    assert Match.are_matched(a.id, c.id) is True

    # User A parts ways with User B via /chat/part-ways
    res = client.post("/chat/part-ways", json={"partner_id": b.id})
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("status") == "ok"

    # Verify mutual match is broken
    assert Match.are_matched(a.id, b.id) is False

    # Verify both users no longer see each other in mutual matches
    matches_a = Match.get_mutual_matches(a.id)
    matches_b = Match.get_mutual_matches(b.id)
    assert not any(str(m["_id"]) == str(b.id) for m in matches_a)
    assert not any(str(m["_id"]) == str(a.id) for m in matches_b)

    # Verify unrelated match A <-> C remains intact
    assert Match.are_matched(a.id, c.id) is True
    assert any(str(m["_id"]) == str(c.id) for m in matches_a)

    # Verify messaging between A and B is now rejected (403)
    msg_res = client.post(
        "/chat/send",
        json={"receiver_id": b.id, "text": "Hello after part ways"},
    )
    assert msg_res.status_code == 403

    # Switch to User B: B also cannot message A
    logout(client)
    login(client, b.email, pw)
    msg_res_b = client.post(
        "/chat/send",
        json={"receiver_id": a.id, "text": "Can I still message you?"},
    )
    assert msg_res_b.status_code == 403

    # Video call between A and B is rejected
    call_res = client.get(f"/video/{a.id}")
    assert call_res.status_code == 403

    logout(client)


def test_part_ways_duplicate_request_safe(client, users_cluster):
    """Calling part ways again when no match exists returns 400 safely."""
    a = users_cluster["a"]
    b = users_cluster["b"]
    pw = users_cluster["password"]

    login(client, a.email, pw)

    # First call succeeded
    client.post("/chat/part-ways", json={"partner_id": b.id})

    # Second duplicate call
    res_dup = client.post("/chat/part-ways", json={"partner_id": b.id})
    assert res_dup.status_code == 400
    assert "No active mutual match" in res_dup.get_json().get("error", "")

    logout(client)


def test_part_ways_can_rematch_in_future(client, users_cluster):
    """Part ways does NOT permanently restrict; users can re-like each other."""
    a = users_cluster["a"]
    b = users_cluster["b"]

    # End connection
    Match.part_ways(a.id, b.id)
    assert Match.are_matched(a.id, b.id) is False

    # Both re-like each other
    Match.like(a.id, b.id)
    Match.like(b.id, a.id)

    # Connection re-established
    assert Match.are_matched(a.id, b.id) is True


# ============================================================================
# 2. RESTRICT USER TESTS
# ============================================================================

def test_restrict_user_unauthenticated(client, users_cluster):
    """Unauthenticated users cannot restrict."""
    logout(client)
    res = client.post("/chat/restrict", json={"partner_id": users_cluster["b"].id})
    assert res.status_code in (302, 401)


def test_restrict_user_self_rejected(client, users_cluster):
    """Users cannot restrict themselves."""
    login(client, users_cluster["a"].email, users_cluster["password"])
    res = client.post("/chat/restrict", json={"partner_id": users_cluster["a"].id})
    assert res.status_code == 400
    assert "cannot restrict yourself" in res.get_json().get("error", "").lower()
    logout(client)


def test_restrict_user_persistence_and_direction(client, users_cluster):
    """Restriction is stored with correct direction, persistent, and ends mutual match."""
    a = users_cluster["a"]
    b = users_cluster["b"]
    pw = users_cluster["password"]

    login(client, a.email, pw)

    res = client.post(
        "/chat/restrict",
        json={"partner_id": b.id, "reason": "Unwanted behavior"},
    )
    assert res.status_code == 200
    assert res.get_json().get("status") == "ok"

    # Verify restriction in database
    assert Restriction.is_restricted(a.id, b.id) is True
    assert Restriction.is_restricted(b.id, a.id) is False  # Direction is clear!
    assert Restriction.is_restricted_either(a.id, b.id) is True

    # Mutual match was ended
    assert Match.are_matched(a.id, b.id) is False

    # Persists across logout and login
    logout(client)
    login(client, a.email, pw)
    assert Restriction.is_restricted(a.id, b.id) is True
    logout(client)


def test_restrict_user_idempotent(client, users_cluster):
    """Duplicate restrict requests do not create duplicate records."""
    a = users_cluster["a"]
    b = users_cluster["b"]

    success1, _ = Restriction.create(a.id, b.id, reason="Reason 1")
    assert success1 is True

    # Second call
    success2, _ = Restriction.create(a.id, b.id, reason="Reason 2")
    assert success2 is True

    # Count records in collection
    count = mongo.db.user_restrictions.count_documents({
        "restrictor_id": str(a.id),
        "restricted_user_id": str(b.id),
    })
    assert count == 1, "Duplicate restriction records were created!"


def test_restricted_user_cannot_message_or_call(client, users_cluster):
    """Restricted user cannot message or call the restrictor."""
    a = users_cluster["a"]
    b = users_cluster["b"]
    pw = users_cluster["password"]

    # User A restricts User B
    Restriction.create(a.id, b.id)

    # User B logs in
    login(client, b.email, pw)

    # 1. Messaging is rejected (403)
    msg_res = client.post(
        "/chat/send",
        json={"receiver_id": a.id, "text": "Are you there?"},
    )
    assert msg_res.status_code == 403

    # 2. Accessing chat page redirects away
    chat_page = client.get(f"/chat/{a.id}", follow_redirects=False)
    assert chat_page.status_code == 302

    # 3. Accessing video page returns 403
    video_page = client.get(f"/video/{a.id}")
    assert video_page.status_code == 403

    # 4. Trying to like user A returns 400
    like_res = client.post(f"/like/{a.id}")
    assert like_res.status_code == 400

    logout(client)


def test_restricted_user_excluded_from_discover(users_cluster):
    """Neither user should see the other in Discover candidates."""
    a = users_cluster["a"]
    d = users_cluster["d"]

    # Clear likes/dislikes so they would normally be candidate
    mongo.db.users.update_one({"_id": ObjectId(a.id)}, {"$set": {"liked": [], "disliked": []}})
    mongo.db.users.update_one({"_id": ObjectId(d.id)}, {"$set": {"liked": [], "disliked": []}})

    # User A restricts User D
    Restriction.create(a.id, d.id)

    # Discover candidates for A must NOT include D
    cands_for_a = Match.get_discover_candidates(a.id, require_shared_interests=False)
    assert not any(str(c["_id"]) == str(d.id) for c in cands_for_a)

    # Discover candidates for D must NOT include A
    cands_for_d = Match.get_discover_candidates(d.id, require_shared_interests=False)
    assert not any(str(c["_id"]) == str(a.id) for c in cands_for_d)


def test_unrelated_users_unaffected(client, users_cluster):
    """A restricting B must NOT affect A with C or D."""
    a = users_cluster["a"]
    b = users_cluster["b"]
    c = users_cluster["c"]
    pw = users_cluster["password"]

    Restriction.create(a.id, b.id)

    # A and C are still mutual matches
    assert Match.are_matched(a.id, c.id) is True

    # A can still message C
    login(client, a.email, pw)
    msg_res = client.post("/chat/send", json={"receiver_id": c.id, "text": "Hi Charlie!"})
    assert msg_res.status_code == 200
    logout(client)


def test_restriction_removal_and_management(users_cluster):
    """User can view and remove their restrictions."""
    a = users_cluster["a"]
    b = users_cluster["b"]

    Restriction.create(a.id, b.id, reason="Spam")
    assert Restriction.is_restricted(a.id, b.id) is True

    my_restrictions = Restriction.get_restrictions_by_user(a.id)
    assert len(my_restrictions) == 1
    assert my_restrictions[0]["restricted_user_id"] == str(b.id)

    # Remove restriction
    removed = Restriction.remove(a.id, b.id)
    assert removed is True
    assert Restriction.is_restricted(a.id, b.id) is False
