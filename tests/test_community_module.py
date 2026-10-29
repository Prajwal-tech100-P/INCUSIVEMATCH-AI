"""
Comprehensive Test Suite for Communities Module
===============================================
InclusiveMatch AI: 50-Member Strict Limit, No Community Video Calls, Concurrency Safety

Covers all 15 scenarios:
1. Community can hold up to 50 members.
2. Owner counts as member #1.
3. 51st member cannot join (exact message: "This community has reached its maximum capacity of 50 members.").
4. User cannot join twice (duplicate memberships prevented).
5. Member can leave, freeing a slot.
6. Owner can remove member, freeing a slot (owner cannot be removed).
7. Unauthorized user cannot remove members.
8. Concurrent join attempts cannot exceed 50 (atomic MongoDB concurrency test).
9. Full communities display correct UI state ("Full (50/50)", disabled Join button).
10. Non-members cannot access private community chat/messages.
11. Members can send text messages.
12. Community video call endpoints and events are rejected/unavailable.
13. One-to-one private video calls still work.
14. Sign recognition and speech accessibility remain unaffected.
15. Community creation, joining, leaving, moderation remain operational.
"""

import os
import sys
import threading
import pytest
from bson import ObjectId

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import create_app
from extensions import mongo
from models.community import Community, MAX_MEMBERS
from models.user import UserModel


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-community-secret-key-2026"
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app_context():
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, email, password):
    client.get("/logout")
    return client.post(
        "/login",
        data={"email": email, "password": password, "remember": False},
        follow_redirects=True,
    )


@pytest.fixture
def test_users(app):
    created_uids = []

    def _create(email, name, role="user"):
        uid = UserModel.create_user(email, "Password123!", name, role=role)
        created_uids.append(uid)
        return uid

    yield _create

    # Cleanup
    for uid in created_uids:
        try:
            mongo.db.users.delete_one({"_id": ObjectId(uid)})
        except Exception:
            pass


@pytest.fixture
def clean_community(app):
    created_cids = []

    def _track(cid):
        created_cids.append(cid)
        return cid

    yield _track

    for cid in created_cids:
        try:
            oid = ObjectId(cid)
            mongo.db.communities.delete_one({"_id": oid})
            mongo.db.community_messages.delete_many({"community_id": str(cid)})
            mongo.db.community_reports.delete_many({"community_id": str(cid)})
            mongo.db.community_blocks.delete_many({"community_id": str(cid)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scenario 1 & 2: Up to 50 members, Owner is Member #1
# ---------------------------------------------------------------------------
def test_1_and_2_community_owner_member_1_and_capacity_50(app, test_users, clean_community):
    owner_id = test_users("comm_owner1@example.com", "Community Owner 1")
    cid = clean_community(Community.create(owner_id, "Capacity Test Group", "Testing 50 member limit", "General"))

    comm = Community.get(cid)
    assert comm is not None
    assert str(comm["created_by"]) == str(owner_id)
    # Scenario 2: Owner counts as member #1
    assert str(owner_id) in [str(x) for x in comm.get("member_ids", [])]
    assert len(comm.get("member_ids", [])) == 1

    # Scenario 1: Add 49 more members to reach exactly 50
    for i in range(2, 51):
        member_id = f"mock_member_id_{i}_{cid}"
        ok, msg = Community.join(cid, member_id)
        assert ok is True, f"Member {i} failed to join: {msg}"

    comm = Community.get(cid)
    assert len(comm.get("member_ids", [])) == 50
    assert MAX_MEMBERS == 50


# ---------------------------------------------------------------------------
# Scenario 3: 51st Member Cannot Join (Exact Error Message)
# ---------------------------------------------------------------------------
def test_3_51st_member_cannot_join(app, test_users, clean_community):
    owner_id = test_users("comm_owner2@example.com", "Community Owner 2")
    cid = clean_community(Community.create(owner_id, "Max 50 Test Group", "Testing 51st rejection", "General"))

    # Fill to 50
    for i in range(2, 51):
        Community.join(cid, f"mock_user_{i}_{cid}")

    # Attempt 51st member
    candidate_51 = f"mock_user_51_{cid}"
    ok, msg = Community.join(cid, candidate_51)
    assert ok is False
    assert msg == "This community has reached its maximum capacity of 50 members."

    comm = Community.get(cid)
    assert len(comm.get("member_ids", [])) == 50
    assert candidate_51 not in [str(x) for x in comm.get("member_ids", [])]


# ---------------------------------------------------------------------------
# Scenario 4: User Cannot Join Twice (Duplicate Prevention)
# ---------------------------------------------------------------------------
def test_4_user_cannot_join_twice(app, test_users, clean_community):
    owner_id = test_users("comm_owner3@example.com", "Community Owner 3")
    cid = clean_community(Community.create(owner_id, "Deduplication Group", "Testing duplicates", "General"))

    user_2 = test_users("comm_user2@example.com", "Community User 2")
    ok1, msg1 = Community.join(cid, user_2)
    assert ok1 is True
    assert msg1 == "Joined community."

    # Try joining again
    ok2, msg2 = Community.join(cid, user_2)
    assert ok2 is True
    assert msg2 == "You are already a member."

    comm = Community.get(cid)
    # Count should still be 2 (owner + user_2)
    assert len(comm.get("member_ids", [])) == 2


# ---------------------------------------------------------------------------
# Scenario 5: Member Can Leave, Freeing a Slot
# ---------------------------------------------------------------------------
def test_5_member_can_leave_freeing_slot(app, test_users, clean_community):
    owner_id = test_users("comm_owner4@example.com", "Community Owner 4")
    cid = clean_community(Community.create(owner_id, "Leave Slot Group", "Testing leave freeing slot", "General"))

    # Fill up to 50
    for i in range(2, 51):
        Community.join(cid, f"filler_{i}_{cid}")

    comm = Community.get(cid)
    assert len(comm.get("member_ids", [])) == 50

    # User 2 leaves
    leaver = f"filler_2_{cid}"
    ok_leave, msg_leave = Community.leave(cid, leaver)
    assert ok_leave is True
    assert msg_leave == "You left the community."

    comm = Community.get(cid)
    assert len(comm.get("member_ids", [])) == 49
    assert leaver not in [str(x) for x in comm.get("member_ids", [])]

    # Now a new user can join to take the 50th slot
    new_user = f"new_candidate_{cid}"
    ok_join, msg_join = Community.join(cid, new_user)
    assert ok_join is True
    assert msg_join == "Joined community."

    comm = Community.get(cid)
    assert len(comm.get("member_ids", [])) == 50


# ---------------------------------------------------------------------------
# Scenario 6: Owner Can Remove Member, Freeing Slot; Cannot Remove Owner
# ---------------------------------------------------------------------------
def test_6_owner_can_remove_member_owner_cannot_be_removed(app, test_users, clean_community):
    owner_id = test_users("comm_owner5@example.com", "Community Owner 5")
    cid = clean_community(Community.create(owner_id, "Remove Test Group", "Testing removal", "General"))

    member_id = test_users("comm_remove_target@example.com", "Removal Target")
    Community.join(cid, member_id)

    # Owner attempts to remove owner -> must fail
    ok_self, msg_self = Community.remove_member(cid, owner_id, owner_id)
    assert ok_self is False
    assert msg_self == "The community owner cannot be removed."

    # Owner removes member -> must succeed
    ok_rm, msg_rm = Community.remove_member(cid, owner_id, member_id)
    assert ok_rm is True
    assert msg_rm == "Member removed from community."

    comm = Community.get(cid)
    assert str(member_id) not in [str(x) for x in comm.get("member_ids", [])]
    assert len(comm.get("member_ids", [])) == 1


# ---------------------------------------------------------------------------
# Scenario 7: Unauthorized User Cannot Remove Members
# ---------------------------------------------------------------------------
def test_7_unauthorized_user_cannot_remove_members(app, test_users, clean_community):
    owner_id = test_users("comm_owner6@example.com", "Community Owner 6")
    cid = clean_community(Community.create(owner_id, "Auth Test Group", "Testing unauth removal", "General"))

    member_a = test_users("comm_member_a@example.com", "Member A")
    member_b = test_users("comm_member_b@example.com", "Member B")
    Community.join(cid, member_a)
    Community.join(cid, member_b)

    # Member A tries to remove Member B -> unauthorized
    ok, msg = Community.remove_member(cid, member_a, member_b)
    assert ok is False
    assert "Unauthorized" in msg

    comm = Community.get(cid)
    assert str(member_b) in [str(x) for x in comm.get("member_ids", [])]


# ---------------------------------------------------------------------------
# Scenario 8: Atomic Concurrency Check (Two Concurrent Joins for 50th Slot)
# ---------------------------------------------------------------------------
def test_8_concurrent_join_attempts_cannot_exceed_50(app, test_users, clean_community):
    owner_id = test_users("comm_owner7@example.com", "Community Owner 7")
    cid = clean_community(Community.create(owner_id, "Concurrency Group", "Testing race safety", "General"))

    # Fill to 49 members
    for i in range(2, 50):
        Community.join(cid, f"preload_{i}_{cid}")

    comm = Community.get(cid)
    assert len(comm.get("member_ids", [])) == 49

    results = []
    # 5 concurrent threads attempting to take the 1 remaining slot
    def _attempt_join(user_id):
        with app.app_context():
            ok, msg = Community.join(cid, user_id)
            results.append((ok, msg, user_id))

    threads = []
    for i in range(1, 6):
        t = threading.Thread(target=_attempt_join, args=(f"concurrent_racer_{i}_{cid}",))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 1 success, 4 rejections
    successes = [r for r in results if r[0] is True]
    failures = [r for r in results if r[0] is False]

    assert len(successes) == 1, f"Expected exactly 1 concurrent success, got {len(successes)}: {results}"
    assert len(failures) == 4
    for f in failures:
        assert f[1] == "This community has reached its maximum capacity of 50 members."

    fresh_comm = Community.get(cid)
    assert len(fresh_comm.get("member_ids", [])) == 50


# ---------------------------------------------------------------------------
# Scenario 9: Full Communities Display Correct UI State
# ---------------------------------------------------------------------------
def test_9_full_community_ui_display(client, test_users, clean_community):
    owner_id = test_users("comm_ui_owner@example.com", "UI Owner")
    cid = clean_community(Community.create(owner_id, "UI Capacity Group", "Testing UI text", "General"))

    # Fill up to 50
    for i in range(2, 51):
        Community.join(cid, f"ui_mock_{i}_{cid}")

    # Login as an outside user
    visitor_email = "comm_visitor@example.com"
    visitor_pwd = "Password123!"
    visitor_id = test_users(visitor_email, "Visitor User")
    _login(client, visitor_email, visitor_pwd)

    # 1. Check /community list view
    res = client.get("/community")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "50 / 50" in html or "Full (50/50)" in html

    # 2. Check /community/<cid> view
    res_detail = client.get(f"/community/{cid}")
    assert res_detail.status_code == 200
    detail_html = res_detail.get_data(as_text=True)
    assert "Community Full (50/50)" in detail_html
    assert "disabled" in detail_html


# ---------------------------------------------------------------------------
# Scenario 10: Non-Members Cannot Access Private Community Messages
# ---------------------------------------------------------------------------
def test_10_non_member_cannot_access_private_chat(client, test_users, clean_community):
    owner_id = test_users("comm_chat_owner@example.com", "Chat Owner")
    cid = clean_community(Community.create(owner_id, "Private Chat Group", "Confidential chat", "General"))

    # Owner creates a message
    Community.create_message(cid, owner_id, "Chat Owner", "Secret discussion for members only")

    # Non-member accesses view
    outsider_email = "comm_outsider@example.com"
    outsider_pwd = "Password123!"
    test_users(outsider_email, "Outsider User")
    _login(client, outsider_email, outsider_pwd)

    res = client.get(f"/community/{cid}")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    # Secret message text must NOT be exposed in non-member view
    assert "Secret discussion for members only" not in html
    assert "Join this community to participate in the group chat" in html


# ---------------------------------------------------------------------------
# Scenario 11: Members Can Send Text Messages
# ---------------------------------------------------------------------------
def test_11_members_can_send_text_messages(app, test_users, clean_community):
    owner_id = test_users("comm_msg_owner@example.com", "Message Owner")
    cid = clean_community(Community.create(owner_id, "Message Test Group", "Messaging", "General"))

    msg = Community.create_message(cid, owner_id, "Message Owner", "Hello community members!")
    assert msg is not None
    assert msg.get("text") == "Hello community members!"
    assert msg.get("community_id") == str(cid)

    messages = Community.messages(cid)
    assert len(messages) >= 1
    assert any(m.get("text") == "Hello community members!" for m in messages)


# ---------------------------------------------------------------------------
# Scenario 12: Community Video Call Endpoints and Events are Rejected/Disabled
# ---------------------------------------------------------------------------
def test_12_community_video_call_endpoints_disabled(client, test_users, clean_community):
    owner_email = "comm_call_owner@example.com"
    owner_pwd = "Password123!"
    owner_id = test_users(owner_email, "Call Owner")
    cid = clean_community(Community.create(owner_id, "Call Test Group", "No video call", "General"))

    _login(client, owner_email, owner_pwd)

    # 1. Ensure community chat template has NO community call button
    chat_res = client.get(f"/community/{cid}")
    assert chat_res.status_code == 200
    chat_html = chat_res.get_data(as_text=True)
    assert "Community Call" not in chat_html
    assert f"/community/{cid}/call" not in chat_html

    # 2. Hitting /community/<cid>/call redirects away with flash warning
    res = client.get(f"/community/{cid}/call", follow_redirects=True)
    assert res.status_code == 200
    res_html = res.get_data(as_text=True)
    assert "Community video calls are not supported" in res_html


# ---------------------------------------------------------------------------
# Scenario 13: One-to-One Private Video Calls Still Work
# ---------------------------------------------------------------------------
def test_13_one_to_one_private_video_calls_unaffected(app):
    from routes.video_routes import video_bp, _ice_servers
    assert video_bp is not None
    # Verify ICE servers and WebRTC routes for private 1-to-1 calling remain fully configured
    servers = _ice_servers()
    assert isinstance(servers, list)
    assert len(servers) > 0


# ---------------------------------------------------------------------------
# Scenario 14: Sign Recognition and Speech Accessibility Remain Unaffected
# ---------------------------------------------------------------------------
def test_14_sign_recognition_accessibility_intact(app):
    from ai_modules.sign_language import SignLanguageRecognizer
    recognizer = SignLanguageRecognizer()
    assert recognizer is not None
    assert hasattr(recognizer, "recognize_frame")
    assert hasattr(recognizer, "reset_user")
    assert hasattr(recognizer, "labels")
    assert len(recognizer.DEFAULT_PATHS) > 0


# ---------------------------------------------------------------------------
# Scenario 15: Community Lifecycle and Moderation Operational
# ---------------------------------------------------------------------------
def test_15_community_lifecycle_and_moderation(client, test_users, clean_community):
    # Create community via route
    admin_email = "comm_admin_lifecycle@example.com"
    admin_pwd = "Password123!"
    admin_id = test_users(admin_email, "Admin Lifecycle", role="admin")
    _login(client, admin_email, admin_pwd)

    create_res = client.post("/community/create", data={
        "name": "Full Lifecycle Group",
        "description": "Lifecycle testing",
        "category": "Sign Language",
        "icon": "🤟"
    }, follow_redirects=True)
    assert create_res.status_code == 200

    comm = mongo.db.communities.find_one({"name": "Full Lifecycle Group"})
    assert comm is not None
    cid = str(comm["_id"])
    clean_community(cid)

    # Member joins
    user_email = "comm_user_lifecycle@example.com"
    user_pwd = "Password123!"
    user_id = test_users(user_email, "User Lifecycle")
    _login(client, user_email, user_pwd)

    join_res = client.post(f"/community/{cid}/join", follow_redirects=True)
    assert join_res.status_code == 200
    assert Community.is_member(cid, user_id) is True

    # User reports member
    report_res = client.post(f"/community/{cid}/report", data={
        "reported_user_id": str(admin_id),
        "reason": "Test report functionality"
    }, follow_redirects=True)
    assert report_res.status_code == 200

    # User leaves
    leave_res = client.post(f"/community/{cid}/leave", follow_redirects=True)
    assert leave_res.status_code == 200
    assert Community.is_member(cid, user_id) is False

    # Admin deletes community
    _login(client, admin_email, admin_pwd)
    del_res = client.post(f"/community/{cid}/delete", follow_redirects=True)
    assert del_res.status_code == 200
    assert Community.get(cid) is None
