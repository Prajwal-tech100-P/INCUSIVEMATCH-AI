"""
Comprehensive Text Chat Module Audit Test Suite
===============================================
MCA Major Project: InclusiveMatch AI

Tests the complete lifecycle of text chat communication:
1. Authenticated message sending and real-time delivery via Socket.IO.
2. Socket.IO chat_message and send_message event handlers.
3. MongoDB persistence, retrieval, and chronological ordering.
4. Unauthorized conversation access rejection (non-matched users).
5. Empty, whitespace-only, and oversized (>2000 chars) message rejection.
6. Blocked and restricted user message rejection.
7. Part-ways (unmatched) user rejection and real-time termination.
8. Message deduplication identifier propagation.
9. Inbox aggregation and partner filtering.
10. Conversation deletion isolation.
"""

import os
os.environ["APP_ENV"] = "testing"
os.environ["SECRET_KEY"] = "test-secret-chat-audit-key"

import pytest
from bson import ObjectId
from datetime import datetime, timedelta

from app import create_app
from extensions import mongo, socketio
from models.user import UserModel
from models.match import Match
from models.message import Message, OnlineTracker
from models.restriction import Restriction


@pytest.fixture(scope="module")
def app():
    test_app = create_app()
    test_app.config["TESTING"] = True
    test_app.config["WTF_CSRF_ENABLED"] = False
    with test_app.app_context():
        mongo.db.app_settings.update_one(
            {"setting_id": "global_config"},
            {"$set": {"maintenance_mode": False}},
            upsert=True
        )
    yield test_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def chat_users(app):
    """Create two mutually matched users (Alice and Bob) and an unrelated user (Charlie)."""
    with app.app_context():
        # User A: Alice
        uid_a = UserModel.create_user(
            email=f"alice_chat_{ObjectId()}@test.com",
            password="Password123!",
            name="Alice Walker",
            comm_prefs=["text_chat"]
        )
        # User B: Bob
        uid_b = UserModel.create_user(
            email=f"bob_chat_{ObjectId()}@test.com",
            password="Password123!",
            name="Bob Davis",
            comm_prefs=["text_chat"]
        )
        # User C: Charlie (unmatched)
        uid_c = UserModel.create_user(
            email=f"charlie_chat_{ObjectId()}@test.com",
            password="Password123!",
            name="Charlie Brown",
            comm_prefs=["text_chat"]
        )

        # Make Alice and Bob mutual matches
        Match.like(uid_a, uid_b)
        Match.like(uid_b, uid_a)

        user_a = UserModel.get_by_id(uid_a)
        user_b = UserModel.get_by_id(uid_b)
        user_c = UserModel.get_by_id(uid_c)

    yield user_a, user_b, user_c

    # Cleanup
    with app.app_context():
        for uid in (uid_a, uid_b, uid_c):
            mongo.db.users.delete_one({"_id": ObjectId(uid)})
            mongo.db.messages.delete_many({"$or": [{"sender_id": uid}, {"receiver_id": uid}]})
            mongo.db.user_restrictions.delete_many({"$or": [{"restrictor_id": uid}, {"restricted_user_id": uid}]})


def _make_socket_client(app, user_id):
    flask_client = app.test_client()
    with flask_client.session_transaction() as sess:
        sess["_user_id"] = str(user_id)
        sess["_fresh"] = True
    sock_client = socketio.test_client(app, flask_test_client=flask_client)
    sock_client.emit("join_user_room", {})
    sock_client.get_received()  # flush connect / join acks
    return flask_client, sock_client


def test_01_authenticated_send_and_realtime_socket_delivery(app, chat_users):
    """Verify an authenticated user sends a message via HTTP, persists to MongoDB, and delivers over Socket.IO."""
    user_a, user_b, _ = chat_users

    client_a, socket_a = _make_socket_client(app, user_a.id)
    client_b, socket_b = _make_socket_client(app, user_b.id)

    # Alice sends a message to Bob
    msg_text = "Hello Bob! How are you doing today?"
    resp = client_a.post("/chat/send", json={
        "receiver_id": user_b.id,
        "text": msg_text
    })

    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.get_json()}"
    data = resp.get_json()
    assert data["status"] == "sent"
    assert "message_id" in data
    msg_id = data["message_id"]
    assert msg_id != ""

    # Verify message in MongoDB
    db_msg = mongo.db.messages.find_one({"_id": ObjectId(msg_id)})
    assert db_msg is not None
    assert db_msg["sender_id"] == user_a.id
    assert db_msg["receiver_id"] == user_b.id
    assert db_msg["text"] == msg_text
    assert db_msg["is_read"] is False

    # Verify real-time receipt by Bob over Socket.IO
    received = socket_b.get_received()
    new_msg_events = [e for e in received if e.get("name") == "new_message"]
    assert len(new_msg_events) == 1
    event_args = new_msg_events[0]["args"][0]
    assert event_args["message_id"] == msg_id
    assert event_args["sender_id"] == user_a.id
    assert event_args["receiver_id"] == user_b.id
    assert event_args["text"] == msg_text
    assert "timestamp" in event_args

    socket_a.disconnect()
    socket_b.disconnect()


def test_02_socket_chat_message_event(app, chat_users):
    """Verify messages can also be sent directly via Socket.IO chat_message event."""
    user_a, user_b, _ = chat_users

    client_a, socket_a = _make_socket_client(app, user_a.id)
    client_b, socket_b = _make_socket_client(app, user_b.id)

    # Alice sends via socket event
    res = socket_a.emit("chat_message", {
        "receiver_id": user_b.id,
        "text": "Sent through Socket.IO!"
    }, callback=True)

    assert res is not None
    assert res.get("ok") is True
    assert res.get("status") == "sent"
    assert res.get("message_id") != ""

    # Bob receives it
    received_b = socket_b.get_received()
    new_msgs = [e for e in received_b if e.get("name") == "new_message"]
    assert len(new_msgs) == 1
    assert new_msgs[0]["args"][0]["text"] == "Sent through Socket.IO!"

    socket_a.disconnect()
    socket_b.disconnect()


def test_03_mongodb_persistence_and_chronological_retrieval(app, chat_users):
    """Verify conversation messages are stored and retrieved in correct chronological order."""
    user_a, user_b, _ = chat_users

    # Clear previous messages
    Message.delete_conversation(user_a.id, user_b.id)

    # Insert messages with staggered timestamps
    m1_id = Message.create(user_a.id, user_b.id, "First message from Alice")
    m2_id = Message.create(user_b.id, user_a.id, "Second message from Bob")
    m3_id = Message.create(user_a.id, user_b.id, "Third message from Alice")

    conversation = Message.get_conversation(user_a.id, user_b.id)
    assert len(conversation) == 3
    assert conversation[0]["text"] == "First message from Alice"
    assert conversation[1]["text"] == "Second message from Bob"
    assert conversation[2]["text"] == "Third message from Alice"

    # Verify conversation page renders all 3 messages
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["_user_id"] = user_a.id

    page_resp = client.get(f"/chat/{user_b.id}")
    assert page_resp.status_code == 200
    html = page_resp.get_data(as_text=True)
    assert "First message from Alice" in html
    assert "Second message from Bob" in html
    assert "Third message from Alice" in html
    assert f'data-msg-id="{m1_id}"' in html
    assert f'data-msg-id="{m2_id}"' in html
    assert f'data-msg-id="{m3_id}"' in html


def test_04_unauthorized_conversation_access_rejected(app, chat_users):
    """Verify an authenticated user cannot view or message someone they are not mutually matched with."""
    user_a, _, user_c = chat_users

    client_a = app.test_client()
    with client_a.session_transaction() as sess:
        sess["_user_id"] = user_a.id

    # Alice tries to view chat with Charlie (not matched)
    view_res = client_a.get(f"/chat/{user_c.id}", follow_redirects=False)
    assert view_res.status_code == 302  # Redirected to dashboard

    # Alice tries to send message to Charlie
    send_res = client_a.post("/chat/send", json={
        "receiver_id": user_c.id,
        "text": "Hey Charlie you don't know me"
    })
    assert send_res.status_code == 403
    data = send_res.get_json()
    assert "mutual match" in data["error"].lower()


def test_05_empty_whitespace_and_invalid_payloads_rejected(app, chat_users):
    """Verify empty strings, whitespace, missing fields, and oversized payloads are rejected."""
    user_a, user_b, _ = chat_users

    client_a = app.test_client()
    with client_a.session_transaction() as sess:
        sess["_user_id"] = user_a.id

    # Empty text
    r1 = client_a.post("/chat/send", json={"receiver_id": user_b.id, "text": ""})
    assert r1.status_code == 400
    assert "empty" in r1.get_json()["error"].lower()

    # Whitespace text
    r2 = client_a.post("/chat/send", json={"receiver_id": user_b.id, "text": "   \n\t   "})
    assert r2.status_code == 400
    assert "empty" in r2.get_json()["error"].lower()

    # Missing receiver_id
    r3 = client_a.post("/chat/send", json={"text": "Valid text but no receiver"})
    assert r3.status_code == 400

    # Oversized payload (> 2000 chars)
    oversized = "A" * 2005
    r4 = client_a.post("/chat/send", json={"receiver_id": user_b.id, "text": oversized})
    assert r4.status_code == 400
    assert "maximum 2000" in r4.get_json()["error"]


def test_06_restricted_users_cannot_access_or_send_messages(app, chat_users):
    """Verify that when a user is restricted, messaging is bidirectionally blocked."""
    user_a, user_b, _ = chat_users

    # Alice restricts Bob
    ok, msg = Restriction.create(user_a.id, user_b.id, reason="Harassment")
    assert ok is True

    # Alice cannot send to Bob
    client_a = app.test_client()
    with client_a.session_transaction() as sess:
        sess["_user_id"] = user_a.id
    res_a = client_a.post("/chat/send", json={"receiver_id": user_b.id, "text": "Are you there?"})
    assert res_a.status_code == 403

    # Bob cannot send to Alice
    client_b = app.test_client()
    with client_b.session_transaction() as sess:
        sess["_user_id"] = user_b.id
    res_b = client_b.post("/chat/send", json={"receiver_id": user_a.id, "text": "Please talk to me"})
    assert res_b.status_code == 403

    # Viewing chat is blocked for both
    assert client_a.get(f"/chat/{user_b.id}").status_code == 302
    assert client_b.get(f"/chat/{user_a.id}").status_code == 302


def test_07_part_ways_unmatches_and_blocks_further_chat(app, chat_users):
    """Verify parting ways immediately ends the mutual match and disables chat."""
    user_a, user_b, _ = chat_users

    # Re-establish match
    Match.like(user_a.id, user_b.id)
    Match.like(user_b.id, user_a.id)
    assert Match.are_matched(user_a.id, user_b.id) is True

    # Alice parts ways
    client_a = app.test_client()
    with client_a.session_transaction() as sess:
        sess["_user_id"] = user_a.id

    part_res = client_a.post("/chat/part-ways", json={"partner_id": user_b.id})
    assert part_res.status_code == 200
    assert Match.are_matched(user_a.id, user_b.id) is False

    # Messaging is now blocked
    send_res = client_a.post("/chat/send", json={"receiver_id": user_b.id, "text": "Hello again?"})
    assert send_res.status_code == 403


def test_08_conversation_deletion_isolated(app, chat_users):
    """Verify deleting a conversation purges only messages between the two users."""
    user_a, user_b, user_c = chat_users

    # Create dummy messages
    Message.create(user_a.id, user_b.id, "Msg A to B")
    Message.create(user_a.id, user_c.id, "Msg A to C")

    deleted_cnt = Message.delete_conversation(user_a.id, user_b.id)
    assert deleted_cnt >= 1

    # Messages between A and B are gone
    assert len(Message.get_conversation(user_a.id, user_b.id)) == 0
    # Messages between A and C remain intact
    assert len(Message.get_conversation(user_a.id, user_c.id)) >= 1


def test_09_inbox_aggregation_and_active_partner_filtering(app, chat_users):
    """Verify inbox lists only active mutual matches and displays latest message."""
    user_a, user_b, user_c = chat_users

    # Re-match A and B
    Match.like(user_a.id, user_b.id)
    Match.like(user_b.id, user_a.id)

    Message.create(user_a.id, user_b.id, "Latest message between A and B")
    Message.create(user_c.id, user_a.id, "Unsolicited message from C")

    client_a = app.test_client()
    with client_a.session_transaction() as sess:
        sess["_user_id"] = user_a.id

    inbox_res = client_a.get("/chat")
    assert inbox_res.status_code == 200
    html = inbox_res.get_data(as_text=True)

    # Bob should appear in inbox
    assert user_b.name in html
    # Charlie should NOT appear in inbox (not a mutual match)
    assert user_c.name not in html


def test_10_template_deduplication_and_error_handling(app, chat_users):
    """Verify chat template includes deduplication IDs, seen set tracking, and failure alerts."""
    user_a, user_b, _ = chat_users

    # Match A and B
    Match.like(user_a.id, user_b.id)
    Match.like(user_b.id, user_a.id)

    msg_id = Message.create(user_a.id, user_b.id, "Hello unique bubble!")

    client_a = app.test_client()
    with client_a.session_transaction() as sess:
        sess["_user_id"] = user_a.id

    chat_view_res = client_a.get(f"/chat/{user_b.id}")
    assert chat_view_res.status_code == 200
    html = chat_view_res.get_data(as_text=True)

    # 1. Bubble has data-msg-id
    assert f'data-msg-id="{msg_id}"' in html or f"data-msg-id='{msg_id}'" in html

    # 2. seenChatMsgIds set is instantiated and seeded
    assert "seenChatMsgIds" in html
    assert "bubble.dataset.msgId" in html or "bubble.getAttribute('data-msg-id')" in html or "seenChatMsgIds.add" in html

    # 3. Socket incoming handler deduplication check
    assert "seenChatMsgIds.has" in html

    # 4. Error banner / alert when message send fails
    assert "alert" in html or "toast" in html or "error" in html
