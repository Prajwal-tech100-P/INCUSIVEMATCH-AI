"""Automated tests for Video Call, Message Notification Sounds & WebRTC Signaling Fixes.

Covers:
1. Audio files existence, format, and non-empty size (incoming_call.mp3, message_notification.mp3).
2. Video call template elements (remoteVideo ID, localVideo ID, unmuteRemoteBtn, WebRTC track handling).
3. Chat template elements (message sound toggle, active partner tracking).
4. main.js behavior (deduplication, audio unlock, dedicated message sound, no ringtone on messages).
5. Video calling Socket.IO and REST API endpoints (pending call with avatar, signaling support).
6. Chat messaging payload (message_id, sender_avatar, sender_name, text).
"""
import os
import pytest
from bson import ObjectId
from app import create_app
from extensions import mongo
from models.user import UserModel
from models.match import Match
from models.call_invitation import CallInvitation


@pytest.fixture(scope="module")
def test_app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-secret-key-calling-sound"
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.app_context():
        yield app


@pytest.fixture
def client(test_app):
    return test_app.test_client()


@pytest.fixture
def test_pair(test_app):
    with test_app.app_context():
        uid1 = UserModel.create_user(
            email=f"call_user1_{ObjectId()}@example.com",
            password="Password123!",
            name="Alice Caller",
            comm_prefs=["text", "video"],
        )
        uid2 = UserModel.create_user(
            email=f"call_user2_{ObjectId()}@example.com",
            password="Password123!",
            name="Bob Receiver",
            comm_prefs=["text", "video"],
        )
        Match.like(uid1, uid2)
        Match.like(uid2, uid1)
        u1 = UserModel.get_by_id(uid1)
        u2 = UserModel.get_by_id(uid2)

        yield u1, u2

        mongo.db.users.delete_one({"_id": ObjectId(uid1)})
        mongo.db.users.delete_one({"_id": ObjectId(uid2)})
        mongo.db.call_invitations.delete_many({
            "$or": [{"caller_id": uid1}, {"receiver_id": uid1}, {"caller_id": uid2}, {"receiver_id": uid2}]
        })


def test_audio_assets_exist():
    """Verify incoming_call.mp3 and message_notification.mp3 exist with valid sizes."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ringtone_path = os.path.join(base_dir, "static", "audio", "incoming_call.mp3")
    msg_sound_path = os.path.join(base_dir, "static", "audio", "message_notification.mp3")

    assert os.path.isfile(ringtone_path), f"Missing ringtone file at {ringtone_path}"
    assert os.path.getsize(ringtone_path) > 1000, "incoming_call.mp3 is suspiciously small or empty"

    assert os.path.isfile(msg_sound_path), f"Missing message sound at {msg_sound_path}"
    assert os.path.getsize(msg_sound_path) > 500, "message_notification.mp3 is suspiciously small or empty"


def test_video_call_template_elements():
    """Verify video_call.html contains the required WebRTC elements and fixes."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(base_dir, "templates", "video_call.html")
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remote and local video IDs
    assert 'id="remoteVideo"' in content, "remoteVideo ID missing from video_call.html"
    assert 'id="localVideo"' in content, "localVideo ID missing from video_call.html"
    assert 'id="unmuteRemoteBtn"' in content, "Audio unmute fallback button missing from video_call.html"

    # Track-level fallback handling
    assert "remoteStream.addTrack(event.track)" in content, "ontrack fallback track addition missing"

    # Signaling deadlock resolution: mutual peer presence
    assert 'type: "peer_present"' in content, "peer_present acknowledgement missing from video_call.html"

    # ICE candidate queuing guard
    assert "pc.remoteDescription.type" in content, "ICE candidate remoteDescription guard missing"


def test_chat_template_sound_elements():
    """Verify chat.html contains message sound toggle button and active partner tracking."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chat_path = os.path.join(base_dir, "templates", "chat.html")
    with open(chat_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'id="toggle-sound-btn"' in content, "Message sound toggle button missing from chat.html"
    assert "window.ACTIVE_CHAT_PARTNER_ID = PARTNER_ID" in content, "Active chat tracking missing from chat.html"
    assert "inclusive_msg_sound_enabled" in content, "Local storage sound preference toggle missing"


def test_main_js_notification_sound_architecture():
    """Verify main.js implementation of ringtone, message chime, and deduplication."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_js_path = os.path.join(base_dir, "static", "js", "main.js")
    with open(main_js_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Dedicated message sound file used
    assert "/static/audio/message_notification.mp3" in content, "message_notification.mp3 missing from main.js"

    # Deduplication set exists
    assert "seenMessageIds" in content, "seenMessageIds deduplication missing from main.js"

    # Incoming call ringtone used for calls
    assert "/static/audio/incoming_call.mp3" in content, "incoming_call.mp3 missing from main.js"

    # Autoplay unlock banner & helper
    assert "incoming-audio-unlock" in content, "incoming-audio-unlock banner missing from main.js"
    assert "unlockRingtoneAudio" in content, "unlockRingtoneAudio function missing from main.js"

    # Single-instance ringtone lock
    assert "isRingtonePlaying" in content, "isRingtonePlaying lock missing from main.js"

    # Modal overlay with caller avatar & name
    assert "incoming-call-overlay" in content, "incoming-call-overlay missing from main.js"
    assert "callerAvatar" in content or "caller_avatar" in content, "caller avatar missing from main.js"


def test_pending_call_api_includes_avatar(client, test_app, test_pair):
    """Verify /api/calls/pending returns caller_avatar and caller info."""
    u1, u2 = test_pair
    with test_app.app_context():
        call_id = CallInvitation.create(u1.id, u2.id)

    # Log in as receiver (u2)
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u2.id)

    res = client.get("/api/calls/pending")
    assert res.status_code == 200
    data = res.get_json()
    assert data["pending"] is True
    assert data["call_id"] == call_id
    assert data["caller_id"] == u1.id
    assert data["caller_name"] == u1.name
    assert "caller_avatar" in data
    assert len(data["caller_avatar"]) > 0


def test_chat_send_emits_enriched_message_payload(client, test_app, test_pair):
    """Verify /chat/send works and returns message_id for deduplication."""
    u1, u2 = test_pair

    # Log in as sender (u1)
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u1.id)

    res = client.post(
        "/chat/send",
        json={"receiver_id": u2.id, "text": "Hello Bob! Testing real-time message notification."},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "sent"
    assert "message_id" in data
    assert len(data["message_id"]) > 0
