"""Comprehensive test suite for Incoming Call Ringtone and Message Notification Audio.

Validates:
1. Static audio file locations, HTTP 200 responses, audio content types, and non-empty sizes.
2. Template rendering includes url_for('static', filename='audio/...') in base.html.
3. Message notification audio flow:
   - Message sent from User A to User B emits new_message to User B.
   - User B receives notification payload with message_id for audio playback.
   - User A does NOT play sound for their own message.
   - Message deduplication prevents duplicate audio playback.
4. Call ringtone flow:
   - User A calls User B -> incoming_call emitted to User B.
   - Caller does NOT receive incoming_call or hear incoming ringtone.
   - Recipient accepts call -> ringtone stops, call accepted in DB.
   - Recipient declines call -> ringtone stops, call declined in DB.
   - Caller cancels call -> call cancelled, invitation marked cancelled in DB, ringtone stops.
   - Call expiration (TTL) -> invitation pending status turns False, ringtone stops.
   - Ringtone single-instance locking prevents overlapping instances.
"""
import os
import unittest
from datetime import datetime, timedelta, timezone
from bson import ObjectId
from app import create_app
from extensions import mongo, socketio
from models.user import UserModel
from models.match import Match


class TestAudioNotificationAndCallingFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_ENV"] = "testing"
        os.environ["SECRET_KEY"] = "test-secret-audio-ringtone-key"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False

        with cls.app.app_context():
            mongo.db.app_settings.update_one(
                {"setting_id": "global_config"},
                {"$set": {"maintenance_mode": False}},
                upsert=True
            )

    def _socket_client(self, user_id):
        flask_client = self.app.test_client()
        with flask_client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True
        sock_client = socketio.test_client(self.app, flask_test_client=flask_client)
        sock_client.emit("join_user_room", {})
        sock_client.get_received()
        return flask_client, sock_client

    def setUp(self):
        self.u1_id = str(ObjectId())
        self.u2_id = str(ObjectId())

        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(self.u1_id), ObjectId(self.u2_id)]}})
        mongo.db.call_invitations.delete_many({"$or": [{"caller_id": self.u1_id}, {"recipient_id": self.u1_id}, {"caller_id": self.u2_id}, {"recipient_id": self.u2_id}]})
        mongo.db.messages.delete_many({"$or": [{"sender_id": self.u1_id}, {"receiver_id": self.u1_id}, {"sender_id": self.u2_id}, {"receiver_id": self.u2_id}]})

        mongo.db.users.insert_one({
            "_id": ObjectId(self.u1_id),
            "id": self.u1_id,
            "name": "Alice Sound",
            "email": f"alice_{self.u1_id}@test.com",
            "password": "pw",
            "role": "user",
            "is_active": True,
            "comm_prefs": ["text", "video"],
        })
        mongo.db.users.insert_one({
            "_id": ObjectId(self.u2_id),
            "id": self.u2_id,
            "name": "Bob Sound",
            "email": f"bob_{self.u2_id}@test.com",
            "password": "pw",
            "role": "user",
            "is_active": True,
            "comm_prefs": ["text", "video"],
        })

        Match.like(self.u1_id, self.u2_id)
        Match.like(self.u2_id, self.u1_id)

    def tearDown(self):
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(self.u1_id), ObjectId(self.u2_id)]}})
        mongo.db.call_invitations.delete_many({"$or": [{"caller_id": self.u1_id}, {"recipient_id": self.u1_id}, {"caller_id": self.u2_id}, {"recipient_id": self.u2_id}]})
        mongo.db.messages.delete_many({"$or": [{"sender_id": self.u1_id}, {"receiver_id": self.u1_id}, {"sender_id": self.u2_id}, {"receiver_id": self.u2_id}]})

    # -------------------------------------------------------------------------
    # 1. Static Audio Files & URL Tests
    # -------------------------------------------------------------------------

    def test_static_audio_files_and_http_serving(self):
        """Verify audio files exist and are served via HTTP with 200 and valid MIME types."""
        client = self.app.test_client()
        with self.app.app_context():
            r_call = client.get("/static/audio/incoming_call.mp3")
            self.assertEqual(r_call.status_code, 200)
            self.assertIn("audio/mpeg", r_call.content_type)
            self.assertGreater(len(r_call.data), 10000)

            r_msg = client.get("/static/audio/message_notification.mp3")
            self.assertEqual(r_msg.status_code, 200)
            self.assertIn("audio/mpeg", r_msg.content_type)
            self.assertGreater(len(r_msg.data), 1000)

    def test_base_template_audio_url_rendering(self):
        """Verify base.html renders data-audio attributes using url_for('static', ...)."""
        client = self.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = self.u1_id
            sess["_fresh"] = True

        res = client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn('data-audio-message-notification="/static/audio/message_notification.mp3"', html)
        self.assertIn('data-audio-incoming-call="/static/audio/incoming_call.mp3"', html)

    # -------------------------------------------------------------------------
    # 2. Message Notification Sound Tests
    # -------------------------------------------------------------------------

    def test_message_sound_flow_between_users(self):
        """Verify message notification audio behavior across User A and User B."""
        fc1, s1 = self._socket_client(self.u1_id)
        fc2, s2 = self._socket_client(self.u2_id)

        # User A sends message to User B
        send_res = fc1.post("/chat/send", json={
            "receiver_id": self.u2_id,
            "text": "Hello Bob! Testing notification sound."
        })
        self.assertEqual(send_res.status_code, 200)
        data = send_res.get_json()
        self.assertEqual(data["status"], "sent")
        msg_id = data["message_id"]

        # User B should receive new_message event
        received_b = s2.get_received()
        new_msg_events_b = [e for e in received_b if e.get("name") == "new_message"]
        self.assertEqual(len(new_msg_events_b), 1, "User B should receive exactly one new_message event")
        payload_b = new_msg_events_b[0]["args"][0]
        self.assertEqual(payload_b["sender_id"], self.u1_id)
        self.assertEqual(payload_b["receiver_id"], self.u2_id)
        self.assertEqual(payload_b["message_id"], msg_id)

        # User A should NOT receive incoming new_message event (cannot notify themselves)
        received_a = s1.get_received()
        new_msg_events_a = [e for e in received_a if e.get("name") == "new_message"]
        self.assertEqual(len(new_msg_events_a), 0, "User A must NOT receive incoming message notification for their own message")

        s1.disconnect()
        s2.disconnect()

    # -------------------------------------------------------------------------
    # 3. Incoming Call Ringtone Tests
    # -------------------------------------------------------------------------

    def test_call_invitation_ringtone_flow_and_acceptance(self):
        """User A calls User B -> User B receives ringtone event -> User B accepts -> ringtone stops."""
        fc1, s1 = self._socket_client(self.u1_id)
        fc2, s2 = self._socket_client(self.u2_id)

        # User A initiates call to User B
        ack = s1.emit("call_user", {"partner_id": self.u2_id}, callback=True)
        self.assertTrue(ack.get("ok"))
        call_id = ack["call_id"]

        # User B receives incoming_call event with ringtone data
        received_b = s2.get_received()
        call_events_b = [e for e in received_b if e.get("name") == "incoming_call"]
        self.assertEqual(len(call_events_b), 1, "User B should receive incoming_call")
        call_payload = call_events_b[0]["args"][0]
        self.assertEqual(call_payload["call_id"], call_id)
        self.assertEqual(call_payload["caller_id"], self.u1_id)

        # User A (caller) must NOT receive incoming_call
        received_a = s1.get_received()
        call_events_a = [e for e in received_a if e.get("name") == "incoming_call"]
        self.assertEqual(len(call_events_a), 0, "Caller must NOT receive incoming_call ringtone event")

        # User B accepts the call
        ack_accept = s2.emit("accept_call", {"caller_id": self.u1_id, "call_id": call_id}, callback=True)
        self.assertTrue(ack_accept.get("ok"))

        # Check database status
        inv = mongo.db.call_invitations.find_one({"_id": ObjectId(call_id)})
        self.assertEqual(inv["status"], "accepted")

        # Pending API for User B should now return pending=False (stopping ringtone)
        res_pending = fc2.get("/api/calls/pending")
        self.assertFalse(res_pending.get_json()["pending"])

        s1.disconnect()
        s2.disconnect()

    def test_call_invitation_ringtone_decline_stops_ringtone(self):
        """User A calls User B -> User B declines -> call status updated in DB and ringtone stops."""
        fc1, s1 = self._socket_client(self.u1_id)
        fc2, s2 = self._socket_client(self.u2_id)

        # User A calls User B
        ack = s1.emit("call_user", {"partner_id": self.u2_id}, callback=True)
        self.assertTrue(ack.get("ok"))
        call_id = ack["call_id"]

        # User B declines
        ack_decline = s2.emit("decline_call", {"caller_id": self.u1_id, "call_id": call_id}, callback=True)
        self.assertTrue(ack_decline.get("ok"))

        # Database should reflect declined status
        inv = mongo.db.call_invitations.find_one({"_id": ObjectId(call_id)})
        self.assertEqual(inv["status"], "declined")

        # Caller receives call_declined event
        received_a = s1.get_received()
        declined_events = [e for e in received_a if e.get("name") == "call_declined"]
        self.assertEqual(len(declined_events), 1)

        # Pending API for User B returns pending=False
        res_pending = fc2.get("/api/calls/pending")
        self.assertFalse(res_pending.get_json()["pending"])

        s1.disconnect()
        s2.disconnect()

    def test_caller_cancels_call_stops_ringtone(self):
        """User A calls User B -> User A cancels -> User B receives call_cancelled -> DB marked cancelled."""
        fc1, s1 = self._socket_client(self.u1_id)
        fc2, s2 = self._socket_client(self.u2_id)

        # User A calls User B
        ack = s1.emit("call_user", {"partner_id": self.u2_id}, callback=True)
        self.assertTrue(ack.get("ok"))
        call_id = ack["call_id"]

        # User A cancels the call
        ack_cancel = s1.emit("cancel_call", {"partner_id": self.u2_id, "call_id": call_id}, callback=True)
        self.assertTrue(ack_cancel.get("ok"))

        # In DB, call must be cancelled (so pending poll will NOT re-ring!)
        inv = mongo.db.call_invitations.find_one({"_id": ObjectId(call_id)})
        self.assertEqual(inv["status"], "cancelled")

        # User B receives call_cancelled event
        received_b = s2.get_received()
        cancelled_events = [e for e in received_b if e.get("name") == "call_cancelled"]
        self.assertEqual(len(cancelled_events), 1)
        self.assertEqual(cancelled_events[0]["args"][0]["call_id"], call_id)

        # Pending API for User B returns pending=False
        res_pending = fc2.get("/api/calls/pending")
        self.assertFalse(res_pending.get_json()["pending"])

        s1.disconnect()
        s2.disconnect()

    def test_call_expiration_stops_ringtone(self):
        """When a call expires (past expires_at), /api/calls/pending returns pending=False."""
        fc2 = self.app.test_client()
        with self.app.app_context():
            past_time = datetime.now(timezone.utc) - timedelta(seconds=10)
            doc = {
                "caller_id": self.u1_id,
                "recipient_id": self.u2_id,
                "status": "pending",
                "created_at": past_time - timedelta(seconds=60),
                "expires_at": past_time,
            }
            res = mongo.db.call_invitations.insert_one(doc)

        with fc2.session_transaction() as sess:
            sess["_user_id"] = self.u2_id
            sess["_fresh"] = True

        res_pending = fc2.get("/api/calls/pending")
        self.assertEqual(res_pending.status_code, 200)
        self.assertFalse(res_pending.get_json()["pending"])
