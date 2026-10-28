"""Comprehensive automated test suite for Reliable Two-Sided Video Call Disconnection in InclusiveMatch AI.

Covers:
1. User A ends call -> User B receives call_ended, call_terminated, and peer_left; CallInvitation in MongoDB marked 'ended'.
2. User B ends call -> User A receives call_ended, call_terminated, and peer_left; CallInvitation in MongoDB marked 'ended'.
3. 'leave_call' event also reliably terminates call and notifies remote participant.
4. Security requirement: Unrelated authenticated user (e.g. Charlie) cannot terminate Alice & Bob's call by guessing call_id.
5. Duplicate end-call / leave-call requests are handled safely without errors.
6. Unexpected socket disconnection of an active participant terminates call and notifies remaining peer.
7. Frontend template WebRTC and DOM teardown validation:
   - All local media tracks stopped via track.stop().
   - Active RTCPeerConnection closed via pc.close().
   - Both localVideo.srcObject and remoteVideo.srcObject set to null.
   - Remote participant receives call termination events and exits to chat.
   - Reconnection timers and duplicate cleanup guards prevent automatic reconnect or errors.
"""
import os
import unittest
from bson import ObjectId
from app import create_app
from extensions import mongo, socketio
from models.user import UserModel
from models.match import Match
from models.call_invitation import CallInvitation


class TestTwoSidedCallDisconnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_ENV"] = "testing"
        os.environ["SECRET_KEY"] = "test-secret-two-sided-call-disconnect"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def setUp(self):
        self.u1_id = UserModel.create_user("alice_term@test.com", "Pw123!test", "Alice Disconnect")
        self.u2_id = UserModel.create_user("bob_term@test.com", "Pw123!test", "Bob Disconnect")
        self.u3_id = UserModel.create_user("charlie_term@test.com", "Pw123!test", "Charlie Stranger")

        Match.like(self.u1_id, self.u2_id)
        Match.like(self.u2_id, self.u1_id)

    def tearDown(self):
        mongo.db.users.delete_many({"_id": {"$in": [
            ObjectId(self.u1_id), ObjectId(self.u2_id), ObjectId(self.u3_id)
        ]}})
        mongo.db.matches.delete_many({"$or": [
            {"user_a": self.u1_id}, {"user_b": self.u1_id},
            {"user_a": self.u2_id}, {"user_b": self.u2_id},
            {"user_a": self.u3_id}, {"user_b": self.u3_id},
        ]})
        mongo.db[CallInvitation.COLLECTION].delete_many({"$or": [
            {"caller_id": self.u1_id}, {"recipient_id": self.u1_id},
            {"caller_id": self.u2_id}, {"recipient_id": self.u2_id},
            {"caller_id": self.u3_id}, {"recipient_id": self.u3_id},
        ]})

    def _socket_client(self, user_id):
        flask_client = self.app.test_client()
        with flask_client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True
        c_val = flask_client.get_cookie("session").value
        return socketio.test_client(self.app, headers={"Cookie": f"session={c_val}"})

    # -------------------------------------------------------------------------
    # 1. User A Ends Call -> User B Disconnected & Notified
    # -------------------------------------------------------------------------

    def test_user_a_ends_call_notifies_and_disconnects_user_b(self):
        """When User A clicks End Call, User B is notified and CallInvitation is marked ended."""
        call_id = CallInvitation.create(self.u1_id, self.u2_id)
        CallInvitation.set_status(call_id, self.u2_id, "accepted")

        s1 = self._socket_client(self.u1_id)
        s2 = self._socket_client(self.u2_id)

        s1.emit("join_call", {"partner_id": self.u2_id, "call_id": call_id})
        s2.emit("join_call", {"partner_id": self.u1_id, "call_id": call_id})
        s1.get_received()
        s2.get_received()

        # User A ends the call
        ack = s1.emit("end_call", {"partner_id": self.u2_id, "call_id": call_id}, callback=True)
        if ack:
            self.assertTrue(ack.get("ok"))

        # User B must receive call termination events
        rec2 = s2.get_received()
        event_names = [e["name"] for e in rec2]
        self.assertIn("call_ended", event_names)
        self.assertIn("call_terminated", event_names)
        self.assertIn("peer_left", event_names)

        ended_event = next(e for e in rec2 if e["name"] == "call_ended")
        self.assertEqual(ended_event["args"][0]["ended_by"], self.u1_id)
        self.assertEqual(ended_event["args"][0]["call_id"], call_id)

        # Database state verification
        doc = mongo.db[CallInvitation.COLLECTION].find_one({"_id": ObjectId(call_id)})
        self.assertIsNotNone(doc)
        self.assertEqual(doc.get("status"), "ended")
        self.assertEqual(doc.get("ended_by"), self.u1_id)
        self.assertIn("ended_at", doc)

        s1.disconnect()
        s2.disconnect()

    # -------------------------------------------------------------------------
    # 2. User B Ends Call -> User A Disconnected & Notified
    # -------------------------------------------------------------------------

    def test_user_b_ends_call_notifies_and_disconnects_user_a(self):
        """When User B clicks End Call, User A is notified and CallInvitation is marked ended."""
        call_id = CallInvitation.create(self.u1_id, self.u2_id)
        CallInvitation.set_status(call_id, self.u2_id, "accepted")

        s1 = self._socket_client(self.u1_id)
        s2 = self._socket_client(self.u2_id)

        s1.emit("join_call", {"partner_id": self.u2_id, "call_id": call_id})
        s2.emit("join_call", {"partner_id": self.u1_id, "call_id": call_id})
        s1.get_received()
        s2.get_received()

        # User B ends the call
        ack = s2.emit("end_call", {"partner_id": self.u1_id, "call_id": call_id}, callback=True)
        if ack:
            self.assertTrue(ack.get("ok"))

        # User A must receive call termination events
        rec1 = s1.get_received()
        event_names = [e["name"] for e in rec1]
        self.assertIn("call_ended", event_names)
        self.assertIn("call_terminated", event_names)
        self.assertIn("peer_left", event_names)

        ended_event = next(e for e in rec1 if e["name"] == "call_ended")
        self.assertEqual(ended_event["args"][0]["ended_by"], self.u2_id)
        self.assertEqual(ended_event["args"][0]["call_id"], call_id)

        # Database state verification
        doc = mongo.db[CallInvitation.COLLECTION].find_one({"_id": ObjectId(call_id)})
        self.assertIsNotNone(doc)
        self.assertEqual(doc.get("status"), "ended")
        self.assertEqual(doc.get("ended_by"), self.u2_id)

        s1.disconnect()
        s2.disconnect()

    # -------------------------------------------------------------------------
    # 3. leave_call Triggers Call Termination
    # -------------------------------------------------------------------------

    def test_leave_call_triggers_two_sided_termination(self):
        """leave_call event terminates the call and notifies remote peer."""
        call_id = CallInvitation.create(self.u1_id, self.u2_id)
        CallInvitation.set_status(call_id, self.u2_id, "accepted")

        s1 = self._socket_client(self.u1_id)
        s2 = self._socket_client(self.u2_id)

        s1.emit("join_call", {"partner_id": self.u2_id, "call_id": call_id})
        s2.emit("join_call", {"partner_id": self.u1_id, "call_id": call_id})
        s1.get_received()
        s2.get_received()

        s1.emit("leave_call", {"partner_id": self.u2_id, "call_id": call_id})

        rec2 = s2.get_received()
        event_names = [e["name"] for e in rec2]
        self.assertIn("call_ended", event_names)
        self.assertIn("call_terminated", event_names)
        self.assertIn("peer_left", event_names)

        doc = mongo.db[CallInvitation.COLLECTION].find_one({"_id": ObjectId(call_id)})
        self.assertEqual(doc.get("status"), "ended")

        s1.disconnect()
        s2.disconnect()

    # -------------------------------------------------------------------------
    # 4. Security Requirement: Unrelated User Cannot Terminate Call
    # -------------------------------------------------------------------------

    def test_unrelated_user_cannot_terminate_another_call(self):
        """An authenticated stranger (Charlie) cannot terminate Alice & Bob's call by guessing call_id."""
        call_id = CallInvitation.create(self.u1_id, self.u2_id)
        CallInvitation.set_status(call_id, self.u2_id, "accepted")

        s1 = self._socket_client(self.u1_id)
        s2 = self._socket_client(self.u2_id)
        s3 = self._socket_client(self.u3_id)

        s1.emit("join_call", {"partner_id": self.u2_id, "call_id": call_id})
        s2.emit("join_call", {"partner_id": self.u1_id, "call_id": call_id})
        s1.get_received()
        s2.get_received()

        # Charlie attempts to terminate Alice & Bob's call
        ack = s3.emit("end_call", {"partner_id": self.u2_id, "call_id": call_id}, callback=True)
        self.assertIsNotNone(ack)
        self.assertFalse(ack.get("ok"))
        self.assertIn("Unauthorized", ack.get("message", ""))

        # Verify call is still active in MongoDB
        doc = mongo.db[CallInvitation.COLLECTION].find_one({"_id": ObjectId(call_id)})
        self.assertEqual(doc.get("status"), "accepted")

        # Verify Alice & Bob received NO termination events
        rec1 = s1.get_received()
        rec2 = s2.get_received()
        self.assertNotIn("call_ended", [e["name"] for e in rec1])
        self.assertNotIn("call_ended", [e["name"] for e in rec2])

        s1.disconnect()
        s2.disconnect()
        s3.disconnect()

    # -------------------------------------------------------------------------
    # 5. Duplicate End Call Handled Safely
    # -------------------------------------------------------------------------

    def test_duplicate_end_call_handled_safely(self):
        """Duplicate end_call calls do not cause errors and safely return ok=True."""
        call_id = CallInvitation.create(self.u1_id, self.u2_id)
        CallInvitation.set_status(call_id, self.u2_id, "accepted")

        s1 = self._socket_client(self.u1_id)
        s1.emit("join_call", {"partner_id": self.u2_id, "call_id": call_id})
        s1.get_received()

        ack1 = s1.emit("end_call", {"partner_id": self.u2_id, "call_id": call_id}, callback=True)
        self.assertTrue(ack1.get("ok"))

        # Second identical call
        ack2 = s1.emit("end_call", {"partner_id": self.u2_id, "call_id": call_id}, callback=True)
        self.assertTrue(ack2.get("ok"))

        s1.disconnect()

    # -------------------------------------------------------------------------
    # 6. Unexpected Disconnection Handled
    # -------------------------------------------------------------------------

    def test_unexpected_socket_disconnect_terminates_call(self):
        """When an active participant disconnects unexpectedly, the peer is notified and call is ended in DB."""
        call_id = CallInvitation.create(self.u1_id, self.u2_id)
        CallInvitation.set_status(call_id, self.u2_id, "accepted")

        s1 = self._socket_client(self.u1_id)
        s2 = self._socket_client(self.u2_id)

        s1.emit("join_call", {"partner_id": self.u2_id, "call_id": call_id})
        s2.emit("join_call", {"partner_id": self.u1_id, "call_id": call_id})
        s1.get_received()
        s2.get_received()

        # Alice disconnects abruptly
        s1.disconnect()

        # Bob receives call termination
        rec2 = s2.get_received()
        event_names = [e["name"] for e in rec2]
        self.assertTrue("call_ended" in event_names or "call_terminated" in event_names or "peer_left" in event_names)

        # Database state marked ended
        doc = mongo.db[CallInvitation.COLLECTION].find_one({"_id": ObjectId(call_id)})
        self.assertEqual(doc.get("status"), "ended")

        s2.disconnect()

    # -------------------------------------------------------------------------
    # 7. Frontend Template Teardown & DOM Verification
    # -------------------------------------------------------------------------

    def test_video_call_template_cleanup_logic(self):
        """Verify video_call.html contains full WebRTC cleanup, track stoppage, and redirection logic."""
        template_path = os.path.join(self.app.root_path, "templates", "video_call.html")
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Track stop
        self.assertIn("track.stop()", content)
        self.assertIn("track.enabled = false", content)

        # 2. Peer connection close
        self.assertIn("pc.close()", content)
        self.assertIn("pc = null", content)

        # 3. Video elements cleared
        self.assertIn("remoteVideo.srcObject = null", content)
        self.assertIn("localVideo.srcObject = null", content)

        # 4. Speech and sign teardown
        self.assertIn("speechRecognizer.stop()", content)
        self.assertIn("clearInterval(signInterval)", content)
        self.assertIn("window.speechSynthesis.cancel()", content)

        # 5. Redirection to chat with partner
        self.assertIn("chat.chat_with", content)

        # 6. Socket termination listeners
        self.assertIn('socket.on("call_ended"', content)
        self.assertIn('socket.on("call_terminated"', content)
        self.assertIn('socket.on("peer_left"', content)

        # 7. Unload listeners
        self.assertIn('window.addEventListener("beforeunload"', content)
        self.assertIn('window.addEventListener("pagehide"', content)

        # 8. Duplicate cleanup guard
        self.assertIn("if (ended) return;", content)
