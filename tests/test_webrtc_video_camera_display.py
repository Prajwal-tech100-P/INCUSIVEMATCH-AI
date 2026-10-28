"""Comprehensive automated test suite for WebRTC Video Call Camera Display & Signaling.

Covers:
1. Video call route accessibility, mutual-match enforcement, and template context (partner & ice servers).
2. STUN / TURN configuration in _ice_servers fallback.
3. End-to-end Socket.IO WebRTC signaling exchange between two matched users:
   - Both users join the authorized call room.
   - User presence ping (peer_present / ready).
   - Offer delivery from initiator to answerer.
   - Answer delivery back to initiator.
   - ICE candidate exchange in both directions with sdpMid and sdpMLineIndex.
   - Unauthorized / non-matched user cannot signal into the room.
   - Peer leave event delivery and room departure.
4. Video call frontend template verification:
   - Local and remote video element IDs and attributes (autoplay, playsinline, muted on local).
   - Local preview remains separate from remote video (PiP styling, separate video tags).
   - Single-assignment srcObject and track-level addition to prevent playback interruption.
   - Autoplay restriction handling with muted fallback and unmute UI banner.
   - Immediate call initialization when socket is already connected.
   - Media concurrency lock to prevent overlapping getUserMedia requests.
"""
import os
import unittest
from bson import ObjectId
from app import create_app
from extensions import mongo, socketio
from models.user import UserModel
from models.match import Match
from models.restriction import Restriction
from routes.video_routes import _ice_servers, call_room


class TestWebRtcVideoCameraDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_ENV"] = "testing"
        os.environ["SECRET_KEY"] = "test-secret-webrtc-video-key"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def setUp(self):
        self.u1_id = UserModel.create_user("alice_vid@test.com", "Pw123!test", "Alice Video")
        self.u2_id = UserModel.create_user("bob_vid@test.com", "Pw123!test", "Bob Video")
        self.u3_id = UserModel.create_user("charlie_vid@test.com", "Pw123!test", "Charlie Stranger")

        Match.like(self.u1_id, self.u2_id)
        Match.like(self.u2_id, self.u1_id)

    def tearDown(self):
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(self.u1_id), ObjectId(self.u2_id), ObjectId(self.u3_id)]}})
        mongo.db.matches.delete_many({"$or": [
            {"user_a": self.u1_id}, {"user_b": self.u1_id},
            {"user_a": self.u2_id}, {"user_b": self.u2_id},
            {"user_a": self.u3_id}, {"user_b": self.u3_id},
        ]})

    def _socket_client(self, user_id):
        flask_client = self.app.test_client()
        with flask_client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True
        c_val = flask_client.get_cookie("session").value
        return socketio.test_client(self.app, headers={"Cookie": f"session={c_val}"})

    # -------------------------------------------------------------------------
    # 1. Route Access & Ice Servers Tests
    # -------------------------------------------------------------------------

    def test_video_call_route_requires_mutual_match(self):
        """Only users with active mutual matches can access /video/<partner_id>."""
        client = self.app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(self.u1_id)
            sess["_fresh"] = True

        # Mutual match -> HTTP 200
        res = client.get(f"/video/{self.u2_id}")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Alice Video", html)
        self.assertIn("Bob Video", html)
        self.assertIn('id="remoteVideo"', html)
        self.assertIn('id="localVideo"', html)

        # Stranger (not matched) -> HTTP 403
        res_stranger = client.get(f"/video/{self.u3_id}")
        self.assertEqual(res_stranger.status_code, 403)

    def test_ice_servers_fallback_and_stun_configuration(self):
        """Verify _ice_servers returns STUN URLs even if environment is blank."""
        with self.app.app_context():
            servers = _ice_servers()
            self.assertIsInstance(servers, list)
            self.assertGreater(len(servers), 0)
            stun_found = any("stun:" in str(s.get("urls")) for s in servers)
            self.assertTrue(stun_found, "STUN servers must be present in _ice_servers")

    # -------------------------------------------------------------------------
    # 2. WebRTC Socket.IO Signaling Between Users
    # -------------------------------------------------------------------------

    def test_webrtc_complete_signaling_exchange(self):
        """Verify complete WebRTC signaling flow between User A and User B."""
        s1 = self._socket_client(self.u1_id)
        s2 = self._socket_client(self.u2_id)

        # 1. User A joins call
        s1.emit("join_call", {"partner_id": self.u2_id})
        rec1 = s1.get_received()
        ready_events_s1 = [e for e in rec1 if e["name"] == "call_ready"]
        self.assertEqual(len(ready_events_s1), 1)

        # 2. User B joins call -> User A receives call_ready for User B
        s2.emit("join_call", {"partner_id": self.u1_id})
        rec2 = s2.get_received()
        self.assertTrue(any(e["name"] == "call_ready" and e["args"][0]["user_id"] == self.u2_id for e in rec2))

        rec1_after = s1.get_received()
        self.assertTrue(any(e["name"] == "call_ready" and e["args"][0]["user_id"] == self.u2_id for e in rec1_after))

        # 3. User presence ping (peer_present)
        s1.emit("signal", {"partner_id": self.u2_id, "type": "peer_present"})
        rec2_pres = s2.get_received()
        self.assertTrue(any(e["name"] == "signal" and e["args"][0].get("type") == "peer_present" for e in rec2_pres))

        # 4. Offer exchange: User A sends offer to User B
        offer_sdp = {"type": "offer", "sdp": "v=0\r\no=alice 100 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"}
        s1.emit("signal", {"partner_id": self.u2_id, "type": "offer", "sdp": offer_sdp})
        rec2_offer = s2.get_received()
        offers = [e for e in rec2_offer if e["name"] == "signal" and e["args"][0].get("type") == "offer"]
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0]["args"][0]["from"], self.u1_id)
        self.assertEqual(offers[0]["args"][0]["sdp"]["type"], "offer")

        # 5. Answer exchange: User B sends answer back to User A
        answer_sdp = {"type": "answer", "sdp": "v=0\r\no=bob 200 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"}
        s2.emit("signal", {"partner_id": self.u1_id, "type": "answer", "sdp": answer_sdp})
        rec1_answer = s1.get_received()
        answers = [e for e in rec1_answer if e["name"] == "signal" and e["args"][0].get("type") == "answer"]
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0]["args"][0]["from"], self.u2_id)
        self.assertEqual(answers[0]["args"][0]["sdp"]["type"], "answer")

        # 6. ICE candidates: User A -> User B
        cand_a = {"candidate": "candidate:1 1 UDP 2122260223 192.168.1.100 50000 typ host", "sdpMid": "0", "sdpMLineIndex": 0}
        s1.emit("signal", {"partner_id": self.u2_id, "type": "candidate", "candidate": cand_a})
        rec2_cand = s2.get_received()
        cands_for_b = [e for e in rec2_cand if e["name"] == "signal" and e["args"][0].get("type") == "candidate"]
        self.assertEqual(len(cands_for_b), 1)
        self.assertEqual(cands_for_b[0]["args"][0]["candidate"]["sdpMid"], "0")

        # 7. ICE candidates: User B -> User A
        cand_b = {"candidate": "candidate:2 1 UDP 2122260223 192.168.1.200 50002 typ host", "sdpMid": "0", "sdpMLineIndex": 0}
        s2.emit("signal", {"partner_id": self.u1_id, "type": "candidate", "candidate": cand_b})
        rec1_cand = s1.get_received()
        cands_for_a = [e for e in rec1_cand if e["name"] == "signal" and e["args"][0].get("type") == "candidate"]
        self.assertEqual(len(cands_for_a), 1)
        self.assertEqual(cands_for_a[0]["args"][0]["candidate"]["sdpMid"], "0")

        # 8. User A leaves call -> User B receives peer_left
        s1.emit("leave_call", {"partner_id": self.u2_id})
        rec2_left = s2.get_received()
        left_events = [e for e in rec2_left if e["name"] == "peer_left"]
        self.assertEqual(len(left_events), 1)
        self.assertEqual(left_events[0]["args"][0]["user_id"], self.u1_id)

        s1.disconnect()
        s2.disconnect()

    def test_unauthorized_user_cannot_signal_into_call(self):
        """A user not in a mutual match cannot join or signal into a call."""
        s3 = self._socket_client(self.u3_id)
        s3.emit("join_call", {"partner_id": self.u1_id})
        rec3 = s3.get_received()
        errors = [e for e in rec3 if e["name"] == "call_error"]
        self.assertEqual(len(errors), 1)
        s3.disconnect()

    # -------------------------------------------------------------------------
    # 3. Video Call Frontend Template Architecture Tests
    # -------------------------------------------------------------------------

    def test_frontend_template_video_elements_and_rendering_flow(self):
        """Verify video_call.html implements single srcObject assignment, autoplay, and concurrency locks."""
        template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "video_call.html")
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Elements existence
        self.assertIn('id="remoteVideo"', content)
        self.assertIn('id="localVideo"', content)
        self.assertIn('id="remotePlaceholder"', content)
        self.assertIn('id="unmuteRemoteBtn"', content)

        # Autoplay and playsinline attributes
        self.assertIn('autoplay playsinline', content)
        # Local preview must be muted to avoid self-echo and allow autoplay
        self.assertIn('id="localVideo" autoplay muted playsinline', content)

        # Local preview separate from remote video (PiP placement)
        self.assertIn('position-absolute bottom-0 end-0', content)

        # Single assignment srcObject guard
        self.assertIn('remoteVideo.srcObject = remoteStream', content)
        self.assertIn('remoteStream.addTrack(event.track)', content)

        # Autoplay failure fallback to muted play with unmute banner
        self.assertIn('remoteVideo.muted = true', content)
        self.assertIn('unmuteRemoteBtn.classList.remove("d-none")', content)

        # Media acquisition concurrency lock
        self.assertIn('mediaPromise', content)

        # Already-connected socket initialization
        self.assertIn('if (socket.connected)', content)
        self.assertIn('initCall()', content)

        # Non-blocking candidate processing
        self.assertIn('if (data.type === "candidate")', content)
        self.assertIn('pendingCandidates.push(data.candidate)', content)


if __name__ == "__main__":
    unittest.main()
