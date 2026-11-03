import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app
from extensions import mongo, socketio
from models.user import UserModel
from models.match import Match
from bson import ObjectId


class TestSocketRelay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_ENV"] = "testing"
        os.environ["SECRET_KEY"] = "test-secret-key"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def _socket_client(self, user_id):
        flask_client = self.app.test_client()
        with flask_client.session_transaction() as sess:
            sess["_user_id"] = str(user_id)
            sess["_fresh"] = True
        return socketio.test_client(self.app, flask_test_client=flask_client)

    def test_relay(self):
        u1_id = str(ObjectId())
        u2_id = str(ObjectId())

        mongo.db.users.insert_one({
            "_id": ObjectId(u1_id),
            "id": u1_id,
            "name": "Signer User A",
            "email": f"a_{u1_id}@test.com",
            "password": "pw",
            "role": "user",
            "is_active": True,
        })
        mongo.db.users.insert_one({
            "_id": ObjectId(u2_id),
            "id": u2_id,
            "name": "Receiver User B",
            "email": f"b_{u2_id}@test.com",
            "password": "pw",
            "role": "user",
            "is_active": True,
        })

        Match.like(u1_id, u2_id)
        Match.like(u2_id, u1_id)

        c1 = self._socket_client(u1_id)
        c2 = self._socket_client(u2_id)

        c1.emit("join_call", {"partner_id": u2_id})
        c2.emit("join_call", {"partner_id": u1_id})

        c2.get_received()  # Flush join messages

        c1.emit("sign_caption", {
            "partner_id": u2_id,
            "text": "NAMASTE",
            "phrase": "NAMASTE",
            "confidence": 98.7,
            "is_final": True
        })

        rec2 = c2.get_received()
        print("Rec2 events:", rec2)
        sign_events = [e for e in rec2 if e["name"] == "sign_caption"]
        self.assertEqual(len(sign_events), 1)
        self.assertEqual(sign_events[0]["args"][0]["text"], "NAMASTE")
        self.assertEqual(sign_events[0]["args"][0]["speaker"], "Signer User A")
        print("SUCCESS: User B received sign_caption event from User A!")

        c1.disconnect()
        c2.disconnect()
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(u1_id), ObjectId(u2_id)]}})
        mongo.db.matches.delete_many({"$or": [{"user_a": u1_id}, {"user_b": u1_id}]})


if __name__ == "__main__":
    unittest.main()

