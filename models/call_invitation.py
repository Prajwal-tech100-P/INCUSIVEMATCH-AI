from datetime import datetime, timedelta, timezone
from bson import ObjectId
from extensions import mongo


class CallInvitation:
    COLLECTION = "call_invitations"
    TTL_SECONDS = 60

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    @classmethod
    def ensure_indexes(cls):
        try:
            mongo.db[cls.COLLECTION].create_index(
                "expires_at", expireAfterSeconds=0, name="call_invitation_ttl"
            )
            mongo.db[cls.COLLECTION].create_index(
                [("recipient_id", 1), ("status", 1), ("created_at", -1)],
                name="call_recipient_status"
            )
        except Exception:
            # Index creation must not prevent the app from starting.
            pass

    @classmethod
    def create(cls, caller_id, recipient_id):
        cls.ensure_indexes()
        now = cls._now()
        mongo.db[cls.COLLECTION].update_many(
            {
                "caller_id": str(caller_id),
                "recipient_id": str(recipient_id),
                "status": "pending",
            },
            {"$set": {"status": "superseded", "updated_at": now}},
        )
        doc = {
            "caller_id": str(caller_id),
            "recipient_id": str(recipient_id),
            "status": "pending",
            "created_at": now,
            "expires_at": now + timedelta(seconds=cls.TTL_SECONDS),
        }
        result = mongo.db[cls.COLLECTION].insert_one(doc)
        return str(result.inserted_id)

    @classmethod
    def pending_for(cls, recipient_id):
        cls.ensure_indexes()
        now = cls._now()
        doc = mongo.db[cls.COLLECTION].find_one(
            {
                "recipient_id": str(recipient_id),
                "status": "pending",
                "expires_at": {"$gt": now},
            },
            sort=[("created_at", -1)],
        )
        if not doc:
            return None
        return {
            "call_id": str(doc["_id"]),
            "caller_id": str(doc["caller_id"]),
            "created_at": doc["created_at"].isoformat(),
            "expires_at": doc["expires_at"].isoformat(),
        }

    @classmethod
    def set_status(cls, call_id, recipient_id, status):
        try:
            oid = ObjectId(call_id)
        except Exception:
            return False
        now = cls._now()
        result = mongo.db[cls.COLLECTION].update_one(
            {
                "_id": oid,
                "recipient_id": str(recipient_id),
                "status": "pending",
                "expires_at": {"$gt": now},
            },
            {"$set": {"status": status, "updated_at": now}},
        )
        return result.modified_count == 1
