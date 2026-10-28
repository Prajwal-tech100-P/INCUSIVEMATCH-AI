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
        update_fields = {"status": status, "updated_at": now}
        if status == "accepted":
            # Extend expires_at by 24h so TTL index doesn't remove ongoing calls prematurely
            update_fields["expires_at"] = now + timedelta(hours=24)
        result = mongo.db[cls.COLLECTION].update_one(
            {
                "_id": oid,
                "recipient_id": str(recipient_id),
                "status": "pending",
                "expires_at": {"$gt": now},
            },
            {"$set": update_fields},
        )
        return result.modified_count == 1

    @classmethod
    def cancel(cls, call_id, caller_id):
        try:
            oid = ObjectId(call_id)
        except Exception:
            return False
        now = cls._now()
        result = mongo.db[cls.COLLECTION].update_one(
            {
                "_id": oid,
                "caller_id": str(caller_id),
                "status": "pending",
            },
            {"$set": {"status": "cancelled", "updated_at": now}},
        )
        return result.modified_count == 1

    @classmethod
    def end_call(cls, call_id=None, user_id=None, partner_id=None):
        """
        Mark an active call invitation as 'ended'.
        Validates that user_id is an actual participant of the call.
        Rejects calls from unauthorized users.
        Handles duplicate/stale termination safely.
        """
        now = cls._now()

        # 1. If call_id is provided, look it up by ID
        if call_id:
            try:
                oid = ObjectId(call_id)
            except Exception:
                oid = None

            if oid:
                doc = mongo.db[cls.COLLECTION].find_one({"_id": oid})
                if doc:
                    # Security check: User must be caller or recipient
                    if user_id is not None:
                        uid_str = str(user_id)
                        if uid_str != str(doc.get("caller_id")) and uid_str != str(doc.get("recipient_id")):
                            return False, "unauthorized"

                    if doc.get("status") == "ended":
                        return True, "already_ended"

                    mongo.db[cls.COLLECTION].update_one(
                        {"_id": oid},
                        {
                            "$set": {
                                "status": "ended",
                                "ended_at": now,
                                "ended_by": str(user_id) if user_id else None,
                                "updated_at": now,
                            }
                        },
                    )
                    return True, str(doc["_id"])

        # 2. If no call_id or doc not found, search by participants
        if user_id and partner_id:
            u1, u2 = str(user_id), str(partner_id)
            doc = mongo.db[cls.COLLECTION].find_one(
                {
                    "$or": [
                        {"caller_id": u1, "recipient_id": u2},
                        {"caller_id": u2, "recipient_id": u1},
                    ],
                    "status": {"$in": ["accepted", "pending"]},
                },
                sort=[("created_at", -1)],
            )
            if doc:
                mongo.db[cls.COLLECTION].update_one(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "status": "ended",
                            "ended_at": now,
                            "ended_by": str(user_id),
                            "updated_at": now,
                        }
                    },
                )
                return True, str(doc["_id"])

        return True, "no_active_call"

    @classmethod
    def get_active_call(cls, user_a, user_b):
        """Find the latest active call invitation (accepted or pending) between user_a and user_b."""
        u1, u2 = str(user_a), str(user_b)
        return mongo.db[cls.COLLECTION].find_one(
            {
                "$or": [
                    {"caller_id": u1, "recipient_id": u2},
                    {"caller_id": u2, "recipient_id": u1},
                ],
                "status": {"$in": ["accepted", "pending"]},
            },
            sort=[("created_at", -1)],
        )


