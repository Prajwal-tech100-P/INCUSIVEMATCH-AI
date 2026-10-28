from datetime import datetime
from bson import ObjectId
from extensions import mongo


class Restriction:
    """Model managing directional and persistent user restrictions."""

    COLLECTION = "user_restrictions"

    @staticmethod
    def _oid(value):
        try:
            return ObjectId(str(value))
        except Exception:
            return None

    @classmethod
    def ensure_indexes(cls):
        """Create required indexes to ensure uniqueness and fast lookup."""
        try:
            col = mongo.db[cls.COLLECTION]
            col.create_index(
                [("restrictor_id", 1), ("restricted_user_id", 1)],
                name="idx_user_restrictions_unique",
                unique=True,
            )
            col.create_index(
                [("restricted_user_id", 1)],
                name="idx_user_restrictions_restricted",
            )
            col.create_index(
                [("restrictor_id", 1)],
                name="idx_user_restrictions_restrictor",
            )
        except Exception:
            # Index errors must not crash the app
            pass

    @classmethod
    def create(cls, restrictor_id, restricted_user_id, reason=None):
        """Create or update a persistent directional restriction.

        Returns (success: bool, message: str).
        """
        r_id = str(restrictor_id).strip()
        t_id = str(restricted_user_id).strip()

        if not r_id or not t_id:
            return False, "Invalid user ID provided."

        if r_id == t_id:
            return False, "You cannot restrict yourself."

        oid_r = cls._oid(r_id)
        oid_t = cls._oid(t_id)
        if not oid_r or not oid_t:
            return False, "Invalid user identifier format."

        # Verify target user exists
        target = mongo.db.users.find_one({"_id": oid_t})
        if not target:
            return False, "Target user not found."

        clean_reason = str(reason).strip()[:500] if reason else None
        now = datetime.utcnow()

        # Idempotent upsert
        mongo.db[cls.COLLECTION].update_one(
            {"restrictor_id": r_id, "restricted_user_id": t_id},
            {
                "$setOnInsert": {
                    "restrictor_id": r_id,
                    "restricted_user_id": t_id,
                    "created_at": now,
                },
                "$set": {
                    "updated_at": now,
                    "reason": clean_reason,
                },
            },
            upsert=True,
        )

        # End mutual match if one exists
        from models.match import Match
        Match.part_ways(r_id, t_id)

        # Invalidate any pending call invitations between these users
        try:
            mongo.db.call_invitations.update_many(
                {
                    "$or": [
                        {"caller_id": r_id, "recipient_id": t_id},
                        {"caller_id": t_id, "recipient_id": r_id},
                    ],
                    "status": "pending",
                },
                {"$set": {"status": "cancelled", "updated_at": now}},
            )
        except Exception:
            pass

        return True, "User restricted successfully."

    @classmethod
    def is_restricted(cls, restrictor_id, restricted_user_id):
        """Check if restrictor_id has explicitly restricted restricted_user_id."""
        r_id = str(restrictor_id).strip()
        t_id = str(restricted_user_id).strip()
        if not r_id or not t_id:
            return False
        return mongo.db[cls.COLLECTION].find_one(
            {"restrictor_id": r_id, "restricted_user_id": t_id}
        ) is not None

    @classmethod
    def is_restricted_either(cls, user_a_id, user_b_id):
        """Check if either user has restricted the other."""
        a_id = str(user_a_id).strip()
        b_id = str(user_b_id).strip()
        if not a_id or not b_id or a_id == b_id:
            return False
        doc = mongo.db[cls.COLLECTION].find_one(
            {
                "$or": [
                    {"restrictor_id": a_id, "restricted_user_id": b_id},
                    {"restrictor_id": b_id, "restricted_user_id": a_id},
                ]
            }
        )
        return doc is not None

    @classmethod
    def get_restricted_user_ids(cls, user_id):
        """Get set of all user IDs where user_id is either restrictor or restricted."""
        uid = str(user_id).strip()
        if not uid:
            return set()
        docs = mongo.db[cls.COLLECTION].find(
            {
                "$or": [
                    {"restrictor_id": uid},
                    {"restricted_user_id": uid},
                ]
            },
            {"restrictor_id": 1, "restricted_user_id": 1},
        )
        result = set()
        for d in docs:
            r = d.get("restrictor_id")
            t = d.get("restricted_user_id")
            if r and r != uid:
                result.add(str(r))
            if t and t != uid:
                result.add(str(t))
        return result

    @classmethod
    def remove(cls, restrictor_id, restricted_user_id):
        """Remove a restriction placed by restrictor_id on restricted_user_id."""
        r_id = str(restrictor_id).strip()
        t_id = str(restricted_user_id).strip()
        if not r_id or not t_id:
            return False
        res = mongo.db[cls.COLLECTION].delete_one(
            {"restrictor_id": r_id, "restricted_user_id": t_id}
        )
        return res.deleted_count > 0

    @classmethod
    def get_restrictions_by_user(cls, user_id):
        """Get list of all users restricted by user_id."""
        uid = str(user_id).strip()
        if not uid:
            return []
        return list(
            mongo.db[cls.COLLECTION]
            .find({"restrictor_id": uid})
            .sort("created_at", -1)
        )
