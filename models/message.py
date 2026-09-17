from extensions import mongo
from datetime import datetime

class Message:
    @staticmethod
    def create(sender_id, receiver_id, text):
        doc = {"sender_id": sender_id, "receiver_id": receiver_id, "text": text,
               "timestamp": datetime.utcnow(), "is_read": False}
        result = mongo.db.messages.insert_one(doc)
        return str(result.inserted_id)

    @staticmethod
    def get_conversation(user_a_id, user_b_id):
        return list(mongo.db.messages.find({"$or": [
            {"sender_id": user_a_id, "receiver_id": user_b_id},
            {"sender_id": user_b_id, "receiver_id": user_a_id}
        ]}).sort("timestamp", 1))

    @staticmethod
    def mark_read(sender_id, receiver_id):
        mongo.db.messages.update_many({"sender_id": sender_id, "receiver_id": receiver_id, "is_read": False},
                                      {"$set": {"is_read": True}})

    @staticmethod
    def get_inbox(user_id):
        return list(mongo.db.messages.aggregate([
            {"$match": {"$or": [{"sender_id": user_id}, {"receiver_id": user_id}]}},
            {"$sort": {"timestamp": -1}},
            {"$group": {"_id": {"$cond": [{"$eq": ["$sender_id", user_id]}, "$receiver_id", "$sender_id"]},
                        "last_message": {"$first": "$$ROOT"}}}
        ]))

    @staticmethod
    def delete_conversation(user_a_id, user_b_id):
        """Delete all messages between two users."""
        result = mongo.db.messages.delete_many({"$or": [
            {"sender_id": user_a_id, "receiver_id": user_b_id},
            {"sender_id": user_b_id, "receiver_id": user_a_id},
        ]})
        return result.deleted_count


class ChatReport:
    """Stores user reports from the chat interface."""

    @staticmethod
    def create(reporter_id, reported_id, reason=""):
        mongo.db.chat_reports.insert_one({
            "reporter_id": str(reporter_id),
            "reported_id": str(reported_id),
            "reason": (reason.strip()[:500]) or "Reported from chat",
            "timestamp": datetime.utcnow(),
            "status": "pending",
        })

    @staticmethod
    def already_reported(reporter_id, reported_id):
        return mongo.db.chat_reports.find_one({
            "reporter_id": str(reporter_id),
            "reported_id": str(reported_id),
            "status": "pending",
        }) is not None


class OnlineTracker:
    """Track online/offline status via Socket.IO sessions."""

    # In-memory set — resets on server restart, which is fine for presence
    _online_users = set()

    @classmethod
    def set_online(cls, user_id):
        cls._online_users.add(str(user_id))

    @classmethod
    def set_offline(cls, user_id):
        cls._online_users.discard(str(user_id))

    @classmethod
    def is_online(cls, user_id):
        return str(user_id) in cls._online_users

