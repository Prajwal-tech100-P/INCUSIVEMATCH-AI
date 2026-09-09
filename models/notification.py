from datetime import datetime
from extensions import mongo


class Notification:
    @staticmethod
    def create(user_id, notification_type, title, message, related_user_id=None):
        doc = {
            "user_id": str(user_id),
            "type": notification_type,
            "title": title,
            "message": message,
            "related_user_id": str(related_user_id) if related_user_id else None,
            "is_read": False,
            "created_at": datetime.utcnow(),
        }
        return mongo.db.notifications.insert_one(doc).inserted_id

    @staticmethod
    def get_for_user(user_id, limit=10):
        return list(
            mongo.db.notifications.find({"user_id": str(user_id)})
            .sort("created_at", -1)
            .limit(limit)
        )

    @staticmethod
    def unread_count(user_id):
        return mongo.db.notifications.count_documents(
            {"user_id": str(user_id), "is_read": False}
        )

    @staticmethod
    def mark_read(notification_id, user_id):
        from bson import ObjectId
        try:
            oid = ObjectId(notification_id)
        except Exception:
            return False
        result = mongo.db.notifications.update_one(
            {"_id": oid, "user_id": str(user_id)},
            {"$set": {"is_read": True}},
        )
        return result.modified_count > 0

    @staticmethod
    def mark_all_read(user_id):
        return mongo.db.notifications.update_many(
            {"user_id": str(user_id), "is_read": False},
            {"$set": {"is_read": True}},
        )
