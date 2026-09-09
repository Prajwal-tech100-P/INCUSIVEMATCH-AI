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
