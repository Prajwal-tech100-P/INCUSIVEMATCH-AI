from datetime import datetime
from bson import ObjectId
from extensions import mongo

MAX_MEMBERS = 5

def _oid(value):
    try:
        return ObjectId(str(value))
    except Exception:
        return None

class Community:
    @staticmethod
    def list_available(user_id):
        uid = str(user_id)
        docs = list(mongo.db.communities.find().sort("created_at", -1))
        result = []
        for d in docs:
            members = d.get("member_ids", [])
            member_ids = [str(x) for x in members]
            d["_member_count"] = len(member_ids)
            d["_is_member"] = uid in member_ids
            result.append(d)
        return result

    @staticmethod
    def get(community_id):
        oid = _oid(community_id)
        return mongo.db.communities.find_one({"_id": oid}) if oid else None

    @staticmethod
    def create(owner_id, name, description, category, icon="🤟"):
        doc = {
            "name": name.strip(),
            "description": description.strip(),
            "category": category.strip() or "General",
            "icon": icon.strip() or "🤟",
            "created_by": str(owner_id),
            "member_ids": [str(owner_id)],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        result = mongo.db.communities.insert_one(doc)
        return str(result.inserted_id)

    @staticmethod
    def is_member(community_id, user_id):
        c = Community.get(community_id)
        return bool(c and str(user_id) in [str(x) for x in c.get("member_ids", [])])

    @staticmethod
    def join(community_id, user_id):
        oid = _oid(community_id)
        if not oid:
            return False, "Invalid community."
        c = mongo.db.communities.find_one({"_id": oid})
        if not c:
            return False, "Community not found."
        uid = str(user_id)
        members = [str(x) for x in c.get("member_ids", [])]
        if uid in members:
            return True, "You are already a member."
        if len(members) >= MAX_MEMBERS:
            return False, "This community is full (maximum 5 members)."
        mongo.db.communities.update_one({"_id": oid}, {"$addToSet": {"member_ids": uid}, "$set": {"updated_at": datetime.utcnow()}})
        return True, "Joined community."

    @staticmethod
    def leave(community_id, user_id):
        oid = _oid(community_id)
        c = Community.get(community_id)
        if not oid or not c:
            return False, "Community not found."
        uid = str(user_id)
        if str(c.get("created_by")) == uid:
            return False, "The community creator cannot leave. Delete the community instead."
        mongo.db.communities.update_one({"_id": oid}, {"$pull": {"member_ids": uid}, "$set": {"updated_at": datetime.utcnow()}})
        return True, "You left the community."

    @staticmethod
    def delete(community_id, user_id):
        oid = _oid(community_id)
        c = Community.get(community_id)
        if not oid or not c:
            return False, "Community not found."
        if str(c.get("created_by")) != str(user_id):
            return False, "Only the community creator can delete this community."
        mongo.db.communities.delete_one({"_id": oid})
        mongo.db.community_messages.delete_many({"community_id": str(community_id)})
        mongo.db.community_reports.delete_many({"community_id": str(community_id)})
        mongo.db.community_blocks.delete_many({"community_id": str(community_id)})
        return True, "Community deleted."

    @staticmethod
    def my_communities(user_id):
        uid = str(user_id)
        docs = list(mongo.db.communities.find({"member_ids": uid}).sort("updated_at", -1))
        for d in docs:
            d["_member_count"] = len(d.get("member_ids", []))
            d["_is_owner"] = str(d.get("created_by")) == uid
        return docs

    @staticmethod
    def messages(community_id, limit=100):
        return list(mongo.db.community_messages.find({"community_id": str(community_id)}).sort("timestamp", 1).limit(limit))

    @staticmethod
    def create_message(community_id, sender_id, sender_name, text):
        doc = {
            "community_id": str(community_id),
            "sender_id": str(sender_id),
            "sender_name": sender_name,
            "text": text,
            "timestamp": datetime.utcnow(),
        }
        result = mongo.db.community_messages.insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    @staticmethod
    def is_blocked(community_id, user_a, user_b):
        ids = {str(user_a), str(user_b)}
        return mongo.db.community_blocks.find_one({
            "community_id": str(community_id),
            "blocker_id": {"$in": list(ids)},
            "blocked_id": {"$in": list(ids)}
        }) is not None

    @staticmethod
    def is_user_blocked(community_id, user_id):
        return mongo.db.community_blocks.find_one({
            "community_id": str(community_id),
            "blocked_id": str(user_id)
        }) is not None

    @staticmethod
    def block_member(community_id, blocker_id, blocked_id):
        mongo.db.community_blocks.update_one(
            {"community_id": str(community_id), "blocker_id": str(blocker_id), "blocked_id": str(blocked_id)},
            {"$set": {"created_at": datetime.utcnow()}},
            upsert=True
        )

    @staticmethod
    def report(community_id, reporter_id, reported_user_id, reason):
        mongo.db.community_reports.insert_one({
            "community_id": str(community_id),
            "reporter_id": str(reporter_id),
            "reported_user_id": str(reported_user_id),
            "reason": reason.strip()[:500] or "Community interaction report",
            "status": "open",
            "created_at": datetime.utcnow(),
        })
