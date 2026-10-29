from datetime import datetime
from bson import ObjectId
from extensions import mongo

MAX_MEMBERS = 50

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
            d["_is_full"] = len(member_ids) >= MAX_MEMBERS
            result.append(d)
        return result

    @staticmethod
    def get(community_id):
        oid = _oid(community_id)
        return mongo.db.communities.find_one({"_id": oid}) if oid else None

    @staticmethod
    def create(owner_id, name, description, category, icon="🤟"):
        owner_str = str(owner_id)
        doc = {
            "name": name.strip(),
            "description": description.strip(),
            "category": category.strip() or "General",
            "icon": icon.strip() or "🤟",
            "created_by": owner_str,
            "member_ids": [owner_str],
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
            return False, f"This community has reached its maximum capacity of {MAX_MEMBERS} members."

        # Atomic concurrency-safe update in MongoDB:
        # Require index MAX_MEMBERS - 1 does not exist (meaning array length < MAX_MEMBERS)
        # and uid is not already in member_ids.
        limit_index_field = f"member_ids.{MAX_MEMBERS - 1}"
        result = mongo.db.communities.update_one(
            {
                "_id": oid,
                limit_index_field: {"$exists": False},
                "member_ids": {"$ne": uid}
            },
            {
                "$addToSet": {"member_ids": uid},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

        if result.modified_count == 0:
            # Re-fetch state to report exact reason
            fresh = mongo.db.communities.find_one({"_id": oid})
            if not fresh:
                return False, "Community not found."
            fresh_members = [str(x) for x in fresh.get("member_ids", [])]
            if uid in fresh_members:
                return True, "You are already a member."
            if len(fresh_members) >= MAX_MEMBERS:
                return False, f"This community has reached its maximum capacity of {MAX_MEMBERS} members."
            return False, "Could not join community. Please try again."

        return True, "Joined community."

    @staticmethod
    def remove_member(community_id, actor_id, target_user_id):
        """
        Removes a member from the community.
        Only the community owner (creator) or an administrator can remove members.
        The owner cannot be removed.
        """
        oid = _oid(community_id)
        c = Community.get(community_id)
        if not oid or not c:
            return False, "Community not found."

        actor_str = str(actor_id)
        target_str = str(target_user_id)
        creator_str = str(c.get("created_by"))

        # Verify permissions: must be owner or admin
        from models.user import UserModel
        actor = UserModel.get_by_id(actor_str)
        is_owner = (creator_str == actor_str)
        is_admin = bool(actor and getattr(actor, "role", "") == "admin")

        if not is_owner and not is_admin:
            return False, "Unauthorized: Only the community owner or an administrator can remove members."

        # Prevent removing the owner
        if target_str == creator_str:
            return False, "The community owner cannot be removed."

        members = [str(x) for x in c.get("member_ids", [])]
        if target_str not in members:
            return False, "User is not a member of this community."

        mongo.db.communities.update_one(
            {"_id": oid},
            {"$pull": {"member_ids": target_str}, "$set": {"updated_at": datetime.utcnow()}}
        )
        return True, "Member removed from community."

    @staticmethod
    def leave(community_id, user_id):
        oid = _oid(community_id)
        c = Community.get(community_id)
        if not oid or not c:
            return False, "Community not found."
        uid = str(user_id)
        if str(c.get("created_by")) == uid:
            return False, "The community creator cannot leave. Delete the community instead."
        members = [str(x) for x in c.get("member_ids", [])]
        if uid not in members:
            return False, "You are not a member of this community."
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
        # Prevent duplicates
        existing = mongo.db.reports.find_one({
            "reporter_id": str(reporter_id),
            "reported_id": str(reported_user_id),
            "status": "pending"
        })
        if existing:
            return False, "Report already submitted."

        from models.user import UserModel
        reporter = UserModel.get_by_id(reporter_id)
        reported = UserModel.get_by_id(reported_user_id)

        mongo.db.reports.insert_one({
            "community_id": str(community_id),
            "reporter_id": str(reporter_id),
            "reporter_name": reporter.name if reporter else "Unknown",
            "reported_id": str(reported_user_id),
            "reported_name": reported.name if reported else "Unknown",
            "reason": reason.strip()[:500] or "Community interaction report",
            "source": "community",
            "status": "pending",
            "created_at": datetime.utcnow(),
        })

        try:
            from models.notification import Notification
            from extensions import socketio
            admins = list(mongo.db.users.find({"role": "admin"}, {"_id": 1}))
            title = "New Community Report"
            msg_body = f"{reporter.name if reporter else 'Unknown'} reported {reported.name if reported else 'Unknown'} in community."
            for admin in admins:
                admin_id = str(admin["_id"])
                Notification.create(
                    user_id=admin_id,
                    notification_type="community_report",
                    title=title,
                    message=msg_body,
                )
                socketio.emit("admin_report", {
                    "title": title,
                    "message": msg_body,
                }, to=f"user:{admin_id}")
        except Exception:
            pass

        return True, "Report submitted successfully."
