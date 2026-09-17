from bson import ObjectId
from extensions import mongo

class Match:
    @staticmethod
    def _oid(value):
        try: return ObjectId(value)
        except Exception: return None

    @staticmethod
    def like(user_id, target_id):
        if user_id == target_id: return False
        target = Match._oid(target_id)
        if not target: return False
        result = mongo.db.users.update_one(
            {"_id": Match._oid(user_id), "role": "user", "is_active": True},
            {"$addToSet": {"liked": target_id}, "$pull": {"disliked": target_id}}
        )
        return result.modified_count > 0

    @staticmethod
    def dislike(user_id, target_id):
        target = Match._oid(target_id)
        if not target or user_id == target_id: return False
        result = mongo.db.users.update_one(
            {"_id": Match._oid(user_id), "role": "user"},
            {"$addToSet": {"disliked": target_id}, "$pull": {"liked": target_id}}
        )
        return result.modified_count > 0

    @staticmethod
    def is_mutual(user_a_id, user_b_id):
        a = mongo.db.users.find_one({"_id": Match._oid(user_a_id), "role": "user"})
        b = mongo.db.users.find_one({"_id": Match._oid(user_b_id), "role": "user"})
        return bool(a and b and user_b_id in a.get("liked", []) and user_a_id in b.get("liked", []))

    @staticmethod
    def are_matched(user_a_id, user_b_id):
        return Match.is_mutual(user_a_id, user_b_id)

    @staticmethod
    def get_discover_candidates(user_id, comm_pref=None):
        user = mongo.db.users.find_one({"_id": Match._oid(user_id)})
        if not user: return []
        excluded = set(user.get("liked", []) + user.get("disliked", []) + [user_id])
        valid_oids = []
        for x in excluded:
            oid = Match._oid(x)
            if oid: valid_oids.append(oid)
        
        # Base query
        query = {"_id": {"$nin": valid_oids}, "role": "user", "is_active": True}
        if comm_pref: query["comm_pref"] = comm_pref
        
        # User's filtering preferences
        u_pref_gender = user.get("pref_gender", "Any")
        if u_pref_gender != "Any":
            query["gender"] = u_pref_gender
            
        u_pref_min_age = user.get("pref_min_age", 18)
        u_pref_max_age = user.get("pref_max_age", 99)
        query["age"] = {"$gte": u_pref_min_age, "$lte": u_pref_max_age}

        candidates = list(mongo.db.users.find(query))
        
        filtered_candidates = []
        u_age = user.get("age", 18)
        u_gender = user.get("gender", "Not Specified")
        mine = set(user.get("interests", []))
        
        for c in candidates:
            # Candidate's filtering preferences (Reverse matching)
            c_pref_gender = c.get("pref_gender", "Any")
            if c_pref_gender != "Any" and c_pref_gender != u_gender:
                continue
                
            c_pref_min_age = c.get("pref_min_age", 18)
            c_pref_max_age = c.get("pref_max_age", 99)
            if not (c_pref_min_age <= u_age <= c_pref_max_age):
                continue

            shared = len(mine & set(c.get("interests", [])))
            c["_match_score"] = min(100, shared * 15 + (10 if c.get("comm_pref") == user.get("comm_pref") else 0))
            filtered_candidates.append(c)
            
        return sorted(filtered_candidates, key=lambda x: x["_match_score"], reverse=True)

    @staticmethod
    def get_mutual_matches(user_id):
        user = mongo.db.users.find_one({"_id": Match._oid(user_id)})
        if not user: return []
        liked = [Match._oid(x) for x in user.get("liked", []) if Match._oid(x)]
        return list(mongo.db.users.find({"_id": {"$in": liked}, "liked": user_id, "role": "user", "is_active": True}))
