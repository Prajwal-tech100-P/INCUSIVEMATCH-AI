from bson import ObjectId
from extensions import mongo
import os
import sys

# Add ai_modules to path so we can import recommender
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_modules'))
try:
    from match_recommender import MatchRecommender
    ai_recommender = MatchRecommender()
except ImportError:
    ai_recommender = None

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
        import logging
        user = mongo.db.users.find_one({"_id": Match._oid(user_id)})
        if not user: return []
        
        excluded = set(user.get("liked", []) + user.get("disliked", []) + [user_id])
        # Include blocked users if the application supports it
        excluded.update(user.get("blocked", []))
        
        valid_oids = []
        for x in excluded:
            oid = Match._oid(x)
            if oid: valid_oids.append(oid)
            
        total_users = mongo.db.users.count_documents({})
        
        # Base query: only filter out admins and explicitly inactive users.
        query = {
            "_id": {"$nin": valid_oids}, 
            "role": {"$in": ["user", None]},
            "is_active": {"$ne": False}
        }
        if comm_pref: query["comm_pref"] = comm_pref
        
        candidates = list(mongo.db.users.find(query))
        logging.info(f"DISCOVER DEBUG: Current user: {user_id}")
        logging.info(f"DISCOVER DEBUG: Total users in DB: {total_users}")
        logging.info(f"DISCOVER DEBUG: Candidates before filters (from DB): {len(candidates)}")
        
        filtered_candidates = []
        u_age = user.get("age", 18)
        u_gender = user.get("gender", "Not Specified")
        u_pref_gender = user.get("pref_gender", "Any")
        u_pref_min_age = user.get("pref_min_age", 18)
        u_pref_max_age = user.get("pref_max_age", 99)
        
        # Apply STRICT filters based on user's preferences before scoring
        for c in candidates:
            c_age = c.get("age", 18)
            c_gender = c.get("gender", "Not Specified")
            
            # Strict age filter
            if c_age < u_pref_min_age or c_age > u_pref_max_age:
                continue
                
            # Strict gender filter
            if u_pref_gender != "Any" and u_pref_gender != c_gender:
                continue
                
            filtered_candidates.append(c)

        if ai_recommender:
            # Use AI for ranking (it sets the '_match_score' key)
            filtered_candidates = ai_recommender.rank_candidates(user, filtered_candidates)
        else:
            # Fallback scoring if AI recommender is not available
            mine_interests = set(user.get("interests", []))
            for c in filtered_candidates:
                score = 0
                shared = len(mine_interests & set(c.get("interests", [])))
                score += min(45, shared * 15)
                
                # Comm prefs list intersection
                u_prefs = user.get("comm_prefs", [])
                c_prefs = c.get("comm_prefs", [])
                if set(u_prefs) & set(c_prefs):
                    score += 10
                    
                score += 30 # Base score for passing strict filters
                c["_match_score"] = min(100, score)
                
            filtered_candidates.sort(key=lambda x: x["_match_score"], reverse=True)
            
        logging.info(f"DISCOVER DEBUG: Final Discover users: {len(filtered_candidates)}")
        
        return filtered_candidates

    @staticmethod
    def get_mutual_matches(user_id):
        user = mongo.db.users.find_one({"_id": Match._oid(user_id)})
        if not user: return []
        liked = [Match._oid(x) for x in user.get("liked", []) if Match._oid(x)]
        return list(mongo.db.users.find({"_id": {"$in": liked}, "liked": user_id, "role": "user", "is_active": True}))
