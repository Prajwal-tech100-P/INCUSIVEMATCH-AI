from bson import ObjectId
from extensions import mongo
import os
import sys

# Add ai_modules to path for optional standalone execution
_ai_dir = os.path.join(os.path.dirname(__file__), '..', 'ai_modules')
if _ai_dir not in sys.path:
    sys.path.append(_ai_dir)

try:
    from ai_modules.match_recommender import MatchRecommender
    ai_recommender = MatchRecommender()
except ImportError:
    try:
        import importlib
        _mod = importlib.import_module("match_recommender")
        ai_recommender = getattr(_mod, "MatchRecommender")()
    except Exception:
        ai_recommender = None
except Exception:
    ai_recommender = None

class Match:
    @staticmethod
    def _oid(value):
        try: return ObjectId(value)
        except Exception: return None

    @staticmethod
    def like(user_id, target_id):
        if str(user_id) == str(target_id): return False
        try:
            from models.restriction import Restriction
            if Restriction.is_restricted_either(user_id, target_id):
                return False
        except Exception:
            pass
        target = Match._oid(target_id)
        if not target: return False
        result = mongo.db.users.update_one(
            {"_id": Match._oid(user_id), "role": "user", "is_active": True},
            {"$addToSet": {"liked": str(target_id)}, "$pull": {"disliked": str(target_id)}}
        )
        return result.modified_count > 0

    @staticmethod
    def dislike(user_id, target_id):
        target = Match._oid(target_id)
        if not target or str(user_id) == str(target_id): return False
        result = mongo.db.users.update_one(
            {"_id": Match._oid(user_id), "role": "user"},
            {"$addToSet": {"disliked": str(target_id)}, "$pull": {"liked": str(target_id)}}
        )
        return result.modified_count > 0

    @staticmethod
    def is_mutual(user_a_id, user_b_id):
        if str(user_a_id) == str(user_b_id):
            return False
        try:
            from models.restriction import Restriction
            if Restriction.is_restricted_either(user_a_id, user_b_id):
                return False
        except Exception:
            pass
        oid_a = Match._oid(user_a_id)
        oid_b = Match._oid(user_b_id)
        a = mongo.db.users.find_one({"_id": oid_a, "role": "user"})
        b = mongo.db.users.find_one({"_id": oid_b, "role": "user"})
        if not a or not b:
            return False
        a_liked = [str(x) for x in a.get("liked", [])]
        b_liked = [str(x) for x in b.get("liked", [])]
        return str(user_b_id) in a_liked and str(user_a_id) in b_liked

    @staticmethod
    def are_matched(user_a_id, user_b_id):
        return Match.is_mutual(user_a_id, user_b_id)

    @staticmethod
    def part_ways(user_a_id, user_b_id):
        """Safely ends a mutual match between two users without permanently restricting either."""
        oid_a = Match._oid(user_a_id)
        oid_b = Match._oid(user_b_id)
        if not oid_a or not oid_b or str(user_a_id) == str(user_b_id):
            return False

        # Pull string and ObjectId variants from 'liked' arrays symmetrically
        mongo.db.users.update_one(
            {"_id": oid_a},
            {"$pull": {"liked": {"$in": [str(user_b_id), oid_b]}}}
        )
        mongo.db.users.update_one(
            {"_id": oid_b},
            {"$pull": {"liked": {"$in": [str(user_a_id), oid_a]}}}
        )

        # Invalidate any pending calls between them
        try:
            from datetime import datetime
            mongo.db.call_invitations.update_many(
                {
                    "$or": [
                        {"caller_id": str(user_a_id), "recipient_id": str(user_b_id)},
                        {"caller_id": str(user_b_id), "recipient_id": str(user_a_id)},
                    ],
                    "status": "pending",
                },
                {"$set": {"status": "cancelled", "updated_at": datetime.utcnow()}},
            )
        except Exception:
            pass

        return True

    @staticmethod
    def get_discover_candidates(
        user_id,
        comm_pref=None,
        filter_gender=None,
        filter_min_age=None,
        filter_max_age=None,
        filter_interest=None,
        online_only=False,
        require_shared_interests=True
    ):
        import logging
        user = mongo.db.users.find_one({"_id": Match._oid(user_id)})
        if not user: return []
        
        excluded = set(user.get("liked", []) + user.get("disliked", []) + [str(user_id)])
        # Include blocked users if the application supports it
        excluded.update(user.get("blocked", []))
        try:
            from models.restriction import Restriction
            excluded.update(Restriction.get_restricted_user_ids(user_id))
        except Exception:
            pass
        
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
        if comm_pref and str(comm_pref).strip().lower() not in {"any", "all", ""}:
            c_term = str(comm_pref).strip().lower()
            query["$or"] = [
                {"communication_preferences": {"$regex": f"^{c_term}", "$options": "i"}},
                {"comm_prefs": {"$regex": f"^{c_term}", "$options": "i"}},
                {"comm_pref": {"$regex": f"^{c_term}", "$options": "i"}}
            ]
        
        candidates = list(mongo.db.users.find(query))
        logging.info(f"DISCOVER DEBUG: Current user: {user_id}")
        logging.info(f"DISCOVER DEBUG: Total users in DB: {total_users}")
        logging.info(f"DISCOVER DEBUG: Candidates before filters (from DB): {len(candidates)}")
        
        filtered_candidates = []
        u_gender = str(user.get("gender", "Not Specified")).strip()
        u_interests = set(str(i).strip().lower() for i in user.get("interests", []) if i and str(i).strip())

        # Extract matching preferences supporting nested matching_preferences and legacy flat fields
        mp = user.get("matching_preferences")
        if not isinstance(mp, dict):
            mp = {}

        if filter_min_age is not None:
            u_pref_min_age = int(filter_min_age)
        else:
            raw_min_age = mp.get("min_age")
            if raw_min_age is None:
                raw_min_age = user.get("pref_min_age", 18)
            try:
                u_pref_min_age = int(raw_min_age) if raw_min_age is not None else 18
            except (ValueError, TypeError):
                u_pref_min_age = 18

        if filter_max_age is not None:
            u_pref_max_age = int(filter_max_age)
        else:
            raw_max_age = mp.get("max_age")
            if raw_max_age is None:
                raw_max_age = user.get("pref_max_age", 99)
            try:
                u_pref_max_age = int(raw_max_age) if raw_max_age is not None else 99
            except (ValueError, TypeError):
                u_pref_max_age = 99

        # Gender filtering:
        if filter_gender and str(filter_gender).strip().lower() not in {"all", "any", ""}:
            pref_genders_normalized = {str(filter_gender).strip().lower()}
            allow_any_gender = False
        else:
            raw_pref_genders = mp.get("preferred_genders")
            if raw_pref_genders is None:
                legacy_pref = user.get("pref_gender")
                if legacy_pref:
                    raw_pref_genders = [legacy_pref]
                else:
                    raw_pref_genders = ["Any"]
            elif isinstance(raw_pref_genders, str):
                raw_pref_genders = [raw_pref_genders]
            elif not isinstance(raw_pref_genders, (list, tuple, set)):
                raw_pref_genders = ["Any"]

            pref_genders_normalized = {
                str(g).strip().lower() for g in raw_pref_genders if g and str(g).strip()
            }
            allow_any_gender = (
                not pref_genders_normalized
                or "any" in pref_genders_normalized
            )

        is_online_checker = None
        if online_only:
            from models.message import OnlineTracker
            is_online_checker = OnlineTracker.is_online

        # Apply STRICT filters based on user's preferences, gender, and interests
        for c in candidates:
            # Safe candidate age extraction
            try:
                c_age = int(c.get("age", 18)) if c.get("age") is not None else 18
            except (ValueError, TypeError):
                c_age = 18

            # Normalized candidate gender
            c_gender = str(c.get("gender", "")).strip().lower()

            # Strict age filter
            if c_age < u_pref_min_age or c_age > u_pref_max_age:
                continue

            # 1. Strict Gender Filter: Candidate gender must match user's preferred gender
            if not allow_any_gender and c_gender not in pref_genders_normalized:
                continue

            # 2. Bidirectional Gender Filter: Candidate's preference must accept user's gender
            c_mp = c.get("matching_preferences") or {}
            c_pref_raw = c_mp.get("preferred_genders") or [c.get("pref_gender", "Any")]
            if isinstance(c_pref_raw, str):
                c_pref_raw = [c_pref_raw]
            c_pref_norm = {str(g).strip().lower() for g in c_pref_raw if g and str(g).strip()}
            if c_pref_norm and "any" not in c_pref_norm and u_gender.lower() not in {"not specified", ""}:
                if u_gender.lower() not in c_pref_norm:
                    continue

            # 3. Interests Filter:
            c_interests_raw = c.get("interests", []) or []
            c_interests = set(str(i).strip().lower() for i in c_interests_raw if i and str(i).strip())
            shared_interests = u_interests & c_interests

            # Explicit interest filter (from user search / dropdown)
            if filter_interest and str(filter_interest).strip():
                f_int = str(filter_interest).strip().lower()
                if f_int not in c_interests:
                    continue

            # Candidate must match based on interest: require at least 1 shared interest
            # if current user has configured interests on their profile
            if require_shared_interests and len(u_interests) > 0:
                if len(shared_interests) == 0:
                    continue

            # Attach shared interests information to candidate dict for UI display
            c["_shared_interests"] = [
                i for i in c_interests_raw
                if str(i).strip().lower() in shared_interests
            ]
            c["_shared_interests_count"] = len(shared_interests)

            # 4. Online-only filter:
            if online_only and is_online_checker:
                if not is_online_checker(str(c["_id"])):
                    continue

            filtered_candidates.append(c)

        if ai_recommender:
            # Use AI / calibrated recommender for ranking (it sets the '_match_score' key)
            filtered_candidates = ai_recommender.rank_candidates(user, filtered_candidates)
        else:
            try:
                from ai_modules.match_recommender import MatchRecommender
                for c in filtered_candidates:
                    c["_match_score"] = MatchRecommender.compute_compatibility_score(user, c)
            except Exception:
                mine_interests = set(str(i).strip().lower() for i in user.get("interests", []) if i)
                for c in filtered_candidates:
                    c_interests = set(str(i).strip().lower() for i in c.get("interests", []) if i)
                    union = mine_interests | c_interests
                    jaccard = (len(mine_interests & c_interests) / len(union)) if union else 0.0
                    c["_match_score"] = int(round(jaccard * 100))
            filtered_candidates.sort(key=lambda x: x.get("_match_score", 0), reverse=True)
            
        logging.info(f"DISCOVER DEBUG: Final Discover users: {len(filtered_candidates)}")
        
        return filtered_candidates

    @staticmethod
    def get_mutual_matches(user_id):
        user = mongo.db.users.find_one({"_id": Match._oid(user_id)})
        if not user: return []
        restricted_ids = set()
        try:
            from models.restriction import Restriction
            restricted_ids = Restriction.get_restricted_user_ids(user_id)
        except Exception:
            pass
        liked = [Match._oid(x) for x in user.get("liked", []) if Match._oid(x) and str(x) not in restricted_ids]
        return list(mongo.db.users.find({"_id": {"$in": liked}, "liked": str(user_id), "role": "user", "is_active": True}))
