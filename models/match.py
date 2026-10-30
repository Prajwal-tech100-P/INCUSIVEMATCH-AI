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
        try:
            return ObjectId(str(value))
        except Exception:
            return None

    @staticmethod
    def like(user_id, target_id):
        if str(user_id) == str(target_id):
            return False
        try:
            from models.restriction import Restriction
            if Restriction.is_restricted_either(user_id, target_id):
                return False
        except Exception:
            pass

        target = Match._oid(target_id)
        u_oid = Match._oid(user_id)
        if not target or not u_oid:
            return False

        # Verify target user exists and is an active regular user
        target_user = mongo.db.users.find_one({"_id": target, "role": "user", "is_active": {"$ne": False}})
        if not target_user:
            return False

        # Verify current user exists and is active
        curr_user = mongo.db.users.find_one({"_id": u_oid, "role": "user", "is_active": {"$ne": False}})
        if not curr_user:
            return False

        # Enforce bidirectional blocking
        u_blocked = [str(x) for x in curr_user.get("blocked", [])]
        t_blocked = [str(x) for x in target_user.get("blocked", [])]
        if str(target_id) in u_blocked or str(user_id) in t_blocked:
            return False

        mongo.db.users.update_one(
            {"_id": u_oid},
            {"$addToSet": {"liked": str(target_id)}, "$pull": {"disliked": str(target_id)}}
        )
        return True

    @staticmethod
    def dislike(user_id, target_id):
        if str(user_id) == str(target_id):
            return False
        target = Match._oid(target_id)
        u_oid = Match._oid(user_id)
        if not target or not u_oid:
            return False
        mongo.db.users.update_one(
            {"_id": u_oid, "role": "user"},
            {"$addToSet": {"disliked": str(target_id)}, "$pull": {"liked": str(target_id)}}
        )
        return True

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
        if not oid_a or not oid_b:
            return False

        a = mongo.db.users.find_one({"_id": oid_a, "role": "user", "is_active": {"$ne": False}})
        b = mongo.db.users.find_one({"_id": oid_b, "role": "user", "is_active": {"$ne": False}})
        if not a or not b:
            return False

        # Bidirectional block check
        a_blocked = [str(x) for x in a.get("blocked", [])]
        b_blocked = [str(x) for x in b.get("blocked", [])]
        if str(user_b_id) in a_blocked or str(user_a_id) in b_blocked:
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
        if not user:
            return []

        uid_str = str(user_id)
        excluded = set([uid_str])
        excluded.update(str(x) for x in user.get("liked", []))
        excluded.update(str(x) for x in user.get("disliked", []))
        excluded.update(str(x) for x in user.get("blocked", []))
        try:
            from models.restriction import Restriction
            excluded.update(str(r) for r in Restriction.get_restricted_user_ids(user_id))
        except Exception:
            pass

        # Bidirectional block safety: Exclude users who have blocked the current user
        try:
            blocked_by_others = mongo.db.users.find({"blocked": uid_str}, {"_id": 1, "blocked": 1})
            for b in blocked_by_others:
                if uid_str in [str(x) for x in b.get("blocked", [])]:
                    excluded.add(str(b["_id"]))
        except Exception:
            pass

        valid_oids = [Match._oid(x) for x in excluded if Match._oid(x)]

        # Base query: active regular users not in exclusion list
        query = {
            "_id": {"$nin": valid_oids},
            "role": {"$in": ["user", None]},
            "is_active": {"$ne": False}
        }

        # Communication preference filter with space and underscore tolerance
        if comm_pref and str(comm_pref).strip().lower() not in {"any", "all", ""}:
            raw_c = str(comm_pref).strip().lower()
            if "sign" in raw_c:
                c_pattern = "(sign[ _]?language|sign)"
            elif "text" in raw_c:
                c_pattern = "(text[ _]?chat|text)"
            elif "voice" in raw_c:
                c_pattern = "(voice|call)"
            else:
                c_pattern = raw_c
            query["$or"] = [
                {"communication_preferences": {"$regex": c_pattern, "$options": "i"}},
                {"comm_prefs": {"$regex": c_pattern, "$options": "i"}},
                {"comm_pref": {"$regex": c_pattern, "$options": "i"}}
            ]

        candidates = list(mongo.db.users.find(query))
        logging.info(f"DISCOVER DEBUG: Current user: {user_id}")
        logging.info(f"DISCOVER DEBUG: Candidates before filters (from DB): {len(candidates)}")

        filtered_candidates = []
        seen_candidate_ids = set()
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

        # Gender filtering
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

        # Normalize filter_interest (supports string, comma-separated string, list, set)
        filter_interests_set = set()
        if filter_interest:
            if isinstance(filter_interest, (list, tuple, set)):
                filter_interests_set = {str(i).strip().lower() for i in filter_interest if i and str(i).strip()}
            elif isinstance(filter_interest, str) and filter_interest.strip():
                filter_interests_set = {str(i).strip().lower() for i in filter_interest.split(',') if i and str(i).strip()}

        # Apply deterministic rule-based candidate filtering
        for c in candidates:
            cid_str = str(c.get("_id", ""))
            # Prevent duplicate cards and excluded candidates
            if not cid_str or cid_str in seen_candidate_ids or cid_str in excluded:
                continue

            # Bidirectional block safety check
            c_blocked = set(str(b) for b in c.get("blocked", []))
            if uid_str in c_blocked:
                continue

            # Safe candidate age extraction
            try:
                c_age = int(c.get("age", 18)) if c.get("age") is not None else 18
            except (ValueError, TypeError):
                c_age = 18

            # Strict age filter
            if c_age < u_pref_min_age or c_age > u_pref_max_age:
                continue

            # Normalized candidate gender
            c_gender = str(c.get("gender", "")).strip().lower()

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

            # 3. Interests Filter
            c_interests_raw = c.get("interests", []) or []
            c_interests = set(str(i).strip().lower() for i in c_interests_raw if i and str(i).strip())
            shared_interests = u_interests & c_interests

            # Explicit interest filter (from user search / checkbox selection)
            if filter_interests_set:
                if not (filter_interests_set & c_interests):
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

            # 4. Online-only filter
            if online_only and is_online_checker:
                if not is_online_checker(cid_str):
                    continue

            seen_candidate_ids.add(cid_str)
            filtered_candidates.append(c)

        # Rank surviving candidates deterministically using Weighted Scoring
        from ai_modules.match_recommender import MatchRecommender
        recommender = MatchRecommender()
        filtered_candidates = recommender.rank_candidates(user, filtered_candidates)

        logging.info(f"DISCOVER DEBUG: Final Discover users: {len(filtered_candidates)}")
        return filtered_candidates

    @staticmethod
    def get_mutual_matches(user_id):
        user = mongo.db.users.find_one({"_id": Match._oid(user_id)})
        if not user:
            return []
        uid_str = str(user_id)
        restricted_ids = set()
        try:
            from models.restriction import Restriction
            restricted_ids.update(str(r) for r in Restriction.get_restricted_user_ids(user_id))
        except Exception:
            pass

        # Exclude bidirectional blocked users
        restricted_ids.update(str(x) for x in user.get("blocked", []))
        blocked_by_others = mongo.db.users.find({"blocked": uid_str}, {"_id": 1})
        for b in blocked_by_others:
            restricted_ids.add(str(b["_id"]))

        liked = [Match._oid(x) for x in user.get("liked", []) if Match._oid(x) and str(x) not in restricted_ids]
        return list(mongo.db.users.find({
            "_id": {"$in": liked},
            "liked": uid_str,
            "role": "user",
            "is_active": {"$ne": False},
            "blocked": {"$ne": uid_str}
        }))
