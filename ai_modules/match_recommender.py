"""
Weighted Scoring Recommendation Algorithm for InclusiveMatch AI
===============================================================
MCA Major Project: InclusiveMatch AI

Replaces the GCN + Transformer neural model in the active user recommendation
pipeline with a transparent, rule-based Weighted Scoring Algorithm.

Formula:
  Final Score = (Interest Score * 0.50) + (Age Score * 0.30) + (Gender Score * 0.20)

Components (each scaled 0 to 100):
1. Interest Score (50%): Jaccard similarity of normalized shared interests:
     (Shared Interests / Total Unique Interests) * 100
2. Age Compatibility (30%): Evaluates age within mandatory preferred age ranges
     with mutual compatibility enforcement. Incompatible ages yield 0.
3. Gender Compatibility (20%): Evaluates mutual gender preferences without
     hardcoded gender categories. Incompatible preferences yield 0.

Note:
  Weighted Scoring is a deterministic rule-based algorithm, not a trained ML model.
  Neural GCN+Transformer model weights are kept on disk for archival / research
  purposes, but are completely removed from the active recommendation path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class WeightedScorer:
    """
    Transparent, deterministic weighted scoring recommendation service.
    """

    WEIGHT_INTERESTS: float = 0.50
    WEIGHT_AGE: float = 0.30
    WEIGHT_GENDER: float = 0.20

    # ------------------------------------------------------------------
    # Data Normalization Helpers (Supports Dicts, Models, Legacy keys)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_field(user: Any, key: str, default: Any = None) -> Any:
        if user is None:
            return default
        if isinstance(user, dict):
            val = user.get(key)
        else:
            val = getattr(user, key, None)
        return val if val is not None else default

    @classmethod
    def _extract_interests(cls, user: Any) -> Set[str]:
        """Extracts normalized (lowercase, trimmed) interests set."""
        raw = cls._extract_field(user, "interests", [])
        if not isinstance(raw, (list, tuple, set)):
            return set()
        return set(
            str(i).strip().lower()
            for i in raw
            if i and isinstance(i, str) and str(i).strip()
        )

    @classmethod
    def _extract_age(cls, user: Any) -> Optional[int]:
        """Safely extracts age as an integer, or None if invalid/missing."""
        raw = cls._extract_field(user, "age")
        if raw is None:
            return None
        try:
            val = int(raw)
            return val if 18 <= val <= 120 else None
        except (ValueError, TypeError):
            return None

    @classmethod
    def _extract_gender(cls, user: Any) -> str:
        """Extracts normalized lowercase gender string."""
        raw = cls._extract_field(user, "gender")
        if not raw:
            return ""
        return str(raw).strip().lower()

    @classmethod
    def _extract_matching_preferences(cls, user: Any) -> Dict[str, Any]:
        """Extracts age range and preferred genders preferences."""
        mp = cls._extract_field(user, "matching_preferences")
        if not isinstance(mp, dict):
            mp = {}

        raw_min = mp.get("min_age")
        if raw_min is None:
            raw_min = cls._extract_field(user, "pref_min_age")
        raw_max = mp.get("max_age")
        if raw_max is None:
            raw_max = cls._extract_field(user, "pref_max_age")

        min_age = 18
        max_age = 99
        try:
            if raw_min is not None:
                min_age = int(raw_min)
        except (ValueError, TypeError):
            pass
        try:
            if raw_max is not None:
                max_age = int(raw_max)
        except (ValueError, TypeError):
            pass

        raw_pref_genders = mp.get("preferred_genders")
        if raw_pref_genders is None:
            raw_pref_genders = cls._extract_field(user, "pref_gender")
        if isinstance(raw_pref_genders, str):
            raw_pref_genders = [raw_pref_genders]
        elif not isinstance(raw_pref_genders, (list, tuple, set)):
            raw_pref_genders = []

        pref_genders = [
            str(g).strip().lower() for g in raw_pref_genders if g and str(g).strip()
        ]

        return {
            "min_age": min_age,
            "max_age": max_age,
            "preferred_genders": pref_genders,
        }

    # ------------------------------------------------------------------
    # Component 1: Shared Interests (50%)
    # ------------------------------------------------------------------
    @classmethod
    def calculate_interest_score(cls, user_a: Any, user_b: Any) -> float:
        """
        Computes Jaccard similarity for shared interests scaled to 0-100:
          Interest Score = (Shared Interests / Total Unique Interests) * 100
        Returns 0.0 if either user has no interests.
        """
        ints_a = cls._extract_interests(user_a)
        ints_b = cls._extract_interests(user_b)
        if not ints_a or not ints_b:
            return 0.0

        shared = ints_a & ints_b
        union = ints_a | ints_b
        if not union:
            return 0.0

        jaccard = len(shared) / len(union)
        return float(jaccard * 100.0)

    # ------------------------------------------------------------------
    # Component 2: Age Compatibility (30%)
    # ------------------------------------------------------------------
    @classmethod
    def is_age_compatible(cls, user_a: Any, user_b: Any) -> Tuple[bool, str]:
        """
        Enforces bidirectional age preference constraints.
        Returns (is_compatible, reason).
        """
        age_a = cls._extract_age(user_a)
        age_b = cls._extract_age(user_b)

        # If either age is missing, compatibility cannot be established
        if age_a is None or age_b is None:
            return False, "Missing age information."

        prefs_a = cls._extract_matching_preferences(user_a)
        prefs_b = cls._extract_matching_preferences(user_b)

        # Direction 1: Does candidate B satisfy user A's preference?
        if age_b < prefs_a["min_age"] or age_b > prefs_a["max_age"]:
            return False, f"Candidate age {age_b} is outside user range [{prefs_a['min_age']}, {prefs_a['max_age']}]."

        # Direction 2: Does user A satisfy candidate B's preference (mutual)?
        if age_a < prefs_b["min_age"] or age_a > prefs_b["max_age"]:
            return False, f"User age {age_a} is outside candidate range [{prefs_b['min_age']}, {prefs_b['max_age']}]."

        return True, "Age is mutually compatible."

    @classmethod
    def calculate_age_score(cls, user_a: Any, user_b: Any) -> float:
        """
        Calculates age compatibility score on a 0-100 scale.
        - Returns 0.0 if mandatory age preference is violated.
        - Inside preferred range, scores smoothly from 70 to 100 based on proximity
          to preferred midpoint.
        """
        is_compat, _ = cls.is_age_compatible(user_a, user_b)
        if not is_compat:
            return 0.0

        age_a = cls._extract_age(user_a) or 25
        age_b = cls._extract_age(user_b) or 25
        prefs_a = cls._extract_matching_preferences(user_a)
        prefs_b = cls._extract_matching_preferences(user_b)

        # Proximity score for A -> B
        mid_a = (prefs_a["min_age"] + prefs_a["max_age"]) / 2.0
        span_a = max(1.0, (prefs_a["max_age"] - prefs_a["min_age"]) / 2.0)
        dist_a = abs(age_b - mid_a)
        score_a = max(0.0, 100.0 - (dist_a / span_a) * 30.0)

        # Proximity score for B -> A
        mid_b = (prefs_b["min_age"] + prefs_b["max_age"]) / 2.0
        span_b = max(1.0, (prefs_b["max_age"] - prefs_b["min_age"]) / 2.0)
        dist_b = abs(age_a - mid_b)
        score_b = max(0.0, 100.0 - (dist_b / span_b) * 30.0)

        return float((score_a + score_b) / 2.0)

    # ------------------------------------------------------------------
    # Component 3: Gender Compatibility (20%)
    # ------------------------------------------------------------------
    @classmethod
    def is_gender_compatible(cls, user_a: Any, user_b: Any) -> Tuple[bool, str]:
        """
        Enforces bidirectional gender preference compatibility without hardcoding genders.
        Returns (is_compatible, reason).
        """
        gen_a = cls._extract_gender(user_a)
        gen_b = cls._extract_gender(user_b)

        prefs_a = cls._extract_matching_preferences(user_a)
        prefs_b = cls._extract_matching_preferences(user_b)

        p_genders_a = prefs_a["preferred_genders"]
        p_genders_b = prefs_b["preferred_genders"]

        # User A's preference check
        a_accepts_any = (not p_genders_a) or ("any" in p_genders_a) or ("all" in p_genders_a)
        # Candidate B's preference check (mutual)
        b_accepts_any = (not p_genders_b) or ("any" in p_genders_b) or ("all" in p_genders_b)

        if not gen_a or not gen_b:
            # If gender is omitted, cannot verify compatibility
            return False, "Missing gender profile information."

        if not a_accepts_any and gen_b not in p_genders_a:
            return False, f"Candidate gender '{gen_b}' does not match user preferences."

        if not b_accepts_any and gen_a not in p_genders_b:
            return False, f"User gender '{gen_a}' does not match candidate preferences."

        return True, "Gender preferences are mutually compatible."

    @classmethod
    def calculate_gender_score(cls, user_a: Any, user_b: Any) -> float:
        """
        Calculates gender compatibility score on a 0-100 scale.
        - Returns 100.0 if mutually compatible.
        - Returns 0.0 if preferences conflict or profile data is missing.
        """
        is_compat, _ = cls.is_gender_compatible(user_a, user_b)
        return 100.0 if is_compat else 0.0

    # ------------------------------------------------------------------
    # Final Compatibility Score Calculation (0 to 100)
    # ------------------------------------------------------------------
    @classmethod
    def compute_compatibility_score(cls, user_a: Any, user_b: Any) -> int:
        """
        Calculates overall compatibility score using the 3-component weighted formula:
          Final Score = (Interest Score * 0.50) + (Age Score * 0.30) + (Gender Score * 0.20)
        Returns an integer percentage in [0, 100].
        """
        if not user_a or not user_b:
            return 0

        int_score = cls.calculate_interest_score(user_a, user_b)
        age_score = cls.calculate_age_score(user_a, user_b)
        gen_score = cls.calculate_gender_score(user_a, user_b)

        weighted = (
            (int_score * cls.WEIGHT_INTERESTS)
            + (age_score * cls.WEIGHT_AGE)
            + (gen_score * cls.WEIGHT_GENDER)
        )

        return int(round(max(0.0, min(100.0, weighted))))

    # ------------------------------------------------------------------
    # Candidate Ranking & Filtering
    # ------------------------------------------------------------------
    def rank_candidates(self, current_user: Any, candidates: List[Any]) -> List[Any]:
        """
        Scores and ranks a list of eligible candidates using the Weighted Scoring Algorithm.
        Attaches _match_score and component scores to candidate dicts.
        Sorts candidates descending by _match_score.
        """
        if not candidates:
            return []

        u_ints = self._extract_interests(current_user)

        for c in candidates:
            c_ints = self._extract_interests(c)
            shared = u_ints & c_ints

            int_score = self.calculate_interest_score(current_user, c)
            age_score = self.calculate_age_score(current_user, c)
            gen_score = self.calculate_gender_score(current_user, c)

            final_score = int(
                round(
                    max(
                        0.0,
                        min(
                            100.0,
                            (int_score * self.WEIGHT_INTERESTS)
                            + (age_score * self.WEIGHT_AGE)
                            + (gen_score * self.WEIGHT_GENDER),
                        ),
                    )
                )
            )

            c["_match_score"] = final_score
            c["_interest_score"] = int(round(int_score))
            c["_age_score"] = int(round(age_score))
            c["_gender_score"] = int(round(gen_score))
            c["_shared_interests_count"] = len(shared)
            c_raw = self._extract_field(c, "interests", [])
            if isinstance(c_raw, (list, tuple, set)):
                c["_shared_interests"] = [
                    i for i in c_raw if str(i).strip().lower() in shared
                ]
            else:
                c["_shared_interests"] = list(shared)

        candidates.sort(key=lambda x: x.get("_match_score", 0), reverse=True)
        return candidates

    def filter_and_rank_candidates(
        self,
        current_user: Any,
        candidates: List[Any],
        require_shared_interests: bool = False,
    ) -> List[Any]:
        """
        Comprehensive candidate filtering and ranking:
        1. Excludes self.
        2. Excludes blocked users in either direction.
        3. Excludes deactivated or ineligible accounts.
        4. Excludes candidates violating mandatory age or gender preferences.
        5. Optionally filters for at least 1 shared interest.
        6. Ranks surviving candidates by weighted compatibility score.
        """
        if not current_user or not candidates:
            return []

        uid_str = str(self._extract_field(current_user, "_id", ""))
        user_blocked = set(str(b) for b in self._extract_field(current_user, "blocked", []))
        user_liked = set(str(l) for l in self._extract_field(current_user, "liked", []))
        user_disliked = set(str(d) for d in self._extract_field(current_user, "disliked", []))

        # Check restriction model if available
        restricted_ids = set()
        try:
            from models.restriction import Restriction
            restricted_ids = set(str(r) for r in Restriction.get_restricted_user_ids(uid_str))
        except Exception:
            pass

        eligible = []
        u_ints = self._extract_interests(current_user)

        for c in candidates:
            cid_str = str(self._extract_field(c, "_id", ""))

            # 1. Exclude self
            if cid_str == uid_str:
                continue

            # 2. Exclude blocked / restricted
            if cid_str in user_blocked or cid_str in restricted_ids:
                continue
            c_blocked = set(str(b) for b in self._extract_field(c, "blocked", []))
            if uid_str in c_blocked:
                continue

            # 3. Exclude already liked or disliked
            if cid_str in user_liked or cid_str in user_disliked:
                continue

            # 4. Exclude deactivated accounts
            if self._extract_field(c, "is_active") is False:
                continue
            if self._extract_field(c, "role") not in ("user", None):
                continue

            # 5. Enforce mandatory age preferences
            age_ok, _ = self.is_age_compatible(current_user, c)
            if not age_ok:
                continue

            # 6. Enforce mandatory gender preferences
            gen_ok, _ = self.is_gender_compatible(current_user, c)
            if not gen_ok:
                continue

            # 7. Optional requirement for shared interests
            c_ints = self._extract_interests(c)
            if require_shared_interests and u_ints and not (u_ints & c_ints):
                continue

            eligible.append(c)

        return self.rank_candidates(current_user, eligible)


# ---------------------------------------------------------------------------
# Recommender Service Alias & Compatibility Layer
# ---------------------------------------------------------------------------

class MatchRecommender(WeightedScorer):
    """
    Direct alias and replacement for MatchRecommender.
    Uses Weighted Scoring exclusively. No PyTorch GCN or Transformer
    checkpoint is loaded or executed during user recommendations.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.num_features = 62
        logger.info(
            "[MatchRecommender] Active matching powered by Weighted Scoring Algorithm "
            "(Interests: 50%, Age: 30%, Gender: 20%). Neural inference is disabled."
        )

    def baseline_rank(self, current_user: Any, candidates: List[Any]) -> List[Any]:
        return self.rank_candidates(current_user, candidates)

    # Legacy feature helpers retained strictly for test / offline training script compatibility
    def _calculate_edge_weight(self, u1: Any, u2: Any) -> float:
        int_score = self.calculate_interest_score(u1, u2) / 100.0
        is_gen, _ = self.is_gender_compatible(u1, u2)
        gen_score = 1.0 if is_gen else 0.0
        return float(min(1.0, max(0.0, (int_score * 0.70) + (gen_score * 0.30))))

    def _extract_features(self, user: Any):
        import numpy as np
        f = np.zeros(self.num_features, dtype=np.float32)
        age = self._extract_age(user)
        if age:
            f[0] = min(max(age, 0) / 100.0, 1.0)
        return f


# ===========================================================================
# Legacy Archival Neural Network Definitions (Kept for Offline Training Code)
# ===========================================================================

class GraphConvolution:
    """Legacy GCN layer placeholder for offline research script compatibility."""
    def __init__(self, *args, **kwargs):
        pass


class MatchPredictorGCNTransformer:
    """Legacy GCN+Transformer placeholder for offline training script compatibility."""
    def __init__(self, *args, **kwargs):
        pass
