"""
Unit and Integration Tests for Weighted Scoring Recommendation Algorithm
========================================================================
MCA Major Project: InclusiveMatch AI

Verifies:
1. Interest score Jaccard similarity (50% weight).
2. Age compatibility calculation and mutual preference enforcement (30% weight).
3. Gender compatibility and mutual preference enforcement (20% weight).
4. Weighted scoring formula integration:
     Final Score = (Interest * 0.50) + (Age * 0.30) + (Gender * 0.20)
5. Strict score bounds [0, 100].
6. Safe handling of missing, empty, or malformed profile fields.
7. Candidate filtering: self-exclusion, blocked users in either direction,
   ineligible accounts, and mandatory preference violators.
8. Candidate ranking order (descending by score).
9. Verification that no neural model (GCN+Transformer) is loaded during matching.
10. Verification of dashboard template compatibility display (no hardcoded 98%).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai_modules.match_recommender import MatchRecommender, WeightedScorer


@pytest.fixture
def scorer():
    return WeightedScorer()


# ---------------------------------------------------------------------------
# 1. Interest Score (Jaccard Similarity - 50% Weight)
# ---------------------------------------------------------------------------

def test_interest_score_empty_lists_handled_safely(scorer):
    """Empty or missing interest lists must return 0.0 safely."""
    u1 = {"interests": []}
    u2 = {"interests": []}
    assert scorer.calculate_interest_score(u1, u2) == 0.0

    u3 = {"interests": None}
    u4 = {}
    assert scorer.calculate_interest_score(u3, u4) == 0.0


def test_interest_score_zero_shared(scorer):
    """Disjoint interest lists yield 0.0 Jaccard similarity."""
    u1 = {"interests": ["Cricket", "Football"]}
    u2 = {"interests": ["Cooking", "Dancing"]}
    assert scorer.calculate_interest_score(u1, u2) == 0.0


def test_interest_score_partial_shared(scorer):
    """Partial overlap yields exact Jaccard similarity percentage."""
    # Intersection = 2 ('coffee', 'reading'), Union = 6
    # Jaccard = 2/6 = 33.333%
    u1 = {"interests": ["Coffee", "Reading", "Hiking", "Art"]}
    u2 = {"interests": ["Coffee", "Reading", "Gaming", "Music"]}
    score = scorer.calculate_interest_score(u1, u2)
    assert pytest.approx(score, 0.01) == 33.33


def test_interest_score_identical_interests(scorer):
    """Identical interests award full 100.0 interest points."""
    u1 = {"interests": ["Coffee", "Technology", "Reading"]}
    u2 = {"interests": ["coffee", "technology", "reading"]}  # case-insensitive
    assert scorer.calculate_interest_score(u1, u2) == 100.0


# ---------------------------------------------------------------------------
# 2. Age Compatibility (30% Weight) & Mutual Enforcement
# ---------------------------------------------------------------------------

def test_age_compatibility_mandatory_preference_violation(scorer):
    """Candidates outside user's mandatory preferred range yield 0.0 age score."""
    # User A prefers ages 20-28; Candidate B is 35 (too old)
    u_a = {"age": 24, "matching_preferences": {"min_age": 20, "max_age": 28}}
    u_b = {"age": 35, "matching_preferences": {"min_age": 18, "max_age": 40}}

    is_compat, _ = scorer.is_age_compatible(u_a, u_b)
    assert is_compat is False
    assert scorer.calculate_age_score(u_a, u_b) == 0.0


def test_age_compatibility_mutual_enforcement(scorer):
    """If candidate B rejects user A's age, mutual compatibility fails."""
    # User A (age 22) accepts Candidate B (age 30, range 20-35)
    # BUT Candidate B only accepts ages 28-35 (rejects 22)
    u_a = {"age": 22, "matching_preferences": {"min_age": 20, "max_age": 35}}
    u_b = {"age": 30, "matching_preferences": {"min_age": 28, "max_age": 35}}

    is_compat, reason = scorer.is_age_compatible(u_a, u_b)
    assert is_compat is False
    assert "User age 22 is outside candidate range" in reason
    assert scorer.calculate_age_score(u_a, u_b) == 0.0


def test_age_compatibility_mutually_compatible(scorer):
    """Mutually compatible ages yield a positive score between 70.0 and 100.0."""
    # User A (25, range 22-30), Candidate B (26, range 20-28)
    u_a = {"age": 25, "matching_preferences": {"min_age": 22, "max_age": 30}}
    u_b = {"age": 26, "matching_preferences": {"min_age": 20, "max_age": 28}}

    is_compat, _ = scorer.is_age_compatible(u_a, u_b)
    assert is_compat is True
    age_score = scorer.calculate_age_score(u_a, u_b)
    assert 70.0 <= age_score <= 100.0


# ---------------------------------------------------------------------------
# 3. Gender Compatibility (20% Weight) & Mutual Enforcement
# ---------------------------------------------------------------------------

def test_gender_compatibility_mutual_match(scorer):
    """Mutually compatible gender preferences award full 100.0 gender score."""
    u_a = {"gender": "Female", "matching_preferences": {"preferred_genders": ["Male"]}}
    u_b = {"gender": "Male", "matching_preferences": {"preferred_genders": ["Female"]}}

    assert scorer.is_gender_compatible(u_a, u_b)[0] is True
    assert scorer.calculate_gender_score(u_a, u_b) == 100.0


def test_gender_compatibility_any_preference(scorer):
    """'Any' preferred gender accepts any gender identity."""
    u_a = {"gender": "Non-binary", "matching_preferences": {"preferred_genders": ["Any"]}}
    u_b = {"gender": "Other", "matching_preferences": {"preferred_genders": ["Non-binary", "Any"]}}

    assert scorer.is_gender_compatible(u_a, u_b)[0] is True
    assert scorer.calculate_gender_score(u_a, u_b) == 100.0


def test_gender_compatibility_mismatch(scorer):
    """Conflicting gender preferences yield 0.0 gender score."""
    # User A wants Female, but candidate B is Male
    u_a = {"gender": "Male", "matching_preferences": {"preferred_genders": ["Female"]}}
    u_b = {"gender": "Male", "matching_preferences": {"preferred_genders": ["Male"]}}

    assert scorer.is_gender_compatible(u_a, u_b)[0] is False
    assert scorer.calculate_gender_score(u_a, u_b) == 0.0


# ---------------------------------------------------------------------------
# 4. Final Weighted Scoring Formula Integration
# ---------------------------------------------------------------------------

def test_final_weighted_formula_exact_calculation(scorer):
    """
    Final Score = (Interest * 0.50) + (Age * 0.30) + (Gender * 0.20)
    Verifies mathematical combination of the 3 components.
    """
    # 1. Perfect Match (100% interests, perfect age midpoint, mutual gender match)
    user_a = {
        "interests": ["Coffee", "Art"],
        "age": 25,
        "gender": "Female",
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Male"]},
    }
    user_b = {
        "interests": ["Coffee", "Art"],
        "age": 25,
        "gender": "Male",
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]},
    }
    # Expected: (100 * 0.50) + (100 * 0.30) + (100 * 0.20) = 50 + 30 + 20 = 100
    assert scorer.compute_compatibility_score(user_a, user_b) == 100

    # 2. Zero Interests, Perfect Age & Gender
    user_c = {
        "interests": ["Gym", "Running"],
        "age": 25,
        "gender": "Male",
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]},
    }
    # Expected: (0 * 0.50) + (100 * 0.30) + (100 * 0.20) = 0 + 30 + 20 = 50
    assert scorer.compute_compatibility_score(user_a, user_c) == 50

    # 3. Partial Interests (50% Jaccard), Perfect Age & Gender
    user_d = {
        "interests": ["Coffee", "Hiking"],  # 1 shared out of 3 total -> Jaccard = 33.33
        "age": 25,
        "gender": "Male",
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]},
    }
    # Expected: (33.33 * 0.50) + (100 * 0.30) + (100 * 0.20) = 16.67 + 30 + 20 = 66.67 -> 67
    assert scorer.compute_compatibility_score(user_a, user_d) == 67


# ---------------------------------------------------------------------------
# 5. Missing / Invalid Profile Data & Boundary Checks
# ---------------------------------------------------------------------------

def test_missing_and_malformed_profile_data(scorer):
    """Missing or invalid fields do not raise exceptions and yield 0."""
    empty_a = {}
    empty_b = {"_id": "123"}
    assert scorer.compute_compatibility_score(empty_a, empty_b) == 0

    malformed_a = {
        "age": "twenty-five",
        "gender": None,
        "interests": None,
        "matching_preferences": "invalid",
    }
    malformed_b = {
        "age": None,
        "gender": "",
        "interests": "not_a_list",
        "matching_preferences": None,
    }
    assert scorer.compute_compatibility_score(malformed_a, malformed_b) == 0


def test_scores_strictly_bounded_between_0_and_100(scorer):
    """Scores are integer percentages strictly bounded within [0, 100]."""
    test_pairs = [
        ({}, {}),
        ({"interests": ["tech"], "age": 24, "gender": "male"}, {"interests": ["tech"], "age": 24, "gender": "female"}),
        ({"age": 999, "gender": "unknown"}, {"age": -5, "gender": "none"}),
    ]
    for u1, u2 in test_pairs:
        s = scorer.compute_compatibility_score(u1, u2)
        assert isinstance(s, int)
        assert 0 <= s <= 100


# ---------------------------------------------------------------------------
# 6. Candidate Filtering Rules
# ---------------------------------------------------------------------------

def test_candidate_filtering_self_and_blocked(scorer):
    """filter_and_rank_candidates excludes self, blocked users in either direction, and age/gender violators."""
    current_user = {
        "_id": "user_current",
        "gender": "Female",
        "age": 25,
        "interests": ["Coffee", "Art"],
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Male"]},
        "blocked": ["user_blocked_by_me"],
        "liked": ["user_already_liked"],
        "disliked": [],
    }

    c_self = {"_id": "user_current", "name": "Self", "age": 25, "gender": "Female"}
    c_blocked_me = {"_id": "user_blocked_by_me", "name": "Blocked1", "age": 25, "gender": "Male"}
    c_blocked_user = {"_id": "user_blocks_me", "name": "Blocked2", "age": 25, "gender": "Male", "blocked": ["user_current"]}
    c_already_liked = {"_id": "user_already_liked", "name": "Liked", "age": 25, "gender": "Male"}
    c_age_violator = {
        "_id": "user_old",
        "name": "Too Old",
        "age": 45,  # User only accepts up to 30
        "gender": "Male",
        "matching_preferences": {"preferred_genders": ["Female"]},
        "interests": ["Coffee"],
    }
    c_valid = {
        "_id": "user_valid",
        "name": "Valid Candidate",
        "age": 26,
        "gender": "Male",
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]},
        "interests": ["Coffee", "Art"],
    }

    candidates = [c_self, c_blocked_me, c_blocked_user, c_already_liked, c_age_violator, c_valid]
    survivors = scorer.filter_and_rank_candidates(current_user, candidates)

    assert len(survivors) == 1
    assert survivors[0]["name"] == "Valid Candidate"
    assert survivors[0]["_match_score"] >= 95


# ---------------------------------------------------------------------------
# 7. Candidate Ranking Order
# ---------------------------------------------------------------------------

def test_rank_candidates_sorting_descending(scorer):
    """rank_candidates sorts candidates descending by _match_score."""
    current_user = {
        "_id": "u_me",
        "age": 25,
        "gender": "Female",
        "interests": ["Coffee", "Technology", "Art"],
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Male"]},
    }
    cand_high = {
        "_id": "c1",
        "name": "High Match",
        "age": 25,
        "gender": "Male",
        "interests": ["Coffee", "Technology", "Art"],  # 100% interests -> 100% total
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]},
    }
    cand_med = {
        "_id": "c2",
        "name": "Medium Match",
        "age": 25,
        "gender": "Male",
        "interests": ["Coffee"],  # 1 shared out of 3 -> 33% interests -> ~67% total
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]},
    }
    cand_low = {
        "_id": "c3",
        "name": "Low Match",
        "age": 25,
        "gender": "Male",
        "interests": ["Gym"],  # 0 shared interests -> 50% total
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]},
    }

    candidates = [cand_low, cand_high, cand_med]
    ranked = scorer.rank_candidates(current_user, candidates)

    assert ranked[0]["name"] == "High Match"
    assert ranked[1]["name"] == "Medium Match"
    assert ranked[2]["name"] == "Low Match"
    assert ranked[0]["_match_score"] > ranked[1]["_match_score"] > ranked[2]["_match_score"]


# ---------------------------------------------------------------------------
# 8. No Neural Model Loading Verification
# ---------------------------------------------------------------------------

def test_no_neural_model_loaded():
    """Verify MatchRecommender has self.model = None and requires no checkpoint."""
    rec = MatchRecommender()
    assert rec.model is None
    # compute_compatibility_score works with zero PyTorch forward pass
    score = rec.compute_compatibility_score(
        {"interests": ["ai"], "age": 25, "gender": "female"},
        {"interests": ["ai"], "age": 25, "gender": "male"}
    )
    assert isinstance(score, int)
    assert score > 0


# ---------------------------------------------------------------------------
# 9. Dashboard Template Dynamic Match Score Display
# ---------------------------------------------------------------------------

def test_dashboard_template_no_hardcoded_98_percent():
    """templates/dashboard.html must not contain hardcoded '98% Match' and use dynamic _match_score."""
    template_path = os.path.join(
        os.path.dirname(__file__), "..", "templates", "dashboard.html"
    )
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "98% Match" not in content
    assert "candidate.get('_match_score', 0)" in content
