import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai_modules.match_recommender import MatchRecommender


@pytest.fixture
def recommender():
    return MatchRecommender()


def test_zero_shared_interests(recommender):
    """Test 1: Zero shared interests yields 0 interest points and lower score."""
    user_a = {
        "_id": "u1",
        "name": "Alice",
        "interests": ["Cricket", "Football"],
        "comm_prefs": ["text"],
    }
    user_b = {
        "_id": "u2",
        "name": "Bob",
        "interests": ["Cooking", "Dancing"],
        "comm_prefs": ["text"],
    }
    score = recommender.compute_compatibility_score(user_a, user_b)
    # Interest Jaccard is 0. Only communication matches (25 pts max).
    assert 0 <= score <= 30
    # Specifically interest score is 0
    ints_a = recommender._extract_interests(user_a)
    ints_b = recommender._extract_interests(user_b)
    assert len(ints_a & ints_b) == 0


def test_partial_shared_interests(recommender):
    """Test 2: Partial shared interests gives proportional Jaccard points."""
    user_a = {
        "_id": "u1",
        "interests": ["Sign Language", "Coffee", "Hiking", "Reading"],
        "comm_prefs": ["text"],
    }
    user_b = {
        "_id": "u2",
        "interests": ["Sign Language", "Coffee", "Photography", "Gaming"],
        "comm_prefs": ["text"],
    }
    score = recommender.compute_compatibility_score(user_a, user_b)
    # Intersection = 2 ('Sign Language', 'Coffee'), Union = 6 -> Jaccard = 2/6 = 0.333
    # Interest points = 0.333 * 45 = 15. Comm points = 25. Total ~ 40.
    assert 35 <= score <= 45


def test_identical_interests(recommender):
    """Test 3: Identical interests awards full 45 interest points."""
    user_a = {
        "_id": "u1",
        "interests": ["Sign Language", "Coffee", "Reading"],
    }
    user_b = {
        "_id": "u2",
        "interests": ["Sign Language", "Coffee", "Reading"],
    }
    score = recommender.compute_compatibility_score(user_a, user_b)
    # Identical interests alone should yield 45 points
    assert score == 45

    # If comm, goal, and preferences also match perfectly, score reaches 100
    user_a_full = {
        "interests": ["Sign Language", "Coffee"],
        "comm_prefs": ["text", "sign_language"],
        "relationship_goal": "Friendship",
        "age": 25,
        "gender": "Female",
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Male"]},
    }
    user_b_full = {
        "interests": ["Sign Language", "Coffee"],
        "comm_prefs": ["text", "sign_language"],
        "relationship_goal": "Friendship",
        "age": 27,
        "gender": "Male",
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]},
    }
    full_score = recommender.compute_compatibility_score(user_a_full, user_b_full)
    assert full_score == 100


def test_missing_relationship_goals(recommender):
    """Test 4: Missing relationship goals must NEVER count as a match."""
    # Both None
    u1 = {"relationship_goal": None}
    u2 = {"relationship_goal": None}
    assert recommender.compute_compatibility_score(u1, u2) == 0
    assert recommender._calculate_edge_weight(u1, u2) == 0.0

    # One None, one set
    u3 = {"relationship_goal": "Friendship"}
    u4 = {"relationship_goal": None}
    assert recommender.compute_compatibility_score(u3, u4) == 0
    assert recommender._calculate_edge_weight(u3, u4) == 0.0

    # Different goals
    u5 = {"relationship_goal": "Friendship"}
    u6 = {"relationship_goal": "Marriage"}
    assert recommender.compute_compatibility_score(u5, u6) == 0
    assert recommender._calculate_edge_weight(u5, u6) == 0.0

    # Matching goals (case-insensitive)
    u7 = {"relationship_goal": "Friendship"}
    u8 = {"relationship_goal": "friendship"}
    assert recommender.compute_compatibility_score(u7, u8) == 15
    assert recommender._calculate_edge_weight(u7, u8) == 0.25


def test_missing_profile_fields(recommender):
    """Test 5: Completely missing or empty profile fields must not crash and yield 0."""
    empty_a = {}
    empty_b = {"_id": "123"}
    score = recommender.compute_compatibility_score(empty_a, empty_b)
    assert score == 0

    malformed_a = {
        "age": None,
        "gender": None,
        "interests": None,
        "comm_prefs": None,
        "relationship_goal": None,
        "matching_preferences": None,
    }
    malformed_b = {
        "age": "invalid_age",
        "gender": "",
        "interests": [],
        "comm_pref": "",
        "relationship_goal": "   ",
        "matching_preferences": "not_a_dict",
    }
    score2 = recommender.compute_compatibility_score(malformed_a, malformed_b)
    assert score2 == 0

    # Feature extraction must also not crash
    feats_a = recommender._extract_features(malformed_a)
    feats_b = recommender._extract_features(malformed_b)
    assert len(feats_a) == recommender.num_features
    assert len(feats_b) == recommender.num_features


def test_scores_stay_between_0_and_100(recommender):
    """Test 6: Scores are always integer and stay strictly within [0, 100]."""
    test_cases = [
        ({}, {}),
        ({"interests": ["art"]}, {"interests": ["art"]}),
        ({"interests": ["art"], "comm_pref": "text"}, {"interests": ["music"], "comm_pref": "sign"}),
        (
            {"interests": ["a", "b"], "comm_prefs": ["text"], "relationship_goal": "Friendship", "age": 22},
            {"interests": ["b", "c"], "comm_prefs": ["text"], "relationship_goal": "Friendship", "age": 24},
        ),
    ]
    for u1, u2 in test_cases:
        score = recommender.compute_compatibility_score(u1, u2)
        assert isinstance(score, int)
        assert 0 <= score <= 100

        edge = recommender._calculate_edge_weight(u1, u2)
        assert isinstance(edge, float)
        assert 0.0 <= edge <= 1.0


def test_dashboard_template_no_hardcoded_98_percent():
    """Test 7: templates/dashboard.html must not contain hardcoded '98% Match'."""
    template_path = os.path.join(
        os.path.dirname(__file__), "..", "templates", "dashboard.html"
    )
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "98% Match" not in content
    assert "candidate.get('_match_score'" in content


def test_rank_candidates_sorting_and_differentiation(recommender):
    """Test 8: rank_candidates assigns _match_score and sorts candidates properly."""
    current_user = {
        "_id": "current",
        "name": "Current",
        "interests": ["Sign Language", "Coffee", "Reading"],
        "comm_prefs": ["text"],
        "relationship_goal": "Friendship",
    }
    strong_match = {
        "_id": "cand_strong",
        "name": "Strong Match",
        "interests": ["Sign Language", "Coffee", "Reading"],
        "comm_prefs": ["text"],
        "relationship_goal": "Friendship",
    }
    partial_match = {
        "_id": "cand_partial",
        "name": "Partial Match",
        "interests": ["Coffee"],
        "comm_prefs": ["text"],
    }
    zero_match = {
        "_id": "cand_zero",
        "name": "Zero Match",
        "interests": ["Gym", "Running"],
        "comm_pref": "voice",
    }

    candidates = [zero_match, strong_match, partial_match]
    ranked = recommender.rank_candidates(current_user, candidates)

    assert ranked[0]["name"] == "Strong Match"
    assert ranked[1]["name"] == "Partial Match"
    assert ranked[2]["name"] == "Zero Match"
    assert ranked[0]["_match_score"] > ranked[1]["_match_score"] > ranked[2]["_match_score"]
    assert ranked[2]["_match_score"] == 0
