"""
Audit and Verification Tests for Rule-Based Match Recommendation System
=======================================================================
InclusiveMatch AI - Major Project: Rule-Based Matching System

Covers:
1. Shared and different interests (Jaccard similarity, 50% weight).
2. Matching and non-matching preferences (Age 30%, Gender 20%).
3. Missing or invalid profile fields (safe fallbacks, zero scores).
4. Like and mutual-match behavior (one-sided vs mutual match, un-like).
5. Dislikes and blocked users (bidirectional exclusion, restrictions).
6. Score boundaries and percentage calculations (strictly [0, 100]).
7. Duplicate recommendation prevention (unique candidates).
8. Communication preferences filtering (Text Chat, Sign Language, Voice).
9. Database error handling and invalid input validation.
"""

import os
import sys
import pytest
from bson import ObjectId

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import create_app
from extensions import mongo
from models.match import Match
from models.user import UserModel
from ai_modules.match_recommender import MatchRecommender, WeightedScorer


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-matching-audit-key-2026"
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app_context():
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def scorer():
    return MatchRecommender()


@pytest.fixture
def test_users_cleanup(app):
    created_ids = []

    def _create(email, name, role="user", age=25, gender="Female", interests=None, matching_prefs=None, comm_prefs=None):
        doc = {
            "email": email,
            "password": "hashed_pass_placeholder",
            "name": name,
            "role": role,
            "is_active": True,
            "age": age,
            "gender": gender,
            "interests": interests or ["Music", "Reading"],
            "matching_preferences": matching_prefs or {"min_age": 20, "max_age": 35, "preferred_genders": ["Any"]},
            "communication_preferences": comm_prefs or ["Text Chat", "Sign Language"],
            "comm_prefs": comm_prefs or ["Text Chat", "Sign Language"],
            "liked": [],
            "disliked": [],
            "blocked": [],
        }
        res = mongo.db.users.insert_one(doc)
        uid = str(res.inserted_id)
        created_ids.append(uid)
        return uid

    yield _create

    for uid in created_ids:
        try:
            mongo.db.users.delete_one({"_id": ObjectId(uid)})
            mongo.db.user_restrictions.delete_many({
                "$or": [{"restrictor_id": uid}, {"restricted_user_id": uid}]
            })
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. Shared & Different Interests (Jaccard 50% Weight)
# ---------------------------------------------------------------------------
def test_interests_jaccard_similarity(scorer):
    # Identical interests -> 100%
    u1 = {"interests": ["Music", "Coding", "Travel"]}
    u2 = {"interests": ["coding", "travel", "music"]}
    assert scorer.calculate_interest_score(u1, u2) == 100.0

    # Partial overlap: intersection = 2, union = 4 -> 50%
    u3 = {"interests": ["Cricket", "Football", "Chess", "Movies"]}
    u4 = {"interests": ["Cricket", "Football", "Gaming", "Anime"]}
    # intersection: 2 ('cricket', 'football'), union: 6 ('cricket', 'football', 'chess', 'movies', 'gaming', 'anime')
    # 2/6 = 33.333%
    assert abs(scorer.calculate_interest_score(u3, u4) - 33.333) < 0.01

    # Disjoint interests -> 0%
    u5 = {"interests": ["Dancing", "Yoga"]}
    u6 = {"interests": ["Programming", "Robotics"]}
    assert scorer.calculate_interest_score(u5, u6) == 0.0

    # Whitespace and casing tolerance
    u7 = {"interests": ["  Artificial Intelligence  ", "ART"]}
    u8 = {"interests": ["artificial intelligence", "art"]}
    assert scorer.calculate_interest_score(u7, u8) == 100.0


# ---------------------------------------------------------------------------
# 2. Matching and Non-Matching Age Preferences (30% Weight)
# ---------------------------------------------------------------------------
def test_age_compatibility_bidirectional(scorer):
    # Mutual match: User A (25) in B's range [20, 30], User B (27) in A's range [22, 32]
    u_a = {"age": 25, "matching_preferences": {"min_age": 22, "max_age": 32}}
    u_b = {"age": 27, "matching_preferences": {"min_age": 20, "max_age": 30}}
    assert scorer.is_age_compatible(u_a, u_b)[0] is True
    assert scorer.calculate_age_score(u_a, u_b) >= 70.0

    # Non-matching: Candidate B (40) outside user A's range [20, 30]
    u_c = {"age": 40, "matching_preferences": {"min_age": 18, "max_age": 50}}
    assert scorer.is_age_compatible(u_a, u_c)[0] is False
    assert scorer.calculate_age_score(u_a, u_c) == 0.0

    # Non-matching reverse: User A (25) outside candidate D's range [30, 40]
    u_d = {"age": 32, "matching_preferences": {"min_age": 30, "max_age": 40}}
    assert scorer.is_age_compatible(u_a, u_d)[0] is False
    assert scorer.calculate_age_score(u_a, u_d) == 0.0


# ---------------------------------------------------------------------------
# 3. Matching and Non-Matching Gender Preferences (20% Weight)
# ---------------------------------------------------------------------------
def test_gender_compatibility_bidirectional(scorer):
    # Mutual match: Female seeks Male, Male seeks Female
    u_fem = {"gender": "Female", "matching_preferences": {"preferred_genders": ["Male"]}}
    u_male = {"gender": "Male", "matching_preferences": {"preferred_genders": ["Female"]}}
    assert scorer.is_gender_compatible(u_fem, u_male)[0] is True
    assert scorer.calculate_gender_score(u_fem, u_male) == 100.0

    # 'Any' preference matches any gender
    u_nb = {"gender": "Non-binary", "matching_preferences": {"preferred_genders": ["Any"]}}
    u_other = {"gender": "Female", "matching_preferences": {"preferred_genders": ["Any"]}}
    assert scorer.is_gender_compatible(u_nb, u_other)[0] is True
    assert scorer.calculate_gender_score(u_nb, u_other) == 100.0

    # Mismatch: Male seeks Male, Female seeks Male -> female is rejected by candidate
    assert scorer.is_gender_compatible(u_fem, {"gender": "Male", "matching_preferences": {"preferred_genders": ["Male"]}})[0] is False


# ---------------------------------------------------------------------------
# 4. Missing and Malformed Profile Fields
# ---------------------------------------------------------------------------
def test_missing_profile_fields_safe_handling(scorer):
    empty_profile = {}
    valid_profile = {
        "age": 25,
        "gender": "Female",
        "interests": ["Music"],
        "matching_preferences": {"min_age": 18, "max_age": 35, "preferred_genders": ["Male"]}
    }

    # Does not throw exception; yields safe integer score 0
    score = scorer.compute_compatibility_score(empty_profile, valid_profile)
    assert score == 0

    # Missing age in one profile yields 0 age score
    no_age = {"gender": "Female", "interests": ["Music"]}
    assert scorer.calculate_age_score(no_age, valid_profile) == 0.0

    # Missing gender in one profile yields 0 gender score
    no_gender = {"age": 25, "interests": ["Music"]}
    assert scorer.calculate_gender_score(no_gender, valid_profile) == 0.0

    # Missing interests yields 0 interest score
    no_interests = {"age": 25, "gender": "Female"}
    assert scorer.calculate_interest_score(no_interests, valid_profile) == 0.0


# ---------------------------------------------------------------------------
# 5. Score Boundaries and Formula Integration
# ---------------------------------------------------------------------------
def test_score_boundaries_and_formula(scorer):
    # Perfect match: 100% interest, 100% age, 100% gender -> 100%
    u1 = {
        "age": 25,
        "gender": "Female",
        "interests": ["Art", "Coding"],
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Male"]}
    }
    u2 = {
        "age": 25,
        "gender": "Male",
        "interests": ["Art", "Coding"],
        "matching_preferences": {"min_age": 20, "max_age": 30, "preferred_genders": ["Female"]}
    }
    score = scorer.compute_compatibility_score(u1, u2)
    assert score == 100

    # Component weights: 50% * 100 + 30% * 100 + 20% * 100 = 100
    # Zero interest match: 0% * 50 + 100% * 30 + 100% * 20 = 50%
    u2_no_common_interest = dict(u2, interests=["Dancing", "Hiking"])
    score_no_common = scorer.compute_compatibility_score(u1, u2_no_common_interest)
    assert score_no_common == 50

    # Bounds check: score is always an int in [0, 100]
    assert isinstance(score, int)
    assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# 6. Like and Mutual-Match Behavior
# ---------------------------------------------------------------------------
def test_like_and_mutual_match_behavior(app, test_users_cleanup):
    user_a = test_users_cleanup("like_test_a@example.com", "Alice")
    user_b = test_users_cleanup("like_test_b@example.com", "Bob")

    # Initially not mutual
    assert Match.is_mutual(user_a, user_b) is False

    # Alice likes Bob (one-sided)
    ok = Match.like(user_a, user_b)
    assert ok is True
    assert Match.is_mutual(user_a, user_b) is False

    # Idempotent like: Alice likes Bob again -> still True and not mutual
    ok_again = Match.like(user_a, user_b)
    assert ok_again is True
    assert Match.is_mutual(user_a, user_b) is False

    # Bob likes Alice back -> Mutual match!
    ok_b = Match.like(user_b, user_a)
    assert ok_b is True
    assert Match.is_mutual(user_a, user_b) is True
    assert Match.are_matched(user_a, user_b) is True

    # Mutual matches query returns each other
    matches_a = Match.get_mutual_matches(user_a)
    assert any(str(m["_id"]) == user_b for m in matches_a)

    matches_b = Match.get_mutual_matches(user_b)
    assert any(str(m["_id"]) == user_a for m in matches_b)

    # Part ways safely ends match
    Match.part_ways(user_a, user_b)
    assert Match.is_mutual(user_a, user_b) is False


# ---------------------------------------------------------------------------
# 7. Dislikes and Bidirectional Blocked Users
# ---------------------------------------------------------------------------
def test_dislike_and_bidirectional_blocks(app, test_users_cleanup):
    user_me = test_users_cleanup("block_test_me@example.com", "Me")
    user_disliked = test_users_cleanup("block_test_disliked@example.com", "Disliked")
    user_i_blocked = test_users_cleanup("block_test_i_blocked@example.com", "I Blocked")
    user_who_blocked_me = test_users_cleanup("block_test_who_blocked_me@example.com", "Blocks Me")
    user_valid = test_users_cleanup("block_test_valid@example.com", "Valid Candidate")

    # 1. Dislike user
    Match.dislike(user_me, user_disliked)

    # 2. Block: Me blocks user_i_blocked
    mongo.db.users.update_one({"_id": ObjectId(user_me)}, {"$addToSet": {"blocked": user_i_blocked}})

    # 3. Block: user_who_blocked_me blocks Me
    mongo.db.users.update_one({"_id": ObjectId(user_who_blocked_me)}, {"$addToSet": {"blocked": user_me}})

    # Retrieve discover candidates
    candidates = Match.get_discover_candidates(user_me, require_shared_interests=False)
    candidate_ids = [str(c["_id"]) for c in candidates]

    # Exclusions verified
    assert user_me not in candidate_ids
    assert user_disliked not in candidate_ids
    assert user_i_blocked not in candidate_ids
    assert user_who_blocked_me not in candidate_ids
    assert user_valid in candidate_ids

    # Blocked user cannot be liked
    assert Match.like(user_me, user_who_blocked_me) is False
    assert Match.like(user_who_blocked_me, user_me) is False


# ---------------------------------------------------------------------------
# 8. Duplicate Recommendation Prevention
# ---------------------------------------------------------------------------
def test_duplicate_recommendation_prevention(app, test_users_cleanup):
    user_id = test_users_cleanup("dedup_user@example.com", "Dedup User")
    candidate_1 = test_users_cleanup("candidate_1@example.com", "Candidate One")
    candidate_2 = test_users_cleanup("candidate_2@example.com", "Candidate Two")

    candidates = Match.get_discover_candidates(user_id, require_shared_interests=False)
    ids = [str(c["_id"]) for c in candidates]

    # No duplicates in candidates list
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 9. Communication Preferences Filtering
# ---------------------------------------------------------------------------
def test_communication_preferences_filter(app, test_users_cleanup):
    user_id = test_users_cleanup("comm_pref_user@example.com", "Comm User")
    
    # User with Sign Language
    sign_user = test_users_cleanup(
        "comm_sign@example.com",
        "Sign Signer",
        comm_prefs=["Sign Language"]
    )
    # User with Text Chat
    text_user = test_users_cleanup(
        "comm_text@example.com",
        "Text Chatter",
        comm_prefs=["Text Chat"]
    )

    # Filter for 'sign_language'
    sign_candidates = Match.get_discover_candidates(user_id, comm_pref="sign_language", require_shared_interests=False)
    sign_ids = [str(c["_id"]) for c in sign_candidates]
    assert sign_user in sign_ids

    # Filter for 'text'
    text_candidates = Match.get_discover_candidates(user_id, comm_pref="text", require_shared_interests=False)
    text_ids = [str(c["_id"]) for c in text_candidates]
    assert text_user in text_ids


# ---------------------------------------------------------------------------
# 10. Database Failures & Invalid User Input Validation
# ---------------------------------------------------------------------------
def test_database_and_invalid_inputs(app):
    # Invalid ObjectId strings
    assert Match.like("invalid_id", "invalid_target") is False
    assert Match.dislike("invalid_id", "invalid_target") is False
    assert Match.is_mutual("invalid_id", "invalid_target") is False
    assert Match.get_discover_candidates("invalid_id") == []
    assert Match.get_mutual_matches("invalid_id") == []

    # Non-existent user IDs
    fake_oid1 = str(ObjectId())
    fake_oid2 = str(ObjectId())
    assert Match.like(fake_oid1, fake_oid2) is False
    assert Match.is_mutual(fake_oid1, fake_oid2) is False

