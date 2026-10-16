"""Tests for Discover filtering and AI matching based on gender and interests."""
import pytest
from unittest.mock import MagicMock, patch
from bson import ObjectId
from models.match import Match


@pytest.fixture
def mock_mongo():
    with patch("models.match.mongo") as mock:
        yield mock


def test_discover_gender_and_interest_filtering(mock_mongo):
    """Verify Discover candidates appear strictly based on gender and shared interests."""
    user_id = str(ObjectId())
    user_doc = {
        "_id": ObjectId(user_id),
        "name": "Alex",
        "gender": "Male",
        "age": 26,
        "interests": ["Cricket", "Technology", "Gaming"],
        "matching_preferences": {
            "min_age": 20,
            "max_age": 35,
            "preferred_genders": ["Female"]
        },
        "liked": [],
        "disliked": [],
        "blocked": []
    }

    # Candidate 1: Female with shared interest ("Cricket") seeking Male -> MATCH
    c1 = {
        "_id": ObjectId(),
        "name": "Sarah",
        "gender": "Female",
        "age": 24,
        "interests": ["Cricket", "Music"],
        "matching_preferences": {"preferred_genders": ["Male"]},
        "role": "user",
        "is_active": True
    }

    # Candidate 2: Female with ZERO shared interests -> MUST BE EXCLUDED
    c2 = {
        "_id": ObjectId(),
        "name": "Emma",
        "gender": "Female",
        "age": 25,
        "interests": ["Yoga", "Cooking", "Baking"],
        "matching_preferences": {"preferred_genders": ["Male"]},
        "role": "user",
        "is_active": True
    }

    # Candidate 3: Male with shared interest ("Gaming") -> MUST BE EXCLUDED (gender mismatch)
    c3 = {
        "_id": ObjectId(),
        "name": "Dave",
        "gender": "Male",
        "age": 27,
        "interests": ["Gaming", "Technology"],
        "matching_preferences": {"preferred_genders": ["Female"]},
        "role": "user",
        "is_active": True
    }

    # Candidate 4: Female with shared interest ("Technology") but seeking Female only -> MUST BE EXCLUDED
    c4 = {
        "_id": ObjectId(),
        "name": "Lisa",
        "gender": "Female",
        "age": 28,
        "interests": ["Technology"],
        "matching_preferences": {"preferred_genders": ["Female"]},
        "role": "user",
        "is_active": True
    }

    # Candidate 5: Female with multiple shared interests ("Gaming", "Technology") seeking Male -> MATCH
    c5 = {
        "_id": ObjectId(),
        "name": "Chloe",
        "gender": "Female",
        "age": 23,
        "interests": ["Gaming", "Technology", "Art"],
        "matching_preferences": {"preferred_genders": ["Male", "Any"]},
        "role": "user",
        "is_active": True
    }

    mock_mongo.db.users.find_one.return_value = user_doc
    mock_mongo.db.users.count_documents.return_value = 5
    mock_mongo.db.users.find.return_value = [c1, c2, c3, c4, c5]

    candidates = Match.get_discover_candidates(user_id)

    # Only c1 and c5 should pass gender and interest matching
    candidate_names = [c["name"] for c in candidates]
    assert "Sarah" in candidate_names
    assert "Chloe" in candidate_names
    assert "Emma" not in candidate_names     # Excluded: zero shared interests
    assert "Dave" not in candidate_names     # Excluded: gender mismatch
    assert "Lisa" not in candidate_names     # Excluded: bidirectional gender mismatch

    # Check shared interests annotations
    chloe = next(c for c in candidates if c["name"] == "Chloe")
    assert set(chloe["_shared_interests"]) == {"Gaming", "Technology"}
    assert chloe["_shared_interests_count"] == 2


def test_discover_specific_interest_override(mock_mongo):
    """Verify specific interest filter narrows down candidates."""
    user_id = str(ObjectId())
    user_doc = {
        "_id": ObjectId(user_id),
        "name": "Alex",
        "gender": "Male",
        "interests": ["Cricket", "Gaming", "Music"],
        "matching_preferences": {"preferred_genders": ["Female"]},
        "liked": [], "disliked": [], "blocked": []
    }

    c_cricket = {
        "_id": ObjectId(),
        "name": "Pooja",
        "gender": "Female",
        "age": 24,
        "interests": ["Cricket"],
        "matching_preferences": {"preferred_genders": ["Any"]},
        "role": "user", "is_active": True
    }
    c_gaming = {
        "_id": ObjectId(),
        "name": "Maya",
        "gender": "Female",
        "age": 24,
        "interests": ["Gaming"],
        "matching_preferences": {"preferred_genders": ["Any"]},
        "role": "user", "is_active": True
    }

    mock_mongo.db.users.find_one.return_value = user_doc
    mock_mongo.db.users.find.return_value = [c_cricket, c_gaming]

    # Filter specifically by "Gaming"
    candidates = Match.get_discover_candidates(user_id, filter_interest="Gaming")
    names = [c["name"] for c in candidates]
    assert names == ["Maya"]


def test_discover_gender_filter_override(mock_mongo):
    """Verify user can dynamically override gender filter from the UI."""
    user_id = str(ObjectId())
    user_doc = {
        "_id": ObjectId(user_id),
        "name": "Jordan",
        "gender": "Non-binary",
        "interests": ["Art", "Music"],
        "matching_preferences": {"preferred_genders": ["Any"]},
        "liked": [], "disliked": [], "blocked": []
    }

    c_female = {
        "_id": ObjectId(),
        "name": "Fiona",
        "gender": "Female",
        "age": 22,
        "interests": ["Art"],
        "matching_preferences": {"preferred_genders": ["Any"]},
        "role": "user", "is_active": True
    }
    c_male = {
        "_id": ObjectId(),
        "name": "Mark",
        "gender": "Male",
        "age": 23,
        "interests": ["Art"],
        "matching_preferences": {"preferred_genders": ["Any"]},
        "role": "user", "is_active": True
    }

    mock_mongo.db.users.find_one.return_value = user_doc
    mock_mongo.db.users.find.return_value = [c_female, c_male]

    # Explicitly filter by Female
    candidates_female = Match.get_discover_candidates(user_id, filter_gender="Female")
    assert [c["name"] for c in candidates_female] == ["Fiona"]

    # Explicitly filter by Male
    candidates_male = Match.get_discover_candidates(user_id, filter_gender="Male")
    assert [c["name"] for c in candidates_male] == ["Mark"]
