import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if "" not in sys.path:
    sys.path.insert(0, "")

import pytest
from unittest.mock import MagicMock, patch
from bson import ObjectId


def test_dashboard_template_interests_ui():
    template_path = os.path.join("templates", "dashboard.html")
    assert os.path.exists(template_path), "templates/dashboard.html must exist"

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Old dropdown must be removed
    assert 'id="filter-interest-select"' not in html, "Old dropdown select must be removed"
    assert '<select name="interest"' not in html, "Old interest select tag must be removed"

    # 2. Add Interests button must exist
    assert 'id="open-interest-picker-btn"' in html, "Add Interests button must exist"
    assert 'id="interest-btn-label"' in html, "Button label span must exist"

    # 3. Selected chips container & hidden inputs
    assert 'id="selected-interests-chips"' in html, "Selected interests chips container must exist"
    assert 'id="interests-hidden-inputs"' in html, "Hidden inputs container must exist"
    assert 'chip-remove-btn' in html, "Chip remove button class must exist"

    # 4. Clear all button
    assert 'id="clear-interests-btn"' in html, "Clear All button must exist"

    # 5. Interest selection modal panel
    assert 'id="interestSelectionModal"' in html, "Interest selection modal must exist"
    assert 'id="interest-search-input"' in html, "Interest search input must exist"
    assert 'interest-checkbox' in html, "Interest checkbox class must exist"
    assert 'interest-checkbox-card' in html, "Interest checkbox card class must exist"
    assert 'id="interest-modal-done-btn"' in html, "Modal Done button must exist"
    assert 'id="interest-modal-cancel-btn"' in html, "Modal Cancel button must exist"
    assert 'id="modal-selected-counter"' in html, "Modal selected counter must exist"

    # 6. Styling: White and purple theme classes
    assert '.interest-chip' in html, "Interest chip CSS must be defined"
    assert '.interest-checkbox-card.selected' in html, "Selected checkbox card CSS must be defined"
    assert '#6366f1' in html, "Primary purple color #6366f1 must be present in styling"


def test_models_match_get_discover_candidates_multiple_interests():
    """Unit test for Match.get_discover_candidates interest filtering logic."""
    from models.match import Match

    uid = ObjectId()
    current_user_doc = {
        "_id": uid,
        "name": "Alex",
        "gender": "Non-binary",
        "age": 25,
        "interests": ["Music", "Reading", "Travel"],
        "matching_preferences": {
            "preferred_genders": ["Any"],
            "min_age": 18,
            "max_age": 50
        },
        "liked": [],
        "disliked": [],
        "blocked": [],
        "is_active": True,
        "role": "user"
    }

    c1 = {
        "_id": ObjectId(),
        "name": "Candidate One (Music & Travel)",
        "gender": "Female",
        "age": 24,
        "interests": ["Music", "Travel"],
        "matching_preferences": {"preferred_genders": ["Any"]},
        "blocked": [],
        "is_active": True,
        "role": "user"
    }

    c2 = {
        "_id": ObjectId(),
        "name": "Candidate Two (Reading only)",
        "gender": "Male",
        "age": 26,
        "interests": ["Reading"],
        "matching_preferences": {"preferred_genders": ["Any"]},
        "blocked": [],
        "is_active": True,
        "role": "user"
    }

    c3 = {
        "_id": ObjectId(),
        "name": "Candidate Three (Cooking only - disjoint)",
        "gender": "Female",
        "age": 28,
        "interests": ["Cooking"],
        "matching_preferences": {"preferred_genders": ["Any"]},
        "blocked": [],
        "is_active": True,
        "role": "user"
    }

    mock_db = MagicMock()
    mock_db.users.find_one.return_value = current_user_doc
    mock_db.users.find.return_value = [c1, c2, c3]

    with patch("models.match.mongo.db", mock_db):
        # Case A: Filter by multiple interests ['Music', 'Reading'] -> c1 and c2 match, c3 excluded
        res_multi = Match.get_discover_candidates(
            user_id=uid,
            filter_interest=["Music", "Reading"],
            require_shared_interests=False
        )
        names_multi = [c["name"] for c in res_multi]
        assert "Candidate One (Music & Travel)" in names_multi
        assert "Candidate Two (Reading only)" in names_multi
        assert "Candidate Three (Cooking only - disjoint)" not in names_multi

        # Case B: Filter by single interest 'Travel' -> only c1 matches
        res_single = Match.get_discover_candidates(
            user_id=uid,
            filter_interest="Travel",
            require_shared_interests=False
        )
        names_single = [c["name"] for c in res_single]
        assert "Candidate One (Music & Travel)" in names_single
        assert "Candidate Two (Reading only)" not in names_single

        # Case C: Filter by comma-separated string 'Music, Travel' -> only c1 matches
        res_csv = Match.get_discover_candidates(
            user_id=uid,
            filter_interest="Music, Travel",
            require_shared_interests=False
        )
        names_csv = [c["name"] for c in res_csv]
        assert "Candidate One (Music & Travel)" in names_csv
        assert "Candidate Two (Reading only)" not in names_csv

        # Case D: Duplicate entries in list ['Music', 'music', 'MUSIC'] -> handles cleanly
        res_dups = Match.get_discover_candidates(
            user_id=uid,
            filter_interest=["Music", "music", "MUSIC"],
            require_shared_interests=False
        )
        names_dups = [c["name"] for c in res_dups]
        assert "Candidate One (Music & Travel)" in names_dups


if __name__ == "__main__":
    test_dashboard_template_interests_ui()
    print("Dashboard template UI assertions PASSED!")
    test_models_match_get_discover_candidates_multiple_interests()
    print("Match model candidate filtering assertions PASSED!")
