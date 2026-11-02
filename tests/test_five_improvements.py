import os
import re
import pytest
from bson import ObjectId
from app import create_app
from extensions import mongo
from models.user import UserModel
from models.match import Match
from routes.profile_routes import INTEREST_CATEGORIES


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-five-improvements-2026"
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app_context():
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, email, password):
    client.get("/logout")
    return client.post(
        "/login",
        data={"email": email, "password": password, "remember": False},
        follow_redirects=True,
    )


@pytest.fixture
def regular_user(app):
    email = "five_user_a@example.com"
    pwd = "TesterPass123!"
    uid = UserModel.create_user(
        email,
        pwd,
        "Five User A",
        role="user"
    )
    mongo.db.users.update_one(
        {"_id": ObjectId(uid)},
        {"$set": {
            "gender": "Male",
            "age": 28,
            "interests": ["Cricket", "Movies"],
            "matching_preferences": {
                "preferred_genders": ["Female"],
                "min_age": 20,
                "max_age": 40
            }
        }}
    )
    yield {"id": uid, "email": email, "password": pwd}
    mongo.db.users.delete_one({"_id": ObjectId(uid)})


@pytest.fixture
def second_user(app):
    email = "five_user_b@example.com"
    pwd = "TesterPass123!"
    uid = UserModel.create_user(
        email,
        pwd,
        "Five User B",
        role="user"
    )
    mongo.db.users.update_one(
        {"_id": ObjectId(uid)},
        {"$set": {
            "gender": "Female",
            "age": 25,
            "interests": ["Movies", "Music"],
            "matching_preferences": {
                "preferred_genders": ["Male"],
                "min_age": 20,
                "max_age": 40
            }
        }}
    )
    yield {"id": uid, "email": email, "password": pwd}
    mongo.db.users.delete_one({"_id": ObjectId(uid)})


def test_1_registration_privacy_consent_alignment(client):
    """Verify registration page has proper flex alignment for privacy consent and preserves mandatory validation."""
    res = client.get("/register")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Checkbox properties
    assert 'id="terms_consent"' in html
    assert 'name="terms_consent"' in html
    assert 'required' in html

    # Label properties
    assert 'for="terms_consent"' in html
    assert "Terms &amp; Conditions" in html or "Terms & Conditions" in html
    assert "Privacy Policy" in html

    # Flex alignment styling
    assert "d-flex align-items-start gap-3" in html
    assert "flex-shrink-0" in html


def test_2_profile_editing_location_manual_only(client, regular_user, app):
    """Verify current location/geolocation controls are removed from profile editing while manual entry is preserved."""
    with app.app_context():
        UserModel.update_profile(
            regular_user["id"],
            {
                "location": {
                    "mode": "manual",
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "city": "Bengaluru",
                    "state": "Karnataka",
                    "country": "India",
                    "display_name": "Bengaluru, Karnataka, India"
                }
            }
        )

    # Log in
    _login(client, regular_user["email"], regular_user["password"])

    # GET profile edit page
    res = client.get("/profile/edit")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Geolocation / detect button must be absent
    assert "btn-get-location" not in html
    assert "Detect My Location" not in html
    assert "location_current_wrapper" not in html
    assert "Use Current Location" not in html
    assert "Enter Location Manually" not in html
    assert "navigator.geolocation" not in html

    # Manual location entry must be present
    assert 'name="location_display"' in html
    assert 'id="val-location"' in html

    # POST profile edit with new manual location
    post_res = client.post("/profile/edit", data={
        "first_name": "UpdatedName",
        "surname": "UpdatedSurname",
        "bio": "Updated bio text",
        "age": "26",
        "gender": "Female",
        "marital_status": "Single",
        "relationship_goal": "Friendship",
        "location_display": "Mysuru, Karnataka, India",
        "interests": ["Cricket", "Movies"]
    }, follow_redirects=True)
    assert post_res.status_code == 200

    # Verify MongoDB preserved coordinates and updated display name
    updated_user = UserModel.get_by_id(regular_user["id"])
    assert updated_user.location["display_name"] == "Mysuru, Karnataka, India"
    assert updated_user.location["mode"] == "manual"
    assert updated_user.location["latitude"] == 12.9716
    assert updated_user.location["longitude"] == 77.5946


def test_3_discover_filters_and_interest_consistency(client, regular_user):
    """Verify 'Online Only' is completely removed from discover filters and interests match INTEREST_CATEGORIES."""
    _login(client, regular_user["email"], regular_user["password"])
    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # 1. 'Online Only' filter removed
    assert "filter-online-switch" not in html
    assert "Online users only" not in html
    assert 'name="online"' not in html

    # 2. Key filters preserved
    assert 'id="filter-gender-group"' in html or 'name="gender"' in html
    assert 'id="min_age_input"' in html or 'name="min_age"' in html
    assert 'id="filter-comm-select"' in html or 'name="comm_pref"' in html
    assert 'id="interestSelectionModal"' in html

    # 3. Interests in discover filter match INTEREST_CATEGORIES
    decoded_html = html.replace("&amp;", "&")
    for cat_name, items in INTEREST_CATEGORIES.items():
        assert cat_name in decoded_html, f"Category '{cat_name}' must be rendered in filter panel"
        for item in items:
            assert f'value="{item}"' in decoded_html, f"Interest item '{item}' must be selectable in filter panel"


def test_4_video_call_audio_echo_cancellation_and_responsive_frames(client, regular_user, second_user, app):
    """Verify audio constraints include echo cancellation and video frame uses responsive contain layout."""
    with app.app_context():
        # Create mutual match
        Match.like(regular_user["id"], second_user["id"])
        Match.like(second_user["id"], regular_user["id"])

    _login(client, regular_user["email"], regular_user["password"])
    res = client.get(f"/video/{second_user['id']}")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Audio echo cancellation constraints
    assert "echoCancellation: true" in html
    assert "noiseSuppression: true" in html
    assert "autoGainControl: true" in html

    # Duplicate audio track prevention in JS
    assert "remoteStream.getAudioTracks()" in html
    assert "remoteStream.removeTrack" in html

    # Video frame responsiveness
    assert "video-stage-container" in html
    assert "remote-video-feed" in html
    assert "object-fit: contain !important" in html
    assert "local-pip-video" in html

    # Aspect-ratio 16/9 removed from rigid stage
    assert 'style="aspect-ratio:16/9; background:#0f172a;"' not in html


def test_5_mobile_back_button_navigation_history_cleanup(client, regular_user, second_user, app):
    """Verify video call cleans up state and uses location.replace to prevent back-button trap."""
    with app.app_context():
        # Create mutual match
        Match.like(regular_user["id"], second_user["id"])
        Match.like(second_user["id"], regular_user["id"])

    _login(client, regular_user["email"], regular_user["password"])
    res = client.get(f"/video/{second_user['id']}")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Uses window.location.replace on hangup and remote termination
    assert 'window.location.replace(' in html

    # Deregisters socket listeners in cleanupCall
    assert 'socket.off("signal")' in html
    assert 'socket.off("call_ended")' in html

    # Back navigation and bfcache protection
    assert 'window.addEventListener("popstate"' in html
    assert 'window.addEventListener("pageshow"' in html
    assert 'handleBackToChat' in html
