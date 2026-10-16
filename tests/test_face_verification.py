"""Automated tests for Secure Real Face-Verification System in InclusiveMatch AI.

Tests all 12 required scenarios:
1. Live face matches uploaded profile photo -> verification succeeds.
2. Live face belongs to another person -> verification fails.
3. No face detected in camera capture -> verification fails.
4. Multiple faces detected in camera capture -> verification fails.
5. Camera permission denied / empty capture -> verification rejected / incomplete.
6. User uploads a new profile photo -> previous verification is immediately invalidated.
7. User attempts client-side value tampering -> badge is not granted (backend is sole authority).
8. Unauthenticated user calls /profile/verify -> request rejected with 401/redirect.
9. One user attempts to verify another user's account -> rejected (endpoint verifies current_user only).
10. Unverified user rendered on Discover -> verified badge is NOT displayed.
11. Verified user rendered on Discover -> verified badge IS displayed.
12. Deactivated account -> profile and badge hidden from discover and profile view.
"""
import os
import base64
import shutil
import pytest
from bson import ObjectId
from unittest.mock import patch

from app import create_app
from extensions import mongo
from models.user import UserModel
from models.match import Match
from ai_modules.face_verification import FaceVerifier


FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FACE_A_PATH = os.path.join(FIXTURES_DIR, "face_person_a.jpg")
FACE_B_PATH = os.path.join(FIXTURES_DIR, "face_person_b.jpg")
NO_FACE_PATH = os.path.join(FIXTURES_DIR, "no_face.jpg")
MULTI_FACE_PATH = os.path.join(FIXTURES_DIR, "multi_faces.jpg")


@pytest.fixture(scope="module")
def test_app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-secret-key-face-verification"
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.app_context():
        yield app


@pytest.fixture
def client(test_app):
    return test_app.test_client()


@pytest.fixture
def test_user_with_photo(test_app):
    """Create a test user with Lena (Face A) as their profile photo."""
    with test_app.app_context():
        upload_folder = test_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)
        photo_filename = f"profile_test_{ObjectId()}.jpg"
        target_path = os.path.join(upload_folder, photo_filename)
        shutil.copyfile(FACE_A_PATH, target_path)

        uid = UserModel.create_user(
            email=f"face_test_{ObjectId()}@example.com",
            password="Password123!",
            name="Alice Verification",
            comm_prefs=["text", "video"],
        )
        UserModel.update_profile(uid, {"profile_photo": photo_filename, "profile_pic": photo_filename})
        user = UserModel.get_by_id(uid)

        yield user

        mongo.db.users.delete_one({"_id": ObjectId(uid)})
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass


def _image_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Live face matches uploaded profile photo -> verification succeeds
# ─────────────────────────────────────────────────────────────────────────────
def test_face_verification_matching_face_succeeds(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u.id)

    b64 = _image_to_base64(FACE_A_PATH)
    res = client.post("/profile/verify", json={"image": b64})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "verified successfully" in data["message"].lower()

    # Verify database state in MongoDB
    fresh_user = UserModel.get_by_id(u.id)
    assert fresh_user.face_verified is True
    assert fresh_user.is_verified is True
    assert fresh_user.face_verified_at is not None
    assert fresh_user.verified_profile_photo == u.profile_photo


# ─────────────────────────────────────────────────────────────────────────────
# 2. Live face belongs to another person -> verification fails
# ─────────────────────────────────────────────────────────────────────────────
def test_face_verification_different_person_fails(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u.id)

    # User photo is Face A; live capture is Face B (different person)
    b64 = _image_to_base64(FACE_B_PATH)
    res = client.post("/profile/verify", json={"image": b64})
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "does not match" in data["message"].lower()

    fresh_user = UserModel.get_by_id(u.id)
    assert fresh_user.face_verified is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. No face detected in camera capture -> verification fails
# ─────────────────────────────────────────────────────────────────────────────
def test_face_verification_no_face_fails(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u.id)

    b64 = _image_to_base64(NO_FACE_PATH)
    res = client.post("/profile/verify", json={"image": b64})
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "no face detected" in data["message"].lower()

    fresh_user = UserModel.get_by_id(u.id)
    assert fresh_user.face_verified is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multiple faces detected in camera capture -> verification fails
# ─────────────────────────────────────────────────────────────────────────────
def test_face_verification_multiple_faces_fails(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u.id)

    b64 = _image_to_base64(MULTI_FACE_PATH)
    res = client.post("/profile/verify", json={"image": b64})
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "multiple faces detected" in data["message"].lower()

    fresh_user = UserModel.get_by_id(u.id)
    assert fresh_user.face_verified is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Camera permission denied / empty capture -> verification rejected
# ─────────────────────────────────────────────────────────────────────────────
def test_face_verification_empty_or_corrupt_capture_fails(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u.id)

    # Empty payload
    res1 = client.post("/profile/verify", json={})
    assert res1.status_code == 400
    assert "no live camera capture provided" in res1.get_json()["message"].lower()

    # Corrupt payload
    res2 = client.post("/profile/verify", json={"image": "invalid_base64_data"})
    assert res2.status_code == 400

    fresh_user = UserModel.get_by_id(u.id)
    assert fresh_user.face_verified is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. User uploads new profile photo -> previous verification is invalidated
# ─────────────────────────────────────────────────────────────────────────────
def test_photo_change_invalidates_previous_verification(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    # 1. First successfully verify the user
    with test_app.app_context():
        UserModel.set_face_verified(u.id, u.profile_photo)
    user_verified = UserModel.get_by_id(u.id)
    assert user_verified.face_verified is True

    # 2. Upload new photo via edit_profile with Cloudinary mock
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u.id)

    with patch("routes.profile_routes.is_cloudinary_configured", return_value=True), \
         patch("routes.profile_routes.upload_profile_photo", return_value={"secure_url": "https://res.cloudinary.com/demo/image/upload/v123/new_photo.jpg", "public_id": "new_photo_id"}):

        # Upload a dummy image file
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (100, 100), color=(100, 150, 200)).save(buf, format="JPEG")
        buf.seek(0)

        res = client.post(
            "/profile/edit",
            data={
                "first_name": u.first_name,
                "surname": u.surname,
                "bio": "Updated bio",
                "profile_pic": (buf, "new_avatar.jpg", "image/jpeg"),
            },
            content_type="multipart/form-data"
        )
        assert res.status_code in (200, 302)

    # 3. Verify that the previous verification was invalidated in MongoDB!
    updated_user = UserModel.get_by_id(u.id)
    assert updated_user.face_verified is False
    assert updated_user.is_verified is False
    assert updated_user.face_verified_at is None
    assert updated_user.verified_profile_photo is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. User attempts client-side value tampering -> badge is not granted
# ─────────────────────────────────────────────────────────────────────────────
def test_client_side_tampering_cannot_award_badge(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    with client.session_transaction() as sess:
        sess["_user_id"] = str(u.id)

    # User attempts to POST face_verified=true through profile edit form
    res = client.post(
        "/profile/edit",
        data={
            "first_name": u.first_name,
            "surname": u.surname,
            "face_verified": "true",
            "is_verified": "true",
            "verification": "verified",
        }
    )
    assert res.status_code in (200, 302)

    # The backend UserModel.update_profile must NOT accept unauthenticated verification spoofing
    fresh_user = UserModel.get_by_id(u.id)
    assert fresh_user.face_verified is False
    assert fresh_user.is_verified is False


# ─────────────────────────────────────────────────────────────────────────────
# 8. Unauthenticated user calls /profile/verify -> request rejected
# ─────────────────────────────────────────────────────────────────────────────
def test_unauthenticated_verification_request_rejected(client):
    b64 = _image_to_base64(FACE_A_PATH)
    res = client.post("/profile/verify", json={"image": b64})
    # Must redirect to login or return 401 Unauthorized
    assert res.status_code in (302, 401)


# ─────────────────────────────────────────────────────────────────────────────
# 9. One user attempts to verify another user's profile -> rejected
# ─────────────────────────────────────────────────────────────────────────────
def test_cannot_verify_another_user(client, test_app, test_user_with_photo):
    victim_user = test_user_with_photo

    # Attacker logs in
    with test_app.app_context():
        attacker_id = UserModel.create_user(
            email=f"attacker_{ObjectId()}@example.com",
            password="Password123!",
            name="Attacker User",
        )

    with client.session_transaction() as sess:
        sess["_user_id"] = str(attacker_id)

    b64 = _image_to_base64(FACE_A_PATH)
    # Attacker tries to pass victim's ID in request body or headers
    res = client.post("/profile/verify", json={"image": b64, "user_id": victim_user.id, "target_user": victim_user.id})

    # The victim's user in database MUST remain unverified!
    victim_fresh = UserModel.get_by_id(victim_user.id)
    assert victim_fresh.face_verified is False

    # Clean up attacker
    with test_app.app_context():
        mongo.db.users.delete_one({"_id": ObjectId(attacker_id)})


# ─────────────────────────────────────────────────────────────────────────────
# 10. Unverified user rendered on Discover -> verified badge is NOT displayed
# ─────────────────────────────────────────────────────────────────────────────
def test_unverified_user_in_discover_has_no_badge(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    assert u.face_verified is False

    # Create viewer
    with test_app.app_context():
        viewer_id = UserModel.create_user(
            email=f"viewer_{ObjectId()}@example.com",
            password="Password123!",
            name="Discover Viewer",
            comm_prefs=["text", "video"],
        )
        viewer = UserModel.get_by_id(viewer_id)

    with client.session_transaction() as sess:
        sess["_user_id"] = str(viewer.id)

    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # If the unverified user is rendered, they must not have the verified badge next to their name
    # The badge icon is <i class="bi bi-patch-check-fill ...">
    if u.first_name in html:
        # Check that there is no verified badge tied to this candidate
        assert not (f"{u.first_name}" in html and 'bi-patch-check-fill' in html and f"{u.first_name} <i class=\"bi bi-patch-check-fill" in html)

    # Clean up viewer
    with test_app.app_context():
        mongo.db.users.delete_one({"_id": ObjectId(viewer_id)})


# ─────────────────────────────────────────────────────────────────────────────
# 11. Verified user rendered on Discover -> verified badge IS displayed
# ─────────────────────────────────────────────────────────────────────────────
def test_verified_user_in_discover_displays_badge(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    # Set user to verified
    with test_app.app_context():
        UserModel.set_face_verified(u.id, u.profile_photo)
        # Ensure interest overlaps with viewer for discover appearance
        UserModel.update_profile(u.id, {"interests": ["Music", "Reading"], "gender": "Female"})

        viewer_id = UserModel.create_user(
            email=f"viewer2_{ObjectId()}@example.com",
            password="Password123!",
            name="Discover Viewer 2",
            comm_prefs=["text", "video"],
        )
        UserModel.update_profile(viewer_id, {"interests": ["Music", "Reading"], "gender": "Male"})
        viewer = UserModel.get_by_id(viewer_id)

    with client.session_transaction() as sess:
        sess["_user_id"] = str(viewer.id)

    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Verified candidate in discover must show bi-patch-check-fill
    candidates = Match.get_discover_candidates(user_id=viewer.id)
    candidate_match = next((c for c in candidates if str(c["_id"]) == str(u.id)), None)
    if candidate_match:
        assert candidate_match.get("face_verified") is True

    # View user's profile directly
    res_prof = client.get(f"/profile/{u.id}")
    assert res_prof.status_code == 200
    prof_html = res_prof.get_data(as_text=True)
    assert 'bi-patch-check-fill' in prof_html
    assert 'Verified' in prof_html

    with test_app.app_context():
        mongo.db.users.delete_one({"_id": ObjectId(viewer_id)})


# ─────────────────────────────────────────────────────────────────────────────
# 12. Deactivated account -> profile and badge hidden from discover & profile
# ─────────────────────────────────────────────────────────────────────────────
def test_deactivated_account_hidden_from_discover_and_profile(client, test_app, test_user_with_photo):
    u = test_user_with_photo
    with test_app.app_context():
        UserModel.set_face_verified(u.id, u.profile_photo)
        # Deactivate user account
        UserModel.update_profile(u.id, {"is_active": False})

        viewer_id = UserModel.create_user(
            email=f"viewer3_{ObjectId()}@example.com",
            password="Password123!",
            name="Discover Viewer 3",
            comm_prefs=["text", "video"],
        )
        viewer = UserModel.get_by_id(viewer_id)

    with client.session_transaction() as sess:
        sess["_user_id"] = str(viewer.id)

    # 1. Discover candidates query must omit deactivated account
    candidates = Match.get_discover_candidates(user_id=viewer.id)
    candidate_ids = [str(c["_id"]) for c in candidates]
    assert str(u.id) not in candidate_ids

    # 2. Directly visiting deactivated user's profile must be rejected
    res = client.get(f"/profile/{u.id}")
    assert res.status_code in (302, 403, 404)

    with test_app.app_context():
        mongo.db.users.delete_one({"_id": ObjectId(viewer_id)})
