"""Automated tests for Cloudinary profile photo storage and MongoDB Atlas indexes.

Tests:
1. Profile photo upload with Cloudinary mock.
2. Profile photo replacement and old image cleanup.
3. Failed Cloudinary upload error handling.
4. Missing Cloudinary configuration error handling (no silent local fallback).
5. Invalid images (format, oversized > 5MB, corrupt bytes).
6. Existing local profile photo backward compatibility.
7. Database update failure cleanup.
8. Index initialization and repeat execution idempotency.
9. Duplicate records detection before unique-index creation.
10. Existing auth, profile, discovery, matching, messaging, and video functionality.
"""
import os
import io
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image
from bson import ObjectId
from pymongo.errors import PyMongoError

from app import create_app
from extensions import mongo
from models.user import UserModel
from models.match import Match
from models.message import Message
from models.notification import Notification
from models.call_invitation import CallInvitation
from utils.cloudinary_storage import (
    validate_image_file,
    is_cloudinary_configured,
    upload_profile_photo,
    delete_profile_photo,
    extract_public_id_from_url,
    get_cloudinary_credentials,
)
from init_db_indexes import (
    initialize_indexes,
    check_duplicate_emails,
    safe_create_index,
)


def create_dummy_image(format="JPEG", size=(100, 100), color=(120, 150, 200)):
    """Generate a valid in-memory image for upload testing."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf


@pytest.fixture(scope="module")
def test_app():
    """Create Flask application configured for testing with MongoDB."""
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-secret-key-for-unit-tests"
    os.environ["CLOUDINARY_CLOUD_NAME"] = "test-cloud"
    os.environ["CLOUDINARY_API_KEY"] = "test-api-key"
    os.environ["CLOUDINARY_API_SECRET"] = "test-api-secret"

    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.app_context():
        yield app


@pytest.fixture
def client(test_app):
    return test_app.test_client()


@pytest.fixture
def clean_test_user(test_app):
    """Create and clean up a test user in MongoDB."""
    with test_app.app_context():
        email = f"test_{ObjectId()}@example.com"
        uid = UserModel.create_user(
            email=email,
            password="Password123!",
            name="Cloudinary Test User",
            comm_prefs=["text"],
        )
        user = UserModel.get_by_id(uid)
        yield user
        # Teardown
        mongo.db.users.delete_one({"_id": ObjectId(uid)})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Profile Photo Upload
# ─────────────────────────────────────────────────────────────────────────────
def test_profile_photo_upload_success(client, test_app, clean_test_user):
    """Test 1: Uploading a new profile photo via Cloudinary."""
    with client.session_transaction() as sess:
        sess["_user_id"] = clean_test_user.id
        sess["_fresh"] = True

    dummy_img = create_dummy_image("PNG")
    mock_upload_res = {
        "secure_url": "https://res.cloudinary.com/test-cloud/image/upload/v12345/inclusivematch/profile_photos/new_pic.png",
        "public_id": "inclusivematch/profile_photos/new_pic",
        "format": "png",
        "width": 100,
        "height": 100,
    }

    with patch("cloudinary.uploader.upload", return_value=mock_upload_res):
        resp = client.post(
            "/profile/edit",
            data={
                "first_name": "Cloud",
                "surname": "User",
                "bio": "Testing Cloudinary upload",
                "age": "26",
                "gender": "Non-binary",
                "relationship_goal": "Friendship",
                "profile_pic": (dummy_img, "avatar.png"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200

    # Verify user doc in MongoDB Atlas
    updated_user = UserModel.get_by_id(clean_test_user.id)
    assert updated_user.profile_photo == mock_upload_res["secure_url"]
    assert updated_user.profile_pic == mock_upload_res["secure_url"]
    assert updated_user.profile_photo_public_id == mock_upload_res["public_id"]
    assert updated_user.avatar_url == mock_upload_res["secure_url"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Profile Photo Replacement
# ─────────────────────────────────────────────────────────────────────────────
def test_profile_photo_replacement_and_old_deletion(client, test_app, clean_test_user):
    """Test 2: Replacing photo uploads new photo, updates DB, then deletes old image."""
    old_public_id = "inclusivematch/profile_photos/old_pic_123"
    old_url = f"https://res.cloudinary.com/test-cloud/image/upload/{old_public_id}.jpg"
    mongo.db.users.update_one(
        {"_id": ObjectId(clean_test_user.id)},
        {"$set": {
            "profile_photo": old_url,
            "profile_pic": old_url,
            "profile_photo_public_id": old_public_id,
        }},
    )

    with client.session_transaction() as sess:
        sess["_user_id"] = clean_test_user.id
        sess["_fresh"] = True

    new_dummy_img = create_dummy_image("JPEG")
    new_public_id = "inclusivematch/profile_photos/new_pic_456"
    new_url = f"https://res.cloudinary.com/test-cloud/image/upload/{new_public_id}.jpg"

    mock_upload_res = {
        "secure_url": new_url,
        "public_id": new_public_id,
        "format": "jpg",
    }

    with patch("cloudinary.uploader.upload", return_value=mock_upload_res) as mock_up, \
         patch("cloudinary.uploader.destroy", return_value={"result": "ok"}) as mock_destroy:

        resp = client.post(
            "/profile/edit",
            data={
                "first_name": "Cloud",
                "surname": "Updated",
                "profile_pic": (new_dummy_img, "new_photo.jpg"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200

        # Upload was called first
        mock_up.assert_called_once()
        # Destroy was called for OLD public id
        mock_destroy.assert_called_once_with(old_public_id, resource_type="image", invalidate=True)

    updated_user = UserModel.get_by_id(clean_test_user.id)
    assert updated_user.profile_photo == new_url
    assert updated_user.profile_photo_public_id == new_public_id


# ─────────────────────────────────────────────────────────────────────────────
# 3. Failed Cloudinary Upload
# ─────────────────────────────────────────────────────────────────────────────
def test_failed_cloudinary_upload_preserves_existing_data(client, test_app, clean_test_user):
    """Test 3: Cloudinary upload failure does not update DB or lose previous photo."""
    initial_photo = clean_test_user.profile_photo

    with client.session_transaction() as sess:
        sess["_user_id"] = clean_test_user.id
        sess["_fresh"] = True

    dummy_img = create_dummy_image("PNG")

    with patch("cloudinary.uploader.upload", side_effect=Exception("Connection timed out")):
        resp = client.post(
            "/profile/edit",
            data={
                "first_name": "Cloud",
                "surname": "User",
                "profile_pic": (dummy_img, "error_test.png"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Failed to upload profile photo to Cloudinary" in resp.data

    # Verify existing profile photo was untouched
    persisted_user = UserModel.get_by_id(clean_test_user.id)
    assert persisted_user.profile_photo == initial_photo


# ─────────────────────────────────────────────────────────────────────────────
# 4. Missing Cloudinary Configuration
# ─────────────────────────────────────────────────────────────────────────────
def test_missing_cloudinary_config_no_silent_fallback(client, test_app, clean_test_user):
    """Test 4: Missing Cloudinary config errors cleanly without silent fallback to local."""
    initial_photo = clean_test_user.profile_photo

    with client.session_transaction() as sess:
        sess["_user_id"] = clean_test_user.id
        sess["_fresh"] = True

    dummy_img = create_dummy_image("PNG")

    with patch.dict(os.environ, {"CLOUDINARY_CLOUD_NAME": "", "CLOUDINARY_API_KEY": ""}):
        assert not is_cloudinary_configured()
        resp = client.post(
            "/profile/edit",
            data={
                "first_name": "Cloud",
                "surname": "User",
                "profile_pic": (dummy_img, "no_config.png"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Cloudinary storage is not configured" in resp.data

    persisted_user = UserModel.get_by_id(clean_test_user.id)
    assert persisted_user.profile_photo == initial_photo


# ─────────────────────────────────────────────────────────────────────────────
# 5. Invalid Images and Oversized Uploads
# ─────────────────────────────────────────────────────────────────────────────
def test_invalid_image_extension_rejected(client, test_app, clean_test_user):
    """Test 5a: Reject disallowed file extensions (e.g. .txt, .pdf, .exe)."""
    with client.session_transaction() as sess:
        sess["_user_id"] = clean_test_user.id
        sess["_fresh"] = True

    bad_file = (io.BytesIO(b"Hello world"), "document.pdf")
    resp = client.post(
        "/profile/edit",
        data={"first_name": "Test", "surname": "User", "profile_pic": bad_file},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Invalid file extension" in resp.data


def test_corrupt_image_content_rejected(client, test_app, clean_test_user):
    """Test 5b: Reject corrupt image content with a spoofed .png extension."""
    with client.session_transaction() as sess:
        sess["_user_id"] = clean_test_user.id
        sess["_fresh"] = True

    corrupt_file = (io.BytesIO(b"Not a real image file content"), "fake.png")
    resp = client.post(
        "/profile/edit",
        data={"first_name": "Test", "surname": "User", "profile_pic": corrupt_file},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Invalid or corrupt image content" in resp.data


def test_oversized_image_rejected(client, test_app, clean_test_user):
    """Test 5c: Reject images exceeding the 5 MB limit."""
    class BigFileStorage:
        filename = "huge.jpg"
        def __init__(self):
            self.pos = 0
            self.size = 6 * 1024 * 1024  # 6 MB
        def seek(self, offset, whence=os.SEEK_SET):
            if whence == os.SEEK_END:
                self.pos = self.size
            else:
                self.pos = offset
        def tell(self):
            return self.pos

    valid, err = validate_image_file(BigFileStorage())
    assert not valid
    assert "exceeds the maximum allowed limit of 5 MB" in err


# ─────────────────────────────────────────────────────────────────────────────
# 6. Existing Local Profile Photo Compatibility
# ─────────────────────────────────────────────────────────────────────────────
def test_backward_compatibility_avatar_urls(test_app):
    """Test 6: Backward compatibility with local paths, default avatar, and Cloudinary URLs."""
    with test_app.app_context():
        # Case A: Legacy local filename
        local_user = UserModel({
            "_id": ObjectId(),
            "name": "Local User",
            "profile_photo": "my_saved_photo.jpg",
        })
        assert local_user.avatar_url == "/static/uploads/my_saved_photo.jpg"

        # Case B: Default avatar
        default_user = UserModel({
            "_id": ObjectId(),
            "name": "Default User",
            "profile_photo": "default.png",
        })
        assert default_user.avatar_url == "/static/uploads/default.png"

        # Case C: Empty/None
        empty_user = UserModel({
            "_id": ObjectId(),
            "name": "Empty User",
            "profile_photo": None,
        })
        assert empty_user.avatar_url == "/static/uploads/default.png"

        # Case D: Cloudinary HTTPS URL
        cloud_url = "https://res.cloudinary.com/test/image/upload/v1/sample.jpg"
        cloud_user = UserModel({
            "_id": ObjectId(),
            "name": "Cloud User",
            "profile_photo": cloud_url,
        })
        assert cloud_user.avatar_url == cloud_url

        # Case E: Jinja template filter resolution for raw dicts and strings
        from app import app
        avatar_filter = app.jinja_env.filters["avatar_url"]
        assert avatar_filter({"profile_pic": "local.png"}) == "/static/uploads/local.png"
        assert avatar_filter({"profile_photo": cloud_url}) == cloud_url
        assert avatar_filter(None) == "/static/uploads/default.png"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Database Update Failure Rollback / Cleanup
# ─────────────────────────────────────────────────────────────────────────────
def test_database_update_failure_cleans_up_new_image(client, test_app, clean_test_user):
    """Test 7: If DB update fails after upload, newly uploaded image is deleted from Cloudinary."""
    with client.session_transaction() as sess:
        sess["_user_id"] = clean_test_user.id
        sess["_fresh"] = True

    dummy_img = create_dummy_image("PNG")
    new_public_id = "inclusivematch/profile_photos/orphan_to_cleanup"
    mock_upload_res = {
        "secure_url": "https://res.cloudinary.com/test/image/upload/orphan.png",
        "public_id": new_public_id,
    }

    with patch("cloudinary.uploader.upload", return_value=mock_upload_res), \
         patch("models.user.UserModel.update_profile", side_effect=PyMongoError("DB connection lost")), \
         patch("cloudinary.uploader.destroy", return_value={"result": "ok"}) as mock_destroy:

        resp = client.post(
            "/profile/edit",
            data={"first_name": "Cloud", "surname": "Fail", "profile_pic": (dummy_img, "test.png")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Database update failed" in resp.data
        # Ensure cleanup occurred
        mock_destroy.assert_called_once_with(new_public_id, resource_type="image", invalidate=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Index Initialization and Repeat Execution
# ─────────────────────────────────────────────────────────────────────────────
def test_index_initialization_repeatability(test_app):
    """Test 8: init_db_indexes creates indexes safely and is fully idempotent when re-run."""
    with test_app.app_context():
        # First execution
        first_run = initialize_indexes(dry_run=False)
        assert first_run is True

        # Second execution (repeatability)
        second_run = initialize_indexes(dry_run=False)
        assert second_run is True

        # Verify key indexes exist
        user_indexes = mongo.db.users.index_information()
        assert "idx_users_email_unique" in user_indexes
        assert "idx_users_role_is_active" in user_indexes
        assert "idx_users_name" in user_indexes
        assert "idx_users_liked" in user_indexes

        msg_indexes = mongo.db.messages.index_information()
        assert "idx_messages_sender_receiver_time" in msg_indexes
        assert "idx_messages_receiver_is_read" in msg_indexes


# ─────────────────────────────────────────────────────────────────────────────
# 9. Duplicate Records Detection Before Unique-Index Creation
# ─────────────────────────────────────────────────────────────────────────────
def test_duplicate_records_detection_skips_unique_index(test_app):
    """Test 9: Duplicate email records are detected and reported without deleting records."""
    with test_app.app_context():
        test_col = mongo.db.test_dup_users
        test_col.drop()

        dup_email = "duplicate_person@test.com"
        test_col.insert_many([
            {"email": dup_email, "name": "First Person"},
            {"email": dup_email.upper(), "name": "Second Person (Same Email)"},
            {"email": "unique_person@test.com", "name": "Third Person"},
        ])

        dups = check_duplicate_emails(test_col)
        assert len(dups) == 1
        assert dups[0]["_id"] == dup_email.lower()
        assert dups[0]["count"] == 2

        # Verify documents were NOT deleted or modified
        assert test_col.count_documents({}) == 3
        test_col.drop()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Existing Platform Functionality Integration
# ─────────────────────────────────────────────────────────────────────────────
def test_existing_platform_functionality(client, test_app):
    """Test 10: Existing auth, profile viewing, discovery, matching, and messages work seamlessly."""
    with test_app.app_context():
        # Create two test users
        uid_a = UserModel.create_user(
            email=f"alice_{ObjectId()}@test.com",
            password="AlicePassword123!",
            name="Alice Wonder",
            comm_prefs=["text"],
        )
        uid_b = UserModel.create_user(
            email=f"bob_{ObjectId()}@test.com",
            password="BobPassword123!",
            name="Bob Builder",
            comm_prefs=["text"],
        )

        user_a = UserModel.get_by_id(uid_a)
        user_b = UserModel.get_by_id(uid_b)

        # Profile viewing
        with client.session_transaction() as sess:
            sess["_user_id"] = uid_a
            sess["_fresh"] = True

        resp = client.get(f"/profile/{uid_b}")
        assert resp.status_code == 200
        assert b"Bob Builder" in resp.data

        # Mutual match flow
        Match.like(uid_a, uid_b)
        Match.like(uid_b, uid_a)
        assert Match.are_matched(uid_a, uid_b) is True

        # Discovery candidates
        candidates = Match.get_discover_candidates(uid_a)
        # Should not include mutual match Bob
        assert uid_b not in [str(c["_id"]) for c in candidates]

        # Messaging
        msg_id = Message.create(uid_a, uid_b, "Hello from Alice!")
        conv = Message.get_conversation(uid_a, uid_b)
        assert len(conv) >= 1
        assert conv[-1]["text"] == "Hello from Alice!"

        # Call invitation
        call_id = CallInvitation.create(uid_a, uid_b)
        pending = CallInvitation.pending_for(uid_b)
        assert pending is not None
        assert pending["caller_id"] == uid_a

        # Cleanup
        mongo.db.users.delete_many({"_id": {"$in": [ObjectId(uid_a), ObjectId(uid_b)]}})
        mongo.db.messages.delete_many({"_id": ObjectId(msg_id)})
        mongo.db.call_invitations.delete_many({"_id": ObjectId(call_id)})


# ─────────────────────────────────────────────────────────────────────────────
# 11. Defensive Environment Variable Handling & URL Public ID Extraction
# ─────────────────────────────────────────────────────────────────────────────
def test_defensive_credentials_parsing_with_quotes_and_spaces():
    """Verify that quotes and accidental whitespace around credentials are trimmed cleanly."""
    with patch.dict(os.environ, {
        "CLOUDINARY_CLOUD_NAME": '  "render-cloud"  ',
        "CLOUDINARY_API_KEY": " '123456789' ",
        "CLOUDINARY_API_SECRET": ' "secret-key-abc" ',
    }):
        assert is_cloudinary_configured() is True
        creds = get_cloudinary_credentials()
        assert creds["cloud_name"] == "render-cloud"
        assert creds["api_key"] == "123456789"
        assert creds["api_secret"] == "secret-key-abc"


def test_cloudinary_url_fallback_support():
    """Verify that CLOUDINARY_URL connection string is supported when individual vars are unset."""
    with patch.dict(os.environ, {
        "CLOUDINARY_CLOUD_NAME": "",
        "CLOUDINARY_API_KEY": "",
        "CLOUDINARY_API_SECRET": "",
        "CLOUDINARY_URL": "cloudinary://12345:abcdef@my-render-cloud",
    }):
        assert is_cloudinary_configured() is True
        creds = get_cloudinary_credentials()
        assert creds["cloudinary_url"] == "cloudinary://12345:abcdef@my-render-cloud"


def test_extract_public_id_from_url():
    """Verify public_id extraction across various Cloudinary URL patterns."""
    url1 = "https://res.cloudinary.com/demo/image/upload/v1570979139/inclusivematch/profile_photos/user_123.jpg"
    assert extract_public_id_from_url(url1) == "inclusivematch/profile_photos/user_123"

    url2 = "https://res.cloudinary.com/demo/image/upload/sample.png"
    assert extract_public_id_from_url(url2) == "sample"

    url3 = "https://res.cloudinary.com/demo/image/upload/s--abc123--/v1/inclusivematch/profile_photos/secure_img.webp"
    assert extract_public_id_from_url(url3) == "inclusivematch/profile_photos/secure_img"

    # Non-cloudinary or invalid URLs
    assert extract_public_id_from_url("/static/uploads/default.png") is None
    assert extract_public_id_from_url(None) is None
    assert extract_public_id_from_url("") is None


def test_delete_profile_photo_accepts_full_url():
    """Verify delete_profile_photo extracts public_id when given a full Cloudinary URL."""
    cloud_url = "https://res.cloudinary.com/test-cloud/image/upload/v1234/inclusivematch/profile_photos/del_test.jpg"
    with patch("cloudinary.uploader.destroy", return_value={"result": "ok"}) as mock_destroy, \
         patch("utils.cloudinary_storage.is_cloudinary_configured", return_value=True), \
         patch("utils.cloudinary_storage.configure_cloudinary"):
        success = delete_profile_photo(cloud_url)
        assert success is True
        mock_destroy.assert_called_once_with(
            "inclusivematch/profile_photos/del_test",
            resource_type="image",
            invalidate=True,
        )


def test_profile_photo_replacement_when_public_id_was_missing_in_db(client, test_app, clean_test_user):
    """Verify replacement correctly deletes old Cloudinary photo even if public_id was not explicitly in MongoDB."""
    old_url = "https://res.cloudinary.com/test-cloud/image/upload/v111/inclusivematch/profile_photos/legacy_photo.jpg"
    mongo.db.users.update_one(
        {"_id": ObjectId(clean_test_user.id)},
        {"$set": {
            "profile_photo": old_url,
            "profile_pic": old_url,
            # Notice: profile_photo_public_id is intentionally omitted
        },
        "$unset": {"profile_photo_public_id": ""}}
    )

    with client.session_transaction() as sess:
        sess["_user_id"] = clean_test_user.id
        sess["_fresh"] = True

    new_dummy_img = create_dummy_image("PNG")
    new_public_id = "inclusivematch/profile_photos/new_pic_789"
    new_url = f"https://res.cloudinary.com/test-cloud/image/upload/{new_public_id}.png"

    mock_upload_res = {
        "secure_url": new_url,
        "public_id": new_public_id,
        "format": "png",
    }

    with patch("cloudinary.uploader.upload", return_value=mock_upload_res), \
         patch("cloudinary.uploader.destroy", return_value={"result": "ok"}) as mock_destroy:

        resp = client.post(
            "/profile/edit",
            data={
                "first_name": "Cloud",
                "surname": "LegacyTest",
                "profile_pic": (new_dummy_img, "new.png"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert resp.status_code == 200

        # Old image was extracted from URL and destroyed!
        mock_destroy.assert_called_once_with(
            "inclusivematch/profile_photos/legacy_photo",
            resource_type="image",
            invalidate=True,
        )

    updated_user = UserModel.get_by_id(clean_test_user.id)
    assert updated_user.profile_photo == new_url
    assert updated_user.profile_photo_public_id == new_public_id

