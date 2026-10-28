"""Automated test suite for Forgot Password & Password Reset in InclusiveMatch AI.

Covers security controls, Brevo email dispatch, token hashing, single-use enforcement,
rate limiting, CSRF protection, and session revocation.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from bson import ObjectId
from app import create_app
from extensions import mongo
from models.user import UserModel
from models.password_reset import PasswordReset
from utils.email_service import send_password_reset_email


@pytest.fixture
def app():
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app_context():
        PasswordReset.ensure_indexes()
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def clean_test_data(client):
    """Clean test users and reset records before and after each test."""
    client.get("/logout")
    test_emails = [
        "reset_user@example.com",
        "another_user@example.com",
        "reg_test@example.com"
    ]
    mongo.db.users.delete_many({"email": {"$in": test_emails}})
    mongo.db.password_resets.delete_many({"email": {"$in": test_emails}})
    yield
    client.get("/logout")
    mongo.db.users.delete_many({"email": {"$in": test_emails}})
    mongo.db.password_resets.delete_many({"email": {"$in": test_emails}})


# ─────────────────────────────────────────────────────────────────────────────
# 1. UI LINK & PAGE LOADING TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_forgot_password_link_appears_on_login(client):
    """Forgot Password link must be clearly visible on the Login page."""
    res = client.get("/login")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Forgot Password?" in html
    assert "/forgot-password" in html


def test_forgot_password_link_does_not_appear_on_registration(client):
    """Forgot Password link must NOT appear on the Registration page."""
    res = client.get("/register")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Forgot Password?" not in html


def test_forgot_password_page_loads(client):
    """Forgot Password page renders properly with instructions and form."""
    res = client.get("/forgot-password")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Forgot Your Password?" in html
    assert "Send Reset Link" in html
    assert "Back to Login" in html
    assert 'type="email"' in html


def test_invalid_email_input_handled(client):
    """Submitting empty or malformed email returns validation error."""
    res = client.post("/forgot-password", data={"email": "not-an-email"}, follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Please enter a valid email address." in html

    res_empty = client.post("/forgot-password", data={"email": ""}, follow_redirects=True)
    assert res_empty.status_code == 200
    html_empty = res_empty.data.decode("utf-8")
    assert "Please enter a valid email address." in html_empty


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMAIL DISPATCH & ANTI-ENUMERATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

@patch("routes.auth_routes.send_password_reset_email")
def test_registered_email_receives_reset_email_mocked(mock_send, client):
    """Registered email triggers a secure reset email and generic response."""
    mock_send.return_value = True

    # Create registered user
    user_id = UserModel.create_user("reset_user@example.com", "OldPassword123!", "Reset User")

    res = client.post(
        "/forgot-password",
        data={"email": "reset_user@example.com"},
        follow_redirects=True
    )
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Generic response must be displayed
    assert "If an account exists for that email address, you will receive instructions to reset your password." in html

    # Verify email was dispatched with recipient and absolute URL
    assert mock_send.called
    call_args = mock_send.call_args[0]
    assert call_args[0] == "reset_user@example.com"
    reset_url = call_args[1]
    assert "/reset-password/" in reset_url

    # Verify token record exists in MongoDB and stores a SHA-256 hash (never plaintext)
    token_doc = mongo.db.password_resets.find_one({"email": "reset_user@example.com"})
    assert token_doc is not None
    assert token_doc["user_id"] == str(user_id)
    assert token_doc["used"] is False
    assert len(token_doc["token_hash"]) == 64  # Valid SHA-256 length


@patch("routes.auth_routes.send_password_reset_email")
def test_unregistered_email_receives_same_generic_response(mock_send, client):
    """Unregistered email receives identical generic response; no email is sent."""
    res = client.post(
        "/forgot-password",
        data={"email": "nonexistent@example.com"},
        follow_redirects=True
    )
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert "If an account exists for that email address, you will receive instructions to reset your password." in html
    assert not mock_send.called


@patch("routes.auth_routes.send_password_reset_email")
def test_email_sending_failure_handled_safely(mock_send, client):
    """SMTP/API errors during sending must not crash the application or expose errors."""
    mock_send.return_value = False

    UserModel.create_user("reset_user@example.com", "OldPassword123!", "Reset User")

    res = client.post(
        "/forgot-password",
        data={"email": "reset_user@example.com"},
        follow_redirects=True
    )
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "If an account exists for that email address, you will receive instructions to reset your password." in html


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOKEN VALIDATION, EXPIRATION, AND SINGLE-USE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_reset_token_accepted(client):
    """A valid, unexpired token renders the Create New Password form."""
    user_id = UserModel.create_user("reset_user@example.com", "OldPassword123!", "Reset User")
    raw_token, _ = PasswordReset.create_token(user_id, "reset_user@example.com")

    res = client.get(f"/reset-password/{raw_token}")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Create New Password" in html
    assert 'name="password"' in html
    assert 'name="confirm_password"' in html


def test_invalid_token_rejected(client):
    """An arbitrary or tampered token renders the invalid link notice."""
    res = client.get("/reset-password/totally-invalid-token-12345")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Invalid or Expired Link" in html
    assert "Request a New Link" in html


def test_expired_token_rejected(client):
    """A token whose expiration timestamp has passed is rejected on GET and POST."""
    user_id = UserModel.create_user("reset_user@example.com", "OldPassword123!", "Reset User")
    raw_token, _ = PasswordReset.create_token(user_id, "reset_user@example.com")

    # Manually backdate the expiration to simulate expiry
    token_hash = PasswordReset.hash_token(raw_token)
    mongo.db.password_resets.update_one(
        {"token_hash": token_hash},
        {"$set": {"expires_at": datetime.utcnow() - timedelta(minutes=5)}}
    )

    # GET must show expired state
    res_get = client.get(f"/reset-password/{raw_token}")
    assert res_get.status_code == 200
    html_get = res_get.data.decode("utf-8")
    assert "Invalid or Expired Link" in html_get

    # POST must reject reset attempt
    res_post = client.post(
        f"/reset-password/{raw_token}",
        data={"password": "NewSecretPass123!", "confirm_password": "NewSecretPass123!"},
        follow_redirects=True
    )
    assert res_post.status_code == 200
    html_post = res_post.data.decode("utf-8")
    assert "expired" in html_post.lower() or "invalid" in html_post.lower()


def test_reused_token_rejected(client):
    """A token that has already been consumed cannot be used a second time."""
    user_id = UserModel.create_user("reset_user@example.com", "OldPassword123!", "Reset User")
    raw_token, _ = PasswordReset.create_token(user_id, "reset_user@example.com")

    # First reset succeeds
    res1 = client.post(
        f"/reset-password/{raw_token}",
        data={"password": "FirstNewPassword123!", "confirm_password": "FirstNewPassword123!"},
        follow_redirects=True
    )
    assert res1.status_code == 200
    assert "Your password has been successfully reset!" in res1.data.decode("utf-8")

    # Second reset attempt with same token must be rejected
    res2 = client.post(
        f"/reset-password/{raw_token}",
        data={"password": "SecondNewPassword123!", "confirm_password": "SecondNewPassword123!"},
        follow_redirects=True
    )
    assert res2.status_code == 200
    html2 = res2.data.decode("utf-8")
    assert "already been used" in html2.lower() or "invalid" in html2.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PASSWORD POLICY, VALIDATION, AND AUTHENTICATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_password_and_confirmation_mismatch_rejected(client):
    """Mismatch between password and confirm_password is rejected."""
    user_id = UserModel.create_user("reset_user@example.com", "OldPassword123!", "Reset User")
    raw_token, _ = PasswordReset.create_token(user_id, "reset_user@example.com")

    res = client.post(
        f"/reset-password/{raw_token}",
        data={"password": "Password12345!", "confirm_password": "DifferentPassword12345!"},
        follow_redirects=True
    )
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Passwords do not match!" in html


def test_password_policy_enforced(client):
    """Password shorter than minimum length of 8 characters is rejected."""
    user_id = UserModel.create_user("reset_user@example.com", "OldPassword123!", "Reset User")
    raw_token, _ = PasswordReset.create_token(user_id, "reset_user@example.com")

    res = client.post(
        f"/reset-password/{raw_token}",
        data={"password": "short", "confirm_password": "short"},
        follow_redirects=True
    )
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Password must be at least 8 characters long." in html


def test_successful_reset_authenticates_with_new_password_and_revokes_old(client):
    """Full lifecycle: Old password ceases working; new password authenticates successfully."""
    user_id = UserModel.create_user("reset_user@example.com", "InitialPassword123!", "Reset User")
    raw_token, _ = PasswordReset.create_token(user_id, "reset_user@example.com")

    # Complete reset
    res = client.post(
        f"/reset-password/{raw_token}",
        data={"password": "BrandNewSecurePassword456!", "confirm_password": "BrandNewSecurePassword456!"},
        follow_redirects=True
    )
    assert res.status_code == 200
    assert "Your password has been successfully reset!" in res.data.decode("utf-8")

    # Attempt login with OLD password -> MUST FAIL
    res_old = client.post(
        "/login",
        data={"email": "reset_user@example.com", "password": "InitialPassword123!"},
        follow_redirects=True
    )
    assert "Invalid email/password" in res_old.data.decode("utf-8")

    # Attempt login with NEW password -> MUST SUCCEED
    res_new = client.post(
        "/login",
        data={"email": "reset_user@example.com", "password": "BrandNewSecurePassword456!"},
        follow_redirects=True
    )
    assert "Welcome back" in res_new.data.decode("utf-8")


def test_existing_sessions_revoked_on_reset(app, client):
    """Active session is invalidated when password reset is executed."""
    user_id = UserModel.create_user("reset_user@example.com", "InitialPassword123!", "Reset User")
    raw_token, _ = PasswordReset.create_token(user_id, "reset_user@example.com")

    # Log user in
    client.post("/login", data={"email": "reset_user@example.com", "password": "InitialPassword123!"})

    # Reset password
    client.post(
        f"/reset-password/{raw_token}",
        data={"password": "BrandNewPassword789!", "confirm_password": "BrandNewPassword789!"},
        follow_redirects=True
    )

    # Verify user is not automatically logged in
    with client.session_transaction() as sess:
        assert "_user_id" not in sess


# ─────────────────────────────────────────────────────────────────────────────
# 5. SECURITY CONTROLS: CSRF, RATE LIMITING & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def test_csrf_protection_works(app, client):
    """When CSRF protection is enabled, requests with missing or invalid token are blocked."""
    app.config["WTF_CSRF_ENABLED"] = True

    # Missing token on forgot-password
    res = client.post("/forgot-password", data={"email": "test@example.com"}, follow_redirects=True)
    assert "security token" in res.data.decode("utf-8").lower()


@patch("routes.auth_routes.send_password_reset_email")
def test_rate_limiting_cooldown(mock_send, client):
    """Multiple rapid reset requests for the same email are throttled to prevent spam."""
    mock_send.return_value = True
    UserModel.create_user("reset_user@example.com", "Pass12345!", "Reset User")

    # First request dispatches email
    client.post("/forgot-password", data={"email": "reset_user@example.com"})
    assert mock_send.call_count == 1

    # Second request immediately after is throttled (email not dispatched again)
    client.post("/forgot-password", data={"email": "reset_user@example.com"})
    assert mock_send.call_count == 1  # Throttled, count remains 1


def test_missing_brevo_config_handled_safely():
    """send_password_reset_email returns False cleanly when credentials are unconfigured."""
    with patch.dict("os.environ", {"BREVO_SMTP_LOGIN": "", "BREVO_SMTP_KEY": ""}, clear=True):
        result = send_password_reset_email("someone@example.com", "https://example.com/reset/123")
        assert result is False


def test_existing_registration_and_login_still_works(client):
    """Ensure normal registration and login flow is completely intact."""
    res_reg = client.post(
        "/register",
        data={
            "name": "Regular User",
            "email": "reg_test@example.com",
            "password": "ValidPassword123!",
            "confirm_password": "ValidPassword123!",
            "comm_pref": "text",
            "terms_consent": "on"
        },
        follow_redirects=True
    )
    assert res_reg.status_code == 200
    assert "Account created! Please log in." in res_reg.data.decode("utf-8")

    res_login = client.post(
        "/login",
        data={"email": "reg_test@example.com", "password": "ValidPassword123!"},
        follow_redirects=True
    )
    assert res_login.status_code == 200
    assert "Welcome back" in res_login.data.decode("utf-8")
