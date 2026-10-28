"""Automated Test Suite for Privacy, Legal Documentation, and User Protection.

Covers:
1. Legal & Policy Pages (all 9 documents return HTTP 200 OK with proper headers and content).
2. Maintenance Mode Exemption (legal routes remain accessible during maintenance).
3. Registration Consent Lifecycle (missing consent blocked, valid consent recorded with version and timestamp in MongoDB).
4. Authenticated Account Deletion (unauthorized access blocked, incorrect password rejected, correct password executes full cleanup).
5. Cloudinary Error Resilience (account deletion succeeds even if Cloudinary API fails).
6. Contact & Grievance Redressal (validation, persistence, unique tracking ID generation).
7. Data Portability / Export (DPDP Act Section 11 machine-readable export).
8. Navigation & UI Integration (footer links, profile settings privacy card).
"""

import os
import json
import pytest
from datetime import datetime
from bson import ObjectId
from unittest.mock import patch

from app import create_app
from extensions import mongo
from models.user import UserModel


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-privacy-legal-key-2026"
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app_context():
        yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def clean_db(client):
    """Clean test collections and session state before and after each test."""
    client.get("/logout")
    test_emails = [
        "consent_test@example.com",
        "delete_test@example.com",
        "other_user@example.com",
        "export_test@example.com",
    ]
    mongo.db.users.delete_many({"email": {"$in": test_emails}})
    mongo.db.grievances.delete_many({"email": {"$in": test_emails + ["complaint@example.com", "valid@example.com"]}})
    mongo.db.app_settings.delete_one({"_id": "app_settings"})
    yield
    client.get("/logout")
    mongo.db.users.delete_many({"email": {"$in": test_emails}})
    mongo.db.grievances.delete_many({"email": {"$in": test_emails + ["complaint@example.com", "valid@example.com"]}})
    mongo.db.app_settings.delete_one({"_id": "app_settings"})


def _login(client, email, password):
    """Helper to cleanly log in via Flask-Login session."""
    client.get("/logout")
    return client.post("/login", data={"email": email, "password": password}, follow_redirects=True)


# ===========================================================================
# 1. LEGAL AND POLICY PAGES HTTP 200 TESTS
# ===========================================================================

LEGAL_ROUTES = [
    ("/privacy", "Privacy Policy", "Digital Personal Data Protection"),
    ("/terms", "Terms and Conditions", "Draft Terms for Legal Review"),
    ("/community-guidelines", "Community Guidelines", "Five Pillars"),
    ("/safety", "User Safety", "Safety & Protection"),
    ("/cookie-policy", "Cookie & Local Storage Policy", "Zero Third-Party Advertising"),
    ("/ai-transparency", "AI Transparency", "Algorithmic Decision Notice"),
    ("/accessibility", "Accessibility Statement", "Assistive Standards"),
    ("/data-rights", "Account Deletion, Retention & Data Subject Rights", "Data Subject Rights"),
    ("/contact-grievance", "Contact & Grievance Redressal Portal", "Grievance Officer"),
]


@pytest.mark.parametrize("route,expected_title,expected_keyword", LEGAL_ROUTES)
def test_legal_page_returns_200_ok(client, route, expected_title, expected_keyword):
    """Verify that every legal and policy document loads cleanly with expected titles."""
    response = client.get(route)
    assert response.status_code == 200, f"Route {route} failed with status {response.status_code}"
    html = response.data.decode("utf-8")
    assert expected_title in html, f"Expected title '{expected_title}' not found in {route}"
    assert expected_keyword in html, f"Expected keyword '{expected_keyword}' not found in {route}"


def test_footer_contains_all_legal_navigation_links(client):
    """Verify that the base footer renders navigation links to key policy pages."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "/privacy" in html
    assert "/terms" in html
    assert "/community-guidelines" in html
    assert "/safety" in html
    assert "/cookie-policy" in html
    assert "/ai-transparency" in html
    assert "/accessibility" in html
    assert "/data-rights" in html
    assert "/contact-grievance" in html


# ===========================================================================
# 2. MAINTENANCE MODE EXEMPTION TESTS
# ===========================================================================

def test_legal_pages_accessible_during_maintenance_mode(app, client):
    """Ensure legal and statutory policy pages remain accessible when maintenance mode is active."""
    mongo.db.app_settings.update_one(
        {"_id": "app_settings"},
        {"$set": {"maintenance_mode": True}},
        upsert=True
    )
    app.config["TESTING"] = False
    try:
        # Non-whitelisted route must be redirected to /maintenance
        res_blocked = client.get("/this-route-does-not-exist")
        assert res_blocked.status_code == 302
        assert "/maintenance" in res_blocked.headers.get("Location", "")

        # Legal pages MUST remain accessible (HTTP 200)
        for route, _, _ in LEGAL_ROUTES:
            res = client.get(route)
            assert res.status_code == 200, f"Legal route {route} was blocked during maintenance mode!"
    finally:
        app.config["TESTING"] = True
        mongo.db.app_settings.delete_one({"_id": "app_settings"})


# ===========================================================================
# 3. REGISTRATION CONSENT LIFECYCLE (PHASE 8)
# ===========================================================================

def test_registration_without_policy_consent_fails(client):
    """Registration must fail if the user does not check the required policy consent box."""
    payload = {
        "name": "Consent Test User",
        "email": "consent_test@example.com",
        "password": "Password123!",
        "confirm_password": "Password123!",
        "comm_pref": "text",
        # terms_consent omitted
    }
    response = client.post("/register", data=payload, follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "You must read and agree to the Terms &amp; Conditions and acknowledge the Privacy Policy" in html or \
           "You must read and agree to the Terms & Conditions and acknowledge the Privacy Policy" in html

    # Verify user was NOT created
    user = UserModel.find_by_email("consent_test@example.com")
    assert user is None


def test_registration_with_policy_consent_records_version_and_timestamp(client):
    """Registration with consent must create the user and persist policy version and timestamp."""
    payload = {
        "name": "Consent Test User",
        "email": "consent_test@example.com",
        "password": "Password123!",
        "confirm_password": "Password123!",
        "comm_pref": "text",
        "terms_consent": "on",
    }
    response = client.post("/register", data=payload, follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Account created! Please log in." in html

    # Verify user was created in MongoDB with policy_consent metadata
    doc = mongo.db.users.find_one({"email": "consent_test@example.com"})
    assert doc is not None
    assert "policy_consent" in doc
    consent = doc["policy_consent"]
    assert consent.get("terms_version") == "2026.1"
    assert consent.get("privacy_version") == "2026.1"
    assert isinstance(consent.get("accepted_at"), datetime)
    assert "ip_address" in consent


# ===========================================================================
# 4. ACCOUNT DELETION & DATA CLEANUP (PHASE 9)
# ===========================================================================

def test_unauthenticated_account_deletion_blocked(client):
    """Unauthenticated POST to /profile/delete-account must be rejected."""
    response = client.post("/profile/delete-account", data={"password": "any"})
    assert response.status_code == 302
    assert "/login" in response.headers.get("Location", "")


def test_account_deletion_with_incorrect_password_rejected(client):
    """Authenticated account deletion with wrong password must be rejected without data loss."""
    uid = UserModel.create_user(
        "delete_test@example.com",
        "CorrectPassword123!",
        "Delete User",
        comm_prefs=["text"]
    )
    _login(client, "delete_test@example.com", "CorrectPassword123!")

    # Submit wrong password
    res = client.post("/profile/delete-account", data={"password": "WrongPassword!"}, follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Incorrect password. Account deletion cancelled." in html

    # User must still exist
    assert UserModel.get_by_id(uid) is not None


def test_account_deletion_with_correct_password_executes_full_cleanup(client):
    """Authenticated account deletion with correct password purges user, messages, and photo."""
    uid1 = UserModel.create_user("delete_test@example.com", "SecretPass123!", "Delete User")
    uid2 = UserModel.create_user("other_user@example.com", "OtherPass123!", "Other User")

    mongo.db.users.update_one(
        {"_id": ObjectId(uid1)},
        {"$set": {"profile_photo_public_id": "inclusivematch/profile_photos/test_del_photo"}}
    )
    mongo.db.users.update_one({"_id": ObjectId(uid2)}, {"$push": {"liked": str(uid1)}})
    mongo.db.messages.insert_one({"sender_id": str(uid1), "receiver_id": str(uid2), "text": "Goodbye"})
    mongo.db.messages.insert_one({"sender_id": str(uid2), "receiver_id": str(uid1), "text": "Farewell"})

    _login(client, "delete_test@example.com", "SecretPass123!")

    with patch("utils.cloudinary_storage.delete_profile_photo") as mock_del_photo:
        mock_del_photo.return_value = True

        res = client.post("/profile/delete-account", data={"password": "SecretPass123!"}, follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert "permanently deleted in accordance with your privacy rights" in html
        mock_del_photo.assert_called_once_with("inclusivematch/profile_photos/test_del_photo")

    # 1. User 1 document must be completely deleted
    assert mongo.db.users.find_one({"_id": ObjectId(uid1)}) is None

    # 2. User 1 must be unlinked from User 2's liked list
    u2_doc = mongo.db.users.find_one({"_id": ObjectId(uid2)})
    assert str(uid1) not in u2_doc.get("liked", [])

    # 3. Messages between them must be deleted
    msg_count = mongo.db.messages.count_documents({"$or": [{"sender_id": str(uid1)}, {"receiver_id": str(uid1)}]})
    assert msg_count == 0


def test_account_deletion_resilient_to_cloudinary_errors(client):
    """Account deletion must succeed in MongoDB even if Cloudinary network call throws."""
    uid = UserModel.create_user("delete_test@example.com", "SecretPass123!", "Delete User")
    mongo.db.users.update_one(
        {"_id": ObjectId(uid)},
        {"$set": {"profile_photo_public_id": "inclusivematch/profile_photos/cloud_err"}}
    )

    _login(client, "delete_test@example.com", "SecretPass123!")

    with patch("utils.cloudinary_storage.delete_profile_photo", side_effect=RuntimeError("Cloudinary timeout")):
        res = client.post("/profile/delete-account", data={"password": "SecretPass123!"}, follow_redirects=True)
        assert res.status_code == 200

    # User must still be deleted from MongoDB despite the Cloudinary error
    assert mongo.db.users.find_one({"_id": ObjectId(uid)}) is None


# ===========================================================================
# 5. CONTACT & GRIEVANCE REDRESSAL (PHASE 10)
# ===========================================================================

def test_grievance_submission_validation_rejects_empty_inputs(client):
    """Grievance submission must validate required fields."""
    res = client.post("/contact-grievance/submit", data={
        "name": "",
        "email": "not-an-email",
        "category": "invalid_category",
        "subject": "",
        "message": "short",
    }, follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Please provide a valid full name" in html

    # Assert no grievance inserted
    assert mongo.db.grievances.count_documents({}) == 0


def test_grievance_submission_persists_with_tracking_id(client):
    """Valid grievance submission must be stored in MongoDB with a unique tracking code."""
    payload = {
        "name": "Jane Doe",
        "email": "valid@example.com",
        "category": "privacy_complaint",
        "subject": "Inquiry regarding data processing",
        "message": "This is a detailed inquiry regarding how my communication preferences are shared with prospective matches.",
    }
    res = client.post("/contact-grievance/submit", data=payload, follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Your request has been registered under Tracking ID:" in html

    # Check MongoDB record
    doc = mongo.db.grievances.find_one({"email": "valid@example.com"})
    assert doc is not None
    assert doc["category"] == "privacy_complaint"
    assert doc["subject"] == "Inquiry regarding data processing"
    assert doc["status"] == "pending"
    assert str(doc["tracking_id"]).startswith("GRV-")
    assert isinstance(doc["created_at"], datetime)


# ===========================================================================
# 6. DATA PORTABILITY / EXPORT (DPDP ACT SECTION 11)
# ===========================================================================

def test_data_export_unauthenticated_blocked(client):
    """Unauthenticated users must not be able to access data export."""
    res = client.get("/data-rights/export-my-data")
    assert res.status_code == 302
    assert "/login" in res.headers.get("Location", "")


def test_data_export_authenticated_returns_valid_json_payload(client):
    """Authenticated user must receive a machine-readable JSON export of their personal data."""
    uid = UserModel.create_user(
        "export_test@example.com",
        "ExportPass123!",
        "Export User",
        comm_prefs=["sign language"],
    )
    _login(client, "export_test@example.com", "ExportPass123!")

    res = client.get("/data-rights/export-my-data")
    assert res.status_code == 200
    assert res.content_type == "application/json"
    assert "attachment; filename=inclusivematch_data_export_" in res.headers.get("Content-Disposition", "")

    payload = json.loads(res.data.decode("utf-8"))
    assert payload["platform"] == "InclusiveMatch AI"
    assert "Digital Personal Data Protection Act, 2023" in payload["statutory_basis"]
    assert payload["account_profile"]["email"] == "export_test@example.com"
    # Password hash must NEVER be included in the export
    assert "password" not in payload["account_profile"]


# ===========================================================================
# 7. PROFILE UI INTEGRATION (PHASE 11)
# ===========================================================================

def test_profile_page_renders_legal_management_section(client):
    """Authenticated user viewing their own profile must see the Privacy & Data Rights section."""
    uid = UserModel.create_user("export_test@example.com", "Pass123!", "Profile User")
    _login(client, "export_test@example.com", "Pass123!")

    res = client.get("/profile")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Privacy, Legal &amp; Data Rights" in html or "Privacy, Legal & Data Rights" in html
    assert "Export My Data" in html
    assert "Delete My Account" in html
    assert "deleteAccountModal" in html


# ===========================================================================
# 8. ADMIN GRIEVANCE MANAGEMENT & USER TRACKING TESTS
# ===========================================================================

def test_admin_grievances_unauthorized_blocked(client):
    """Unauthenticated or regular user cannot access admin grievance control center."""
    # 1. Unauthenticated
    res = client.get("/admin/grievances")
    assert res.status_code in [302, 403]

    # 2. Regular user (non-admin)
    UserModel.create_user("regular_user@example.com", "Password123!", "Regular User", role="user")
    _login(client, "regular_user@example.com", "Password123!")
    res2 = client.get("/admin/grievances")
    assert res2.status_code in [302, 403]
    mongo.db.users.delete_one({"email": "regular_user@example.com"})


def test_admin_grievance_management_and_response_lifecycle(client):
    """Authorized administrator can review grievances, change status, and send response notes."""
    # Create admin user
    admin_id = UserModel.create_user("admin_test@example.com", "AdminPass123!", "Admin User", role="admin")

    # Insert test grievance
    payload = {
        "name": "Jane Grievant",
        "email": "complaint@example.com",
        "category": "privacy_complaint",
        "subject": "Data processing inquiry",
        "message": "Please explain how my profile data is processed and stored.",
    }
    client.post("/contact-grievance/submit", data=payload)
    doc = mongo.db.grievances.find_one({"email": "complaint@example.com"})
    assert doc is not None
    tracking_id = doc["tracking_id"]

    # Log in as admin
    _login(client, "admin_test@example.com", "AdminPass123!")

    # 1. Admin views grievance list
    res_list = client.get("/admin/grievances")
    assert res_list.status_code == 200
    assert tracking_id in res_list.data.decode("utf-8")
    assert "Jane Grievant" in res_list.data.decode("utf-8")

    # 2. Admin updates status to resolved with response notes
    res_update = client.post(
        f"/admin/grievances/{str(doc['_id'])}/status",
        data={
            "status": "resolved",
            "response_notes": "We have completed your inquiry. Your profile data is stored with strict encryption."
        },
        follow_redirects=True
    )
    assert res_update.status_code == 200
    updated_doc = mongo.db.grievances.find_one({"_id": doc["_id"]})
    assert updated_doc["status"] == "resolved"
    assert "strict encryption" in updated_doc["resolution_notes"]
    assert isinstance(updated_doc["resolved_at"], datetime)

    # 3. User can track resolution via /contact-grievance/track
    client.get("/logout")
    res_track = client.post("/contact-grievance/track", data={
        "tracking_id": tracking_id,
        "email": "complaint@example.com"
    })
    assert res_track.status_code == 200
    track_html = res_track.data.decode("utf-8")
    assert "RESOLVED" in track_html
    assert "We have completed your inquiry" in track_html

    # Cleanup
    mongo.db.users.delete_one({"_id": ObjectId(admin_id)})
    mongo.db.grievances.delete_one({"_id": doc["_id"]})


def test_legal_route_aliases_redirect_correctly(client):
    """Verify route aliases redirect to canonical legal endpoints."""
    res_contact = client.get("/contact")
    assert res_contact.status_code == 302
    assert "/contact-grievance" in res_contact.headers.get("Location", "")

    res_support = client.get("/support")
    assert res_support.status_code == 302
    assert "/contact-grievance" in res_support.headers.get("Location", "")

    res_terms = client.get("/terms-of-use")
    assert res_terms.status_code == 302
    assert "/terms" in res_terms.headers.get("Location", "")


def test_admin_dashboard_navbar_excludes_discover_messages_community(client):
    """Verify admin dashboard excludes Discover, Community, and Messages navigation bar."""
    admin_id = UserModel.create_user("admin_nav_test@example.com", "AdminPass123!", "Admin Nav User", role="admin")
    try:
        _login(client, "admin_nav_test@example.com", "AdminPass123!")

        # 1. On Admin Dashboard
        admin_res = client.get("/admin/")
        assert admin_res.status_code == 200
        admin_html = admin_res.data.decode("utf-8")

        # Discover, Messages, Community user navbar items must NOT be present
        assert ">Discover</a>" not in admin_html
        assert ">Community</a>" not in admin_html
        assert ">Messages</a>" not in admin_html
        assert "mobile-bottom-nav" not in admin_html
        assert "Exit Admin" in admin_html
        assert "Admin Portal" in admin_html

        # 2. On User Dashboard
        user_res = client.get("/dashboard")
        assert user_res.status_code == 200
        user_html = user_res.data.decode("utf-8")

        # User navbar items MUST be present on user dashboard
        assert ">Discover</a>" in user_html
        assert ">Community</a>" in user_html
        assert ">Messages</a>" in user_html
        assert "mobile-bottom-nav" in user_html
    finally:
        client.get("/logout")
        mongo.db.users.delete_one({"_id": ObjectId(admin_id)})


