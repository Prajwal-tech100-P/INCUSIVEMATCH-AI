import os
import pytest
from bson import ObjectId
from app import create_app
from extensions import mongo
from models.user import UserModel


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-admin-nav-key-2026"
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
def admin_user(app):
    email = "admin_nav_suite_test@example.com"
    pwd = "AdminSecurePass123!"
    uid = UserModel.create_user(email, pwd, "Admin Suite Tester", role="admin")
    yield {"id": uid, "email": email, "password": pwd}
    mongo.db.users.delete_one({"_id": ObjectId(uid)})


@pytest.fixture
def regular_user(app):
    email = "user_nav_suite_test@example.com"
    pwd = "UserSecurePass123!"
    uid = UserModel.create_user(email, pwd, "Regular Suite Tester", role="user")
    yield {"id": uid, "email": email, "password": pwd}
    mongo.db.users.delete_one({"_id": ObjectId(uid)})


def test_1_admin_navigation_excludes_discover_messages_community(client, admin_user):
    """Confirm all representative admin pages consistently exclude Discover, Messages, and Community."""
    _login(client, admin_user["email"], admin_user["password"])

    admin_routes = [
        "/admin/",
        "/admin/users",
        "/admin/matches",
        "/admin/reports",
        "/admin/grievances",
        "/admin/messages",
        "/admin/statistics",
        "/admin/settings",
    ]

    for route in admin_routes:
        res = client.get(route)
        assert res.status_code == 200, f"Route {route} failed to render with 200"
        html = res.data.decode("utf-8")

        # Discover, User Messages, Community must NOT appear
        assert ">Discover</a>" not in html, f"Discover link found on admin route {route}"
        assert ">Messages</a>" not in html, f"Messages link found on admin route {route}"
        assert ">Community</a>" not in html, f"Community link found on admin route {route}"


def test_2_admin_navigation_excludes_redundant_admin_button(client, admin_user):
    """Confirm the redundant top-nav Admin button is hidden while inside the admin section."""
    _login(client, admin_user["email"], admin_user["password"])

    admin_routes = ["/admin/", "/admin/users", "/admin/reports", "/admin/settings"]
    for route in admin_routes:
        res = client.get(route)
        assert res.status_code == 200
        html = res.data.decode("utf-8")

        # The warning-styled top-nav link '<a class="nav-link text-warning fw-semibold" href="/admin/">...Admin</a>' must not appear
        assert 'class="nav-link text-warning fw-semibold"' not in html, f"Redundant Admin button found on {route}"


def test_3_admin_pages_do_not_render_mobile_bottom_navigation(client, admin_user):
    """Confirm that the user-facing mobile bottom navigation bar is suppressed on admin pages."""
    _login(client, admin_user["email"], admin_user["password"])

    admin_routes = ["/admin/", "/admin/users", "/admin/reports", "/admin/grievances"]
    for route in admin_routes:
        res = client.get(route)
        assert res.status_code == 200
        html = res.data.decode("utf-8")

        assert "mobile-bottom-nav" not in html, f"mobile-bottom-nav found on {route}"


def test_4_normal_user_pages_display_expected_navigation(client, regular_user):
    """Confirm normal user pages still display Discover, Messages, Community, and mobile bottom nav."""
    _login(client, regular_user["email"], regular_user["password"])

    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert ">Discover</a>" in html
    assert ">Messages</a>" in html
    assert ">Community</a>" in html
    assert "mobile-bottom-nav" in html
    assert "Exit Admin" not in html
    assert "Admin Portal" not in html
    assert 'class="nav-link text-warning fw-semibold"' not in html


def test_5_exit_admin_link_resolves_correctly_and_preserves_auth(client, admin_user):
    """Confirm Exit Admin link appears on admin pages, resolves to /dashboard without loops, and preserves admin status."""
    _login(client, admin_user["email"], admin_user["password"])

    # Verify Exit Admin appears on admin routes
    res = client.get("/admin/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Exit Admin" in html
    assert "/dashboard" in html

    # Clicking Exit Admin navigates to /dashboard
    exit_res = client.get("/dashboard")
    assert exit_res.status_code == 200
    exit_html = exit_res.data.decode("utf-8")

    # On user dashboard, admin sees user navigation AND the Admin return button
    assert ">Discover</a>" in exit_html
    assert ">Community</a>" in exit_html
    assert 'class="nav-link text-warning fw-semibold"' in exit_html
    assert "Admin" in exit_html

    # Admin authority remains intact: can navigate straight back to /admin/
    admin_return_res = client.get("/admin/")
    assert admin_return_res.status_code == 200


def test_6_unauthorized_users_blocked_from_admin_routes(client, regular_user):
    """Verify unauthorized users (unauthenticated and non-admin) cannot access admin routes."""
    client.get("/logout")
    admin_routes = ["/admin/", "/admin/users", "/admin/reports", "/admin/settings"]

    # 1. Unauthenticated users are redirected to login
    for route in admin_routes:
        res = client.get(route)
        assert res.status_code == 302, f"Unauthenticated request to {route} did not redirect"
        assert "/login" in res.headers.get("Location", "")

    # 2. Authenticated non-admin users get 403 Forbidden
    _login(client, regular_user["email"], regular_user["password"])
    for route in admin_routes:
        res = client.get(route)
        assert res.status_code == 403, f"Non-admin user received status {res.status_code} instead of 403 on {route}"


def test_7_representative_admin_routes_render_successfully(client, admin_user):
    """Verify representative admin routes render without HTTP 500 or template errors."""
    _login(client, admin_user["email"], admin_user["password"])

    routes_to_test = [
        "/admin/",
        "/admin/users",
        "/admin/matches",
        "/admin/reports",
        "/admin/grievances",
        "/admin/messages",
        "/admin/statistics",
        "/admin/settings",
    ]

    for route in routes_to_test:
        res = client.get(route)
        assert res.status_code == 200, f"Route {route} failed with status {res.status_code}"
        assert b"Admin Portal" in res.data, f"Admin Portal badge missing on {route}"
        assert b"InclusiveMatch AI" in res.data
