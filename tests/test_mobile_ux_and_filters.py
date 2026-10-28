import os
import re
import pytest
from bson import ObjectId
from app import create_app
from extensions import mongo
from models.user import UserModel


@pytest.fixture(scope="module")
def app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-mobile-filters-key-2026"
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
    email = "mobile_filter_tester@example.com"
    pwd = "TesterPass123!"
    uid = UserModel.create_user(
        email,
        pwd,
        "Mobile Tester",
        role="user"
    )
    mongo.db.users.update_one(
        {"_id": ObjectId(uid)},
        {"$set": {
            "gender": "Male",
            "age": 28,
            "interests": ["Coffee", "Travel", "Photography"],
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
def admin_user(app):
    email = "admin_mobile_tester@example.com"
    pwd = "AdminPass123!"
    uid = UserModel.create_user(
        email,
        pwd,
        "Admin Mobile Tester",
        role="admin"
    )
    yield {"id": uid, "email": email, "password": pwd}
    mongo.db.users.delete_one({"_id": ObjectId(uid)})


# ==========================================
# 1. FILTER INTERFACE & BOTTOM SHEET TESTS
# ==========================================

def test_filter_button_replaces_large_sidebar(client, regular_user):
    """Verify the large desktop sidebar is removed and replaced by a compact filter button."""
    _login(client, regular_user["email"], regular_user["password"])
    res = client.get("/dashboard")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # 1. The old oversized sidebar must NOT exist
    assert 'id="filters-sidebar"' not in html, "Old oversized sidebar #filters-sidebar should be removed"

    # 2. Compact filter button must exist in toolbar
    assert 'id="open-filter-btn"' in html, "Compact filter button #open-filter-btn must be present"
    assert 'data-bs-toggle="offcanvas"' in html
    assert 'data-bs-target="#filterPanel"' in html
    assert 'bi-funnel-fill' in html, "Filter button must use bi-funnel icon"
    assert "Filters" in html

    # 3. Main discovery area is expanded to col-12
    assert 'id="main-content-area"' in html
    assert 'col-12' in html


def test_filter_panel_structure_and_controls(client, regular_user):
    """Verify filter panel contains all required bottom sheet elements and controls."""
    _login(client, regular_user["email"], regular_user["password"])
    res = client.get("/dashboard")
    html = res.data.decode("utf-8")

    # Offcanvas container
    assert 'id="filterPanel"' in html, "Unified filter panel #filterPanel must exist"
    assert 'filter-panel' in html

    # Mobile drag handle
    assert 'bottom-sheet-drag-handle' in html, "Mobile bottom sheet drag handle must exist"

    # Header and Close button
    assert 'data-bs-dismiss="offcanvas"' in html

    # Form with all 5 supported filters
    assert 'id="discover-filters-form"' in html

    # Gender radio pills
    assert 'value="Any"' in html
    assert 'value="Female"' in html
    assert 'value="Male"' in html
    assert 'value="Non-binary"' in html

    # Age inputs
    assert 'name="min_age"' in html
    assert 'name="max_age"' in html

    # Shared interests select
    assert 'name="interest"' in html

    # Communication preference select
    assert 'name="comm_pref"' in html

    # Online status switch
    assert 'name="online"' in html

    # Sticky Footer actions
    assert 'id="apply-filters-btn"' in html
    assert 'type="submit"' in html
    assert 'id="reset-filters-btn"' in html
    assert 'href="/dashboard"' in html


def test_filters_preservation_and_active_badge(client, regular_user):
    """Verify applied filters preserve selections and display active count badge."""
    _login(client, regular_user["email"], regular_user["password"])

    query_params = {
        "gender": "Female",
        "min_age": 22,
        "max_age": 35,
        "comm_pref": "text",
        "online": "1"
    }
    res = client.get("/dashboard", query_string=query_params)
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # 1. Active count badge on trigger button
    assert 'badge bg-primary text-white rounded-pill' in html

    # 2. Preserved values
    assert 'value="Female" checked' in html or ('value="Female"' in html and 'checked' in html)
    assert 'value="22"' in html
    assert 'value="35"' in html
    assert 'value="text" selected' in html or ('value="text"' in html and 'selected' in html)
    assert 'id="filter-online-switch" checked' in html or ('filter-online-switch' in html and 'checked' in html)


def test_empty_state_when_filters_yield_no_matches(client, regular_user):
    """Verify a clean empty state with Reset Filters button appears when no matches found."""
    _login(client, regular_user["email"], regular_user["password"])

    res = client.get("/dashboard?interest=ExtremelyRareHobbyXYZ999&min_age=98&max_age=99")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert "No matching profiles found" in html
    assert "Reset Filters" in html
    assert 'href="/dashboard"' in html


# ==========================================
# 2. HAMBURGER MENU & AUTHENTICATION TESTS
# ==========================================

def test_hamburger_menu_on_unauthenticated_pages(client):
    """Verify the hamburger toggler and mobile menu works properly on login/register/forgot pages."""
    client.get("/logout")

    auth_pages = [
        "/login",
        "/register",
        "/forgot-password"
    ]

    for path in auth_pages:
        res = client.get(path)
        assert res.status_code == 200, f"Page {path} failed to render"
        html = res.data.decode("utf-8")

        # 1. Toggler button
        assert 'class="navbar-toggler' in html, f"Navbar toggler button missing on {path}"
        assert 'data-bs-toggle="collapse"' in html
        assert 'data-bs-target="#navbarNav"' in html
        assert 'aria-controls="navbarNav"' in html
        assert 'aria-label="Toggle navigation"' in html

        # 2. Navbar collapse container
        assert 'id="navbarNav"' in html, f"#navbarNav missing on {path}"

        # 3. CRITICAL BUG CHECK: desktop-nav-links must NOT be hiding the unauthenticated menu
        assert 'desktop-nav-links' not in html, (
            f"Broken desktop-nav-links class found on {path}, which caused mobile menu to vanish!"
        )

        # 4. Login and Sign Up links exist for unauthenticated visitors
        assert 'href="/login"' in html, f"Login link missing in navbar on {path}"
        assert 'href="/register"' in html, f"Sign Up link missing in navbar on {path}"


def test_hamburger_menu_on_authenticated_user_pages(client, regular_user):
    """Verify mobile hamburger menu contains proper authenticated items."""
    _login(client, regular_user["email"], regular_user["password"])
    res = client.get("/dashboard")
    html = res.data.decode("utf-8")

    assert 'id="navbarNav"' in html
    assert 'desktop-nav-links' not in html

    # User links in collapse
    assert 'href="/dashboard"' in html
    assert 'href="/chat"' in html
    assert 'href="/community"' in html
    assert 'id="userDropdown"' in html


def test_admin_navigation_separation_preserved_in_hamburger(client, admin_user):
    """Verify admin section continues to strictly exclude user dating nav from mobile collapse."""
    _login(client, admin_user["email"], admin_user["password"])
    res = client.get("/admin/")
    html = res.data.decode("utf-8")

    assert 'id="navbarNav"' in html

    # Dating links must NOT exist in navbar
    assert '>Discover</a>' not in html
    assert '>Messages</a>' not in html
    assert '>Community</a>' not in html

    # Exit Admin must exist in navbar
    assert 'Exit Admin' in html
    assert 'href="/dashboard"' in html


# ==========================================
# 3. ANDROID RESPONSIVENESS & ASSETS TESTS
# ==========================================

def test_viewport_meta_tag_and_safe_areas(client):
    """Verify viewport meta tag includes viewport-fit=cover for Android devices."""
    client.get("/logout")
    res = client.get("/login", follow_redirects=True)
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert 'viewport-fit=cover' in html, "Viewport meta tag should include viewport-fit=cover"


def test_responsive_css_and_bottom_sheet_rules(app):
    """Verify style.css contains Android bottom sheet rules and overflow protection."""
    css_path = os.path.join(app.static_folder, "css", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    # No desktop-nav-links hiding rule
    assert ".desktop-nav-links { display: none !important; }" not in css

    # Overflow protection
    assert "overflow-x: hidden" in css

    # Android bottom sheet rules
    assert ".filter-panel.offcanvas" in css
    assert "max-height: 60vh" in css
    assert "border-radius: 24px 24px 0 0" in css
    assert "env(safe-area-inset-bottom" in css

    # Touch target minimums
    assert "min-height: 44px" in css


def test_javascript_event_listeners_and_dismiss(app):
    """Verify main.js contains outside-click, Escape key, and filter validation."""
    js_path = os.path.join(app.static_folder, "js", "main.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    assert "navbarTogglerBtn" in js or "navbar-toggler" in js
    assert "Escape" in js
    assert "filter-min-age" in js
    assert "Minimum age cannot be greater than maximum age" in js
