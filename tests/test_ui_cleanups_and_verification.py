import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if "" not in sys.path:
    sys.path.insert(0, "")

import pytest


def test_no_online_offline_status_in_chat():
    """Verify that online and offline status indicators are removed from chat UI."""
    chat_path = os.path.join("templates", "chat.html")
    assert os.path.exists(chat_path)

    with open(chat_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Header status indicators
    assert 'id="partner-status"' not in html, "Chat header partner-status element must be removed"
    assert '> Online</span>' not in html, "Online status text must be removed from chat header"
    assert '> Offline</span>' not in html, "Offline status text must be removed from chat header"
    assert 'Online</span>' not in html, "Online span must not exist in chat template"
    assert 'Offline</span>' not in html, "Offline span must not exist in chat template"

    # 2. Status polling loop
    assert '/api/user/online/' not in html, "Online/offline polling loop must be removed"

    # 3. Verified profile badge remains intact
    assert 'verified-badge' in html or 'bi-patch-check-fill' in html, "Verified badge should remain in chat header"

    print("CHAT ONLINE/OFFLINE STATUS REMOVAL VERIFIED!")


def test_test_sound_button_removed_from_base():
    """Verify that Test Sound button is removed while preserving notification bell and profile dropdown."""
    base_path = os.path.join("templates", "base.html")
    assert os.path.exists(base_path)

    with open(base_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Test Sound button removed
    assert 'id="desktopAudioTestBtn"' not in html, "Test Sound button #desktopAudioTestBtn must be removed"
    assert 'Test Sound</span>' not in html, "Test Sound text must be removed"

    # 2. Notification bell and badge preserved
    assert 'id="notificationDropdown"' in html, "Notification bell dropdown must be preserved"
    assert 'bi-bell' in html, "Bell icon must be preserved"
    assert 'id="notificationBadge"' in html, "Unread notification count badge must be preserved"

    # 3. User profile dropdown preserved
    assert 'id="userDropdown"' in html, "User profile dropdown must be preserved"
    assert 'My Profile' in html, "My Profile link must be preserved"
    assert 'Edit Profile' in html, "Edit Profile link must be preserved"
    assert 'Logout' in html, "Logout link must be preserved"

    print("TEST SOUND BUTTON REMOVAL & NAVBAR PRESERVATION VERIFIED!")


def test_profile_unwanted_options_removed():
    """Verify that Data Portability, Grievance, and Delete Account are removed from profile page."""
    profile_path = os.path.join("templates", "profile.html")
    assert os.path.exists(profile_path)

    with open(profile_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Unwanted options removed from profile page
    assert 'Data Portability</h6>' not in html, "Data Portability card heading must be removed from profile"
    assert 'Export My Data' not in html, "Export My Data button must be removed from profile"
    assert 'Grievance Redressal</h6>' not in html, "Grievance card heading must be removed from profile"
    assert 'Grievance Portal' not in html, "Grievance Portal button must be removed from profile"
    assert 'Account Erasure</h6>' not in html, "Account Erasure card heading must be removed from profile"
    assert 'id="deleteAccountModal"' not in html, "deleteAccountModal must be removed from profile"

    # 2. User profile details preserved
    assert 'Edit Profile' in html, "Edit Profile button must be preserved"
    assert 'Interests' in html, "User interests section must be preserved"
    assert 'Communication' in html, "Communication preferences must be preserved"

    print("PROFILE UNWANTED OPTIONS REMOVAL VERIFIED!")


def test_audio_system_and_legal_pages_preserved():
    """Verify that backend notification audio and legal compliance pages remain intact."""
    main_js_path = os.path.join("static", "js", "main.js")
    assert os.path.exists(main_js_path)

    with open(main_js_path, "r", encoding="utf-8") as f:
        js = f.read()

    # 1. Notification audio logic preserved
    assert 'playMessageNotificationSound' in js, "Message sound player must be preserved"
    assert 'showIncomingCall' in js, "Incoming call notification must be preserved"
    assert 'getSharedAudioContext' in js, "Shared audio context must be preserved"
    assert 'preloadAudioBuffers' in js, "Audio buffer preloader must be preserved"
    assert 'stopRingtone' in js, "stopRingtone function must be preserved"

    # 2. Check legal routes exist
    legal_routes_path = os.path.join("routes", "legal_routes.py")
    assert os.path.exists(legal_routes_path)

    with open(legal_routes_path, "r", encoding="utf-8") as f:
        legal_py = f.read()

    assert 'def contact_grievance' in legal_py, "contact_grievance legal route must be preserved"
    assert 'def export_my_data' in legal_py, "export_my_data route must be preserved"
    assert 'def data_rights' in legal_py, "data_rights route must be preserved"

    print("NOTIFICATION AUDIO & COMPLIANCE PRESERVATION VERIFIED!")


if __name__ == "__main__":
    test_no_online_offline_status_in_chat()
    test_test_sound_button_removed_from_base()
    test_profile_unwanted_options_removed()
    test_audio_system_and_legal_pages_preserved()
    print("\n--- ALL CLEANUP VERIFICATION TESTS PASSED SUCCESSFULLY ---")
