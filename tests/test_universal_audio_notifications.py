"""
Automated tests verifying the removal of Desktop Audio Settings while preserving
the universal notification sounds and incoming call ringtones across both Android and Desktop.
"""
import os
import re
import pytest
from app import create_app


@pytest.fixture(scope="module")
def test_app():
    os.environ["APP_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-secret-universal-audio"
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    return app


def test_no_desktop_audio_settings_or_testing_controls_in_main_js():
    """Verify that desktop-specific audio settings, banners, and test controls are completely removed."""
    main_js_path = os.path.join("static", "js", "main.js")
    with open(main_js_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Desktop-specific functions and banners must not exist
    assert "testDesktopAudio" not in content, "testDesktopAudio must be removed"
    assert "showDesktopAudioBannerIfNeeded" not in content, "showDesktopAudioBannerIfNeeded must be removed"
    assert "hideDesktopAudioBanner" not in content, "hideDesktopAudioBanner must be removed"
    assert "desktop-audio-banner" not in content, "desktop-audio-banner DOM element must be removed"
    assert "isAndroidDevice" not in content, "isAndroidDevice discriminator must be removed"


def test_no_desktop_audio_settings_in_templates():
    """Verify no desktop audio settings controls or banners are hardcoded in templates."""
    templates_dir = "templates"
    for root, _, files in os.walk(templates_dir):
        for file in files:
            if file.endswith(".html"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    html = f.read()
                assert "testDesktopAudio" not in html, f"testDesktopAudio found in {filepath}"
                assert "desktop-audio-banner" not in html, f"desktop-audio-banner found in {filepath}"
                assert "Desktop Audio Settings" not in html, f"Desktop Audio Settings found in {filepath}"


def test_universal_audio_functionality_preserved():
    """Verify universal audio pipeline is preserved in main.js."""
    main_js_path = os.path.join("static", "js", "main.js")
    with open(main_js_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Audio assets
    assert "/static/audio/message_notification.mp3" in content
    assert "/static/audio/incoming_call.mp3" in content

    # Universal Web Audio preloading and decoding
    assert "preloadAudioBuffers" in content
    assert "getSharedAudioContext" in content

    # Message sound triggers and deduplication
    assert "playMessageNotificationSound" in content
    assert "isMessageSoundEnabled" in content
    assert "seenMessageIds" in content

    # Ringtone continuous playback and stopping
    assert "playRingtone" in content
    assert "stopRingtone" in content
    assert "isRingtonePlaying" in content
    assert "unlockRingtoneAudio" in content
    assert "startSyntheticRingtone" in content
    assert "stopSyntheticRingtone" in content

    # User gesture unlocking
    assert "unlockAudio" in content
    assert "['click', 'keydown', 'touchstart', 'pointerdown']" in content


def test_shared_notification_preference_toggle_preserved():
    """Verify chat.html contains the shared message notification sound toggle."""
    chat_path = os.path.join("templates", "chat.html")
    with open(chat_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert 'id="toggle-sound-btn"' in content
    assert "inclusive_msg_sound_enabled" in content

