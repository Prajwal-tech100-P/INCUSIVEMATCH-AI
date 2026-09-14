import os
import secrets
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


def _bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name):
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


class Config:
    APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()

    # Never use a predictable secret in production.
    if APP_ENV == "production":
        SECRET_KEY = os.environ.get("SECRET_KEY")
    else:
        SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_urlsafe(48)

    MONGO_URI = os.environ.get(
        "MONGO_URI",
        "mongodb://localhost:27017/inclusive_match_db"
    )

    UPLOAD_FOLDER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "static", "uploads"
    )
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

    # Browser session hardening.
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _bool_env("SESSION_COOKIE_SECURE", APP_ENV == "production")
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)

    # Socket.IO is same-origin by default. If you deploy behind a proxy/domain
    # and need an explicit allow-list, set this to comma-separated origins.
    SOCKETIO_CORS_ALLOWED_ORIGINS = _csv_env("SOCKETIO_CORS_ALLOWED_ORIGINS")

    # WebRTC: STUN is useful for discovery; TURN is what makes calls reliable
    # across restrictive NAT/firewalls. Use a real TURN provider in production.
    WEBRTC_STUN_URLS = _csv_env("WEBRTC_STUN_URLS") or [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
    ]
    WEBRTC_TURN_URL = os.environ.get("WEBRTC_TURN_URL", "").strip()
    WEBRTC_TURN_USERNAME = os.environ.get("WEBRTC_TURN_USERNAME", "").strip()
    WEBRTC_TURN_CREDENTIAL = os.environ.get("WEBRTC_TURN_CREDENTIAL", "").strip()
    WEBRTC_TURN_TLS_URL = os.environ.get("WEBRTC_TURN_TLS_URL", "").strip()
