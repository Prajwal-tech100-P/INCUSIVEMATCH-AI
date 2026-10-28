import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from bson import ObjectId
from pymongo import ReturnDocument
from extensions import mongo


class PasswordReset:
    """Manages secure, single-use password reset tokens in MongoDB."""

    COLLECTION = "password_resets"
    EXPIRATION_MINUTES = 30
    RATE_LIMIT_SECONDS = 60  # Minimum cooldown between email dispatches per account
    IP_RATE_LIMIT_MAX = 10   # Max requests per IP within 15 minutes
    IP_RATE_LIMIT_WINDOW = 15

    @classmethod
    def _collection(cls):
        return mongo.db[cls.COLLECTION]

    @staticmethod
    def hash_token(raw_token: str) -> str:
        """Compute SHA-256 hash of the raw token for secure database storage."""
        return hashlib.sha256(raw_token.strip().encode("utf-8")).hexdigest()

    @classmethod
    def ensure_indexes(cls) -> None:
        """Create required indexes for fast lookup, uniqueness, and asynchronous TTL expiry."""
        try:
            col = cls._collection()
            # Unique index on token_hash ensures no duplicate tokens can exist
            col.create_index(
                [("token_hash", 1)],
                name="idx_password_resets_token_hash",
                unique=True
            )
            # Index on user_id for rapid retrieval and bulk invalidation
            col.create_index(
                [("user_id", 1)],
                name="idx_password_resets_user_id"
            )
            # Index on email for rate limiting checks
            col.create_index(
                [("email", 1), ("created_at", -1)],
                name="idx_password_resets_email_created_at"
            )
            # TTL index on expires_at for automated MongoDB cleanup
            # Note: Application code still strictly validates expires_at explicitly
            col.create_index(
                [("expires_at", 1)],
                name="idx_password_resets_expires_at",
                expireAfterSeconds=0
            )
        except Exception:
            # Index initialization errors must not crash the application
            pass

    @classmethod
    def is_email_rate_limited(cls, email: str) -> bool:
        """Check if a password reset was requested for this email within the cooldown window."""
        if not email:
            return False
        normalized_email = email.strip().lower()
        cutoff = datetime.utcnow() - timedelta(seconds=cls.RATE_LIMIT_SECONDS)
        recent = cls._collection().find_one({
            "email": normalized_email,
            "created_at": {"$gt": cutoff}
        })
        return recent is not None

    @classmethod
    def is_ip_rate_limited(cls, ip_address: Optional[str]) -> bool:
        """Check if an IP address exceeded request limits within the rate-limit window."""
        if not ip_address:
            return False
        cutoff = datetime.utcnow() - timedelta(minutes=cls.IP_RATE_LIMIT_WINDOW)
        count = cls._collection().count_documents({
            "ip_address": ip_address,
            "created_at": {"$gt": cutoff}
        })
        return count >= cls.IP_RATE_LIMIT_MAX

    @classmethod
    def create_token(cls, user_id: str, email: str, ip_address: Optional[str] = None) -> Tuple[str, datetime]:
        """Generate a cryptographically secure token, store its hash, and return (raw_token, expires_at).

        Any previous unconsumed tokens for this user are invalidated.
        """
        raw_token = secrets.token_urlsafe(32)
        token_hash = cls.hash_token(raw_token)
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=cls.EXPIRATION_MINUTES)

        col = cls._collection()

        # Invalidate any previously active unused tokens for this user
        col.update_many(
            {"user_id": str(user_id), "used": False},
            {"$set": {"used": True, "invalidated_at": now}}
        )

        doc = {
            "user_id": str(user_id),
            "email": email.strip().lower(),
            "token_hash": token_hash,
            "created_at": now,
            "expires_at": expires_at,
            "used": False,
            "used_at": None,
            "ip_address": ip_address or ""
        }
        col.insert_one(doc)

        return raw_token, expires_at

    @classmethod
    def verify_token(cls, raw_token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Verify token validity without consuming it (useful for GET request rendering).

        Returns (token_doc, None) on success or (None, error_message) on failure.
        """
        if not raw_token or not isinstance(raw_token, str):
            return None, "Invalid or missing password reset token."

        token_hash = cls.hash_token(raw_token)
        doc = cls._collection().find_one({"token_hash": token_hash})

        if not doc:
            return None, "This password reset link is invalid or does not exist."

        if doc.get("used", False):
            return None, "This password reset link has already been used. Please request a new one."

        expires_at = doc.get("expires_at")
        if not expires_at or expires_at < datetime.utcnow():
            return None, "This password reset link has expired. Please request a new one."

        return doc, None

    @classmethod
    def consume_token(cls, raw_token: str) -> Tuple[Optional[str], Optional[str]]:
        """Atomically validate and consume a reset token.

        Returns (user_id, None) on success or (None, error_message) on failure.
        """
        if not raw_token or not isinstance(raw_token, str):
            return None, "Invalid or missing password reset token."

        token_hash = cls.hash_token(raw_token)
        now = datetime.utcnow()

        # Atomic find-and-update ensures race-condition safety and single-use enforcement
        updated_doc = cls._collection().find_one_and_update(
            {
                "token_hash": token_hash,
                "used": False,
                "expires_at": {"$gt": now}
            },
            {
                "$set": {
                    "used": True,
                    "used_at": now
                }
            },
            return_document=ReturnDocument.AFTER
        )

        if not updated_doc:
            # Inspect existing doc to return an informative message
            existing = cls._collection().find_one({"token_hash": token_hash})
            if not existing:
                return None, "This password reset link is invalid or does not exist."
            if existing.get("used", False):
                return None, "This password reset link has already been used. Please request a new one."
            if existing.get("expires_at", now) <= now:
                return None, "This password reset link has expired. Please request a new one."
            return None, "Unable to validate reset token. Please try again."

        return str(updated_doc["user_id"]), None
