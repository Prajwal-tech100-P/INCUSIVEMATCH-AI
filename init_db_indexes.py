#!/usr/bin/env python3
"""MongoDB Atlas Index Optimization and Initialization Script.

Repeatable, safe index management utility for the InclusiveMatch AI project.
Connects to the application's configured MongoDB Atlas database, detects duplicate
records before creating unique indexes, avoids dropping existing indexes, and
creates optimized indexes supporting actual queries across all application models.
"""
import os
import sys
import re
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional
import pymongo
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IndexInitializer")


def mask_connection_uri(uri: str) -> str:
    """Mask credentials in MongoDB connection string to ensure credentials are never exposed."""
    if not uri:
        return "<empty>"
    # Mask password and username in mongodb:// or mongodb+srv://
    masked = re.sub(r"://([^:@]+):([^@]+)@", r"://\1:****@", uri)
    return masked


def get_configured_mongo_uri() -> str:
    """Read MongoDB connection string from environment or application config."""
    from config import Config
    return os.environ.get("MONGO_URI", Config.MONGO_URI)


def check_duplicate_emails(collection) -> List[Dict[str, Any]]:
    """Detect duplicate normalized email records in the users collection.

    Does NOT delete or modify any documents; reports duplicates for manual review.
    """
    pipeline = [
        {"$match": {"email": {"$exists": True, "$ne": None}}},
        {
            "$group": {
                "_id": {"$toLower": {"$trim": {"input": "$email"}}},
                "count": {"$sum": 1},
                "docs": {
                    "$push": {
                        "id": {"$toString": "$_id"},
                        "email": "$email",
                        "name": {"$ifNull": ["$name", ""]},
                    }
                },
            }
        },
        {"$match": {"count": {"$gt": 1}, "_id": {"$ne": ""}}},
    ]
    return list(collection.aggregate(pipeline))


def check_duplicate_community_blocks(collection) -> List[Dict[str, Any]]:
    """Detect duplicate blocker-blocked pairs within the same community."""
    pipeline = [
        {
            "$group": {
                "_id": {
                    "community_id": "$community_id",
                    "blocker_id": "$blocker_id",
                    "blocked_id": "$blocked_id",
                },
                "count": {"$sum": 1},
                "ids": {"$push": {"$toString": "$_id"}},
            }
        },
        {
            "$match": {
                "count": {"$gt": 1},
                "_id.community_id": {"$ne": None},
                "_id.blocker_id": {"$ne": None},
                "_id.blocked_id": {"$ne": None},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def index_matches_spec(existing_spec: Dict[str, Any], keys: List[Tuple[str, int]]) -> bool:
    """Check if existing index spec matches the requested keys sequence."""
    existing_keys = existing_spec.get("key", [])
    if isinstance(existing_keys, dict):
        existing_keys = list(existing_keys.items())
    elif hasattr(existing_keys, "items"):
        existing_keys = list(existing_keys.items())

    # Format of keys: list of (field, direction)
    norm_existing = [(k, int(v)) for k, v in existing_keys]
    norm_requested = [(k, int(v)) for k, v in keys]
    return norm_existing == norm_requested


def safe_create_index(
    collection,
    keys: List[Tuple[str, int]],
    name: str,
    unique: bool = False,
    expireAfterSeconds: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Create an index if it does not already exist, with meaningful name and error safety.

    Avoids dropping or modifying existing indexes.
    """
    col_name = collection.name
    existing_indexes = collection.index_information()

    # 1. Check if index with same name already exists
    if name in existing_indexes:
        logger.info("  [EXISTS] '%s.%s' already exists with name '%s'.", col_name, name, name)
        return {"status": "exists", "name": name, "collection": col_name}

    # 2. Check if an index with identical key pattern already exists under a legacy/different name
    for ex_name, ex_spec in existing_indexes.items():
        if index_matches_spec(ex_spec, keys):
            # Check unique/TTL constraints if applicable
            ex_unique = ex_spec.get("unique", False)
            ex_ttl = ex_spec.get("expireAfterSeconds")
            if ex_unique == unique and ex_ttl == expireAfterSeconds:
                logger.info(
                    "  [EXISTS] '%s' already has identical index under existing name '%s'. Keeping it.",
                    col_name,
                    ex_name,
                )
                return {"status": "exists", "name": ex_name, "collection": col_name}

    if dry_run:
        logger.info(
            "  [DRY-RUN] Would create index '%s.%s' (keys=%s, unique=%s, ttl=%s)",
            col_name,
            name,
            keys,
            unique,
            expireAfterSeconds,
        )
        return {"status": "dry_run", "name": name, "collection": col_name}

    # 3. Create index safely
    kwargs: Dict[str, Any] = {"name": name, "unique": unique}
    if expireAfterSeconds is not None:
        kwargs["expireAfterSeconds"] = expireAfterSeconds

    try:
        created_name = collection.create_index(keys, **kwargs)
        logger.info("  [CREATED] '%s.%s' created successfully.", col_name, created_name)
        return {"status": "created", "name": created_name, "collection": col_name}
    except Exception as e:
        logger.error("  [ERROR] Failed to create index '%s.%s': %s", col_name, name, e)
        return {"status": "error", "name": name, "collection": col_name, "error": str(e)}


def initialize_indexes(mongo_uri: Optional[str] = None, dry_run: bool = False) -> bool:
    """Connect to MongoDB Atlas, detect duplicates, and initialize all required indexes safely.

    Returns:
        True if all operations completed cleanly without unhandled errors, False otherwise.
    """
    uri = mongo_uri or get_configured_mongo_uri()
    masked = mask_connection_uri(uri)
    logger.info("Connecting to MongoDB at: %s", masked)

    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=8000)
        # Verify connection
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        logger.error(
            "Connection failed: Server selection timeout. Please verify network access, "
            "Atlas IP Access List, and credentials: %s",
            e,
        )
        return False
    except PyMongoError as e:
        logger.error("MongoDB Atlas connection error: %s", e)
        return False

    # Resolve database: prefer default from connection URI, fallback to inclusive_match_db
    try:
        db = client.get_default_database()
    except Exception:
        db = None
    if db is None:
        db = client["inclusive_match_db"]

    db_name = db.name
    logger.info("Connected successfully. Target database: '%s'", db_name)
    logger.info("Executing safe index initialization (dry_run=%s)...", dry_run)

    results = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. USERS COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("--- Inspecting collection: 'users' ---")
    users_col = db.users

    # Check for duplicate emails BEFORE creating unique index
    dup_emails = check_duplicate_emails(users_col)
    if dup_emails:
        logger.warning(
            "  [WARNING] %d duplicate email group(s) detected in 'users' collection! "
            "Unique index 'idx_users_email_unique' SKIPPED to preserve user data.",
            len(dup_emails),
        )
        for dup in dup_emails:
            logger.warning(
                "    - Normalized Email: '%s' (Count: %d). Document records: %s",
                dup["_id"],
                dup["count"],
                dup["docs"],
            )
        results.append({
            "collection": "users",
            "name": "idx_users_email_unique",
            "status": "skipped_due_to_duplicates",
            "duplicates": len(dup_emails),
        })
    else:
        results.append(
            safe_create_index(
                users_col,
                [("email", pymongo.ASCENDING)],
                name="idx_users_email_unique",
                unique=True,
                dry_run=dry_run,
            )
        )

    # Compound index for candidate discovery and admin stats (role + is_active)
    results.append(
        safe_create_index(
            users_col,
            [("role", pymongo.ASCENDING), ("is_active", pymongo.ASCENDING)],
            name="idx_users_role_is_active",
            dry_run=dry_run,
        )
    )

    # Index for user directory listing sorted by name
    results.append(
        safe_create_index(
            users_col,
            [("name", pymongo.ASCENDING)],
            name="idx_users_name",
            dry_run=dry_run,
        )
    )

    # Multikey index on 'liked' array for mutual matches queries
    results.append(
        safe_create_index(
            users_col,
            [("liked", pymongo.ASCENDING)],
            name="idx_users_liked",
            dry_run=dry_run,
        )
    )

    # Index on face_verified for verified badge filtering and discovery
    results.append(
        safe_create_index(
            users_col,
            [("face_verified", pymongo.ASCENDING)],
            name="idx_users_face_verified",
            dry_run=dry_run,
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. MESSAGES COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("--- Inspecting collection: 'messages' ---")
    messages_col = db.messages

    # Conversation retrieval and chronological sorting (both directions)
    results.append(
        safe_create_index(
            messages_col,
            [
                ("sender_id", pymongo.ASCENDING),
                ("receiver_id", pymongo.ASCENDING),
                ("timestamp", pymongo.ASCENDING),
            ],
            name="idx_messages_sender_receiver_time",
            dry_run=dry_run,
        )
    )
    results.append(
        safe_create_index(
            messages_col,
            [
                ("receiver_id", pymongo.ASCENDING),
                ("sender_id", pymongo.ASCENDING),
                ("timestamp", pymongo.ASCENDING),
            ],
            name="idx_messages_receiver_sender_time",
            dry_run=dry_run,
        )
    )

    # Unread messages lookup and mark-as-read queries
    results.append(
        safe_create_index(
            messages_col,
            [("receiver_id", pymongo.ASCENDING), ("is_read", pymongo.ASCENDING)],
            name="idx_messages_receiver_is_read",
            dry_run=dry_run,
        )
    )

    # Global message timestamp for monitoring and statistics
    results.append(
        safe_create_index(
            messages_col,
            [("timestamp", pymongo.DESCENDING)],
            name="idx_messages_timestamp",
            dry_run=dry_run,
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. NOTIFICATIONS COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("--- Inspecting collection: 'notifications' ---")
    notifications_col = db.notifications

    # User notifications ordered by newest first
    results.append(
        safe_create_index(
            notifications_col,
            [("user_id", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)],
            name="idx_notifications_user_created",
            dry_run=dry_run,
        )
    )

    # Unread notifications count and mark-all-read queries
    results.append(
        safe_create_index(
            notifications_col,
            [("user_id", pymongo.ASCENDING), ("is_read", pymongo.ASCENDING)],
            name="idx_notifications_user_unread",
            dry_run=dry_run,
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CALL INVITATIONS COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("--- Inspecting collection: 'call_invitations' ---")
    call_col = db.call_invitations

    # TTL index: Automatically purges invitations after expiration
    results.append(
        safe_create_index(
            call_col,
            [("expires_at", pymongo.ASCENDING)],
            name="idx_call_invitations_ttl",
            expireAfterSeconds=0,
            dry_run=dry_run,
        )
    )

    # Pending call retrieval by recipient and status
    results.append(
        safe_create_index(
            call_col,
            [
                ("recipient_id", pymongo.ASCENDING),
                ("status", pymongo.ASCENDING),
                ("created_at", pymongo.DESCENDING),
            ],
            name="idx_call_recipient_status",
            dry_run=dry_run,
        )
    )

    # Superseding existing invitations from same caller to same recipient
    results.append(
        safe_create_index(
            call_col,
            [
                ("caller_id", pymongo.ASCENDING),
                ("recipient_id", pymongo.ASCENDING),
                ("status", pymongo.ASCENDING),
            ],
            name="idx_call_caller_recipient_status",
            dry_run=dry_run,
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. COMMUNITIES COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("--- Inspecting collection: 'communities' ---")
    communities_col = db.communities

    # Membership and updated time
    results.append(
        safe_create_index(
            communities_col,
            [("member_ids", pymongo.ASCENDING), ("updated_at", pymongo.DESCENDING)],
            name="idx_communities_member_ids_updated",
            dry_run=dry_run,
        )
    )

    # Available communities sorted by newest
    results.append(
        safe_create_index(
            communities_col,
            [("created_at", pymongo.DESCENDING)],
            name="idx_communities_created_at",
            dry_run=dry_run,
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. COMMUNITY MESSAGES COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("--- Inspecting collection: 'community_messages' ---")
    community_msgs_col = db.community_messages

    # Messages within a community sorted chronologically
    results.append(
        safe_create_index(
            community_msgs_col,
            [("community_id", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)],
            name="idx_community_messages_cid_time",
            dry_run=dry_run,
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. COMMUNITY BLOCKS COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("--- Inspecting collection: 'community_blocks' ---")
    blocks_col = db.community_blocks

    # Duplicate block check before unique index
    dup_blocks = check_duplicate_community_blocks(blocks_col)
    if dup_blocks:
        logger.warning(
            "  [WARNING] %d duplicate block relationship(s) detected in 'community_blocks'. "
            "Unique index 'idx_community_blocks_unique' SKIPPED to preserve records.",
            len(dup_blocks),
        )
        for db_item in dup_blocks:
            logger.warning("    - Pair: %s (Count: %d)", db_item["_id"], db_item["count"])
        results.append({
            "collection": "community_blocks",
            "name": "idx_community_blocks_unique",
            "status": "skipped_due_to_duplicates",
            "duplicates": len(dup_blocks),
        })
    else:
        results.append(
            safe_create_index(
                blocks_col,
                [
                    ("community_id", pymongo.ASCENDING),
                    ("blocker_id", pymongo.ASCENDING),
                    ("blocked_id", pymongo.ASCENDING),
                ],
                name="idx_community_blocks_unique",
                unique=True,
                dry_run=dry_run,
            )
        )

    # Blocked member check within community
    results.append(
        safe_create_index(
            blocks_col,
            [("community_id", pymongo.ASCENDING), ("blocked_id", pymongo.ASCENDING)],
            name="idx_community_blocks_cid_blocked",
            dry_run=dry_run,
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 8. REPORTS COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("--- Inspecting collection: 'reports' ---")
    reports_col = db.reports

    # Duplicate report check query
    results.append(
        safe_create_index(
            reports_col,
            [
                ("reporter_id", pymongo.ASCENDING),
                ("reported_id", pymongo.ASCENDING),
                ("status", pymongo.ASCENDING),
            ],
            name="idx_reports_reporter_reported_status",
            dry_run=dry_run,
        )
    )

    # Status and chronological sorting for admin review
    results.append(
        safe_create_index(
            reports_col,
            [("status", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)],
            name="idx_reports_status_created",
            dry_run=dry_run,
        )
    )

    # Global created_at index for admin panel listing
    results.append(
        safe_create_index(
            reports_col,
            [("created_at", pymongo.DESCENDING)],
            name="idx_reports_created_at",
            dry_run=dry_run,
        )
    )

    # Summary report
    created_count = sum(1 for r in results if r.get("status") == "created")
    exists_count = sum(1 for r in results if r.get("status") == "exists")
    error_count = sum(1 for r in results if r.get("status") == "error")
    skipped_count = sum(1 for r in results if "skipped" in r.get("status", ""))

    logger.info("==================================================")
    logger.info("Index Optimization Summary for '%s':", db_name)
    logger.info("  Created: %d", created_count)
    logger.info("  Already Exists: %d", exists_count)
    logger.info("  Skipped (Duplicates): %d", skipped_count)
    logger.info("  Errors: %d", error_count)
    logger.info("==================================================")

    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Initialize and optimize MongoDB Atlas indexes for InclusiveMatch AI."
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="Optional MongoDB connection URI. Defaults to MONGO_URI in .env or Config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect existing indexes and planned actions without creating any indexes.",
    )
    args = parser.parse_args()

    success = initialize_indexes(mongo_uri=args.uri, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
