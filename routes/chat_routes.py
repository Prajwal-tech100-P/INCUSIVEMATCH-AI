from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from models.message import Message, ChatReport, OnlineTracker
from models.user import UserModel
from models.match import Match
from models.call_invitation import CallInvitation
from ai_modules.nlp_safety import NLPModerator
from extensions import socketio, mongo
from datetime import datetime

import hashlib
import time
import threading

chat_bp = Blueprint("chat", __name__)
moderator = NLPModerator()

# Backward compatibility aliases for process-local references
_warned_requests_cache = {}
_warned_cache_lock = threading.Lock()

def _handle_warning_with_dedup(user, receiver_id: str, text: str):
    """Atomically deduplicate warnings across multiple WSGI worker processes using MongoDB.

    - Computes a deterministic SHA-256 hash of sender:receiver:text.
    - Uses a 5-second logical deduplication window enforced in application code.
    - Atomically checks or acquires a warning slot to guarantee concurrent requests cannot
      both increment the user's warning counter.
    - If identical violation occurred within last 5 seconds, returns cached warning count without re-incrementing.
    - If 5 seconds elapsed, treats as a genuinely new violation.
    - Fails safely: DB errors block the message and return a warning without corrupting counters.
    """
    from bson import ObjectId
    from datetime import timedelta
    from pymongo import ReturnDocument
    from pymongo.errors import DuplicateKeyError

    user_id = str(user.id)
    req_hash = hashlib.sha256(f"{user_id}:{receiver_id}:{text}".encode("utf-8")).hexdigest()
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=5)

    try:
        # Fast path: Check if an active, completed warning was already issued within 5 seconds
        existing = mongo.db.warned_request_dedup.find_one({
            "_id": req_hash,
            "created_at": {"$gt": cutoff},
            "status": "completed"
        })
        if existing:
            cached_count = existing.get("warning_count", 1)
            if cached_count >= 5:
                return jsonify(error="Your account has been temporarily restricted for 7 days because you received 5 harmful-language warnings."), 403
            return jsonify(error=f"Warning {cached_count}/5: Your message contains potentially harmful language. Please use respectful language."), 400

        # Attempt atomic reservation: insert new claim or conditionally update an expired record
        is_new_claim = False
        try:
            mongo.db.warned_request_dedup.insert_one({
                "_id": req_hash,
                "sender_id": user_id,
                "receiver_id": receiver_id,
                "created_at": now,
                "status": "processing",
                "expire_at": now + timedelta(seconds=60)
            })
            is_new_claim = True
        except DuplicateKeyError:
            # Document exists. Check if it is expired (> 5s old) and update atomically
            res = mongo.db.warned_request_dedup.find_one_and_update(
                {
                    "_id": req_hash,
                    "created_at": {"$lte": cutoff}
                },
                {
                    "$set": {
                        "created_at": now,
                        "status": "processing",
                        "expire_at": now + timedelta(seconds=60)
                    }
                },
                return_document=ReturnDocument.AFTER
            )
            if res is not None:
                is_new_claim = True

        if not is_new_claim:
            # Another worker claimed this warning slot! Wait briefly for completion
            for _ in range(10):  # wait up to 500ms
                time.sleep(0.05)
                doc = mongo.db.warned_request_dedup.find_one({"_id": req_hash})
                if doc and doc.get("status") == "completed":
                    cached_count = doc.get("warning_count", 1)
                    if cached_count >= 5:
                        return jsonify(error="Your account has been temporarily restricted for 7 days because you received 5 harmful-language warnings."), 403
                    return jsonify(error=f"Warning {cached_count}/5: Your message contains potentially harmful language. Please use respectful language."), 400

            # Fallback if processing timed out: read user's current DB warning count
            user_doc = mongo.db.users.find_one({"_id": ObjectId(user_id)})
            curr_cnt = user_doc.get("warning_count", 1) if user_doc else 1
            if curr_cnt >= 5:
                return jsonify(error="Your account has been temporarily restricted for 7 days because you received 5 harmful-language warnings."), 403
            return jsonify(error=f"Warning {curr_cnt}/5: Your message contains potentially harmful language. Please use respectful language."), 400

        # This worker won the atomic reservation! Perform atomic increment on user in MongoDB
        updated_user = mongo.db.users.find_one_and_update(
            {"_id": ObjectId(user_id)},
            {"$inc": {"warning_count": 1}},
            return_document=ReturnDocument.AFTER
        )
        new_count = updated_user.get("warning_count", 1) if updated_user else ((user.warning_count or 0) + 1)
        user.warning_count = new_count

        if new_count >= 5:
            # 5th warning: Block for 7 days
            block_until = datetime.utcnow() + timedelta(days=7)
            user.block_expiry = block_until
            user._is_active = False
            mongo.db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"is_active": False, "block_expiry": block_until}}
            )
            mongo.db.warned_request_dedup.update_one(
                {"_id": req_hash},
                {"$set": {"status": "completed", "warning_count": 5, "updated_at": datetime.utcnow()}}
            )
            return jsonify(error="Your account has been temporarily restricted for 7 days because you received 5 harmful-language warnings."), 403

        mongo.db.warned_request_dedup.update_one(
            {"_id": req_hash},
            {"$set": {"status": "completed", "warning_count": new_count, "updated_at": datetime.utcnow()}}
        )
        return jsonify(error=f"Warning {new_count}/5: Your message contains potentially harmful language. Please use respectful language."), 400

    except Exception:
        # Fail safely: Prohibited content is ALWAYS blocked; do not leak exception or allow message through
        user_doc = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        cnt = user_doc.get("warning_count", 1) if user_doc else 1
        if cnt >= 5:
            return jsonify(error="Your account has been temporarily restricted for 7 days because you received 5 harmful-language warnings."), 403
        return jsonify(error=f"Warning {cnt}/5: Your message contains potentially harmful language. Please use respectful language."), 400


def _inbox(user_id):
    out = []
    from models.restriction import Restriction
    for item in Message.get_inbox(user_id):
        partner = UserModel.get_by_id(item["_id"])
        if partner and Match.are_matched(user_id, partner.id) and not Restriction.is_restricted_either(user_id, partner.id):
            is_online = OnlineTracker.is_online(partner.id)
            out.append({"partner": partner, "last_message": item["last_message"], "is_online": is_online})
    return out

@chat_bp.route("/chat")
@login_required
def chat_home():
    return render_template("chat.html", inbox=_inbox(current_user.id), messages=[], active_partner=None, current_user_id=current_user.id)

@chat_bp.route("/chat/<partner_id>")
@login_required
def chat_with(partner_id):
    from models.restriction import Restriction
    partner = UserModel.get_by_id(partner_id)
    if not partner or partner.id == current_user.id or not Match.are_matched(current_user.id, partner_id) or Restriction.is_restricted_either(current_user.id, partner_id):
        flash("You can only communicate with an active mutual match.", "warning")
        return redirect(url_for("main.dashboard"))
    Message.mark_read(partner_id, current_user.id)
    is_partner_online = OnlineTracker.is_online(partner_id)
    return render_template("chat.html", inbox=_inbox(current_user.id),
                           messages=Message.get_conversation(current_user.id, partner_id),
                           active_partner=partner, current_user_id=current_user.id,
                           is_partner_online=is_partner_online)

@chat_bp.route("/chat/send", methods=["POST"])
@login_required
def send_message():
    data=request.get_json(silent=True) or {}
    receiver_id=str(data.get("receiver_id",""))
    text=str(data.get("text","")).strip()
    receiver=UserModel.get_by_id(receiver_id)
    
    # Check if sender restriction has expired and lift if needed
    current_user.check_and_lift_restriction()

    # Check if sender is currently blocked from sending messages
    if current_user.block_expiry and datetime.utcnow() < current_user.block_expiry:
        return jsonify(error=f"You are temporarily blocked from sending messages until {current_user.block_expiry.strftime('%Y-%m-%d %H:%M:%S')} UTC."), 403
        
    if not text or not receiver_id: return jsonify(error="Message cannot be empty."),400
    from models.restriction import Restriction
    if not receiver or not Match.are_matched(current_user.id, receiver_id) or Restriction.is_restricted_either(current_user.id, receiver_id):
        return jsonify(error="You can only message a mutual match."),403
    if len(text)>2000: return jsonify(error="Message is too long (maximum 2000 characters)."),400
    
    result = moderator.analyze_message(text)

    # If message is ambiguous, log for audit/review without punishing the sender
    if result.get("action") == "flag_for_review":
        try:
            mongo.db.moderation_logs.insert_one({
                "sender_id": current_user.id,
                "receiver_id": receiver_id,
                "action": "flag_for_review",
                "scores": result.get("scores", {}),
                "context_classification": result.get("decision", {}).get("context_classification", "ambiguous"),
                "timestamp": datetime.utcnow()
            })
        except Exception:
            pass

    # Check if moderation policy requires a punitive warning/block
    if not result["is_safe"] or result.get("requires_warning", False):
        return _handle_warning_with_dedup(current_user, receiver_id, text)
        
    msg_id = Message.create(current_user.id, receiver_id, text)
    sender_avatar = getattr(current_user, "avatar_url", "/static/uploads/default.png")
    socketio.emit("new_message", {
        "message_id": str(msg_id) if msg_id else "",
        "sender_id": current_user.id,
        "receiver_id": receiver_id,
        "sender_name": current_user.name,
        "sender_avatar": sender_avatar,
        "text": text,
    }, to=f"user:{receiver_id}")
    return jsonify(status="sent", message_id=str(msg_id) if msg_id else "")

# ── Delete chat ────────────────────────────────────────────────────────────

@chat_bp.route("/chat/delete", methods=["POST"])
@login_required
def delete_chat():
    data = request.get_json(silent=True) or {}
    partner_id = str(data.get("partner_id", ""))
    if not partner_id:
        return jsonify(error="Missing partner_id."), 400
    deleted = Message.delete_conversation(current_user.id, partner_id)
    return jsonify(status="deleted", deleted_count=deleted)

# ── Part ways with a mutual match ─────────────────────────────────────────

@chat_bp.route("/chat/part-ways", methods=["POST"])
@login_required
def part_ways():
    data = request.get_json(silent=True) or {}
    partner_id = str(data.get("partner_id", "")).strip()

    if not partner_id:
        return jsonify(error="Missing partner_id."), 400

    if partner_id == str(current_user.id):
        return jsonify(error="Cannot part ways with yourself."), 400

    partner = UserModel.get_by_id(partner_id)
    if not partner:
        return jsonify(error="User not found."), 404

    # Verify that a mutual match currently exists
    if not Match.are_matched(current_user.id, partner_id):
        return jsonify(error="No active mutual match exists with this user."), 400

    # Safely remove mutual match relationship (does not create permanent block)
    Match.part_ways(current_user.id, partner_id)

    # Broadcast real-time match termination over Socket.IO to both users
    from routes.video_routes import call_room
    socketio.emit("match_ended", {"partner_id": str(partner_id)}, to=f"user:{current_user.id}")
    socketio.emit("match_ended", {"partner_id": str(current_user.id)}, to=f"user:{partner_id}")
    try:
        socketio.emit("call_terminated", {"reason": "connection_ended"}, to=call_room(current_user.id, partner_id))
    except Exception:
        pass

    return jsonify(status="ok", message="You have parted ways with this user.")

# ── Restrict user from chat ───────────────────────────────────────────────

@chat_bp.route("/chat/restrict", methods=["POST"])
@login_required
def restrict_user():
    data = request.get_json(silent=True) or {}
    partner_id = str(data.get("partner_id", "")).strip()
    reason = str(data.get("reason", "")).strip()

    if not partner_id:
        return jsonify(error="Missing partner_id."), 400

    if partner_id == str(current_user.id):
        return jsonify(error="You cannot restrict yourself."), 400

    from models.restriction import Restriction
    success, msg = Restriction.create(current_user.id, partner_id, reason=reason)
    if not success:
        return jsonify(error=msg), 400

    # Broadcast real-time match termination over Socket.IO to both users
    from routes.video_routes import call_room
    socketio.emit("match_ended", {"partner_id": str(partner_id)}, to=f"user:{current_user.id}")
    socketio.emit("match_ended", {"partner_id": str(current_user.id)}, to=f"user:{partner_id}")
    try:
        socketio.emit("call_terminated", {"reason": "connection_ended"}, to=call_room(current_user.id, partner_id))
    except Exception:
        pass

    return jsonify(status="ok", message="User restricted successfully.")

# ── Report user from chat ─────────────────────────────────────────────────

@chat_bp.route("/chat/report", methods=["POST"])
@login_required
def report_user():
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Report request received from user {current_user.id}")
    data = request.get_json(silent=True) or {}
    partner_id = str(data.get("partner_id", ""))
    reason = str(data.get("reason", "")).strip()
    if not partner_id:
        logger.warning(f"Validation failed: Missing partner_id for user {current_user.id}")
        return jsonify(error="Missing partner_id."), 400
    if ChatReport.already_reported(current_user.id, partner_id):
        logger.info(f"Duplicate report detected: User {current_user.id} already reported {partner_id}")
        return jsonify(error="Report already submitted."), 409

    try:
        reported_user = UserModel.get_by_id(partner_id)
        reported_name = reported_user.name if reported_user else "Unknown"
        report_reason = reason or "Reported from private chat"
        
        # Store in main reports collection
        mongo.db.reports.insert_one({
            "reporter_id": current_user.id,
            "reporter_name": current_user.name,
            "reported_id": partner_id,
            "reported_name": reported_name,
            "reason": report_reason,
            "source": "private_chat",
            "status": "pending",
            "created_at": datetime.utcnow(),
        })
    
        # Notify every admin via in-app notification + real-time Socket.IO
        _notify_admins_of_report(current_user.name, reported_name, report_reason)
        logger.info(f"Admin notification emitted for report by {current_user.id} on {partner_id}")
        
        return jsonify(status="reported", message="Report submitted successfully.")
    except Exception as e:
        logger.error(f"MongoDB/Server error while saving report: {e}")
        return jsonify(error="Unable to submit the report right now. Please try again."), 500


def _notify_admins_of_report(reporter_name, reported_name, reason):
    """Send a notification to all admin users about a new chat report."""
    try:
        from models.notification import Notification

        admins = list(mongo.db.users.find({"role": "admin"}, {"_id": 1}))
        title = "New Chat Report"
        message = f"{reporter_name} reported {reported_name}: {reason[:120]}"

        for admin in admins:
            admin_id = str(admin["_id"])
            # Persistent notification (shows in admin notification bell)
            Notification.create(
                user_id=admin_id,
                notification_type="chat_report",
                title=title,
                message=message,
            )
            # Real-time push (admin sees it instantly if online)
            socketio.emit("admin_report", {
                "title": title,
                "message": message,
            }, to=f"user:{admin_id}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to send admin notifications for report: {e}")

# ── Online status API ──────────────────────────────────────────────────────

@chat_bp.route("/api/user/online/<user_id>", methods=["GET"])
@login_required
def check_online(user_id):
    return jsonify(online=OnlineTracker.is_online(user_id))

# ── Socket.IO events ──────────────────────────────────────────────────────

@socketio.on("join_user_room")
def join_user_room(data):
    from flask_socketio import join_room
    from flask_login import current_user as socket_user
    if socket_user.is_authenticated:
        join_room(f"user:{socket_user.id}")


@chat_bp.route("/api/calls/pending", methods=["GET"])
@login_required
def pending_call():
    invitation = CallInvitation.pending_for(current_user.id)
    if not invitation:
        return jsonify(pending=False)
    caller = UserModel.get_by_id(invitation["caller_id"])
    if not caller or not Match.are_matched(current_user.id, invitation["caller_id"]):
        return jsonify(pending=False)
    return jsonify(
        pending=True,
        call_id=invitation["call_id"],
        caller_id=invitation["caller_id"],
        caller_name=caller.name,
        caller_avatar=getattr(caller, "avatar_url", "/static/uploads/default.png"),
        expires_at=invitation["expires_at"],
    )


@socketio.on("connect")
def authenticated_socket_connect(auth=None):
    """Authenticate every Socket.IO connection, join private room, set online."""
    from flask_login import current_user as socket_user
    from flask_socketio import join_room
    if not socket_user.is_authenticated:
        return False
    join_room(f"user:{socket_user.id}")
    OnlineTracker.set_online(socket_user.id)

@socketio.on("disconnect")
def on_disconnect(reason=None):
    """Mark user offline on Socket.IO disconnect."""
    from flask_login import current_user as socket_user
    if socket_user.is_authenticated:
        OnlineTracker.set_offline(socket_user.id)
