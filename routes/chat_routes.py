from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from models.message import Message, ChatReport, OnlineTracker
from models.user import UserModel
from models.match import Match
from models.call_invitation import CallInvitation
from ai_modules.nlp_safety import NLPModerator
from extensions import socketio, mongo
from datetime import datetime

chat_bp = Blueprint("chat", __name__)
moderator = NLPModerator()

def _inbox(user_id):
    out=[]
    for item in Message.get_inbox(user_id):
        partner=UserModel.get_by_id(item["_id"])
        if partner: 
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
    partner=UserModel.get_by_id(partner_id)
    if not partner or partner.id == current_user.id or not Match.are_matched(current_user.id, partner_id):
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
    
    # Check if sender is currently blocked from sending messages
    if current_user.block_expiry and datetime.utcnow() < current_user.block_expiry:
        return jsonify(error=f"You are temporarily blocked from sending messages until {current_user.block_expiry.strftime('%Y-%m-%d %H:%M:%S')} UTC."), 403
        
    if not text or not receiver_id: return jsonify(error="Message cannot be empty."),400
    if not receiver or not Match.are_matched(current_user.id, receiver_id):
        return jsonify(error="You can only message a mutual match."),403
    if len(text)>2000: return jsonify(error="Message is too long (maximum 2000 characters)."),400
    
    result=moderator.analyze_message(text)
    if not result["is_safe"]: 
        from extensions import mongo
        from bson import ObjectId
        from datetime import timedelta
        
        # Increment warning count
        current_warning_count = current_user.warning_count if current_user.warning_count is not None else 0
        new_count = current_warning_count + 1
        current_user.warning_count = new_count
        
        updates = {"warning_count": new_count}
        
        if new_count >= 5:
            # 5th warning: Block for 7 days
            block_until = datetime.utcnow() + timedelta(days=7)
            updates["is_active"] = False
            updates["block_expiry"] = block_until
            mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": updates})
            return jsonify(error="Your account has been temporarily restricted for 7 days because you received 5 harmful-language warnings."), 403
        
        mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": updates})
        return jsonify(error=f"Warning {new_count}/5: Your message contains potentially harmful language. Please use respectful language."), 400
        
    Message.create(current_user.id, receiver_id, text)
    socketio.emit("new_message", {"sender_id":current_user.id,"receiver_id":receiver_id,
                                   "sender_name":current_user.name,"text":text},
                  room=f"user:{receiver_id}")
    return jsonify(status="sent")

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
            }, room=f"user:{admin_id}")
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
