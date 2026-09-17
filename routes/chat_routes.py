from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from models.message import Message, ChatReport, OnlineTracker
from models.user import UserModel
from models.match import Match
from models.call_invitation import CallInvitation
from ai_modules.nlp_safety import NLPModerator
from extensions import socketio
from datetime import datetime

chat_bp = Blueprint("chat", __name__)
moderator = NLPModerator()

def _inbox(user_id):
    out=[]
    for item in Message.get_inbox(user_id):
        partner=UserModel.get_by_id(item["_id"])
        if partner: out.append({"partner": partner, "last_message": item["last_message"]})
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
    if not text or not receiver_id: return jsonify(error="Message cannot be empty."),400
    if not receiver or not Match.are_matched(current_user.id, receiver_id):
        return jsonify(error="You can only message a mutual match."),403
    if len(text)>2000: return jsonify(error="Message is too long (maximum 2000 characters)."),400
    result=moderator.analyze_message(text)
    if not result["is_safe"]: return jsonify(error=f"Message blocked: {result['reason']}"),400
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
    data = request.get_json(silent=True) or {}
    partner_id = str(data.get("partner_id", ""))
    reason = str(data.get("reason", "")).strip()
    if not partner_id:
        return jsonify(error="Missing partner_id."), 400
    if ChatReport.already_reported(current_user.id, partner_id):
        return jsonify(error="You have already reported this user. We are reviewing it."), 409

    # Store in chat_reports collection
    ChatReport.create(current_user.id, partner_id, reason)

    # Also mirror to the admin `reports` collection (so it shows on /admin/reports)
    reported_user = UserModel.get_by_id(partner_id)
    reported_name = reported_user.name if reported_user else "Unknown"
    report_reason = reason or "Reported from chat"
    mongo.db.reports.insert_one({
        "reporter_id": current_user.id,
        "reporter_name": current_user.name,
        "reported_id": partner_id,
        "reported_name": reported_name,
        "reason": report_reason,
        "source": "chat",
        "status": "pending",
        "created_at": datetime.utcnow(),
    })

    # Notify every admin via in-app notification + real-time Socket.IO
    _notify_admins_of_report(current_user.name, reported_name, report_reason)

    return jsonify(status="reported", message="Report submitted. Our team will review it.")


def _notify_admins_of_report(reporter_name, reported_name, reason):
    """Send a notification to all admin users about a new chat report."""
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
    # Broadcast online status to all connected clients
    socketio.emit("user_status", {"user_id": socket_user.id, "online": True}, broadcast=True)


@socketio.on("disconnect")
def on_disconnect():
    """Mark user offline on Socket.IO disconnect."""
    from flask_login import current_user as socket_user
    if socket_user.is_authenticated:
        OnlineTracker.set_offline(socket_user.id)
        socketio.emit("user_status", {"user_id": socket_user.id, "online": False}, broadcast=True)
