from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from models.community import Community, MAX_MEMBERS
from models.user import UserModel
from ai_modules.nlp_safety import NLPModerator
from extensions import socketio
from routes.video_routes import _ice_servers
from datetime import datetime

community_bp = Blueprint("community", __name__)
moderator = NLPModerator()

def _community_or_404(community_id):
    community = Community.get(community_id)
    if not community:
        return None
    return community

@community_bp.route("/community")
@login_required
def home():
    return render_template(
        "community.html",
        my_communities=Community.my_communities(current_user.id),
        available=Community.list_available(current_user.id),
        max_members=MAX_MEMBERS
    )

@community_bp.route("/community/create", methods=["POST"])
@login_required
def create():
    name = str(request.form.get("name", "")).strip()
    description = str(request.form.get("description", "")).strip()
    category = str(request.form.get("category", "General")).strip()
    icon = str(request.form.get("icon", "🤟")).strip()
    if not name:
        flash("Community name is required.", "warning")
        return redirect(url_for("community.home"))
    if len(name) > 60 or len(description) > 300:
        flash("Community name/description is too long.", "warning")
        return redirect(url_for("community.home"))
    cid = Community.create(current_user.id, name, description, category, icon)
    flash("Community created. You are the first member.", "success")
    return redirect(url_for("community.view", community_id=cid))

@community_bp.route("/community/<community_id>")
@login_required
def view(community_id):
    community = _community_or_404(community_id)
    if not community:
        flash("Community not found.", "danger")
        return redirect(url_for("community.home"))
    is_member = Community.is_member(community_id, current_user.id)
    members = [UserModel.get_by_id(str(uid)) for uid in community.get("member_ids", [])]
    members = [m for m in members if m]
    return render_template(
        "community_chat.html",
        community=community,
        members=members,
        messages=Community.messages(community_id),
        is_member=is_member,
        is_owner=str(community.get("created_by")) == str(current_user.id),
        max_members=MAX_MEMBERS,
        current_user_id=current_user.id
    )

@community_bp.route("/community/<community_id>/join", methods=["POST"])
@login_required
def join(community_id):
    ok, msg = Community.join(community_id, current_user.id)
    flash(msg, "success" if ok else "warning")
    return redirect(url_for("community.view", community_id=community_id))

@community_bp.route("/community/<community_id>/leave", methods=["POST"])
@login_required
def leave(community_id):
    ok, msg = Community.leave(community_id, current_user.id)
    flash(msg, "success" if ok else "warning")
    return redirect(url_for("community.home"))

@community_bp.route("/community/<community_id>/delete", methods=["POST"])
@login_required
def delete(community_id):
    ok, msg = Community.delete(community_id, current_user.id)
    flash(msg, "success" if ok else "warning")
    return redirect(url_for("community.home"))

@community_bp.route("/community/<community_id>/report", methods=["POST"])
@login_required
def report(community_id):
    if not Community.is_member(community_id, current_user.id):
        return jsonify(error="Join the community first."), 403
    reported = str(request.form.get("reported_user_id", ""))
    reason = str(request.form.get("reason", ""))
    if not reported or reported == current_user.id:
        return jsonify(error="Invalid report target."), 400
    if not Community.is_member(community_id, reported):
        return jsonify(error="User is not a member of this community."), 400
    Community.report(community_id, current_user.id, reported, reason)
    flash("Report submitted to moderators.", "success")
    return redirect(url_for("community.view", community_id=community_id))

@community_bp.route("/community/<community_id>/block", methods=["POST"])
@login_required
def block(community_id):
    blocked = str(request.form.get("blocked_user_id", ""))
    if not Community.is_member(community_id, current_user.id) or not Community.is_member(community_id, blocked):
        return jsonify(error="Both users must be community members."), 403
    if blocked == current_user.id:
        return jsonify(error="You cannot block yourself."), 400
    Community.block_member(community_id, current_user.id, blocked)
    flash("Member blocked in this community.", "success")
    return redirect(url_for("community.view", community_id=community_id))

@socketio.on("join_community")
def join_community_socket(data):
    from flask_socketio import join_room
    from flask_login import current_user as socket_user
    if not socket_user.is_authenticated:
        return
    cid = str((data or {}).get("community_id", ""))
    if Community.is_member(cid, socket_user.id):
        join_room(f"community:{cid}")

@socketio.on("community_message")
def community_message(data):
    from flask_login import current_user as socket_user
    if not socket_user.is_authenticated:
        return
    data = data or {}
    cid = str(data.get("community_id", ""))
    text = str(data.get("text", "")).strip()
    if not cid or not text or len(text) > 2000:
        return
    community = Community.get(cid)
    if not community or not Community.is_member(cid, socket_user.id):
        return
    if Community.is_user_blocked(cid, socket_user.id):
        return
    result = moderator.analyze_message(text)
    if not result["is_safe"]:
        socket_user._community_last_error = result["reason"]
        socketio.emit("community_message_error", {"message": f"Message blocked: {result['reason']}"}, room=f"user:{socket_user.id}")
        return
    msg = Community.create_message(cid, socket_user.id, socket_user.name, text)
    stamp = msg["timestamp"].strftime("%I:%M %p")
    socketio.emit("community_message", {
        "community_id": cid,
        "sender_id": socket_user.id,
        "sender_name": socket_user.name,
        "text": text,
        "timestamp": stamp,
    }, room=f"community:{cid}")

@socketio.on("join_community_user_room")
def join_community_user_room(data):
    from flask_socketio import join_room
    from flask_login import current_user as socket_user
    if socket_user.is_authenticated:
        join_room(f"user:{socket_user.id}")


# --- Optional 5-person community video call signaling (WebRTC mesh) ---
_community_call_members = {}

@community_bp.route("/community/<community_id>/call")
@login_required
def community_call(community_id):
    community = Community.get(community_id)
    if not community or not Community.is_member(community_id, current_user.id):
        return "Join the community before starting a community call.", 403
    return render_template("community_video_call.html", community=community, max_members=MAX_MEMBERS, webrtc_ice_servers=_ice_servers())

@socketio.on("join_community_call")
def join_community_call(data):
    from flask_socketio import join_room
    from flask_login import current_user as socket_user
    if not socket_user.is_authenticated:
        return
    cid = str((data or {}).get("community_id", ""))
    if not Community.is_member(cid, socket_user.id):
        emit("community_call_error", {"message": "Only community members can join the call."})
        return
    room = f"community_call:{cid}"
    sid = getattr(request, "sid", None)
    members = _community_call_members.setdefault(room, {})
    if len(members) >= MAX_MEMBERS and str(socket_user.id) not in members:
        emit("community_call_error", {"message": "Community video call is full (maximum 5 participants)."})
        return
    existing = [{"user_id": uid, "name": name} for uid, name in members.values()]
    join_room(room)
    members[str(socket_user.id)] = (str(socket_user.id), socket_user.name)
    emit("community_call_members", {"members": existing})
    emit("community_call_user_joined", {"user_id": str(socket_user.id), "name": socket_user.name}, room=room, include_self=False)

@socketio.on("community_call_signal")
def community_call_signal(data):
    from flask_login import current_user as socket_user
    if not socket_user.is_authenticated:
        return
    cid = str((data or {}).get("community_id", ""))
    if not Community.is_member(cid, socket_user.id):
        return
    target = str((data or {}).get("target_id", ""))
    if target not in _community_call_members.get(f"community_call:{cid}", {}):
        return
    payload = {
        "from": str(socket_user.id),
        "from_name": socket_user.name,
        "type": data.get("type"),
        "sdp": data.get("sdp"),
        "candidate": data.get("candidate"),
    }
    # User-room delivery avoids requiring the browser to know Socket.IO sids.
    socketio.emit("community_call_signal", payload, room=f"user:{target}")

@socketio.on("community_call_user_room")
def community_call_user_room(data):
    from flask_socketio import join_room
    from flask_login import current_user as socket_user
    if socket_user.is_authenticated:
        join_room(f"user:{socket_user.id}")

@socketio.on("leave_community_call")
def leave_community_call(data):
    from flask_socketio import leave_room
    from flask_login import current_user as socket_user
    if not socket_user.is_authenticated:
        return
    cid = str((data or {}).get("community_id", ""))
    room = f"community_call:{cid}"
    members = _community_call_members.get(room, {})
    if str(socket_user.id) in members:
        members.pop(str(socket_user.id), None)
    leave_room(room)
    emit("community_call_user_left", {"user_id": str(socket_user.id)}, room=room)
    if not members:
        _community_call_members.pop(room, None)
