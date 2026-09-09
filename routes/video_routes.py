from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from flask_socketio import join_room, leave_room, emit
from models.user import UserModel
from models.match import Match
from models.call_invitation import CallInvitation
from ai_modules.sign_language import SignLanguageRecognizer
from extensions import socketio
from flask import current_app

video_bp = Blueprint("video", __name__)
recognizer = SignLanguageRecognizer()


def call_room(user_a, user_b):
    return "call_" + "_".join(sorted([str(user_a), str(user_b)]))


def _authorized_call(data):
    data = data or {}
    partner_id = str(data.get("partner_id", "")).strip()
    if not partner_id or partner_id == str(current_user.id):
        return None
    if not Match.are_matched(current_user.id, partner_id):
        return None
    return call_room(current_user.id, partner_id)


def _ice_servers():
    cfg = current_app.config
    servers = [{"urls": url} for url in cfg.get("WEBRTC_STUN_URLS", [])]
    turn_url = cfg.get("WEBRTC_TURN_URL")
    turn_user = cfg.get("WEBRTC_TURN_USERNAME")
    turn_credential = cfg.get("WEBRTC_TURN_CREDENTIAL")
    turn_tls = cfg.get("WEBRTC_TURN_TLS_URL")

    if turn_url and turn_user and turn_credential:
        urls = [turn_url]
        if turn_tls and turn_tls not in urls:
            urls.append(turn_tls)
        servers.append({
            "urls": urls,
            "username": turn_user,
            "credential": turn_credential,
        })
    return servers


@video_bp.route("/video/<partner_id>")
@login_required
def video_call(partner_id):
    partner = UserModel.get_by_id(partner_id)
    if not partner or partner.id == current_user.id or not Match.are_matched(current_user.id, partner_id):
        return "Video calls are available only between active mutual matches.", 403
    return render_template(
        "video_call.html",
        partner=partner,
        webrtc_ice_servers=_ice_servers(),
    )


@video_bp.route("/api/sign/recognize", methods=["POST"])
@login_required
def recognize_sign():
    if "frame" not in request.files:
        return jsonify(text="", confidence=0, message="No frame supplied."), 400
    frame = request.files["frame"].read()
    if len(frame) > 2 * 1024 * 1024:
        return jsonify(text="", confidence=0, message="Frame is too large."), 413
    result = recognizer.recognize_frame(frame, user_id=current_user.id)
    return jsonify(result)


@socketio.on("call_user")
def call_user(data):
    """Create a short-lived authenticated invitation and notify the recipient."""
    if not current_user.is_authenticated:
        return {"ok": False, "message": "Please log in again."}

    data = data or {}
    partner_id = str(data.get("partner_id", "")).strip()
    if not partner_id or partner_id == str(current_user.id):
        return {"ok": False, "message": "Invalid call recipient."}

    partner = UserModel.get_by_id(partner_id)
    if not partner or not Match.are_matched(current_user.id, partner_id):
        return {"ok": False, "message": "You can only call an active mutual match."}

    call_id = CallInvitation.create(current_user.id, partner_id)
    socketio.emit(
        "incoming_call",
        {
            "call_id": call_id,
            "caller_id": str(current_user.id),
            "caller_name": current_user.name,
            "message": f"{current_user.name} is calling you.",
        },
        room=f"user:{partner_id}",
    )
    return {"ok": True, "call_id": call_id, "message": "Calling the user."}


@socketio.on("accept_call")
def accept_call(data):
    if not current_user.is_authenticated:
        return {"ok": False, "message": "Please log in again."}

    data = data or {}
    caller_id = str(data.get("caller_id", "")).strip()
    call_id = str(data.get("call_id", "")).strip()

    caller = UserModel.get_by_id(caller_id)
    if not caller or not Match.are_matched(current_user.id, caller_id):
        return {"ok": False, "message": "This call is no longer available."}

    if not CallInvitation.set_status(call_id, current_user.id, "accepted"):
        return {"ok": False, "message": "This call has expired or was already handled."}

    socketio.emit(
        "call_accepted",
        {"user_id": str(current_user.id), "user_name": current_user.name, "call_id": call_id},
        room=f"user:{caller_id}",
    )
    return {"ok": True, "message": "Call accepted."}


@socketio.on("decline_call")
def decline_call(data):
    if not current_user.is_authenticated:
        return {"ok": False, "message": "Please log in again."}

    data = data or {}
    caller_id = str(data.get("caller_id", "")).strip()
    call_id = str(data.get("call_id", "")).strip()
    caller = UserModel.get_by_id(caller_id)

    if not caller or not Match.are_matched(current_user.id, caller_id):
        return {"ok": False, "message": "This call is no longer available."}

    ok = CallInvitation.set_status(call_id, current_user.id, "declined")
    if not ok:
        return {"ok": False, "message": "This call has expired or was already handled."}

    socketio.emit(
        "call_declined",
        {"user_id": str(current_user.id), "user_name": current_user.name, "call_id": call_id},
        room=f"user:{caller_id}",
    )
    return {"ok": True, "message": "Call declined."}


@socketio.on("join_call")
def join_call(data):
    if not current_user.is_authenticated:
        return
    room = _authorized_call(data)
    if not room:
        emit("call_error", {"message": "You can only call an active mutual match."})
        return
    join_room(room)
    emit("call_ready", {"user_id": str(current_user.id)}, room=room)


@socketio.on("signal")
def signal(data):
    if not current_user.is_authenticated:
        return
    data = data or {}
    signal_type = data.get("type")
    if signal_type not in {"offer", "answer", "candidate"}:
        return

    room = _authorized_call(data)
    if not room:
        return

    if signal_type in {"offer", "answer"}:
        sdp = data.get("sdp")
        if not isinstance(sdp, dict):
            return
        if not isinstance(sdp.get("sdp"), str) or len(sdp["sdp"]) > 100_000:
            return
        payload = {
            "type": signal_type,
            "sdp": sdp,
            "from": str(current_user.id),
        }
    else:
        candidate = data.get("candidate")
        if not isinstance(candidate, dict):
            return
        if len(str(candidate)) > 20_000:
            return
        payload = {
            "type": "candidate",
            "candidate": candidate,
            "from": str(current_user.id),
        }

    emit("signal", payload, room=room, include_self=False)


@socketio.on("leave_call")
def leave_call(data):
    if not current_user.is_authenticated:
        return
    room = _authorized_call(data)
    if room:
        leave_room(room)
        emit("peer_left", {"user_id": str(current_user.id)}, room=room, include_self=False)
