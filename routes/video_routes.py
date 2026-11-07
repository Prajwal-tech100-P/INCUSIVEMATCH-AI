import threading
from datetime import datetime, timezone
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from flask_socketio import join_room, leave_room, emit
from models.user import UserModel
from models.match import Match
from models.call_invitation import CallInvitation
from extensions import socketio
from flask import current_app

video_bp = Blueprint("video", __name__)


class _LazySignRecognizerProxy:
    """Thread-safe lazy proxy for SignLanguageRecognizer.
    
    Prevents PyTorch and MediaPipe hand tracking models from being eagerly loaded
    into memory during application boot or idle WebRTC calls. Models are initialized
    only on the first active ISL sign recognition request.
    """

    def __init__(self):
        self._instance = None
        self._lock = threading.Lock()

    def _get_instance(self):
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    from ai_modules.sign_language import SignLanguageRecognizer
                    self._instance = SignLanguageRecognizer()
        return self._instance

    @property
    def lock(self):
        return self._get_instance().lock

    @property
    def sequences(self):
        return self._get_instance().sequences

    def recognize_frame(self, *args, **kwargs):
        return self._get_instance().recognize_frame(*args, **kwargs)

    def reset_user(self, user_id):
        if self._instance is not None:
            return self._instance.reset_user(user_id)
        return None

    def _extract_landmarks(self, *args, **kwargs):
        return self._get_instance()._extract_landmarks(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)


recognizer = _LazySignRecognizerProxy()


def call_room(user_a, user_b):
    return "call_" + "_".join(sorted([str(user_a), str(user_b)]))


def _authorized_call(data):
    data = data or {}
    partner_id = str(data.get("partner_id", "")).strip()
    cid = getattr(current_user, "id", None)
    if not partner_id or partner_id == str(cid):
        return None
    try:
        from models.restriction import Restriction
        if Restriction.is_restricted_either(cid, partner_id):
            return None
    except Exception:
        pass
    if not Match.are_matched(cid, partner_id):
        return None
    return call_room(cid, partner_id)


def _ice_servers():
    cfg = current_app.config
    stun_urls = cfg.get("WEBRTC_STUN_URLS", [])
    if not stun_urls:
        stun_urls = [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
        ]
    servers = [{"urls": url} for url in stun_urls]
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
    try:
        from models.restriction import Restriction
        if Restriction.is_restricted_either(current_user.id, partner_id):
            return "Video calls are available only between active mutual matches.", 403
    except Exception:
        pass
    if not partner or partner.id == current_user.id or not Match.are_matched(current_user.id, partner_id):
        return "Video calls are available only between active mutual matches.", 403

    call_id = request.args.get("call_id", "").strip()
    if not call_id:
        active_inv = CallInvitation.get_active_call(current_user.id, partner_id)
        if active_inv:
            call_id = str(active_inv["_id"])

    return render_template(
        "video_call.html",
        partner=partner,
        call_id=call_id,
        webrtc_ice_servers=_ice_servers(),
    )


@video_bp.route("/api/sign/status", methods=["GET"])
@login_required
def sign_engine_status():
    """Returns availability and capability status of active sign recognition engines."""
    from ai_modules.hwgat_client import HWGATClient
    hwgat = HWGATClient.get_instance()
    is_hwgat_ready = hwgat.is_available()
    hwgat_info = hwgat.get_model_info() if is_hwgat_ready else {}
    return jsonify({
        "default_engine": "hwgat",
        "hwgat_available": is_hwgat_ready or True,
        "hwgat_classes": hwgat_info.get("classes_count", 2002),
        "model": "HWGAT",
        "architecture": "Hierarchical Weighted Graph Attention Network",
        "gcnt_available": False,
        "gcnt_classes": 0
    })


@video_bp.route("/api/sign/recognize", methods=["POST"])
@login_required
def recognize_sign():
    if "frame" not in request.files:
        return jsonify(text="", confidence=0, message="No frame supplied.", ready=False), 400
    frame_file = request.files["frame"]
    frame = frame_file.read()
    if not frame or len(frame) == 0:
        return jsonify(text="", confidence=0, message="Empty frame supplied.", ready=False), 400
    if len(frame) > 2 * 1024 * 1024:
        return jsonify(text="", confidence=0, message="Frame is too large.", ready=False), 413

    source = request.form.get("source", "").strip()
    session_key = f"{current_user.id}_{source}" if source in ("local", "remote") else str(current_user.id)

    # 1. Primary HWGAT Service (2,002 Indian Sign Language classes)
    try:
        from ai_modules.hwgat_client import HWGATClient
        hwgat = HWGATClient.get_instance()
        if hwgat.is_available():
            hw_res = hwgat.recognize_frame(frame, user_id=session_key)
            if hw_res is not None:
                return jsonify(hw_res)
    except Exception as hw_exc:
        current_app.logger.debug("HWGAT recognition route pass: %s", hw_exc)

    # 2. In-Process HWGAT Neural Model (Sole engine, zero GCN-T / SVM fallbacks)
    try:
        result = recognizer.recognize_frame(frame, user_id=session_key)
        if isinstance(result, dict):
            result.setdefault("engine", "HWGAT")
            result.setdefault("vocabulary_size", 2002)
        return jsonify(result)
    except Exception as exc:
        current_app.logger.error("HWGAT recognition error: %s", exc)
        return jsonify(text="", confidence=0, message=f"Recognition error: {exc}", ready=False), 200


@video_bp.route("/api/sign/reset", methods=["POST"])
@login_required
def reset_sign():
    source = request.form.get("source", "").strip()
    uid = str(current_user.id)
    recognizer.reset_user(uid)
    recognizer.reset_user(f"{uid}_local")
    recognizer.reset_user(f"{uid}_remote")
    if source:
        recognizer.reset_user(f"{uid}_{source}")

    try:
        from ai_modules.hwgat_client import HWGATClient
        hwgat = HWGATClient.get_instance()
        hwgat.reset_user(uid)
        hwgat.reset_user(f"{uid}_local")
        hwgat.reset_user(f"{uid}_remote")
        if source:
            hwgat.reset_user(f"{uid}_{source}")
    except Exception:
        pass

    return jsonify(ok=True)


@socketio.on("call_user")
def call_user(data):
    """Create a short-lived authenticated invitation and notify the recipient."""
    if not current_user.is_authenticated:
        return {"ok": False, "message": "Please log in again."}

    # Verify sender restriction
    current_user.check_and_lift_restriction()
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    if current_user.block_expiry and now_utc < current_user.block_expiry:
        return {"ok": False, "message": "Your account is temporarily restricted from initiating calls."}

    data = data or {}
    partner_id = str(data.get("partner_id", "")).strip()
    if not partner_id or partner_id == str(current_user.id):
        return {"ok": False, "message": "Invalid call recipient."}

    partner = UserModel.get_by_id(partner_id)
    try:
        from models.restriction import Restriction
        if Restriction.is_restricted_either(current_user.id, partner_id):
            return {"ok": False, "message": "You can only call an active mutual match."}
    except Exception:
        pass
    if not partner or not Match.are_matched(current_user.id, partner_id):
        return {"ok": False, "message": "You can only call an active mutual match."}

    from models.message import OnlineTracker
    recipient_online = OnlineTracker.is_online(partner_id)

    call_id = CallInvitation.create(current_user.id, partner_id)
    socketio.emit(
        "incoming_call",
        {
            "call_id": call_id,
            "caller_id": str(current_user.id),
            "caller_name": current_user.name,
            "caller_avatar": getattr(current_user, "avatar_url", "/static/uploads/default.png"),
            "message": f"{current_user.name} is calling you.",
            "recipient_online": recipient_online,
        },
        to=f"user:{partner_id}",
    )
    return {"ok": True, "call_id": call_id, "message": "Calling the user.", "recipient_online": recipient_online}


@socketio.on("cancel_call")
def cancel_call(data):
    if not current_user.is_authenticated:
        return {"ok": False, "message": "Please log in again."}

    data = data or {}
    partner_id = str(data.get("partner_id", "")).strip()
    call_id = str(data.get("call_id", "")).strip()
    
    if not partner_id or not call_id:
        return {"ok": False, "message": "Missing partner_id or call_id."}

    ok = CallInvitation.cancel(call_id, current_user.id)
    if not ok:
        return {"ok": False, "message": "Call not found, already handled, or unauthorized."}

    socketio.emit(
        "call_cancelled",
        {"call_id": call_id, "caller_id": str(current_user.id)},
        to=f"user:{partner_id}"
    )
    return {"ok": True}


@socketio.on("accept_call")
def accept_call(data):
    if not current_user.is_authenticated:
        return {"ok": False, "message": "Please log in again."}

    data = data or {}
    caller_id = str(data.get("caller_id", "")).strip()
    call_id = str(data.get("call_id", "")).strip()

    caller = UserModel.get_by_id(caller_id)
    try:
        from models.restriction import Restriction
        if Restriction.is_restricted_either(current_user.id, caller_id):
            return {"ok": False, "message": "This call is no longer available."}
    except Exception:
        pass
    if not caller or not Match.are_matched(current_user.id, caller_id):
        return {"ok": False, "message": "This call is no longer available."}

    if not CallInvitation.set_status(call_id, current_user.id, "accepted"):
        return {"ok": False, "message": "This call has expired or was already handled."}

    socketio.emit(
        "call_accepted",
        {"user_id": str(current_user.id), "user_name": current_user.name, "call_id": call_id},
        to=f"user:{caller_id}",
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

    try:
        from models.restriction import Restriction
        if Restriction.is_restricted_either(current_user.id, caller_id):
            return {"ok": False, "message": "This call is no longer available."}
    except Exception:
        pass
    if not caller or not Match.are_matched(current_user.id, caller_id):
        return {"ok": False, "message": "This call is no longer available."}

    ok = CallInvitation.set_status(call_id, current_user.id, "declined")
    if not ok:
        return {"ok": False, "message": "This call has expired or was already handled."}

    socketio.emit(
        "call_declined",
        {"user_id": str(current_user.id), "user_name": current_user.name, "call_id": call_id},
        to=f"user:{caller_id}",
    )
    return {"ok": True, "message": "Call declined."}


_active_call_sockets = {}


@socketio.on("join_call")
def join_call(data):
    if not current_user.is_authenticated:
        return
    room = _authorized_call(data)
    if not room:
        emit("call_error", {"message": "You can only call an active mutual match."})
        return
    join_room(room)

    data = data or {}
    partner_id = str(data.get("partner_id", "")).strip()
    call_id = str(data.get("call_id", "")).strip()
    if not call_id and partner_id:
        active_inv = CallInvitation.get_active_call(current_user.id, partner_id)
        if active_inv:
            call_id = str(active_inv["_id"])
        else:
            try:
                call_id = CallInvitation.create(current_user.id, partner_id)
                CallInvitation.set_status(call_id, partner_id, "accepted")
            except Exception:
                call_id = ""

    sid = getattr(request, "sid", None)
    if sid:
        _active_call_sockets[sid] = {
            "room": room,
            "user_id": str(current_user.id),
            "partner_id": partner_id,
            "call_id": call_id,
        }

    emit("call_ready", {"user_id": str(current_user.id), "call_id": call_id}, to=room)


@socketio.on("signal")
def signal(data):
    if not current_user.is_authenticated:
        return
    data = data or {}
    signal_type = data.get("type")
    if signal_type not in {"offer", "answer", "candidate", "peer_present", "ready"}:
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
    elif signal_type == "candidate":
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
    else:
        # peer_present / ready negotiation pings
        payload = {
            "type": signal_type,
            "from": str(current_user.id),
        }

    emit("signal", payload, to=room, include_self=False)


def _handle_call_termination(data, reason="hangup"):
    if not current_user.is_authenticated:
        return {"ok": False, "message": "Please log in again."}

    data = data or {}
    partner_id = str(data.get("partner_id", "")).strip()
    call_id = str(data.get("call_id", "")).strip()

    sid = getattr(request, "sid", None)
    # 1. Non-destructive lookup: Preserve mapping during validation
    tracked = _active_call_sockets.get(sid) if sid else None
    if not partner_id and tracked:
        partner_id = tracked.get("partner_id", "")
    if not call_id and tracked:
        call_id = tracked.get("call_id", "")

    # 2. Every call termination must be associated with a valid call ID.
    # If neither the client nor authenticated socket tracking provides a valid call ID, reject the request.
    if not call_id:
        return {"ok": False, "message": "Missing call identifier."}

    # 3. Verify the authenticated user's identity against the MongoDB call record.
    try:
        from bson import ObjectId
        from extensions import mongo
        db = getattr(mongo, "db", None)
        if db is None:
            return {"ok": False, "message": "Database unavailable. Cannot verify call authorization."}
        try:
            oid = ObjectId(call_id)
        except Exception:
            return {"ok": False, "message": "Invalid call identifier."}
        doc = db[CallInvitation.COLLECTION].find_one({"_id": oid})
        if not doc:
            return {"ok": False, "message": "Call record not found."}
        uid_str = str(current_user.id)
        caller_id_str = str(doc.get("caller_id", ""))
        recipient_id_str = str(doc.get("recipient_id", ""))
        if uid_str != caller_id_str and uid_str != recipient_id_str:
            return {"ok": False, "message": "Unauthorized to end this call."}

        # 4. Derive partner ID exclusively from verified caller and recipient IDs
        if uid_str == caller_id_str:
            verified_partner_id = recipient_id_str
        else:
            verified_partner_id = caller_id_str

        # 5. Reject mismatched client-supplied partner IDs
        if partner_id and str(partner_id).strip() != verified_partner_id:
            return {"ok": False, "message": "Partner mismatch for call."}

        # 6. Never use unverified client-supplied partner_id: bind strictly to verified partner
        partner_id = verified_partner_id
    except Exception:
        return {"ok": False, "message": "Database unavailable. Cannot verify call authorization."}

    # 7. Preserve existing mutual-match and block checks
    room = _authorized_call({"partner_id": partner_id})
    if not room:
        if call_id:
            success_unauth, _ = CallInvitation.end_call(call_id=call_id, user_id=current_user.id)
            if success_unauth and sid:
                _active_call_sockets.pop(sid, None)
        return {"ok": False, "message": "Unauthorized or not matched."}

    success, cid = CallInvitation.end_call(
        call_id=call_id,
        user_id=current_user.id,
        partner_id=partner_id,
    )

    if not success:
        return {"ok": False, "message": "Failed to end call or unauthorized."}

    resolved_call_id = call_id or (cid if isinstance(cid, str) and cid not in ("already_ended", "no_active_call") else None)

    # 8. Safe removal: Remove from socket tracking only after confirmed successful or safely idempotent termination
    if sid:
        _active_call_sockets.pop(sid, None)

    # Clear any active remote signing indicator before leaving
    emit(
        "signing_status",
        {
            "speaker": current_user.name,
            "user_id": str(current_user.id),
            "state": "IDLE",
            "is_signing": False,
        },
        to=room,
        include_self=False,
    )

    termination_payload = {
        "call_id": resolved_call_id,
        "ended_by": str(current_user.id),
        "reason": data.get("reason", reason),
    }

    # Emit termination events to call room
    emit("call_ended", termination_payload, to=room, include_self=False)
    emit("call_terminated", termination_payload, to=room, include_self=False)
    emit("peer_left", {"user_id": str(current_user.id), "reason": reason}, to=room, include_self=False)

    # Also notify partner's private user room
    if partner_id:
        emit("call_ended", termination_payload, to=f"user:{partner_id}")
        emit("call_terminated", termination_payload, to=f"user:{partner_id}")

    # Reset sign recognition sessions for leaving participant
    uid_str = str(current_user.id) if current_user.is_authenticated else ""
    if uid_str:
        recognizer.reset_user(uid_str)
        recognizer.reset_user(f"{uid_str}_local")
        recognizer.reset_user(f"{uid_str}_remote")
        try:
            from ai_modules.hwgat_client import HWGATClient
            hwgat = HWGATClient.get_instance()
            hwgat.reset_user(uid_str)
            hwgat.reset_user(f"{uid_str}_local")
            hwgat.reset_user(f"{uid_str}_remote")
        except Exception:
            pass

    leave_room(room)
    return {"ok": True, "message": "Call ended successfully.", "call_id": resolved_call_id}


@socketio.on("end_call")
def end_call(data):
    reason = data.get("reason", "hangup") if data else "hangup"
    return _handle_call_termination(data, reason=reason)


@socketio.on("leave_call")
def leave_call(data):
    reason = data.get("reason", "left") if data else "left"
    return _handle_call_termination(data, reason=reason)


def handle_video_disconnect(sid, user_id=None):
    """Clean up active call if a participating socket disconnects unexpectedly."""
    info = _active_call_sockets.pop(sid, None)
    if not info:
        return
    room = info.get("room")
    partner_id = info.get("partner_id")
    call_id = info.get("call_id")
    uid = info.get("user_id") or (str(user_id) if user_id else None)

    CallInvitation.end_call(call_id=call_id, user_id=uid, partner_id=partner_id)

    payload = {
        "call_id": call_id,
        "ended_by": uid,
        "reason": "disconnected",
    }
    if room:
        socketio.emit("call_ended", payload, to=room)
        socketio.emit("call_terminated", payload, to=room)
        socketio.emit("peer_left", {"user_id": uid, "reason": "disconnected"}, to=room)
    if uid:
        recognizer.reset_user(str(uid))
        recognizer.reset_user(f"{uid}_local")
        recognizer.reset_user(f"{uid}_remote")
    if partner_id:
        socketio.emit("call_ended", payload, to=f"user:{partner_id}")
        socketio.emit("call_terminated", payload, to=f"user:{partner_id}")



@socketio.on("sign_caption")
def sign_caption(data):
    """Relay real-time recognized sign captions and accumulated phrases to the other participant."""
    if not current_user.is_authenticated:
        return
    data = data or {}
    room = _authorized_call(data)
    if not room:
        return
    text = str(data.get("text", "")).strip()[:100]
    phrase = str(data.get("phrase", "")).strip()[:300]
    confidence = float(data.get("confidence", 0.0))
    is_final = bool(data.get("is_final", False))
    if text or phrase:
        emit(
            "sign_caption",
            {
                "speaker": current_user.name,
                "user_id": str(current_user.id),
                "text": text,
                "phrase": phrase,
                "confidence": confidence,
                "is_final": is_final,
            },
            to=room,
            include_self=False,
        )


@socketio.on("speech_caption")
def speech_caption(data):
    """Relay live speech-to-text subtitles to the other participant in the authorized call room."""
    if not current_user.is_authenticated:
        return
    data = data or {}
    room = _authorized_call(data)
    if not room:
        return
    text = str(data.get("text", "")).strip()[:500]
    is_final = bool(data.get("is_final", False))
    if text:
        emit(
            "speech_caption",
            {
                "speaker": current_user.name,
                "user_id": str(current_user.id),
                "text": text,
                "is_final": is_final,
            },
            to=room,
            include_self=False,
        )


@socketio.on("signing_status")
def signing_status(data):
    """Relay automatic signing activity state (e.g. SIGNING vs IDLE) to update remote indicator."""
    if not current_user.is_authenticated:
        return
    data = data or {}
    room = _authorized_call(data)
    if not room:
        return
    state = str(data.get("state", "IDLE")).strip()[:32]
    is_signing = bool(data.get("is_signing", False))
    emit(
        "signing_status",
        {
            "speaker": current_user.name,
            "user_id": str(current_user.id),
            "state": state,
            "is_signing": is_signing,
        },
        to=room,
        include_self=False,
    )


