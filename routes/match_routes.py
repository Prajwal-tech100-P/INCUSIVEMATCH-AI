from flask import Blueprint, jsonify
from flask_login import login_required, current_user
from models.match import Match
from models.user import UserModel
from models.notification import Notification
from extensions import socketio

match_bp=Blueprint("match",__name__)


@match_bp.route("/like/<target_id>",methods=["POST"])
@login_required
def like(target_id):
    target=UserModel.get_by_id(target_id)
    if not target or target.role!="user" or not target.is_active or target.id==current_user.id:
        return jsonify(error="Invalid profile."),400

    # Only create a match notification when the relationship changes
    # from one-sided to mutual. This prevents duplicate notifications.
    was_mutual = Match.is_mutual(current_user.id, target_id)
    Match.like(current_user.id,target_id)
    mutual=Match.is_mutual(current_user.id,target_id)

    if mutual and not was_mutual:
        Notification.create(
            current_user.id,
            "match",
            "It's a match! 💕",
            f"You and {target.name} liked each other. You can now chat and call.",
            target.id,
        )
        Notification.create(
            target.id,
            "match",
            "It's a match! 💕",
            f"You and {current_user.name} liked each other. You can now chat and call.",
            current_user.id,
        )

        # Push the event to both users when they are online.
        payload = {
            "type": "match",
            "title": "It's a mutual match! 💕",
            "message": f"You matched with {target.name}!" if current_user.id != target.id else "You have a new mutual match!",
            "partner_id": target.id,
            "partner_name": target.name,
        }
        socketio.emit("match_notification", payload, room=f"user:{current_user.id}")
        socketio.emit(
            "match_notification",
            {
                **payload,
                "message": f"You matched with {current_user.name}!",
                "partner_id": current_user.id,
                "partner_name": current_user.name,
            },
            room=f"user:{target.id}",
        )

    return jsonify(status="liked",mutual=mutual)


@match_bp.route("/dislike/<target_id>",methods=["POST"])
@login_required
def dislike(target_id):
    target=UserModel.get_by_id(target_id)
    if not target or target.role!="user" or target.id==current_user.id:
        return jsonify(error="Invalid profile."),400
    Match.dislike(current_user.id,target_id)
    return jsonify(status="disliked")
