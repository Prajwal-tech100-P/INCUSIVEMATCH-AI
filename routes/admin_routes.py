from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import current_user
from utils.auth import admin_required
from models.user import UserModel
from extensions import mongo
from bson import ObjectId
from datetime import datetime, timedelta

admin_bp=Blueprint("admin",__name__)


def _mutual_match_count():
    """Count each mutual pair once."""
    users = list(mongo.db.users.find(
        {"role": "user", "is_active": True},
        {"_id": 1, "liked": 1}
    ))
    liked_map = {str(u["_id"]): set(u.get("liked", [])) for u in users}
    pairs = set()
    for a, liked in liked_map.items():
        for b in liked:
            if b in liked_map and a in liked_map.get(b, set()):
                pairs.add(tuple(sorted((a, b))))
    return len(pairs)


@admin_bp.route("/")
@admin_required
def dashboard():
    stats={
        "user_count":mongo.db.users.count_documents({"role":"user"}),
        "admin_count":mongo.db.users.count_documents({"role":"admin"}),
        "active_count":mongo.db.users.count_documents({"is_active":True,"role":"user"}),
        "message_count":mongo.db.messages.count_documents({}),
        "match_count":_mutual_match_count(),
        "report_count":mongo.db.reports.count_documents({}),
        "unread_message_count":mongo.db.messages.count_documents({"is_read":False}),
    }
    return render_template("admin/dashboard.html",stats=stats)


@admin_bp.route("/users")
@admin_required
def users():
    return render_template("admin/users.html",users=UserModel.get_all_users())


@admin_bp.route("/matches")
@admin_required
def matches():
    # Build a clean list of mutual pairs for the admin view.
    users = list(mongo.db.users.find({"role": "user"}, {"name": 1, "email": 1, "liked": 1}))
    by_id = {str(u["_id"]): u for u in users}
    seen = set()
    match_rows = []
    for a in users:
        aid = str(a["_id"])
        for bid in a.get("liked", []):
            if bid not in by_id or aid not in set(by_id[bid].get("liked", [])):
                continue
            key = tuple(sorted((aid, bid)))
            if key in seen:
                continue
            seen.add(key)
            b = by_id[bid]
            match_rows.append({
                "user_a": a,
                "user_b": b,
            })
    return render_template("admin/matches.html", matches=match_rows)


@admin_bp.route("/reports")
@admin_required
def reports():
    reports = list(mongo.db.reports.find().sort("created_at", -1))
    return render_template("admin/reports.html", reports=reports)


@admin_bp.route("/messages")
@admin_required
def messages():
    messages = list(mongo.db.messages.find().sort("timestamp", -1).limit(200))
    user_ids = set()
    for m in messages:
        user_ids.add(str(m.get("sender_id", "")))
        user_ids.add(str(m.get("receiver_id", "")))
    users = {u.id: u for u in [UserModel.get_by_id(uid) for uid in user_ids] if u}
    return render_template("admin/messages.html", messages=messages, users=users)


@admin_bp.route("/statistics")
@admin_required
def statistics():
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    stats = {
        "user_count": mongo.db.users.count_documents({"role": "user"}),
        "active_count": mongo.db.users.count_documents({"role": "user", "is_active": True}),
        "match_count": _mutual_match_count(),
        "message_count": mongo.db.messages.count_documents({}),
        "message_count_7d": mongo.db.messages.count_documents({"timestamp": {"$gte": seven_days_ago}}),
        "report_count": mongo.db.reports.count_documents({}),
        "unread_notifications": mongo.db.notifications.count_documents({"is_read": False}),
    }
    return render_template("admin/statistics.html", stats=stats)


@admin_bp.route("/settings", methods=["GET", "POST"])
@admin_required
def settings():
    if request.method == "POST":
        # Keep this lightweight: settings are stored in a single admin config document.
        maintenance = request.form.get("maintenance_mode") == "on"
        mongo.db.app_settings.update_one(
            {"_id": "app_settings"},
            {"$set": {"maintenance_mode": maintenance, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
        flash("Admin settings updated.", "success")
        return redirect(url_for("admin.settings"))
    settings = mongo.db.app_settings.find_one({"_id": "app_settings"}) or {"maintenance_mode": False}
    return render_template("admin/settings.html", settings=settings)


@admin_bp.route("/users/<user_id>/toggle-active",methods=["POST"])
@admin_required
def toggle_active(user_id):
    if user_id==current_user.id:
        flash("You cannot disable your own account.","danger")
        return redirect(url_for("admin.users"))
    try: oid=ObjectId(user_id)
    except Exception:
        flash("Invalid user.","danger"); return redirect(url_for("admin.users"))
    u=mongo.db.users.find_one({"_id":oid})
    if not u: flash("User not found.","danger")
    else:
        mongo.db.users.update_one({"_id":oid},{"$set":{"is_active":not u.get("is_active",True)}})
        flash("User status updated.","success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<user_id>/delete",methods=["POST"])
@admin_required
def delete_user(user_id):
    if user_id==current_user.id:
        flash("You cannot delete your own account.","danger"); return redirect(url_for("admin.users"))
    UserModel.delete_user(user_id); flash("User deleted.","success"); return redirect(url_for("admin.users"))


@admin_bp.route("/users/<user_id>/toggle-role",methods=["POST"])
@admin_required
def toggle_role(user_id):
    if user_id==current_user.id:
        flash("You cannot change your own role.","danger"); return redirect(url_for("admin.users"))
    UserModel.toggle_role(user_id); flash("User role updated.","success"); return redirect(url_for("admin.users"))
