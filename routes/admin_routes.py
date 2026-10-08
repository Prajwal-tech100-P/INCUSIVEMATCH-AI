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
    for report in reports:
        dt = report.get("created_at")
        if isinstance(dt, datetime):
            ist_dt = dt + timedelta(hours=5, minutes=30)
            report["formatted_date"] = ist_dt.strftime("%d %b %Y, %I:%M %p")
        else:
            report["formatted_date"] = str(dt) if dt else "N/A"
    return render_template("admin/reports.html", reports=reports)


@admin_bp.route("/reports/<report_id>/status", methods=["POST"])
@admin_required
def update_report_status(report_id):
    new_status = request.form.get("status")
    action = request.form.get("action_taken")

    if new_status == "resolved" and not action:
        flash("You must select an action when resolving a report.", "danger")
        return redirect(url_for("admin.reports"))

    if new_status in ["pending", "reviewed", "resolved", "dismissed"]:
        try:
            report = mongo.db.reports.find_one({"_id": ObjectId(report_id)})
            if not report:
                flash("Report not found.", "danger")
                return redirect(url_for("admin.reports"))

            update_data = {"status": new_status}
            if new_status == "resolved":
                update_data["action_taken"] = action
                update_data["resolved_at"] = datetime.utcnow()

                reported_id = report.get("reported_id")
                if reported_id:
                    if action == "User Blocked for 1 Week":
                        mongo.db.users.update_one(
                            {"_id": ObjectId(reported_id)},
                            {"$set": {
                                "is_active": False,
                                "block_expiry": datetime.utcnow() + timedelta(days=7)
                            }}
                        )
                    elif action == "Warning Issued":
                        mongo.db.users.update_one(
                            {"_id": ObjectId(reported_id)},
                            {"$inc": {"warning_count": 1}}
                        )
                    elif action == "Message Removed":
                        msg_id = report.get("message_id")
                        if msg_id:
                            mongo.db.messages.delete_one({"_id": ObjectId(msg_id)})
                        else:
                            reporter_id = report.get("reporter_id")
                            if reporter_id:
                                last_msg = mongo.db.messages.find_one(
                                    {"sender_id": reported_id, "receiver_id": reporter_id},
                                    sort=[("timestamp", -1)]
                                )
                                if last_msg:
                                    mongo.db.messages.delete_one({"_id": last_msg["_id"]})

            mongo.db.reports.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": update_data}
            )
            flash("Report status updated.", "success")
        except Exception:
            flash("Invalid report ID or database error.", "danger")
    return redirect(url_for("admin.reports"))


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
        maintenance = request.form.get("maintenance_mode") == "on"
        try:
            mongo.db.app_settings.update_one(
                {"_id": "app_settings"},
                {"$set": {"maintenance_mode": maintenance, "updated_at": datetime.utcnow()}},
                upsert=True,
            )
            if maintenance:
                flash("Maintenance mode enabled.", "success")
            else:
                flash("Maintenance mode disabled.", "success")
        except Exception as e:
            import logging
            logging.error(f"Error updating maintenance mode: {e}")
            flash("Unable to update maintenance mode. Please try again.", "danger")
            
        return redirect(url_for("admin.settings"))
        
    try:
        settings_data = mongo.db.app_settings.find_one({"_id": "app_settings"}) or {"maintenance_mode": False}
    except Exception:
        settings_data = {"maintenance_mode": False}
        
    return render_template("admin/settings.html", settings=settings_data)


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
