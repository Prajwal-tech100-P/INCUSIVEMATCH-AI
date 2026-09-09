from flask import Blueprint, render_template, redirect, url_for, jsonify
from flask_login import current_user, login_required
from models.match import Match
from models.notification import Notification

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    candidates = Match.get_discover_candidates(current_user.id, comm_pref=None)
    mutual_matches = Match.get_mutual_matches(current_user.id)
    notifications = Notification.get_for_user(current_user.id, limit=10)
    unread_notifications = Notification.unread_count(current_user.id)
    return render_template(
        'dashboard.html',
        candidates=candidates,
        mutual_matches=mutual_matches,
        notifications=notifications,
        unread_notifications=unread_notifications,
    )


@main_bp.route('/notifications/<notification_id>/read', methods=['POST'])
@login_required
def mark_notification_read(notification_id):
    ok = Notification.mark_read(notification_id, current_user.id)
    return jsonify(status="ok" if ok else "not_found")


@main_bp.route('/notifications/read-all', methods=['POST'])
@login_required
def mark_all_notifications_read():
    Notification.mark_all_read(current_user.id)
    return jsonify(status="ok")
