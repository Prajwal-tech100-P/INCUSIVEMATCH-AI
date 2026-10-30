from flask import Blueprint, render_template, redirect, url_for, jsonify, request
from flask_login import current_user, login_required
from models.match import Match
from models.notification import Notification
from routes.profile_routes import INTEREST_CATEGORIES

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    gender_filter = request.args.get('gender', '').strip()
    min_age_filter = request.args.get('min_age', type=int)
    max_age_filter = request.args.get('max_age', type=int)
    comm_filter = request.args.get('comm_pref', '').strip()
    online_filter = request.args.get('online', '').lower() in {'1', 'true', 'yes'}

    # Support multiple interest selections (checkboxes/chips)
    raw_interests = request.args.getlist('interests') or request.args.getlist('interest')
    if not raw_interests:
        single_val = request.args.get('interests') or request.args.get('interest', '')
        if single_val:
            raw_interests = [s.strip() for s in single_val.split(',') if s.strip()]

    selected_interests = []
    seen_interests = set()
    for item in raw_interests:
        val = str(item).strip()
        if val and val.lower() not in seen_interests:
            seen_interests.add(val.lower())
            selected_interests.append(val)

    # Determine default gender preference if no override filter provided
    user_mp = getattr(current_user, 'matching_preferences', {}) or {}
    default_genders = user_mp.get('preferred_genders') or ['Any']
    default_gender = default_genders[0] if default_genders else 'Any'
    default_min_age = user_mp.get('min_age', 18)
    default_max_age = user_mp.get('max_age', 99)

    candidates = Match.get_discover_candidates(
        user_id=current_user.id,
        comm_pref=comm_filter or None,
        filter_gender=gender_filter or None,
        filter_min_age=min_age_filter,
        filter_max_age=max_age_filter,
        filter_interest=selected_interests or None,
        online_only=online_filter,
        require_shared_interests=True
    )
    mutual_matches = Match.get_mutual_matches(current_user.id)
    notifications = Notification.get_for_user(current_user.id, limit=10)
    unread_notifications = Notification.unread_count(current_user.id)

    active_filters = {
        'gender': gender_filter if gender_filter else default_gender,
        'has_gender_override': bool(gender_filter),
        'min_age': min_age_filter if min_age_filter is not None else default_min_age,
        'max_age': max_age_filter if max_age_filter is not None else default_max_age,
        'comm_pref': comm_filter,
        'interest': selected_interests[0] if len(selected_interests) == 1 else (','.join(selected_interests) if selected_interests else ''),
        'interests': selected_interests,
        'online': online_filter
    }

    return render_template(
        'dashboard.html',
        candidates=candidates,
        mutual_matches=mutual_matches,
        notifications=notifications,
        unread_notifications=unread_notifications,
        interest_categories=INTEREST_CATEGORIES,
        active_filters=active_filters,
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


@main_bp.route('/maintenance')
def maintenance():
    return render_template('maintenance.html')
