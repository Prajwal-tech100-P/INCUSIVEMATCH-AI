from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
from urllib.parse import urlparse
import re
import logging
from flask_login import login_user, logout_user, login_required, current_user
from models.user import UserModel
from models.password_reset import PasswordReset
from utils.email_service import send_password_reset_email

logger = logging.getLogger("AuthRoutes")
auth_bp = Blueprint('auth', __name__)
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_csrf():
    """Verify CSRF token if enabled in application configuration."""
    if current_app.config.get("WTF_CSRF_ENABLED", True) is False:
        return True
    form_csrf = request.form.get("csrf_token")
    session_csrf = session.get("_csrf_token")
    return bool(form_csrf and session_csrf and form_csrf == session_csrf)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        comm_pref = request.form.get('comm_pref', 'text')

        if comm_pref == 'sign':
            comm_pref = 'sign language'

        if not all([name, email, password, confirm_password]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('auth.register'))

        terms_consent = request.form.get('terms_consent')
        if not terms_consent:
            flash('You must read and agree to the Terms & Conditions and acknowledge the Privacy Policy to create an account.', 'danger')
            return redirect(url_for('auth.register'))

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('auth.register'))

        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'danger')
            return redirect(url_for('auth.register'))

        existing_user = UserModel.find_by_email(email)

        if existing_user:
            flash('That email address is already registered.', 'danger')
            return redirect(url_for('auth.register'))

        from datetime import datetime
        policy_consent = {
            "terms_version": "2026.1",
            "privacy_version": "2026.1",
            "accepted_at": datetime.utcnow(),
            "ip_address": request.headers.get("X-Forwarded-For", request.remote_addr),
            "user_agent": request.headers.get("User-Agent", "")[:250],
        }

        UserModel.create_user(
            email,
            password,
            name,
            comm_prefs=[comm_pref],
            policy_consent=policy_consent
        )

        flash('Account created! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = UserModel.find_by_email(email)

        if user and user.is_active and UserModel.verify_password(
            user.password,
            password
        ):
            login_user(user)
            flash(f'Welcome back, {user.name}!', 'success')

            next_page = request.args.get('next')

            if next_page:
                parsed = urlparse(next_page)

                if (
                    parsed.scheme
                    or parsed.netloc
                    or not next_page.startswith('/')
                ):
                    next_page = None

            if user.is_admin():
                return redirect(next_page or url_for('admin.dashboard'))
            return redirect(next_page or url_for('main.dashboard'))

        else:
            flash(
                'Invalid email/password or your account is disabled.',
                'danger'
            )

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))


@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Request a password reset link sent to the user's registered email address."""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        if not _validate_csrf():
            flash('Invalid or expired security token. Please try again.', 'danger')
            return redirect(url_for('auth.forgot_password'))

        email = request.form.get('email', '').strip().lower()
        if not email or not EMAIL_REGEX.match(email):
            flash('Please enter a valid email address.', 'danger')
            return redirect(url_for('auth.forgot_password'))

        raw_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        ip_address = raw_ip.split(",")[0].strip() if raw_ip and "," in raw_ip else (raw_ip or "").strip()

        # Strict generic response to prevent account/email enumeration
        generic_response = "If an account exists for that email address, you will receive instructions to reset your password."

        # Verify IP rate limits
        if PasswordReset.is_ip_rate_limited(ip_address):
            logger.warning("IP rate limit exceeded for password reset request: %s", ip_address)
            flash(generic_response, 'info')
            return redirect(url_for('auth.login'))

        user = UserModel.find_by_email(email)
        if user and getattr(user, 'is_active', True):
            # Throttle rapid consecutive dispatches to the same account
            if not PasswordReset.is_email_rate_limited(email):
                raw_token, _ = PasswordReset.create_token(user.id, user.email, ip_address)

                # Resolve application base URL securely
                base_url = current_app.config.get("APP_BASE_URL", "").strip()
                if not base_url:
                    if current_app.config.get("APP_ENV") == "production":
                        base_url = "https://incusivematch-ai.onrender.com"
                    else:
                        base_url = request.url_root.rstrip('/')
                else:
                    base_url = base_url.rstrip('/')

                reset_url = f"{base_url}/reset-password/{raw_token}"
                sent = send_password_reset_email(user.email, reset_url, user_name=user.name)
                if not sent:
                    logger.error("Password reset email delivery could not be completed.")
            else:
                logger.info("Password reset request throttled for email cooldown.")

        flash(generic_response, 'info')
        return redirect(url_for('auth.login'))

    return render_template('forgot_password.html')


@auth_bp.route('/reset-password', methods=['GET', 'POST'])
@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token=None):
    """Validate token and allow the user to securely set a new password."""
    if not token:
        token = request.args.get('token', '').strip()

    # GET request: validate token to determine view state
    if request.method == 'GET':
        if not token:
            return render_template(
                'reset_password.html',
                token_valid=False,
                token="",
                error_message="No reset token provided. Please request a new link."
            )

        doc, error = PasswordReset.verify_token(token)
        if error:
            return render_template(
                'reset_password.html',
                token_valid=False,
                token=token,
                error_message=error
            )

        return render_template(
            'reset_password.html',
            token_valid=True,
            token=token,
            error_message=None
        )

    # POST request: process password update
    if not token:
        token = request.form.get('token', '').strip()

    if not _validate_csrf():
        flash('Invalid or expired security token. Please try again.', 'danger')
        return render_template('reset_password.html', token_valid=True, token=token, error_message=None)

    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')

    if not password or not confirm_password:
        flash('All password fields are required.', 'danger')
        return render_template('reset_password.html', token_valid=True, token=token, error_message=None)

    if password != confirm_password:
        flash('Passwords do not match!', 'danger')
        return render_template('reset_password.html', token_valid=True, token=token, error_message=None)

    if len(password) < 8:
        flash('Password must be at least 8 characters long.', 'danger')
        return render_template('reset_password.html', token_valid=True, token=token, error_message=None)

    # Atomically validate and consume the token
    user_id, error = PasswordReset.consume_token(token)
    if error:
        flash(error, 'danger')
        return redirect(url_for('auth.forgot_password'))

    # Update password using standard project hasher
    UserModel.update_password(user_id, password)

    # Invalidate existing authenticated sessions for safety
    if current_user.is_authenticated:
        logout_user()
    session.clear()

    flash('Your password has been successfully reset! Please log in with your new password.', 'success')
    return redirect(url_for('auth.login'))