from flask import Blueprint, render_template, request, redirect, url_for, flash
from urllib.parse import urlparse
from flask_login import login_user, logout_user, login_required, current_user
from models.user import UserModel

auth_bp = Blueprint('auth', __name__)


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

        UserModel.create_user(
            email,
            password,
            name,
            comm_pref=comm_pref
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

        print("LOGIN EMAIL:", email)
        print("PASSWORD RECEIVED:", bool(password))

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

            return redirect(
                next_page or url_for('main.dashboard')
            )

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