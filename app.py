import os
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from extensions import mongo, login_manager, socketio


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    if app.config.get("APP_ENV") == "production" and not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY must be set in production.")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Render/Cloudflare/reverse proxies terminate HTTPS before forwarding to
    # Flask. Trust the standard proxy headers so Flask knows the original scheme.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    mongo.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "warning"

    origins = app.config.get("SOCKETIO_CORS_ALLOWED_ORIGINS") or None
    socketio.init_app(app, cors_allowed_origins=origins)

    from models.user import UserModel

    @login_manager.user_loader
    def load_user(user_id):
        return UserModel.get_by_id(user_id)

    @app.template_filter("avatar_url")
    def avatar_url_filter(user_or_photo):
        """Resolve Cloudinary HTTPS URL, legacy local upload, or default avatar."""
        if not user_or_photo:
            return "/static/uploads/default.png"
        if hasattr(user_or_photo, "avatar_url"):
            return user_or_photo.avatar_url
        photo = None
        if isinstance(user_or_photo, str):
            photo = user_or_photo
        elif isinstance(user_or_photo, dict):
            photo = user_or_photo.get("profile_photo") or user_or_photo.get("profile_pic")
        elif hasattr(user_or_photo, "profile_photo"):
            photo = user_or_photo.profile_photo or getattr(user_or_photo, "profile_pic", None)

        if not photo or photo == "default.png":
            return "/static/uploads/default.png"
        if str(photo).startswith(("http://", "https://")):
            return photo
        return f"/static/uploads/{photo}"

    app.jinja_env.globals["avatar_url"] = avatar_url_filter

    def generate_csrf_token():
        from flask import session
        import secrets
        if "_csrf_token" not in session:
            session["_csrf_token"] = secrets.token_hex(32)
        return session["_csrf_token"]

    app.jinja_env.globals["csrf_token"] = generate_csrf_token

    @app.before_request
    def check_maintenance():
        from flask import request, redirect, url_for, jsonify
        from flask_login import current_user
        
        # Define allowed endpoints during maintenance
        allowed_endpoints = [
            'static', 
            'auth.login', 
            'auth.logout', 
            'auth.forgot_password',
            'auth.reset_password',
            'main.maintenance', 
            'health_check'
        ]
        
        # Admins must always be able to access the admin panel to turn off maintenance.
        # Legal and policy pages remain accessible so users can always read policies.
        if (
            app.config.get("TESTING")
            or getattr(request, 'endpoint', None) in allowed_endpoints
            or getattr(request, 'blueprint', None) in ('admin', 'legal')
        ):
            return

        # Fetch settings from MongoDB safely
        try:
            settings = mongo.db.app_settings.find_one({"_id": "app_settings"})
        except Exception:
            settings = None
            
        if settings and settings.get("maintenance_mode") is True:
            # Admins are exempt from maintenance mode on all other routes too
            if current_user.is_authenticated and getattr(current_user, 'role', '') == 'admin':
                return
                
            # Block Socket.IO and APIs directly with a JSON error
            if request.path.startswith('/socket.io/') or request.path.startswith('/api/'):
                return jsonify({"error": "Maintenance mode active"}), 503
                
            # Redirect normal users to the maintenance page
            return redirect(url_for('main.maintenance'))

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(self), microphone=(self), geolocation=()"
        )
        if app.config.get("SESSION_COOKIE_SECURE"):
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains"
            )
        return response

    from routes.auth_routes import auth_bp
    from routes.main_routes import main_bp
    from routes.profile_routes import profile_bp
    from routes.match_routes import match_bp
    from routes.chat_routes import chat_bp
    from routes.admin_routes import admin_bp
    from routes.video_routes import video_bp
    from routes.community_routes import community_bp
    from routes.legal_routes import legal_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(match_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(community_bp)
    app.register_blueprint(legal_bp)
    app.register_blueprint(admin_bp, url_prefix="/admin")

    @app.route("/health")
    def health_check():
        return {"status": "healthy", "service": "Inclusive Match AI"}

    with app.app_context():
        try:
            from models.call_invitation import CallInvitation
            from models.restriction import Restriction
            from models.password_reset import PasswordReset
            CallInvitation.ensure_indexes()
            Restriction.ensure_indexes()
            PasswordReset.ensure_indexes()
        except Exception:
            pass

    return app


app = create_app()

if __name__ == "__main__":
    # Local development only. Production uses Gunicorn with threaded workers.
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, allow_unsafe_werkzeug=True)
