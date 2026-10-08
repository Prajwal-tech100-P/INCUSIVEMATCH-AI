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

    @app.before_request
    def check_maintenance():
        from flask import request, redirect, url_for, jsonify
        from flask_login import current_user
        
        # Define allowed endpoints during maintenance
        allowed_endpoints = [
            'static', 
            'auth.login', 
            'auth.logout', 
            'main.maintenance', 
            'health_check'
        ]
        
        # Admins must always be able to access the admin panel to turn off maintenance.
        # If unauthenticated, the @admin_required decorator will redirect them to login.
        if getattr(request, 'endpoint', None) in allowed_endpoints or getattr(request, 'blueprint', None) == 'admin':
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

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(match_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(community_bp)
    app.register_blueprint(admin_bp, url_prefix="/admin")

    @app.route("/health")
    def health_check():
        return {"status": "healthy", "service": "Inclusive Match AI"}

    return app


app = create_app()

if __name__ == "__main__":
    # Local development only. Production uses Gunicorn with threaded workers.
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, allow_unsafe_werkzeug=True)
