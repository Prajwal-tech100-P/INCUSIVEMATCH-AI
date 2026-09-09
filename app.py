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
    socketio.run(
    app,
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000)),
    debug=False
    )


