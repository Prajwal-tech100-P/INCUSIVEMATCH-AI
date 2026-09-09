"""Shared Flask extension objects."""
from flask_pymongo import PyMongo
from flask_login import LoginManager
from flask_socketio import SocketIO

mongo = PyMongo()
login_manager = LoginManager()
# Same-origin Socket.IO is the secure default. app.py may supply an explicit
# production allow-list through configuration.
socketio = SocketIO(async_mode="threading")
