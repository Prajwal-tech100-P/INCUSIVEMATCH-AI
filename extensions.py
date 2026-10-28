"""Shared Flask extension objects."""
from flask_pymongo import PyMongo
from flask_login import LoginManager
from flask_socketio import SocketIO


class RetentiveSocketIO(SocketIO):
    """SocketIO subclass that ensures event handlers are retained across init_app calls.

    In Flask-SocketIO, if handlers are registered via @socketio.on after an initial
    app creation, they are attached directly to self.server and not stored in
    self.handlers. If a test or factory invokes init_app again (e.g. creating test
    app fixtures), self.server is re-instantiated and without this retention,
    the event handlers would be wiped out.
    """

    def on(self, message, namespace=None):
        decorator = super().on(message, namespace=namespace)

        def wrapper(handler):
            res = decorator(handler)
            ns = namespace or "/"
            if self.server and ns in self.server.handlers and message in self.server.handlers[ns]:
                wrapped_handler = self.server.handlers[ns][message]
                entry = (message, wrapped_handler, ns)
                if entry not in self.handlers:
                    self.handlers.append(entry)
            return res

        return wrapper


mongo = PyMongo()
login_manager = LoginManager()
# Same-origin Socket.IO is the secure default. app.py may supply an explicit
# production allow-list through configuration.
socketio = RetentiveSocketIO(async_mode="threading")

