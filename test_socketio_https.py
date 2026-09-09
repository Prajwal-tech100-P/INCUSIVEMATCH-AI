from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def home():
    return "SocketIO HTTPS TEST OK"

if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=False,
        allow_unsafe_werkzeug=True,
        ssl_context=(
            "10.13.125.62+2.pem",
            "10.13.125.62+2-key.pem"
        )
    )