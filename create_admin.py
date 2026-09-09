from app import create_app
from extensions import mongo
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    email = input("Admin email: ").strip().lower()
    name = input("Admin name: ").strip()
    password = input("Admin password (min 6 chars): ")
    if len(password) < 6:
        raise SystemExit("Password must be at least 6 characters.")
    if mongo.db.users.find_one({"email": email}):
        raise SystemExit("Email already exists.")
    mongo.db.users.insert_one({
        "email": email, "name": name, "password": generate_password_hash(password),
        "role": "admin", "bio": "Platform administrator", "interests": [],
        "comm_pref": "mixed", "profile_pic": "default.png", "is_active": True,
        "liked": [], "disliked": []
    })
    print("Admin account created.")
