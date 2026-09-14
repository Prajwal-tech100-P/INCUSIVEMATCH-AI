from app import create_app
from extensions import mongo
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():

    # Admin login details
    email = "admin@inclusivematch.com"
    password = "Admin@123"
    name = "Admin"

    # Create or update admin account
    result = mongo.db.users.update_one(
        {"email": email},
        {
            "$set": {
                "email": email,
                "name": name,
                "password": generate_password_hash(password),
                "role": "admin",
                "is_active": True
            }
        },
        upsert=True
    )

    if result.upserted_id:
        print("Admin account created successfully.")
    elif result.modified_count > 0:
        print("Admin account updated successfully.")
    else:
        print("Admin account already exists and is correct.")

    print("--------------------------------")
    print("Admin Email:", email)
    print("Admin Password:", password)
    print("Admin Role: admin")
    print("--------------------------------")