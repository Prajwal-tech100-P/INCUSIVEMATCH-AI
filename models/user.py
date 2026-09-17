from bson import ObjectId
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import mongo

class UserModel(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc["_id"])
        self.name = user_doc.get("name", "")
        self.email = user_doc.get("email", "")
        self.password = user_doc.get("password", "")
        self.role = user_doc.get("role", "user")
        self.bio = user_doc.get("bio", "")
        self.interests = user_doc.get("interests", [])
        self.comm_pref = user_doc.get("comm_pref", "text")
        self.profile_pic = user_doc.get("profile_pic", "default.png")
        self._is_active = user_doc.get("is_active", True)
        self.liked = user_doc.get("liked", [])
        self.disliked = user_doc.get("disliked", [])
        
        # New multi-step profile fields
        self.age = user_doc.get("age", 18)
        self.gender = user_doc.get("gender", "Not Specified")
        self.location = user_doc.get("location", "")
        self.relationship_goal = user_doc.get("relationship_goal", "Friendship")
        
        # New matching preferences
        self.pref_min_age = user_doc.get("pref_min_age", 18)
        self.pref_max_age = user_doc.get("pref_max_age", 99)
        self.pref_gender = user_doc.get("pref_gender", "Any")
        
        # Verification
        self.is_verified = user_doc.get("is_verified", False)

    @property
    def is_active(self):
        return self._is_active

    def is_admin(self): return self.role == "admin"

    @staticmethod
    def create_user(email, password, name, comm_pref="text", role="user"):
        doc = {
            "email": email, 
            "password": generate_password_hash(password), 
            "name": name,
            "role": role, 
            "bio": "", 
            "interests": [], 
            "comm_pref": comm_pref,
            "profile_pic": "default.png", 
            "is_active": True, 
            "liked": [], 
            "disliked": [],
            "age": 18,
            "gender": "Not Specified",
            "location": "",
            "relationship_goal": "Friendship",
            "pref_min_age": 18,
            "pref_max_age": 99,
            "pref_gender": "Any",
            "is_verified": False
        }
        result = mongo.db.users.insert_one(doc)
        return str(result.inserted_id)

    @staticmethod
    def find_by_email(email):
        doc = mongo.db.users.find_one({"email": email})
        return UserModel(doc) if doc else None

    @staticmethod
    def get_by_id(user_id):
        try:
            doc = mongo.db.users.find_one({"_id": ObjectId(user_id)})
            return UserModel(doc) if doc else None
        except Exception:
            return None

    @staticmethod
    def get_all_users():
        return [UserModel(d) for d in mongo.db.users.find().sort("name", 1)]

    @staticmethod
    def update_profile(user_id, update_data):
        mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})

    @staticmethod
    def delete_user(user_id):
        try:
            oid = ObjectId(user_id)
        except Exception:
            return
        mongo.db.users.delete_one({"_id": oid})
        mongo.db.messages.delete_many({"$or": [{"sender_id": user_id}, {"receiver_id": user_id}]})
        mongo.db.users.update_many({}, {"$pull": {"liked": user_id, "disliked": user_id}})

    @staticmethod
    def toggle_role(user_id):
        user = UserModel.get_by_id(user_id)
        if user:
            mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"role": "admin" if user.role == "user" else "user"}})

    @staticmethod
    def verify_password(hashed_password, plain_password):
        return check_password_hash(hashed_password, plain_password)
