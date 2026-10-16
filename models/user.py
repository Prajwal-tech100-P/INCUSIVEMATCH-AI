from bson import ObjectId
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import mongo
from datetime import datetime

class UserModel(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc["_id"])
        
        # Legacy name support, extract first/last
        self.name = user_doc.get("name", "")
        self.first_name = user_doc.get("first_name", self.name.split(' ')[0] if self.name else "")
        self.surname = user_doc.get("surname", " ".join(self.name.split(' ')[1:]) if self.name else "")
        
        self.email = user_doc.get("email", "")
        self.password = user_doc.get("password", "")
        self.role = user_doc.get("role", "user")
        self.bio = user_doc.get("bio", "")
        self.interests = user_doc.get("interests", [])
        
        self.block_expiry = user_doc.get("block_expiry")
        self.warning_count = user_doc.get("warning_count", 0)
        
        # Communication preferences
        legacy_comm = user_doc.get("comm_pref", "text")
        legacy_comm_prefs = user_doc.get("comm_prefs", [legacy_comm] if isinstance(legacy_comm, str) else ["text"])
        self.communication_preferences = user_doc.get("communication_preferences", legacy_comm_prefs)
        
        # Backward compatibility aliases
        self.comm_prefs = self.communication_preferences
        
        self.profile_photo = user_doc.get("profile_photo", user_doc.get("profile_pic", "default.png"))
        self.profile_pic = self.profile_photo
        self.profile_photo_public_id = user_doc.get("profile_photo_public_id")
        
        self._is_active = user_doc.get("is_active", True)
        self.liked = user_doc.get("liked", [])
        self.disliked = user_doc.get("disliked", [])
        
        # Profile fields
        self.age = user_doc.get("age", 18)
        self.gender = user_doc.get("gender", "Not Specified")
        self.marital_status = user_doc.get("marital_status", "Prefer not to say")
        
        # Location
        loc = user_doc.get("location", {})
        if isinstance(loc, str):
            self.location = {
                "mode": "manual",
                "latitude": None,
                "longitude": None,
                "city": "",
                "state": "",
                "country": "",
                "display_name": loc
            }
        else:
            self.location = loc
            
        self.relationship_goal = user_doc.get("relationship_goal", "Friendship")
        
        # Matching Preferences
        mp = user_doc.get("matching_preferences")
        if not isinstance(mp, dict):
            mp = {}

        pref_genders = mp.get("preferred_genders")
        if not pref_genders:
            legacy_g = user_doc.get("pref_gender")
            pref_genders = [legacy_g] if legacy_g else ["Any"]
        elif isinstance(pref_genders, str):
            pref_genders = [pref_genders]
        elif not isinstance(pref_genders, (list, tuple, set)):
            pref_genders = ["Any"]

        raw_min_age = mp.get("min_age", user_doc.get("pref_min_age", 18))
        raw_max_age = mp.get("max_age", user_doc.get("pref_max_age", 99))
        try:
            min_age = int(raw_min_age) if raw_min_age is not None else 18
        except (ValueError, TypeError):
            min_age = 18
        try:
            max_age = int(raw_max_age) if raw_max_age is not None else 99
        except (ValueError, TypeError):
            max_age = 99

        self.matching_preferences = {
            "min_age": min_age,
            "max_age": max_age,
            "preferred_genders": list(pref_genders)
        }
        
        # Verification
        self.face_verified = bool(user_doc.get("face_verified", False))
        self.is_verified = self.face_verified
        self.face_verified_at = user_doc.get("face_verified_at")
        self.verified_profile_photo = user_doc.get("verified_profile_photo")
        self.verified_profile_photo_id = user_doc.get("verified_profile_photo_id")

    @property
    def avatar_url(self):
        """Return the resolved avatar URL for Cloudinary HTTPS, local uploads, or default avatar."""
        photo = self.profile_photo or self.profile_pic
        if not photo or photo == "default.png":
            return "/static/uploads/default.png"
        if str(photo).startswith(("http://", "https://")):
            return photo
        return f"/static/uploads/{photo}"

    @property
    def is_active(self):
        if self.block_expiry:
            if datetime.utcnow() > self.block_expiry:
                # Auto-unblock
                mongo.db.users.update_one({"_id": ObjectId(self.id)}, {"$set": {"is_active": True}, "$unset": {"block_expiry": ""}})
                self._is_active = True
                self.block_expiry = None
            else:
                return False
        return self._is_active

    def is_admin(self): return self.role == "admin"

    @staticmethod
    def create_user(email, password, name, comm_prefs=None, role="user"):
        if comm_prefs is None:
            comm_prefs = ["text"]
        
        first_name = name.split(' ')[0] if name else ""
        surname = " ".join(name.split(' ')[1:]) if name else ""
        
        doc = {
            "email": email, 
            "password": generate_password_hash(password), 
            "name": name,
            "first_name": first_name,
            "surname": surname,
            "role": role, 
            "bio": "", 
            "interests": [], 
            "communication_preferences": comm_prefs,
            "comm_prefs": comm_prefs,
            "profile_photo": "default.png", 
            "is_active": True, 
            "liked": [], 
            "disliked": [],
            "age": 18,
            "gender": "Not Specified",
            "marital_status": "Prefer not to say",
            "location": {
                "mode": "",
                "latitude": None,
                "longitude": None,
                "city": "",
                "state": "",
                "country": "",
                "display_name": ""
            },
            "relationship_goal": "Friendship",
            "pref_min_age": 18,
            "pref_max_age": 99,
            "pref_gender": "Any",
            "matching_preferences": {
                "min_age": 18,
                "max_age": 99,
                "preferred_genders": ["Any"]
            },
            "face_verified": False,
            "is_verified": False,
            "face_verified_at": None,
            "verified_profile_photo": None,
            "verified_profile_photo_id": None
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
        user_doc = mongo.db.users.find_one({"_id": oid})
        if user_doc:
            photo_ref = user_doc.get("profile_photo_public_id") or user_doc.get("profile_photo")
            if photo_ref:
                try:
                    from utils.cloudinary_storage import delete_profile_photo
                    delete_profile_photo(photo_ref)
                except Exception:
                    pass
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

    @staticmethod
    def set_face_verified(user_id, photo_url, photo_public_id=None):
        try:
            oid = ObjectId(user_id)
        except Exception:
            return False
        now = datetime.utcnow()
        result = mongo.db.users.update_one(
            {"_id": oid},
            {
                "$set": {
                    "face_verified": True,
                    "is_verified": True,
                    "face_verified_at": now,
                    "verified_profile_photo": photo_url,
                    "verified_profile_photo_id": photo_public_id,
                    "verification": {
                        "status": "verified",
                        "verified_at": now,
                        "photo": photo_url
                    }
                }
            }
        )
        return result.modified_count > 0

    @staticmethod
    def invalidate_face_verification(user_id):
        try:
            oid = ObjectId(user_id)
        except Exception:
            return False
        result = mongo.db.users.update_one(
            {"_id": oid},
            {
                "$set": {
                    "face_verified": False,
                    "is_verified": False,
                    "face_verified_at": None,
                    "verified_profile_photo": None,
                    "verified_profile_photo_id": None,
                    "verification": {
                        "status": "unverified",
                        "verified_at": None
                    }
                }
            }
        )
        return result.modified_count > 0

