import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from datetime import datetime
from bson import ObjectId
import pymongo

from app import create_app
from extensions import mongo
from models.user import UserModel
from models.match import Match
from models.message import Message
from models.notification import Notification
from ai_modules.match_recommender import MatchRecommender
from config import Config

def run_all_checks():
    print("=" * 70)
    print("STARTING COMPREHENSIVE END-TO-END VERIFICATION")
    print("=" * 70)

    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    client = app.test_client()

    results = {}

    with app.app_context():
        # ---------------------------------------------------------------------
        # 1. MongoDB Connectivity Check
        # ---------------------------------------------------------------------
        print("\n[CHECK 1] Confirming MongoDB Connectivity...")
        try:
            mongo_uri = Config.MONGO_URI
            is_atlas = "mongodb+srv" in mongo_uri or "mongodb.net" in mongo_uri
            server_info = mongo.db.command("ping")
            server_build = mongo.db.command("buildInfo")
            collections = mongo.db.list_collection_names()
            user_count = mongo.db.users.count_documents({})
            
            print(f" -> Ping response: {server_info}")
            print(f" -> MongoDB Version: {server_build.get('version', 'unknown')}")
            print(f" -> Active Collections ({len(collections)}): {collections}")
            print(f" -> Target DB: {mongo.db.name}")
            print(f" -> Target Type: {'MongoDB Atlas (Cloud)' if is_atlas else 'Local/Dedicated MongoDB Instance'}")
            print(f" -> Total existing users in DB: {user_count}")
            
            results["mongodb"] = {
                "status": "PASS",
                "is_atlas": is_atlas,
                "version": server_build.get("version"),
                "ping_ok": server_info.get("ok") == 1.0,
                "collections_count": len(collections),
            }
        except Exception as e:
            print(f" -> MongoDB Connection FAILED: {e}")
            results["mongodb"] = {"status": "FAIL", "error": str(e)}

        # ---------------------------------------------------------------------
        # 2. Registration and Login Test
        # ---------------------------------------------------------------------
        print("\n[CHECK 2] Testing User Registration and Login Flow...")
        user1_record = None
        uid_2 = None
        uid_3 = None
        try:
            test_ts = int(datetime.utcnow().timestamp())
            email_1 = f"test_user1_{test_ts}@example.com"
            email_2 = f"test_user2_{test_ts}@example.com"
            email_3 = f"test_user3_{test_ts}@example.com"
            password = "SecurePassword123!"

            # Test Registration via POST /register
            reg_resp = client.post("/register", data={
                "name": "Jordan Doe",
                "email": email_1,
                "password": password,
                "confirm_password": password,
                "comm_pref": "text"
            }, follow_redirects=True)
            
            assert reg_resp.status_code == 200
            user1_record = UserModel.find_by_email(email_1)
            assert user1_record is not None, "User 1 was not saved in DB after registration"
            print(f" -> Registered User 1 successfully (ID: {user1_record.id})")

            # Test Login with Wrong Password
            bad_login = client.post("/login", data={
                "email": email_1,
                "password": "WrongPassword999!"
            }, follow_redirects=True)
            assert b"Invalid email/password" in bad_login.data or b"disabled" in bad_login.data, "Failed to reject invalid password"
            print(" -> Invalid password properly rejected with error notice")

            # Test Login with Correct Password
            good_login = client.post("/login", data={
                "email": email_1,
                "password": password
            }, follow_redirects=True)
            assert good_login.status_code == 200
            assert b"Logout" in good_login.data or b"Dashboard" in good_login.data
            print(" -> Valid login authenticated successfully and opened dashboard")

            # Create User 2 and User 3 for matching/chat tests
            uid_2 = UserModel.create_user(
                email=email_2,
                password=password,
                name="Taylor Smith",
                comm_prefs=["text", "sign language"]
            )
            uid_3 = UserModel.create_user(
                email=email_3,
                password=password,
                name="Morgan Lee",
                comm_prefs=["audio"]
            )
            user2_record = UserModel.get_by_id(uid_2)
            user3_record = UserModel.get_by_id(uid_3)
            print(f" -> Created User 2 (ID: {uid_2}) and User 3 (ID: {uid_3})")

            results["auth"] = {"status": "PASS", "details": "Registration and Login fully functional"}
        except Exception as e:
            print(f" -> Auth FAILED: {e}")
            results["auth"] = {"status": "FAIL", "error": str(e)}

        # ---------------------------------------------------------------------
        # 3. Profile Editing Test
        # ---------------------------------------------------------------------
        print("\n[CHECK 3] Testing Profile Editing...")
        try:
            with client.session_transaction() as sess:
                sess["_user_id"] = user1_record.id
                sess["_fresh"] = True

            edit_resp = client.post("/profile/edit", data={
                "first_name": "Jordan",
                "surname": "Doe",
                "bio": "Enthusiastic developer who loves sign language and open-source AI.",
                "interests": ["Technology", "AI & Machine Learning", "Gaming"],
                "communication_preferences": ["text", "sign language"],
                "age": "26",
                "gender": "Non-binary",
                "marital_status": "Single",
                "relationship_goal": "Long-term",
                "location_mode": "manual",
                "location_display": "San Francisco, CA"
            }, follow_redirects=True)

            assert edit_resp.status_code == 200
            updated_u1 = UserModel.get_by_id(user1_record.id)
            assert updated_u1.bio == "Enthusiastic developer who loves sign language and open-source AI."
            assert "Technology" in updated_u1.interests
            assert updated_u1.age == 26
            assert updated_u1.gender == "Non-binary"
            print(f" -> Profile edited successfully. Updated bio: '{updated_u1.bio}'")
            print(f" -> Interests persisted: {updated_u1.interests}, Age: {updated_u1.age}")

            # Also set interests for User 2 (similar) and User 3 (dissimilar)
            mongo.db.users.update_one(
                {"_id": ObjectId(uid_2)},
                {"$set": {
                    "interests": ["Technology", "AI & Machine Learning", "Gaming", "Reading"],
                    "bio": "AI enthusiast and gamer.",
                    "age": 27,
                    "gender": "Female"
                }}
            )
            mongo.db.users.update_one(
                {"_id": ObjectId(uid_3)},
                {"$set": {
                    "interests": ["Baking", "Cooking", "Gardening"],
                    "bio": "Loves peaceful gardening and cakes.",
                    "age": 45,
                    "gender": "Male"
                }}
            )
            print(" -> Seeded profiles for compatibility and matching evaluation.")
            results["profile_editing"] = {"status": "PASS", "details": "Profile updates verified and persisted in DB"}
        except Exception as e:
            print(f" -> Profile Editing FAILED: {e}")
            results["profile_editing"] = {"status": "FAIL", "error": str(e)}

        # ---------------------------------------------------------------------
        # 4. Likes, Dislikes, and Mutual Matching Test
        # ---------------------------------------------------------------------
        print("\n[CHECK 4] Testing Likes, Dislikes, and Mutual Matching...")
        try:
            # 1. User 1 likes User 2
            with client.session_transaction() as sess:
                sess["_user_id"] = user1_record.id
                sess["_fresh"] = True

            like_res1 = client.post(f"/like/{uid_2}")
            data1 = like_res1.get_json()
            assert like_res1.status_code == 200
            assert data1["status"] == "liked"
            assert data1["mutual"] is False, "Should NOT be mutual yet"
            print(" -> User 1 liked User 2 (One-way like recorded, mutual=False)")

            # 2. User 1 dislikes User 3
            dislike_res = client.post(f"/dislike/{uid_3}")
            data_dis = dislike_res.get_json()
            assert dislike_res.status_code == 200
            assert data_dis["status"] == "disliked"
            u1_after_dis = mongo.db.users.find_one({"_id": Match._oid(user1_record.id)})
            assert uid_3 in u1_after_dis.get("disliked", []), "User 3 should be in disliked list"
            print(" -> User 1 disliked User 3 (Dislike recorded, User 3 filtered from future matches)")

            # 3. User 2 logs in and likes User 1 back -> Mutual Match!
            client.get("/logout")
            u2_login = client.post("/login", data={
                "email": email_2,
                "password": password
            }, follow_redirects=True)
            assert u2_login.status_code == 200

            like_res2 = client.post(f"/like/{user1_record.id}")
            data2 = like_res2.get_json()
            if like_res2.status_code != 200:
                print(f"DEBUG: like_res2 status={like_res2.status_code}, data={like_res2.data}")
            assert like_res2.status_code == 200, f"Expected 200, got {like_res2.status_code}"
            assert data2["status"] == "liked"
            assert data2["mutual"] is True, "Should be MUTUAL now"
            print(" -> User 2 liked User 1 back (Mutual match established, mutual=True)")

            # Verify in Match model
            assert Match.are_matched(user1_record.id, str(uid_2)) is True
            assert Match.are_matched(user1_record.id, str(uid_3)) is False
            print(" -> Verified Match.are_matched(U1, U2) == True and Match.are_matched(U1, U3) == False")

            results["matching"] = {"status": "PASS", "details": "Likes, dislikes, and mutual matching verified"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f" -> Matching FAILED: {e}")
            results["matching"] = {"status": "FAIL", "error": str(e)}

        # ---------------------------------------------------------------------
        # 5. Compatibility Scores Variation Test
        # ---------------------------------------------------------------------
        print("\n[CHECK 5] Testing Compatibility Score Variation...")
        try:
            recommender = MatchRecommender()
            u1_doc = mongo.db.users.find_one({"_id": ObjectId(user1_record.id)})
            u2_doc = mongo.db.users.find_one({"_id": ObjectId(uid_2)})
            u3_doc = mongo.db.users.find_one({"_id": ObjectId(uid_3)})

            score_high = recommender.compute_compatibility_score(u1_doc, u2_doc)
            score_low = recommender.compute_compatibility_score(u1_doc, u3_doc)

            print(f" -> Compatibility between User 1 (Tech/Gaming) & User 2 (Tech/Gaming): {score_high}%")
            print(f" -> Compatibility between User 1 (Tech/Gaming) & User 3 (Gardening/Baking): {score_low}%")

            diff = score_high - score_low
            print(f" -> Score Difference: {diff:.1f} points")
            assert score_high > score_low, "Matching score should be higher for shared interests & preferences"
            assert diff >= 15.0, f"Expected meaningful score variance, got diff={diff}"
            print(" -> Compatibility scores vary meaningfully across diverse profiles!")

            results["compatibility"] = {
                "status": "PASS",
                "score_high": score_high,
                "score_low": score_low,
                "difference": diff
            }
        except Exception as e:
            print(f" -> Compatibility Score FAILED: {e}")
            results["compatibility"] = {"status": "FAIL", "error": str(e)}

        # ---------------------------------------------------------------------
        # 6. Private Messaging Test
        # ---------------------------------------------------------------------
        print("\n[CHECK 6] Testing Private Messaging...")
        try:
            # 1. User 1 logs in
            client.get("/logout")
            client.post("/login", data={"email": email_1, "password": password}, follow_redirects=True)

            # Try sending to User 3 (NOT a mutual match) -> Must be rejected (403)
            blocked_send = client.post("/chat/send", json={
                "receiver_id": str(uid_3),
                "text": "Hey non-match!"
            })
            assert blocked_send.status_code == 403, f"Expected 403 Forbidden for non-match, got {blocked_send.status_code}"
            print(" -> Non-mutual match chat blocked with 403 Forbidden (Privacy protected)")

            # 2. User 1 sends to User 2 (Mutual Match) -> Allowed!
            allowed_send = client.post("/chat/send", json={
                "receiver_id": str(uid_2),
                "text": "Hello Taylor, great to match with you!"
            })
            assert allowed_send.status_code == 200, f"Expected 200 OK for mutual match, got {allowed_send.status_code} ({allowed_send.data})"
            print(" -> Mutual match message sent successfully (Status 200)")

            # 3. User 2 logs in and replies to User 1 -> Allowed!
            client.get("/logout")
            client.post("/login", data={"email": email_2, "password": password}, follow_redirects=True)

            reply_send = client.post("/chat/send", json={
                "receiver_id": str(user1_record.id),
                "text": "Hey Jordan! Excited to connect!"
            })
            assert reply_send.status_code == 200
            print(" -> Mutual match reply sent successfully (Status 200)")

            # 4. Verify conversation retrieval from DB
            conv = Message.get_conversation(user1_record.id, uid_2)
            assert len(conv) == 2
            assert conv[0]["text"] == "Hello Taylor, great to match with you!"
            assert conv[1]["text"] == "Hey Jordan! Excited to connect!"
            print(f" -> Verified conversation retrieved from DB (Total: {len(conv)} messages)")

            # 5. Test Safety Filter on message content
            flagged_send = client.post("/chat/send", json={
                "receiver_id": user1_record.id,
                "text": "You idiot, I hate you so much!"
            })
            assert flagged_send.status_code == 400
            err_msg = flagged_send.get_json().get("error", "").lower()
            assert "harmful" in err_msg or "safety" in err_msg or "inappropriate" in err_msg, f"Unexpected error msg: {err_msg}"
            print(" -> Inappropriate/toxic message blocked by AI Safety Moderator")

            results["messaging"] = {"status": "PASS", "details": "Messaging permissions, delivery, and safety confirmed"}
        except Exception as e:
            print(f" -> Private Messaging FAILED: {e}")
            results["messaging"] = {"status": "FAIL", "error": str(e)}

        # ---------------------------------------------------------------------
        # CLEANUP
        # ---------------------------------------------------------------------
        print("\n[CLEANUP] Cleaning up test records...")
        u_ids = [x for x in [getattr(user1_record, "id", None), uid_2, uid_3] if x]
        if u_ids:
            mongo.db.users.delete_many({"_id": {"$in": [ObjectId(x) for x in u_ids]}})
            mongo.db.matches.delete_many({"$or": [
                {"user1_id": {"$in": u_ids}},
                {"user2_id": {"$in": u_ids}},
                {"user_id": {"$in": u_ids}},
                {"target_id": {"$in": u_ids}}
            ]})
            mongo.db.messages.delete_many({"$or": [
                {"sender_id": {"$in": u_ids}},
                {"receiver_id": {"$in": u_ids}}
            ]})
        print(" -> Test users, matches, and messages cleaned up successfully.")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY OF VERIFICATION RESULTS:")
    print("=" * 70)
    all_passed = True
    for check_name, check_data in results.items():
        status = check_data["status"]
        if status != "PASS":
            all_passed = False
        print(f"[{status}] {check_name}: {check_data}")
    
    print("=" * 70)
    if all_passed:
        print("ALL 6 SPECIFIED FEATURES ARE WORKING PROPERLY!")
    else:
        print("SOME CHECKS REPORTED ISSUES.")
    print("=" * 70)

if __name__ == "__main__":
    run_all_checks()
