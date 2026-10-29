import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from bson import ObjectId
from extensions import mongo
from models.user import UserModel
from utils.cloudinary_storage import (
    is_cloudinary_configured,
    validate_image_file,
    upload_profile_photo,
    delete_profile_photo,
    extract_public_id_from_url,
)

profile_bp = Blueprint('profile', __name__)

INTEREST_CATEGORIES = {
    "Sports & Fitness": ["Cricket", "Football", "Basketball", "Badminton", "Tennis", "Swimming", "Running", "Gym", "Yoga", "Cycling"],
    "Entertainment": ["Movies", "Music", "Singing", "Dancing", "Photography", "Gaming", "Reading", "Watching TV", "Anime"],
    "Creative": ["Drawing", "Painting", "Writing", "Cooking", "Baking", "Crafts", "Design"],
    "Travel & Lifestyle": ["Travelling", "Hiking", "Nature", "Adventure", "Food", "Fashion", "Shopping"],
    "Technology & Learning": ["Technology", "Programming", "AI & Machine Learning", "Science", "Education", "Learning Languages"],
    "Social & Other": ["Pets", "Volunteering", "Community Activities", "Meditation", "Spirituality"]
}

@profile_bp.route('/profile')
@login_required
def my_profile():
    return render_template('profile.html', user=current_user, interests_cats=INTEREST_CATEGORIES, edit_mode=False)


@profile_bp.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        surname = request.form.get('surname', '').strip()
        bio = request.form.get('bio', '').strip()
        
        interests = request.form.getlist('interests')
        communication_preferences = request.form.getlist('communication_preferences')
        if not communication_preferences:
            communication_preferences = current_user.communication_preferences
            
        age_str = request.form.get('age')
        age = int(age_str) if age_str and age_str.isdigit() else current_user.age
        
        gender = request.form.get('gender', current_user.gender)
        marital_status = request.form.get('marital_status', current_user.marital_status)
        relationship_goal = request.form.get('relationship_goal', current_user.relationship_goal)
        
        # Location logic
        location_mode = request.form.get('location_mode', 'manual')
        loc_display = request.form.get('location_display', '').strip()
        
        location_data = {
            "mode": location_mode,
            "latitude": None,
            "longitude": None,
            "city": "",
            "state": "",
            "country": "",
            "display_name": loc_display
        }
        
        if location_mode == 'current':
            try:
                location_data["latitude"] = float(request.form.get('latitude', 0))
                location_data["longitude"] = float(request.form.get('longitude', 0))
            except ValueError:
                pass
            location_data["city"] = request.form.get('city', '').strip()
            location_data["state"] = request.form.get('state', '').strip()
            location_data["country"] = request.form.get('country', '').strip()
        
        # Matching Preferences
        pref_min_age_str = request.form.get('pref_min_age')
        pref_max_age_str = request.form.get('pref_max_age')
        pref_min_age = int(pref_min_age_str) if pref_min_age_str and pref_min_age_str.isdigit() else current_user.matching_preferences.get('min_age', 18)
        pref_max_age = int(pref_max_age_str) if pref_max_age_str and pref_max_age_str.isdigit() else current_user.matching_preferences.get('max_age', 99)
        preferred_genders = request.form.getlist('preferred_genders')
        if not preferred_genders:
            preferred_genders = current_user.matching_preferences.get('preferred_genders', ['Any'])

        update_data = {
            'first_name': first_name,
            'surname': surname,
            'name': f"{first_name} {surname}".strip(),
            'bio': bio,
            'interests': interests,
            'communication_preferences': communication_preferences,
            'comm_prefs': communication_preferences,
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'location': location_data,
            'relationship_goal': relationship_goal,
            'pref_min_age': pref_min_age,
            'pref_max_age': pref_max_age,
            'pref_gender': preferred_genders[0] if len(preferred_genders) == 1 else ("Any" if "Any" in preferred_genders else ",".join(preferred_genders)),
            'matching_preferences': {
                'min_age': pref_min_age,
                'max_age': pref_max_age,
                'preferred_genders': preferred_genders
            }
        }

        # Handle profile picture upload with Cloudinary persistence
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename:
                # 1. Validate image format, size (max 5 MB), and content integrity
                is_valid, error_msg = validate_image_file(file)
                if not is_valid:
                    flash(error_msg, 'danger')
                    return redirect(url_for('profile.edit_profile'))

                # 2. Check Cloudinary availability (do not silently fall back to local storage)
                if not is_cloudinary_configured():
                    flash(
                        'Cloudinary storage is not configured. Profile photo upload failed. '
                        'Existing profile data was preserved.',
                        'danger',
                    )
                    return redirect(url_for('profile.edit_profile'))

                # 3. Upload the new photo first
                try:
                    upload_res = upload_profile_photo(file)
                    new_photo_url = upload_res['secure_url']
                    new_public_id = upload_res['public_id']
                except Exception as upload_err:
                    flash(
                        f'Failed to upload profile photo to Cloudinary: {upload_err}. '
                        'Existing profile data was preserved.',
                        'danger',
                    )
                    return redirect(url_for('profile.edit_profile'))

                # 4. Attach new photo details to update payload
                old_public_id = getattr(current_user, 'profile_photo_public_id', None)
                if not old_public_id and getattr(current_user, 'profile_photo', None):
                    old_public_id = extract_public_id_from_url(current_user.profile_photo)
                update_data['profile_photo'] = new_photo_url
                update_data['profile_pic'] = new_photo_url  # Backward compat
                update_data['profile_photo_public_id'] = new_public_id

                # Immediately invalidate previous verification because profile photo changed
                update_data['face_verified'] = False
                update_data['is_verified'] = False
                update_data['face_verified_at'] = None
                update_data['verified_profile_photo'] = None
                update_data['verified_profile_photo_id'] = None
                update_data['verification'] = {
                    'status': 'unverified',
                    'verified_at': None
                }

                # 5. Update database after successful upload
                try:
                    UserModel.update_profile(current_user.id, update_data)
                except Exception as db_err:
                    # Clean up newly uploaded image to avoid orphaned files
                    try:
                        delete_profile_photo(new_public_id)
                    except Exception:
                        pass
                    flash(
                        f'Database update failed: {db_err}. Existing profile photo was preserved.',
                        'danger',
                    )
                    return redirect(url_for('profile.edit_profile'))

                # 6. Delete old Cloudinary image only AFTER the database update succeeds
                if old_public_id and old_public_id != new_public_id:
                    try:
                        delete_profile_photo(old_public_id)
                    except Exception as del_err:
                        current_app.logger.warning(
                            f'Failed to delete old Cloudinary image {old_public_id}: {del_err}'
                        )

                flash('Profile updated successfully!', 'success')
                return redirect(url_for('profile.my_profile'))

        # If no new photo was submitted, update profile attributes as normal
        UserModel.update_profile(current_user.id, update_data)
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile.my_profile'))

    return render_template('profile.html', user=current_user, interests_cats=INTEREST_CATEGORIES, edit_mode=True)


@profile_bp.route('/profile/verify', methods=['POST'])
@login_required
def verify_profile():
    if not current_user.is_active:
        return {'success': False, 'message': 'Account is deactivated or suspended.'}, 403

    data = request.get_json(silent=True) or {}
    image_data_b64 = data.get('image', '')
    
    if not image_data_b64:
        return {'success': False, 'message': 'No live camera capture provided.'}, 400

    user_doc = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}) or {}
    photo = user_doc.get('profile_photo') or user_doc.get('profile_pic') or getattr(current_user, 'profile_photo', None)
    if not photo or photo == 'default.png':
        return {'success': False, 'message': 'Please upload a clear profile photo first before verifying.'}, 400

    try:
        import base64
        import os
        import urllib.request
        from werkzeug.utils import secure_filename
        from ai_modules.face_verification import FaceVerifier

        # Handle base64 image data from the live camera canvas
        if ',' in image_data_b64:
            image_data_b64 = image_data_b64.split(',')[1]

        try:
            image_data = base64.b64decode(image_data_b64)
        except Exception:
            return {'success': False, 'message': 'Invalid camera image format.'}, 400

        if not image_data or len(image_data) < 100:
            return {'success': False, 'message': 'Live capture is empty or corrupt.'}, 400

        # Retrieve reference profile image data
        if str(photo).startswith(('http://', 'https://')):
            from urllib.parse import urlparse
            parsed = urlparse(str(photo))
            hostname = (parsed.hostname or "").lower()

            if parsed.scheme != "https" and not current_app.config.get("TESTING"):
                return {'success': False, 'message': 'Profile photo must be hosted over secure HTTPS.'}, 400

            is_trusted_host = (
                hostname.endswith("cloudinary.com") or
                hostname.endswith("res.cloudinary.com") or
                (current_app.config.get("TESTING") and hostname in ("localhost", "127.0.0.1", "testserver"))
            )
            if not is_trusted_host:
                return {'success': False, 'message': 'Invalid profile photo storage host.'}, 400

            try:
                req = urllib.request.Request(photo, headers={'User-Agent': 'InclusiveMatch-AI'})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    if resp.status != 200:
                        return {'success': False, 'message': f'Failed to retrieve profile photo (HTTP {resp.status}).'}, 400
                    profile_image_data = resp.read()
            except Exception as e:
                return {'success': False, 'message': f'Failed to download profile photo for verification: {e}'}, 400
        else:
            safe_photo = secure_filename(photo)
            photo_path = os.path.join(current_app.config['UPLOAD_FOLDER'], safe_photo)
            if not os.path.isfile(photo_path):
                return {'success': False, 'message': 'Reference profile photo not found.'}, 400
            with open(photo_path, 'rb') as f:
                profile_image_data = f.read()

        verifier = FaceVerifier()
        result = verifier.verify(image_data, profile_image_data)

        if result.get("verified"):
            UserModel.set_face_verified(
                current_user.id,
                photo_url=photo,
                photo_public_id=getattr(current_user, 'profile_photo_public_id', None)
            )
            current_user.face_verified = True
            current_user.is_verified = True
            return {
                'success': True,
                'message': 'Face verified successfully! Verified badge granted.',
                'score': result.get('score'),
                'threshold': result.get('threshold')
            }
        else:
            return {
                'success': False,
                'message': result.get('reason', 'Verification failed. Face does not match profile photo.'),
                'score': result.get('score'),
                'threshold': result.get('threshold')
            }, 400

    except Exception as e:
        current_app.logger.error("Verification endpoint error: %s", e)
        return {'success': False, 'message': f'An unexpected error occurred during verification: {e}'}, 500


@profile_bp.route('/profile/<user_id>')
@login_required
def view_profile(user_id):
    user = UserModel.get_by_id(user_id)
    if not user or not user.is_active:
        flash('User profile is not available.', 'danger')
        return redirect(url_for('main.dashboard'))
    return render_template('profile.html', user=user, interests_cats=INTEREST_CATEGORIES, edit_mode=False)


@profile_bp.route('/profile/delete-account', methods=['POST'])
@login_required
def delete_account():
    """Authenticated self-service account deletion under DPDP Act Section 12."""
    from flask_login import logout_user

    password = request.form.get('password', '')
    if not password:
        flash('Password confirmation is required to delete your account.', 'danger')
        return redirect(url_for('profile.my_profile'))

    # Re-authenticate user credentials before destructive action
    if not UserModel.verify_password(current_user.password, password):
        flash('Incorrect password. Account deletion cancelled.', 'danger')
        return redirect(url_for('profile.my_profile'))

    user_id = current_user.id
    success = UserModel.delete_user(user_id)
    if success:
        logout_user()
        flash(
            'Your account and associated personal data have been permanently deleted in accordance with your privacy rights. Thank you for using InclusiveMatch AI.',
            'info'
        )
        return redirect(url_for('main.index'))
    else:
        flash('Failed to delete account. Please try again or contact our Grievance Officer.', 'danger')
        return redirect(url_for('profile.my_profile'))

