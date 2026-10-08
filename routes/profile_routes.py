import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from models.user import UserModel
from utils.auth import allowed_file

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
            'matching_preferences': {
                'min_age': pref_min_age,
                'max_age': pref_max_age,
                'preferred_genders': preferred_genders
            }
        }

        # Handle profile picture upload
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
                filename = secure_filename(f"{current_user.id}_{file.filename}")
                file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
                update_data['profile_photo'] = filename
                update_data['profile_pic'] = filename  # Backward compat

        UserModel.update_profile(current_user.id, update_data)
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile.my_profile'))

    return render_template('profile.html', user=current_user, interests_cats=INTEREST_CATEGORIES, edit_mode=True)


@profile_bp.route('/profile/verify', methods=['POST'])
@login_required
def verify_profile():
    data = request.get_json(silent=True) or {}
    image_data_b64 = data.get('image', '')
    
    if not image_data_b64:
        return {'success': False, 'message': 'No image provided.'}
        
    try:
        import base64
        import os
        from datetime import datetime
        from ai_modules.face_verification import FaceVerifier

        # Handle base64 image data from the canvas
        if ',' in image_data_b64:
            image_data_b64 = image_data_b64.split(',')[1]
        image_data = base64.b64decode(image_data_b64)
        
        if not current_user.profile_photo:
            return {'success': False, 'message': 'Please upload a profile photo first.'}
            
        photo_path = os.path.join(current_app.config['UPLOAD_FOLDER'], current_user.profile_photo)
        if not os.path.exists(photo_path):
            return {'success': False, 'message': 'Profile photo not found.'}
            
        with open(photo_path, 'rb') as f:
            profile_image_data = f.read()
            
        verifier = FaceVerifier()
        result = verifier.verify(image_data, profile_image_data)
        
        if result.get("verified"):
            UserModel.update_profile(current_user.id, {
                'face_verified': True, 
                'is_verified': True,
                'verification': {
                    'status': 'verified',
                    'verified_at': datetime.utcnow()
                }
            })
            return {'success': True, 'message': 'Profile verified successfully!'}
        else:
            return {'success': False, 'message': result.get("reason", "Verification failed. Please make sure your face is clearly visible and matches your profile photo.")}
            
    except Exception as e:
        import logging
        logging.error(f"Verification error: {e}")
        return {'success': False, 'message': 'An error occurred during verification.'}


@profile_bp.route('/profile/<user_id>')
@login_required
def view_profile(user_id):
    user = UserModel.get_by_id(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('main.dashboard'))
    return render_template('profile.html', user=user, interests_cats=INTEREST_CATEGORIES, edit_mode=False)
