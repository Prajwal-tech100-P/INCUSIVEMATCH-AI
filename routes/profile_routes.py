import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from models.user import UserModel
from utils.auth import allowed_file

profile_bp = Blueprint('profile', __name__)

INTEREST_CHOICES = [
    'Sign Language', 'Photography', 'Music', 'Art', 'Cooking', 'Hiking',
    'Travel', 'Reading', 'Gaming', 'Fitness', 'Movies', 'Dogs', 'Cats',
    'Coffee', 'Dancing', 'Technology', 'Fashion', 'Sports', 'Yoga'
]


@profile_bp.route('/profile')
@login_required
def my_profile():
    return render_template('profile.html', user=current_user, interests=INTEREST_CHOICES, edit_mode=False)


@profile_bp.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        bio = request.form.get('bio', '').strip()
        interests = request.form.getlist('interests')
        comm_pref = request.form.get('comm_pref', current_user.comm_pref)
        
        # New fields
        age = int(request.form.get('age', current_user.age))
        gender = request.form.get('gender', current_user.gender)
        location = request.form.get('location', current_user.location)
        relationship_goal = request.form.get('relationship_goal', current_user.relationship_goal)
        
        pref_min_age = int(request.form.get('pref_min_age', current_user.pref_min_age))
        pref_max_age = int(request.form.get('pref_max_age', current_user.pref_max_age))
        pref_gender = request.form.get('pref_gender', current_user.pref_gender)

        update_data = {
            'bio': bio,
            'interests': interests,
            'comm_pref': comm_pref,
            'age': age,
            'gender': gender,
            'location': location,
            'relationship_goal': relationship_goal,
            'pref_min_age': pref_min_age,
            'pref_max_age': pref_max_age,
            'pref_gender': pref_gender,
        }

        # Handle profile picture upload
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
                filename = secure_filename(f"{current_user.id}_{file.filename}")
                file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
                update_data['profile_pic'] = filename

        UserModel.update_profile(current_user.id, update_data)
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile.my_profile'))

    return render_template('profile.html', user=current_user, interests=INTEREST_CHOICES, edit_mode=True)


@profile_bp.route('/profile/verify', methods=['POST'])
@login_required
def verify_profile():
    # Simulated verification endpoint for the face scan
    UserModel.update_profile(current_user.id, {'is_verified': True})
    return {'success': True, 'message': 'Profile verified successfully!'}


@profile_bp.route('/profile/<user_id>')
@login_required
def view_profile(user_id):
    user = UserModel.get_by_id(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('main.dashboard'))
    return render_template('profile.html', user=user, interests=INTEREST_CHOICES, edit_mode=False)
