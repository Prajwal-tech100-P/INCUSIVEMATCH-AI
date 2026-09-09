# Inclusive Match AI — Corrected MVP

An accessible dating and communication web application for Deaf and Hard-of-Hearing (DHH) users.

## Modules
- User registration/login and profile management
- Browse/discover profiles
- AI-style compatibility scoring using interests + communication preference
- Mutual likes before private communication
- Real-time chat with Socket.IO
- Speech-to-text and text-to-speech using browser Web Speech APIs
- WebRTC video calling between mutual matches
- Sign-language recognition API hook (no fabricated predictions)
- Baseline NLP safety moderation
- Admin dashboard: user counts, activate/disable, role management and deletion

## Important AI note
The original ZIP contained **stub AI functions**: sign recognition always returned `HELLO`, and face verification always returned `True`. Those are not real AI features. This corrected version removes those false results.

For the final academic AI implementation, connect:
1. MediaPipe/OpenCV hand landmarks
2. A trained sign classifier (static signs) or LSTM/Transformer (video sequences)
3. A real face-verification model for profile/live-image comparison
4. A trained toxicity classifier for stronger NLP moderation

## Database
MongoDB is used. The database is created automatically when data is first inserted.

## Run on Windows
1. Install MongoDB Community Server and make sure MongoDB is running.
2. Create a virtual environment:
   `python -m venv venv`
3. Activate:
   `venv\Scripts\activate`
4. Install:
   `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and set your values.
6. Run:
   `python app.py`
7. Open `http://127.0.0.1:5000`

## Create the first admin
After MongoDB is running:
`python create_admin.py`

## Architecture
Browser → Flask routes → MongoDB
              ↘ Socket.IO → real-time chat/signaling
              ↘ AI modules → matching/safety/sign-language/verification

WebRTC carries live audio/video peer-to-peer; MongoDB stores users, likes and chat history.


## Updated Admin & Match Notifications
- Admin Control Center now includes Users, Matches, Reports, Messages / Monitoring, Statistics, Settings and Logout.
- Mutual matches create persistent notifications for both users and real-time Socket.IO match alerts when online.
- User dashboard includes unread match notifications and a notification bell.
- New MongoDB collections used: `notifications`, `reports` (if reports are submitted), and `app_settings`.

## Real sign recognition setup
The video call is WebRTC; OpenCV is used to decode frames and MediaPipe extracts 21 hand landmarks. The primary classifier is a bidirectional GRU trained from your own landmark sequence CSV dataset. The runtime keeps 24 frames, uses normalized 21-point landmarks plus temporal motion features, and applies prediction smoothing. An old SVM model remains as a compatibility fallback.

1. Install the runtime dependency if compatible with your Python: `python -m pip install mediapipe`.
2. Prepare a CSV with 42 columns in order `x0,y0,...,x20,y20` and a `label` column.
3. Train: `python ai_modules/train_sign_model.py path\\to\\landmarks.csv`.
4. Confirm the console prints the held-out test accuracy and creates `models/sign_classifier.joblib`.
5. Start the app with `python app.py`.

The UI reports classifier confidence, not a fabricated accuracy. No model means no prediction. A high confidence score is not the same as guaranteed correctness; use the held-out test accuracy to report model performance in the project.


## Secure video-call deployment

See `SECURE_VIDEO_DEPLOYMENT.md`.

For final laptop + mobile testing, deploy the app on a trusted HTTPS hostname. Do not use a self-signed certificate on a LAN IP as the production solution. Configure MongoDB Atlas and a TURN provider through environment variables.
