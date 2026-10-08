import cv2
import numpy as np

class FaceVerifier:
    """Integration point for identity verification.

    Uses OpenCV Haarcascades for face detection and ORB feature matching 
    for face comparison.
    """
    def __init__(self):
        # Load OpenCV's pre-trained Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _extract_face(self, img_array):
        # Convert byte array to cv2 image
        np_arr = np.frombuffer(img_array, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None

        # Take the largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        return cv2.resize(face_img, (150, 150))

    def verify(self, live_image_data, profile_image_data):
        live_face = self._extract_face(live_image_data)
        if live_face is None:
            return {"verified": False, "reason": "No face detected in live camera."}

        profile_face = self._extract_face(profile_image_data)
        if profile_face is None:
            return {"verified": False, "reason": "No face detected in profile photo."}

        # Compute ORB keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(live_face, None)
        kp2, des2 = self.orb.detectAndCompute(profile_face, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return {"verified": False, "reason": "Not enough facial features found to compare."}

        # Match descriptors
        matches = self.matcher.match(des1, des2)
        
        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate a similarity score based on good matches
        # A good match might have distance < 60
        good_matches = [m for m in matches if m.distance < 65]
        
        score = len(good_matches) / min(len(des1), len(des2))

        # We set a threshold for matching
        threshold = 0.12 # heuristic for ORB face matching

        if score >= threshold:
            return {"verified": True, "reason": f"Face matches (score: {score:.2f})"}
        else:
            return {"verified": False, "reason": "Faces do not match."}

