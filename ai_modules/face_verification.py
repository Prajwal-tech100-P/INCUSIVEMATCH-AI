"""Secure Facial Verification Module for InclusiveMatch AI.

Uses:
- YuNet (cv2.FaceDetectorYN): Fast deep CNN for face detection and 5-point facial landmark alignment.
- SFace (cv2.FaceRecognizerSF): Deep CNN extracting 128-dimensional facial feature embeddings.

Strict Verification Rules:
1. Both images (live capture and reference profile photo) must be valid, decodable images.
2. Require exactly ONE clear face in the reference profile photo.
3. Require exactly ONE clear face in the live camera capture.
4. Reject images with no detectable face (0 faces) with clear feedback.
5. Reject images with multiple faces (>1 faces) to prevent spoofing and identity confusion.
6. Identity match is verified using SFace cosine similarity with a validated threshold (0.363).
7. Never uses Haar-cascades or ORB keypoint matching for the final identity decision.
"""
import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Model paths in the models/ directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Validated SFace cosine threshold recommended by OpenCV Zoo
DEFAULT_COSINE_THRESHOLD = 0.363
# Score threshold for YuNet face detector confidence
DEFAULT_DETECTION_SCORE_THRESHOLD = 0.65
DEFAULT_NMS_THRESHOLD = 0.3


class FaceVerifier:
    """Production-grade face detection and deep facial embedding verification."""

    def __init__(self, yunet_path=None, sface_path=None, cosine_threshold=DEFAULT_COSINE_THRESHOLD):
        self.yunet_path = yunet_path or YUNET_PATH
        self.sface_path = sface_path or SFACE_PATH
        self.cosine_threshold = cosine_threshold
        self.detector = None
        self.recognizer = None
        self._initialized = False
        self._init_models()

    def _init_models(self):
        """Load deep neural network models for face detection and recognition."""
        if not os.path.exists(self.yunet_path) or not os.path.exists(self.sface_path):
            logger.warning("YuNet or SFace model file not found in %s", MODELS_DIR)
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                self.yunet_path,
                "",
                (320, 320),
                score_threshold=DEFAULT_DETECTION_SCORE_THRESHOLD,
                nms_threshold=DEFAULT_NMS_THRESHOLD,
            )
            self.recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")
            self._initialized = True
            logger.info("FaceVerifier initialized with YuNet and SFace models.")
        except Exception as e:
            logger.error("Failed to initialize YuNet / SFace models: %s", e)
            self._initialized = False

    def _decode_image(self, image_data):
        """Decode raw bytes into a BGR OpenCV numpy image with dimension and size validation."""
        if not image_data or not isinstance(image_data, (bytes, bytearray)):
            return None, "Empty or invalid image data."
        if len(image_data) > 10 * 1024 * 1024:
            return None, "Image exceeds maximum allowed size (10 MB)."

        try:
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return None, "Failed to decode image. Format may be corrupt or unsupported."
            h, w = img.shape[:2]
            if h < 40 or w < 40:
                return None, "Image resolution is too low for reliable facial verification."
            return img, None
        except Exception as e:
            return None, f"Image decoding error: {e}"

    def detect_and_extract(self, img, source_label="image"):
        """Detect faces, enforce single-face constraint, align and extract 128-D embedding."""
        if not self._initialized:
            return None, None, {"verified": False, "reason": "Face verification service is currently unavailable."}

        h, w = img.shape[:2]
        # YuNet requires explicit input size set matching image dimensions
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)

        num_faces = len(faces) if faces is not None else 0

        if num_faces == 0:
            if "live" in source_label.lower():
                reason = "No face detected in live camera capture. Please face the camera directly with good lighting."
            else:
                reason = "No face detected in reference profile photo. Please upload a clear photo of your face."
            return None, None, {"verified": False, "reason": reason}

        if num_faces > 1:
            if "live" in source_label.lower():
                reason = "Multiple faces detected in live camera. Please ensure only your face is visible."
            else:
                reason = "Multiple faces detected in profile photo. Please upload a photo containing only yourself."
            return None, None, {"verified": False, "reason": reason}

        face = faces[0]
        conf = float(face[-1]) if len(face) > 0 else 0.0
        if conf < DEFAULT_DETECTION_SCORE_THRESHOLD:
            return None, None, {"verified": False, "reason": f"Face in {source_label} is not clear enough (confidence {conf:.2f})."}

        try:
            aligned_face = self.recognizer.alignCrop(img, face)
            embedding = self.recognizer.feature(aligned_face)
            return face, embedding, None
        except Exception as e:
            logger.error("Error extracting face features: %s", e)
            return None, None, {"verified": False, "reason": f"Could not process facial landmarks in {source_label}."}

    def verify(self, live_image_data, profile_image_data):
        """Perform end-to-end identity comparison between live capture and profile photo.

        Returns:
            dict: {
                "verified": bool,
                "score": float (optional),
                "threshold": float (optional),
                "reason": str
            }
        """
        if not self._initialized:
            return {"verified": False, "reason": "Face recognition models are not loaded on the server."}

        # 1. Decode live camera capture
        live_img, live_err = self._decode_image(live_image_data)
        if live_err:
            return {"verified": False, "reason": f"Live camera capture error: {live_err}"}

        # 2. Decode reference profile photo
        profile_img, profile_err = self._decode_image(profile_image_data)
        if profile_err:
            return {"verified": False, "reason": f"Profile photo error: {profile_err}"}

        # 3. Detect & extract single face from live capture
        live_face, live_emb, live_fail = self.detect_and_extract(live_img, source_label="live camera")
        if live_fail:
            return live_fail

        # 4. Detect & extract single face from profile photo
        profile_face, profile_emb, profile_fail = self.detect_and_extract(profile_img, source_label="profile photo")
        if profile_fail:
            return profile_fail

        # 5. Compute Deep Cosine Similarity
        try:
            cosine_score = float(self.recognizer.match(live_emb, profile_emb, cv2.FaceRecognizerSF_FR_COSINE))
            l2_distance = float(self.recognizer.match(live_emb, profile_emb, cv2.FaceRecognizerSF_FR_NORM_L2))
        except Exception as e:
            logger.error("Error matching facial embeddings: %s", e)
            return {"verified": False, "reason": "Failed to compare facial embeddings."}

        logger.info(
            "Face verification comparison completed: cosine_score=%.4f (threshold=%.4f), l2_dist=%.4f",
            cosine_score, self.cosine_threshold, l2_distance
        )

        # 6. Verification decision based strictly on deep feature similarity threshold
        if cosine_score >= self.cosine_threshold:
            return {
                "verified": True,
                "score": round(cosine_score, 4),
                "threshold": self.cosine_threshold,
                "reason": f"Face verified successfully (similarity: {cosine_score:.2f})."
            }
        else:
            return {
                "verified": False,
                "score": round(cosine_score, 4),
                "threshold": self.cosine_threshold,
                "reason": f"Face does not match your profile photo (similarity: {cosine_score:.2f}, required: {self.cosine_threshold}). Please try again with clear lighting and a frontal angle."
            }
