class FaceVerifier:
    """Integration point for identity verification.

    Do not claim a user is verified until a real face-verification model/service
    compares a live capture with the profile image.
    """
    def verify(self, image_data, profile_image_data):
        return {"verified": False, "reason": "Face verification model is not configured."}
