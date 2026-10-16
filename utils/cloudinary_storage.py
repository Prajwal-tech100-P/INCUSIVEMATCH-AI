"""Cloudinary profile photo storage and validation utility.

Handles persistent image storage on Cloudinary, secure HTTPS URL generation,
file validation (format, size, content verification via Pillow), and safe
replacement/deletion workflows.
"""
import os
import logging
from typing import Tuple, Optional, Dict, Any
from io import BytesIO
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api

logger = logging.getLogger(__name__)

CLOUDINARY_FOLDER = "inclusivematch/profile_photos"
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
PERMITTED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
PERMITTED_PIL_FORMATS = {"PNG", "JPEG", "GIF", "WEBP"}


def _clean_str(val: Optional[str]) -> str:
    """Strip whitespace and surrounding single/double quotes from a string."""
    if not val:
        return ""
    s = str(val).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def get_cloudinary_credentials() -> Dict[str, str]:
    """Retrieve and clean Cloudinary credentials from environment variables."""
    cloud_name = _clean_str(os.environ.get("CLOUDINARY_CLOUD_NAME"))
    api_key = _clean_str(os.environ.get("CLOUDINARY_API_KEY"))
    api_secret = _clean_str(os.environ.get("CLOUDINARY_API_SECRET"))
    if cloud_name and api_key and api_secret:
        return {
            "cloud_name": cloud_name,
            "api_key": api_key,
            "api_secret": api_secret,
        }

    cloudinary_url = _clean_str(os.environ.get("CLOUDINARY_URL"))
    if cloudinary_url and cloudinary_url.startswith("cloudinary://"):
        return {"cloudinary_url": cloudinary_url}

    return {}


def is_cloudinary_configured() -> bool:
    """Return True if all required Cloudinary environment variables are present and non-empty."""
    return bool(get_cloudinary_credentials())


def configure_cloudinary() -> None:
    """Configure Cloudinary SDK with environment variables."""
    creds = get_cloudinary_credentials()
    if not creds:
        raise RuntimeError(
            "Cloudinary credentials are not configured. "
            "Please set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET."
        )

    if "cloudinary_url" in creds:
        cloudinary.config(cloudinary_url=creds["cloudinary_url"], secure=True)
    else:
        cloudinary.config(
            cloud_name=creds["cloud_name"],
            api_key=creds["api_key"],
            api_secret=creds["api_secret"],
            secure=True,
        )


def extract_public_id_from_url(url: Optional[str]) -> Optional[str]:
    """Extract Cloudinary public_id from a Cloudinary HTTPS/HTTP URL.

    Example:
        'https://res.cloudinary.com/cloud/image/upload/v12345/inclusivematch/profile_photos/xyz.png'
        -> 'inclusivematch/profile_photos/xyz'
    """
    if not url or not isinstance(url, str) or "res.cloudinary.com" not in url:
        return None
    import re
    match = re.search(r"/image/upload/(?:s--[^/]+--/)?(?:v\d+/)?([^\?\#]+)", url)
    if match:
        full_path = match.group(1)
        if "." in full_path:
            full_path = full_path.rsplit(".", 1)[0]
        return full_path
    return None


def validate_image_file(file_storage) -> Tuple[bool, Optional[str]]:
    """Validate uploaded file for filename extension, max 5 MB size, and image content integrity.

    Returns:
        (is_valid, error_message): Tuple where is_valid is a boolean and error_message is
        None if valid, or a descriptive error string if invalid.
    """
    if not file_storage or not file_storage.filename:
        return False, "No file was selected for upload."

    filename = file_storage.filename
    if "." not in filename:
        return False, "File must have an extension."

    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in PERMITTED_EXTENSIONS:
        return (
            False,
            f"Invalid file extension '.{ext}'. Permitted formats are: "
            + ", ".join(sorted(PERMITTED_EXTENSIONS)).upper()
            + ".",
        )

    # Validate file size (max 5 MB)
    try:
        file_storage.seek(0, os.SEEK_END)
        file_size = file_storage.tell()
        file_storage.seek(0)
    except Exception as e:
        return False, f"Could not determine file size: {e}"

    if file_size > MAX_IMAGE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return (
            False,
            f"File size ({size_mb:.2f} MB) exceeds the maximum allowed limit of 5 MB.",
        )

    if file_size == 0:
        return False, "Uploaded file is empty (0 bytes)."

    # Validate actual image content using Pillow
    try:
        # Read file into buffer or verify directly
        img = Image.open(file_storage)
        img.verify()
        detected_format = (img.format or "").upper()
        file_storage.seek(0)

        if detected_format not in PERMITTED_PIL_FORMATS:
            return (
                False,
                f"Image content format '{detected_format}' is not permitted. "
                "Only PNG, JPEG, GIF, and WEBP image files are allowed.",
            )
    except Exception as e:
        file_storage.seek(0)
        return False, f"Invalid or corrupt image content: {e}"

    return True, None


def upload_profile_photo(file_storage, folder: str = CLOUDINARY_FOLDER) -> Dict[str, Any]:
    """Upload an image file to Cloudinary in the designated folder.

    Args:
        file_storage: File-like object (e.g. Werkzeug FileStorage or BytesIO)
        folder: Cloudinary destination folder (default: inclusivematch/profile_photos)

    Returns:
        Dict containing 'secure_url' and 'public_id'.

    Raises:
        RuntimeError: If Cloudinary is not configured.
        Exception: If the upload to Cloudinary fails.
    """
    configure_cloudinary()
    file_storage.seek(0)

    result = cloudinary.uploader.upload(
        file_storage,
        folder=folder,
        resource_type="image",
        overwrite=False,
        unique_filename=True,
    )

    secure_url = result.get("secure_url")
    public_id = result.get("public_id")

    if not secure_url or not public_id:
        raise RuntimeError("Cloudinary upload did not return a valid secure_url and public_id.")

    return {
        "secure_url": secure_url,
        "public_id": public_id,
        "format": result.get("format"),
        "width": result.get("width"),
        "height": result.get("height"),
    }


def delete_profile_photo(public_id_or_url: Optional[str]) -> bool:
    """Delete a profile photo from Cloudinary by its public ID or full URL.

    Args:
        public_id_or_url: The Cloudinary public ID string or full Cloudinary URL.

    Returns:
        True if deleted or skipped safely, False on failure.
    """
    if not public_id_or_url or not isinstance(public_id_or_url, str):
        return True

    target_id = public_id_or_url
    if "res.cloudinary.com" in public_id_or_url:
        extracted = extract_public_id_from_url(public_id_or_url)
        if extracted:
            target_id = extracted
        else:
            logger.warning("Could not extract public ID from Cloudinary URL: %s", public_id_or_url)
            return False

    if not is_cloudinary_configured():
        logger.warning(
            "Cannot delete Cloudinary image '%s': Cloudinary is not configured.",
            target_id,
        )
        return False

    try:
        configure_cloudinary()
        res = cloudinary.uploader.destroy(target_id, resource_type="image", invalidate=True)
        status = res.get("result")
        logger.info("Cloudinary deletion for '%s': result=%s", target_id, status)
        return status in {"ok", "not found"}
    except Exception as e:
        logger.error("Error deleting image '%s' from Cloudinary: %s", target_id, e)
        return False
