"""
extract_landmarks.py - MediaPipe Holistic Landmark Extraction and Preprocessing

Transforms raw sign language video into compact, normalized (192, 64, 2) NumPy arrays.
Strictly isolated within experimental/hwgat/
"""
import os
import sys
import argparse
import av
import numpy as np
from typing import Tuple, Dict, Any, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from demo_utils import CFG
from pose_modules.Mediapipe import Model as HolisticPoseModel
import dataset_utils as du


def validate_landmarks_array(arr: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that the extracted landmark array meets HWGAT tensor requirements:
    - Shape must be exactly (192, 64, 2)
    - Must not contain NaN or Inf values
    - Must contain non-zero motion/landmarks
    """
    if not isinstance(arr, np.ndarray):
        return False, f"Expected numpy.ndarray, got {type(arr)}"

    if arr.shape != (192, 64, 2):
        return False, f"Shape mismatch: expected (192, 64, 2), got {arr.shape}"

    if np.isnan(arr).any():
        return False, "Array contains NaN values"

    if np.isinf(arr).any():
        return False, "Array contains Inf values"

    non_zero_ratio = np.count_nonzero(arr) / arr.size
    if non_zero_ratio < 0.01:
        return False, f"Insufficient landmark detections (non-zero ratio: {non_zero_ratio:.4f})"

    return True, "Valid"


def extract_landmarks_from_video(
    video_path: str,
    output_npy_path: Optional[str] = None,
    pose_model: Optional[HolisticPoseModel] = None,
    cfg: Optional[CFG] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process a video file with PyAV + MediaPipe Holistic and apply CFG test_transform.
    Returns the normalized (192, 64, 2) array and extraction metadata.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if cfg is None:
        cfg = CFG()

    if pose_model is None:
        pose_model = HolisticPoseModel()

    container = av.open(video_path)
    stream = container.streams.video[0]

    raw_features = []
    frame_count = 0
    duration_sec = float(stream.duration * stream.time_base) if stream.duration else 0.0

    for frame in container.decode(stream):
        img_rgb = frame.to_ndarray(format="rgb24")
        h, w, _ = img_rgb.shape
        feature = pose_model.forward(img_rgb, h, w)
        raw_features.append(feature)
        frame_count += 1

    container.close()

    if frame_count == 0 or len(raw_features) == 0:
        raise ValueError(f"No frames could be decoded from {video_path}")

    # Apply HWGAT test_transform pipeline (LinearInterpolation, Preprocessing, Normalization)
    landmarks_array = np.array(cfg.test_transform(raw_features), dtype=np.float32)

    is_valid, reason = validate_landmarks_array(landmarks_array)
    if not is_valid:
        raise ValueError(f"Extracted landmarks failed validation: {reason}")

    if output_npy_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_npy_path)), exist_ok=True)
        np.save(output_npy_path, landmarks_array)

    meta = {
        "frames": frame_count,
        "duration_sec": duration_sec,
        "shape": list(landmarks_array.shape),
        "is_valid": is_valid,
        "validation_message": reason,
    }

    return landmarks_array, meta


def process_video_entry(
    video_path: str,
    class_name: str,
    delete_raw_confirm: bool = False,
    pose_model: Optional[HolisticPoseModel] = None,
    cfg: Optional[CFG] = None,
) -> str:
    """
    Extract landmarks from a video, save to custom_dataset/landmarks/<class_name>/,
    record in manifest, and optionally delete the raw video if confirmed and verified.
    """
    du.init_dataset_structure()
    class_info = du.register_or_get_class(class_name)

    sample_id = os.path.splitext(os.path.basename(video_path))[0]
    out_npy_dir = os.path.join(du.LANDMARKS_DIR, class_name)
    out_npy_path = os.path.join(out_npy_dir, f"{sample_id}.npy")

    arr, meta = extract_landmarks_from_video(
        video_path=video_path,
        output_npy_path=out_npy_path,
        pose_model=pose_model,
        cfg=cfg,
    )

    # Register in manifest
    du.record_sample_in_manifest(
        sample_id=sample_id,
        class_name=class_name,
        class_id=class_info["class_id"],
        is_custom=class_info["is_custom"],
        video_path=video_path,
        npy_path=out_npy_path,
        frames=meta["frames"],
        duration_sec=meta["duration_sec"],
        valid_landmarks=meta["is_valid"],
    )
    du.increment_class_sample_count(class_name)
    du.update_collection_summary()

    # Safety-guarded raw video deletion
    if delete_raw_confirm:
        # Strict validation checks before deletion
        if (
            os.path.exists(out_npy_path)
            and os.path.getsize(out_npy_path) > 0
            and meta["is_valid"]
        ):
            loaded = np.load(out_npy_path)
            val_ok, _ = validate_landmarks_array(loaded)
            if val_ok:
                os.remove(video_path)
                print(f"[STORAGE SAVED] Deleted verified raw video: {video_path}")
            else:
                print(f"[SAFETY PRESERVED] Saved .npy failed reload check. Raw video kept: {video_path}")
        else:
            print(f"[SAFETY PRESERVED] Landmark verification incomplete. Raw video kept: {video_path}")

    return out_npy_path


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic landmarks for HWGAT")
    parser.add_argument("--input", type=str, help="Path to single raw video file")
    parser.add_argument("--class_name", type=str, help="Class/sign label name")
    parser.add_argument("--all", action="store_true", help="Batch extract all videos in custom_dataset/raw_videos/")
    parser.add_argument(
        "--delete_raw_confirm",
        action="store_true",
        help="CONFIRMATION REQUIRED: Delete raw video only after successful .npy extraction and verification",
    )
    args = parser.parse_args()

    du.init_dataset_structure()

    if args.input:
        if not args.class_name:
            print("ERROR: --class_name is required when --input is specified.")
            sys.exit(1)
        print(f"Extracting landmarks from: {args.input} (Class: {args.class_name})")
        out_path = process_video_entry(
            video_path=args.input,
            class_name=args.class_name,
            delete_raw_confirm=args.delete_raw_confirm,
        )
        print(f"[SUCCESS] Saved landmarks to: {out_path}")

    elif args.all:
        raw_root = du.RAW_DIR
        video_files = []
        for root, _, files in os.walk(raw_root):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    cname = os.path.basename(root)
                    if cname and cname != "raw_videos":
                        video_files.append((os.path.join(root, f), cname))

        if not video_files:
            print(f"No raw video files found in {raw_root}")
            return

        print(f"Found {len(video_files)} video(s) to extract...")
        pose_model = HolisticPoseModel()
        cfg = CFG()

        for idx, (vpath, cname) in enumerate(video_files, 1):
            sample_id = os.path.splitext(os.path.basename(vpath))[0]
            expected_npy = os.path.join(du.LANDMARKS_DIR, cname, f"{sample_id}.npy")
            if os.path.exists(expected_npy):
                print(f"[{idx}/{len(video_files)}] Already extracted: {sample_id}.npy")
                continue

            try:
                print(f"[{idx}/{len(video_files)}] Processing {vpath}...")
                process_video_entry(
                    video_path=vpath,
                    class_name=cname,
                    delete_raw_confirm=args.delete_raw_confirm,
                    pose_model=pose_model,
                    cfg=cfg,
                )
            except Exception as e:
                print(f"[ERROR] Failed extracting {vpath}: {e}")

        print("Batch landmark extraction complete.")
        stats = du.get_storage_statistics()
        print(f"Total Storage: Raw {stats['raw_videos_mb']} MB vs Landmarks {stats['landmarks_mb']} MB")
        print(f"Storage Savings: {stats['savings_percentage']}%")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
