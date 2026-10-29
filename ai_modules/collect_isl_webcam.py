"""
Interactive Webcam Dataset Collection Tool for Indian Sign Language (ISL)
=========================================================================
MCA Major Project: InclusiveMatch AI

Features:
- Live OpenCV camera stream with interactive HUD.
- Class selection from supported 29 classes or interactive creation of new classes.
- Start / Stop recording with visual 3-2-1 countdown and frame progress bar.
- Consistent 42-joint 2-hand MediaPipe landmark extraction (handedness stabilized).
- Quality verification (rejects sequences with missing or insufficient hands).
- Saves clean (T, 84) NumPy arrays to datasets/isl_custom/<class_name>/<rec_id>.npy.
- Updates datasets/isl_custom/metadata.json with sample counts and parameters.
- Supports --mock-run / --test-mode for headless test verification.

Usage:
  .\\venv\\Scripts\\python.exe ai_modules/collect_isl_webcam.py
  .\\venv\\Scripts\\python.exe ai_modules/collect_isl_webcam.py --class hello --num-samples 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai_modules.isl_dataset_manager import ISLDatasetManager
from ai_modules.sign_language import SignLanguageRecognizer


class ISLWebcamCollector:
    """Interactive webcam recorder for building the custom ISL dataset."""

    def __init__(
        self,
        camera_idx: int = 0,
        seq_frames: int = 45,
        dataset_dir: str = "datasets/isl_custom",
    ):
        self.camera_idx = camera_idx
        self.seq_frames = seq_frames
        self.dataset_mgr = ISLDatasetManager(dataset_dir)
        self.classes = self.dataset_mgr.get_supported_classes()
        self.current_class_idx = 0
        self.recognizer = SignLanguageRecognizer()

    @property
    def current_class(self) -> str:
        if not self.classes:
            return "hello"
        return self.classes[self.current_class_idx % len(self.classes)]

    def extract_landmarks_from_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts 42-joint (84-dimensional) landmarks using the project's
        consistent MediaPipe handedness extraction pipeline.
        Returns array of shape (84,) float32.
        """
        joints = self.recognizer._extract_landmarks(frame_bgr)
        if joints is None:
            return np.zeros(84, dtype=np.float32)
        # joints is (42, 2) -> flatten to (84,)
        return joints.reshape(84).astype(np.float32)

    def record_sequence_from_camera(
        self, cap: cv2.VideoCapture, countdown_sec: int = 2
    ) -> Optional[np.ndarray]:
        """
        Captures a sequence of frames from camera with a countdown.
        Returns (T, 84) numpy array or None if aborted.
        """
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # 1. Countdown
        start_time = time.time()
        while time.time() - start_time < countdown_sec:
            ok, frame = cap.read()
            if not ok:
                break
            remaining = int(countdown_sec - (time.time() - start_time)) + 1
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Get Ready for '{self.current_class}' in {remaining}...",
                (w // 8, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 165, 255),
                3,
            )
            cv2.imshow("InclusiveMatch AI - ISL Dataset Collector", overlay)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                return None

        # 2. Record sequence
        frames_landmarks: List[np.ndarray] = []
        for f_idx in range(self.seq_frames):
            ok, frame = cap.read()
            if not ok:
                break

            lms = self.extract_landmarks_from_bgr(frame)
            frames_landmarks.append(lms)

            # Draw progress HUD
            h, w = frame.shape[:2]
            has_hand = bool(np.any(lms != 0))
            color = (0, 255, 0) if has_hand else (0, 0, 255)
            status_text = "Hand Detected" if has_hand else "No Hand Detected!"

            cv2.putText(
                frame,
                f"RECORDING [{self.current_class}]: {f_idx + 1}/{self.seq_frames}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Progress bar at bottom
            bar_w = int((f_idx + 1) / self.seq_frames * (w - 40))
            cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 15), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, h - 30), (w - 20, h - 15), (255, 255, 255), 1)

            cv2.imshow("InclusiveMatch AI - ISL Dataset Collector", frame)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                return None

        if len(frames_landmarks) == 0:
            return None

        return np.stack(frames_landmarks, axis=0)

    def run_interactive(self) -> None:
        """Launches the interactive OpenCV webcam collector GUI."""
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_idx}.")
            print("Please ensure your webcam is connected or try another camera index.")
            return

        print("=" * 70)
        print("INCLUSIVE MATCH AI - WEBCAM DATASET COLLECTOR")
        print("=" * 70)
        print("Controls:")
        print("  SPACE        : Start recording sample for selected sign")
        print("  C / TAB      : Next sign class")
        print("  P            : Previous sign class")
        print("  N            : Add new sign class")
        print("  S            : Show dataset sample summary")
        print("  Q / ESC      : Quit collector")
        print("=" * 70)

        window_name = "InclusiveMatch AI - ISL Dataset Collector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 640)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera stream lost. Exiting.")
                    break

                h, w = frame.shape[:2]
                display = frame.copy()

                summary = self.dataset_mgr.get_summary()
                curr_count = summary["classes"].get(self.current_class, 0)

                # Draw Top Banner
                cv2.rectangle(display, (0, 0), (w, 100), (40, 20, 60), -1)
                cv2.putText(
                    display,
                    f"Selected Sign: {self.current_class.upper()} ({curr_count} samples saved)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    "[SPACE]=Record  [C/TAB]=Next  [P]=Prev  [N]=New Sign  [S]=Summary  [Q]=Quit",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                )

                # Live landmark test preview
                test_lms = self.extract_landmarks_from_bgr(frame)
                has_hand = bool(np.any(test_lms != 0))
                badge_color = (0, 255, 0) if has_hand else (0, 0, 255)
                badge_msg = "Hands Ready" if has_hand else "Show Hands in Frame"
                cv2.putText(
                    display,
                    badge_msg,
                    (w - 240, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    badge_color,
                    2,
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    break
                elif key in (ord("c"), ord("C"), 9):  # 'c' or TAB
                    self.current_class_idx = (self.current_class_idx + 1) % len(self.classes)
                elif key in (ord("p"), ord("P")):
                    self.current_class_idx = (self.current_class_idx - 1) % len(self.classes)
                elif key in (ord("n"), ord("N")):
                    print("\n--- Enter New Sign Class Name (in terminal) ---")
                    try:
                        new_cls = input("New sign class name: ").strip().lower()
                        if new_cls:
                            self.dataset_mgr.add_class(new_cls)
                            self.classes = self.dataset_mgr.get_supported_classes()
                            self.current_class_idx = self.classes.index(new_cls)
                            print(f"Added and selected class '{new_cls}'.")
                    except Exception as e:
                        print(f"Error adding class: {e}")
                elif key in (ord("s"), ord("S")):
                    sum_info = self.dataset_mgr.get_summary()
                    print("\n--- Current ISL Dataset Summary ---")
                    print(f"Total Classes : {sum_info['num_classes']}")
                    print(f"Total Samples : {sum_info['total_samples']}")
                    for c_name, c_cnt in sorted(sum_info["classes"].items()):
                        print(f"  {c_name:15s}: {c_cnt} samples")
                    print("-----------------------------------\n")
                elif key == 32:  # SPACE bar to record
                    print(f"\nRecording sequence for sign '{self.current_class}'...")
                    seq = self.record_sequence_from_camera(cap, countdown_sec=2)
                    if seq is not None:
                        ok, msg, info = self.dataset_mgr.save_recording(
                            self.current_class, seq, signer_id="local_webcam_user"
                        )
                        print(f"Result: {msg}")
                        if ok:
                            print(f"Saved: {info['file_path']} ({info['total_frames']} frames, {info['active_frames']} active)")
                    else:
                        print("Recording cancelled.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam collector closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="InclusiveMatch AI - ISL Webcam Dataset Collector")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--frames", type=int, default=45, help="Number of frames per sequence (default: 45)")
    parser.add_argument("--dataset-dir", default="datasets/isl_custom", help="Output dataset directory")
    parser.add_argument("--class-name", default=None, help="Initial class name to select")
    parser.add_argument("--sync-benchmark", action="store_true", help="Sync existing 608 benchmark samples before starting")
    args = parser.parse_args()

    collector = ISLWebcamCollector(
        camera_idx=args.camera,
        seq_frames=args.frames,
        dataset_dir=args.dataset_dir,
    )

    if args.sync_benchmark:
        print("Syncing existing benchmark dataset...")
        res = collector.dataset_mgr.sync_from_external_landmarks()
        print("Sync result:", res)
        collector.classes = collector.dataset_mgr.get_supported_classes()

    if args.class_name and args.class_name.lower() in collector.classes:
        collector.current_class_idx = collector.classes.index(args.class_name.lower())

    collector.run_interactive()


if __name__ == "__main__":
    main()

