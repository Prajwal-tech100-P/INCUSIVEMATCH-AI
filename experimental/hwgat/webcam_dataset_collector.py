"""
webcam_dataset_collector.py - Interactive Webcam Sign Language Dataset Collector

Features:
- DirectShow (cv2.CAP_DSHOW) fast webcam initialization
- Vocabulary search across 2,002 base classes + Custom class creation
- Live sample tracking and <10 samples warning
- Real-time on-screen HUD (Standby, Recording, Processing, Saved)
- Automated landmark extraction and manifest updates
- Storage-efficient immediate .npy generation

Strictly isolated within experimental/hwgat/
"""
import os
import sys
import time
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import dataset_utils as du
from extract_landmarks import process_video_entry


def select_or_create_label() -> str:
    """CLI menu to search base vocabulary or enter a custom sign label."""
    du.init_dataset_structure()
    base_vocab = du.load_base_vocabulary()
    mapping = du.load_class_mapping()

    print("\n" + "=" * 65)
    print("  HWGAT SIGN LANGUAGE DATASET COLLECTOR - LABEL SELECTION")
    print("=" * 65)
    print("1. Search from 2,002 base vocabulary classes")
    print("2. Enter a new Custom Sign class name")
    print("3. View current collected classes & sample counts")
    print("=" * 65)

    choice = input("Select an option (1/2/3) [default: 2]: ").strip()

    if choice == "1":
        query = input("Enter search keyword (e.g. 'book', 'hello', 'water'): ").strip().lower()
        matches = [name for name in base_vocab.values() if query in name.lower()]
        if not matches:
            print(f"No classes found matching '{query}'.")
            return select_or_create_label()

        print(f"\nFound {len(matches)} matching classes (showing up to 20):")
        for i, m in enumerate(matches[:20], 1):
            info = mapping.get(m, {})
            cnt = info.get("sample_count", 0)
            print(f"  {i}. {m} (Current samples: {cnt})")

        sel = input("Select number or type full name: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(matches[:20]):
            return matches[int(sel) - 1]
        elif sel in matches:
            return sel
        else:
            print("Invalid selection.")
            return select_or_create_label()

    elif choice == "3":
        collected = {c: info["sample_count"] for c, info in mapping.items() if info["sample_count"] > 0}
        if not collected:
            print("No samples collected yet.")
        else:
            print(f"\nCurrently collected classes ({len(collected)}):")
            for c, cnt in sorted(collected.items(), key=lambda x: x[1], reverse=True):
                warn = " [WARNING: < 10 samples]" if cnt < 10 else " [READY]"
                print(f"  - {c}: {cnt} samples{warn}")
        return select_or_create_label()

    else:
        label = input("Enter custom sign name (e.g. 'Namaste', 'ThankYou', 'Friend'): ").strip()
        if not label:
            print("Label name cannot be empty.")
            return select_or_create_label()
        # Clean label name
        clean_label = label.replace(" ", "_")
        return clean_label


def record_sign_clip(
    class_name: str,
    camera_index: int = 0,
    record_duration: float = 3.5,
    auto_extract: bool = True,
    delete_raw_confirm: bool = False,
) -> None:
    """
    Open webcam with DirectShow, show HUD, and record a sign clip on SPACE press.
    """
    du.init_dataset_structure()
    class_info = du.register_or_get_class(class_name)
    current_count = class_info["sample_count"]

    print("\n" + "=" * 65)
    print(f"Target Sign Class: {class_name}")
    print(f"Current Samples:   {current_count}")
    if current_count < 10:
        print(f">>> ATTENTION: Only {current_count}/10 recommended samples collected for '{class_name}'.")
    else:
        print(f">>> Good! {current_count} samples collected for '{class_name}'.")
    print("=" * 65)
    print("Controls in Camera Window:")
    print("  [SPACE] - Start recording sign gesture")
    print("  [Q]     - Quit / Finish session")
    print("=" * 65)

    # Initialize DirectShow camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index} with DirectShow. Falling back to default...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[FATAL] Cannot access webcam on index {camera_index}.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    state = "STANDBY"
    recorded_frames = []
    record_start_time = 0.0

    class_raw_dir = os.path.join(du.RAW_DIR, class_name)
    os.makedirs(class_raw_dir, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame.")
                break

            display = frame.copy()
            now = time.time()

            # Handle states
            if state == "STANDBY":
                cv2.rectangle(display, (10, 10), (w - 10, 80), (0, 0, 0), -1)
                status_text = f"Class: {class_name} | Samples: {current_count}"
                sample_color = (0, 165, 255) if current_count < 10 else (0, 255, 0)
                cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sample_color, 2)
                cv2.putText(display, "Press SPACE to Record | Q to Quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            elif state == "RECORDING":
                elapsed = now - record_start_time
                remaining = max(0.0, record_duration - elapsed)
                recorded_frames.append(frame.copy())

                # Recording HUD
                cv2.rectangle(display, (10, 10), (w - 10, 80), (0, 0, 180), -1)
                cv2.circle(display, (35, 45), 12, (0, 0, 255), -1)
                cv2.putText(display, f"RECORDING '{class_name}' ({remaining:.1f}s)", (60, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Progress bar
                prog = min(1.0, elapsed / record_duration)
                cv2.rectangle(display, (10, 82), (int(10 + (w - 20) * prog), 88), (0, 255, 0), -1)

                if elapsed >= record_duration:
                    state = "SAVING"

            elif state == "SAVING":
                cv2.rectangle(display, (10, 10), (w - 10, 80), (180, 100, 0), -1)
                cv2.putText(display, "Processing & Extracting Landmarks...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("HWGAT Dataset Collector", display)
                cv2.waitKey(1)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                sample_id = f"{class_name}_{timestamp}"
                video_filename = f"{sample_id}.mp4"
                video_path = os.path.join(class_raw_dir, video_filename)

                # Write out video via OpenCV VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                for f in recorded_frames:
                    writer.write(f)
                writer.release()

                print(f"\n[SAVED VIDEO] {video_path} ({len(recorded_frames)} frames)")

                # Extract landmarks immediately
                if auto_extract:
                    try:
                        print("Extracting MediaPipe landmarks to .npy...")
                        npy_path = process_video_entry(
                            video_path=video_path,
                            class_name=class_name,
                            delete_raw_confirm=delete_raw_confirm,
                        )
                        print(f"[EXTRACTED LANDMARKS] Saved: {npy_path}")
                        current_count += 1
                    except Exception as e:
                        print(f"[ERROR] Landmark extraction failed: {e}")

                recorded_frames = []
                state = "STANDBY"

            cv2.imshow("HWGAT Dataset Collector", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # ESC or 'q'
                break
            elif key == ord(" ") and state == "STANDBY":
                state = "RECORDING"
                record_start_time = time.time()
                recorded_frames = [frame.copy()]

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nCollection session ended.")
    summary = du.update_collection_summary()
    print(f"Total samples for '{class_name}': {summary['class_distribution'].get(class_name, 0)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HWGAT Webcam Sign Dataset Collector")
    parser.add_argument("--class_name", type=str, help="Sign class name (if not provided, interactive prompt will ask)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--duration", type=float, default=3.5, help="Recording duration per clip in seconds (default: 3.5s)")
    parser.add_argument("--delete_raw", action="store_true", help="Delete raw video after successful verified .npy extraction")
    args = parser.parse_args()

    class_name = args.class_name
    if not class_name:
        class_name = select_or_create_label()

    record_sign_clip(
        class_name=class_name,
        camera_index=args.camera,
        record_duration=args.duration,
        auto_extract=True,
        delete_raw_confirm=args.delete_raw,
    )


if __name__ == "__main__":
    main()

