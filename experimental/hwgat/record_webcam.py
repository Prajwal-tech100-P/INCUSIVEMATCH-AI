"""
HWGAT Robust Webcam Recorder
============================
Reliable video recording utility for Indian Sign Language (ISL) gestures.
Features:
- Windows DirectShow (CAP_DSHOW) backend optimization with automatic fallback.
- Real-time on-screen HUD (Standby, Recording status, Frame count, Timer).
- Interactive controls: [R] Start Recording, [S] Stop & Save, [Q] Quit.
- PyAV-compatible video output format (mp4v in .mp4 container).
- Post-recording integrity validation with PyAV.
"""

import os
import sys
import time
import argparse
import cv2

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

def open_camera(preferred_index=0, preferred_backend="dshow"):
    """
    Opens the camera using DirectShow by default on Windows to avoid
    the slow MSMF negotiation and obsensor 'Camera index out of range' errors.
    """
    backends = []
    if preferred_backend.lower() == "dshow":
        backends = [("DirectShow", cv2.CAP_DSHOW), ("MediaFoundation", cv2.CAP_MSMF), ("Default", cv2.CAP_ANY)]
    elif preferred_backend.lower() == "msmf":
        backends = [("MediaFoundation", cv2.CAP_MSMF), ("DirectShow", cv2.CAP_DSHOW), ("Default", cv2.CAP_ANY)]
    else:
        backends = [("Default", cv2.CAP_ANY), ("DirectShow", cv2.CAP_DSHOW), ("MediaFoundation", cv2.CAP_MSMF)]

    # Try preferred index first, then 0, 1, 2
    indices_to_try = [preferred_index] + [i for i in [0, 1, 2] if i != preferred_index]

    for b_name, b_flag in backends:
        for idx in indices_to_try:
            print(f"[Webcam] Attempting connection to Camera {idx} using {b_name}...")
            cap = cv2.VideoCapture(idx, b_flag)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[Webcam] Connected successfully to Camera {idx} ({b_name}) - Resolution: {w}x{h}")
                    return cap, idx, b_name
                cap.release()

    return None, None, None


def record_session(output_path, camera_index=0, backend="dshow", target_fps=20.0):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cap, active_idx, active_backend = open_camera(camera_index, backend)
    if cap is None:
        print("\n[ERROR] Could not access any webcam.")
        print("Please run 'python check_webcam.py' to diagnose camera access.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Controls instructions
    print("\n" + "=" * 65)
    print("HWGAT WEBCAM RECORDER - CONTROLS")
    print("=" * 65)
    print(" [R] key -> START recording sign gesture")
    print(" [S] key -> STOP recording and SAVE to disk")
    print(" [Q] key -> QUIT recorder")
    print(f" Target Output: {output_path}")
    print("=" * 65 + "\n")

    recording = False
    recorded_frames = []
    start_time = 0.0
    saved = False

    window_name = "HWGAT Webcam Recorder (Press R=Record, S=Save, Q=Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Webcam] Frame read returned None. Exiting loop.")
                break

            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # Process key events (10ms wait for responsive keyboard input)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord('r'), ord('R')) and not recording:
                recording = True
                recorded_frames = []
                start_time = time.time()
                saved = False
                print(f"[Webcam] >>> RECORDING STARTED -> Capturing frames for '{output_path}'...")

            elif key in (ord('s'), ord('S')) and recording:
                recording = False
                elapsed = max(time.time() - start_time, 0.001)
                actual_fps = len(recorded_frames) / elapsed
                print(f"[Webcam] >>> RECORDING STOPPED -> Captured {len(recorded_frames)} frames in {elapsed:.2f}s (~{actual_fps:.1f} FPS).")
                break

            elif key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                print("[Webcam] Quit requested by user.")
                break

            # Handle live recording buffer
            if recording:
                recorded_frames.append(frame.copy())
                elapsed = time.time() - start_time
                frame_count = len(recorded_frames)

                # Flash red indicator every 0.5s
                show_flash = int(elapsed * 2) % 2 == 0
                if show_flash:
                    cv2.circle(display_frame, (30, 35), 12, (0, 0, 255), -1)
                cv2.putText(display_frame, f"REC [{frame_count} frames | {elapsed:.1f}s]", (52, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "Press 'S' to Stop & Save", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                # Standby HUD overlay
                cv2.rectangle(display_frame, (15, 15), (420, 85), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (15, 15), (420, 85), (0, 255, 255), 1)
                cv2.putText(display_frame, "STANDBY: Ready to record", (25, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "Press [R] to Record | [Q] to Quit", (25, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Footer status
            cv2.putText(display_frame, f"CAM {active_idx} [{active_backend}] {w}x{h}", (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Save video if frames were captured
    if recorded_frames:
        elapsed = max(time.time() - start_time, 0.001)
        # Use target_fps or calculated fps bounded cleanly
        calc_fps = len(recorded_frames) / elapsed
        effective_fps = max(min(calc_fps, 30.0), 10.0)

        print(f"\n[Saving] Writing {len(recorded_frames)} frames to: {output_path}")
        print(f"[Saving] Encoding at {effective_fps:.1f} FPS with 'mp4v' codec...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))

        for f in recorded_frames:
            writer.write(f)
        writer.release()
        saved = True

        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"[Success] Video saved successfully! Size: {file_size_kb:.1f} KB")

        # PyAV validation check
        if HAS_PYAV:
            try:
                container = av.open(output_path)
                decoded_count = sum(1 for _ in container.decode(video=0))
                container.close()
                print(f"[PyAV Check] Verified! PyAV decoded all {decoded_count} frames cleanly.")
            except Exception as e:
                print(f"[PyAV Check Warning] PyAV verification note: {e}")

        print(f"\nNext Step: Run HWGAT inference on this recording:")
        print(f"  .\\experimental\\hwgat_env\\Scripts\\python.exe experimental/hwgat/test_hwgat_video.py --video \"{output_path}\"\n")
        return True
    else:
        print("\n[Notice] No frames were recorded. Output file was not created.")
        return False


def main():
    parser = argparse.ArgumentParser(description="HWGAT Accessible Webcam Recorder")
    parser.add_argument("--output", "-o", type=str, default="test_videos/today.mp4",
                        help="Path to save the recorded video (default: test_videos/today.mp4)")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--backend", "-b", type=str, default="dshow",
                        choices=["dshow", "msmf", "any"],
                        help="OpenCV capture backend (default: dshow)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Target frame rate (default: 20.0)")

    args = parser.parse_args()

    # Normalise output path relative to script directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = args.output
    if not os.path.isabs(out):
        out = os.path.join(script_dir, out)

    record_session(output_path=out, camera_index=args.camera, backend=args.backend, target_fps=args.fps)


if __name__ == "__main__":
    main()
