"""
HWGAT Webcam Diagnostics Utility
=================================
Diagnoses available camera devices and OpenCV backends on Windows.
Identifies camera index, resolution, backend compatibility (DirectShow, MSMF),
and provides actionable troubleshooting instructions if access is restricted.
"""

import sys
import time
import cv2

BACKENDS = [
    ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
    ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
    ("Auto / Default (CAP_ANY)", cv2.CAP_ANY)
]

def scan_cameras(max_index=4):
    print("=" * 75)
    print("HWGAT WEBCAM DIAGNOSTICS - CAMERA SCAN & BACKEND VERIFICATION")
    print("=" * 75)
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Scanning camera indices 0 through {max_index - 1} across backends...\n")

    working_devices = []

    for backend_name, backend_id in BACKENDS:
        print(f"--- Testing Backend: {backend_name} ---")
        found_any = False
        for idx in range(max_index):
            sys.stdout.write(f"  Probing index {idx}... ")
            sys.stdout.flush()
            t0 = time.time()
            cap = cv2.VideoCapture(idx, backend_id)
            opened = cap.isOpened()
            elapsed = time.time() - t0

            if opened:
                ret, frame = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if ret and frame is not None:
                    found_any = True
                    print(f"SUCCESS [Opened in {elapsed:.2f}s] - Resolution: {w}x{h}, Reported FPS: {fps:.1f}")
                    working_devices.append({
                        "index": idx,
                        "backend_name": backend_name,
                        "backend_id": backend_id,
                        "resolution": (w, h),
                        "fps": fps
                    })
                else:
                    print(f"OPENED but frame read returned False ({elapsed:.2f}s)")
            else:
                print(f"Not available ({elapsed:.2f}s)")
        if not found_any:
            print(f"  No working camera found with {backend_name}.")
        print()

    print("=" * 75)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 75)

    if working_devices:
        print(f"[+] Found {len(working_devices)} operational camera configuration(s):")
        for i, dev in enumerate(working_devices, 1):
            print(f"   Option {i}: Index {dev['index']} using {dev['backend_name']} ({dev['resolution'][0]}x{dev['resolution'][1]})")

        best = working_devices[0]
        # Prefer DirectShow if available
        for dev in working_devices:
            if dev["backend_id"] == cv2.CAP_DSHOW:
                best = dev
                break

        print(f"\n[RECOMMENDED SETTING]:")
        print(f"   Camera Index: {best['index']}")
        print(f"   Backend:      {best['backend_name']}")
        print(f"   Command:      python record_webcam.py --camera {best['index']}\n")
    else:
        print("[-] ERROR: No working camera detected on this system!")
        print("\n[TROUBLESHOOTING WINDOWS CAMERA ACCESS]:")
        print("1. WINDOWS PRIVACY SETTINGS:")
        print("   - Open Windows Settings -> 'Privacy & security' -> 'Camera'")
        print("   - Ensure 'Camera access' is toggled ON.")
        print("   - Scroll down to 'Let desktop apps access your camera' and toggle ON.")
        print("   - Ensure Python or your IDE is not blocked in the list.")
        print("2. CAMERA IN USE BY ANOTHER PROCESS:")
        print("   - Close other applications that may hold the webcam lock:")
        print("     (Teams, Zoom, Discord, Google Meet, Skype, or existing Python processes).")
        print("3. PHYSICAL PRIVACY SHUTTER:")
        print("   - Ensure your laptop webcam hardware privacy shutter is open.")
        print("4. DRIVER CHECK:")
        print("   - Open Device Manager -> 'Cameras' or 'Imaging devices'")
        print("   - Verify your webcam is enabled and functioning normally.\n")

    return working_devices

if __name__ == "__main__":
    scan_cameras()

