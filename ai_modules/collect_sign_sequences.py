"""Collect MediaPipe hand-landmark sequences for GRU sign training.

Usage:
  python ai_modules/collect_sign_sequences.py --labels HELLO YES NO --per-label 40 --output data/sign_sequences.csv

Press SPACE to record a 24-frame sequence; ESC quits. The camera window shows
the current label. Collect different distances, lighting and hand positions.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np


def features(hand_landmarks):
    pts = np.array([(p.x, p.y) for p in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale <= 1e-6:
        return None
    return (pts / scale).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--per-label", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--output", default="data/sign_sequences.csv")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    try:
        import mediapipe as mp
    except Exception as exc:
        raise SystemExit(f"MediaPipe is required: {exc}")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")] + ["label", "sequence_id"]
    exists = out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.60, min_tracking_confidence=0.60)
    with out.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists: writer.writerow(cols)
        sequence_id = 0
        try:
            for label in args.labels:
                collected = 0
                while collected < args.per_label:
                    ok, frame = cap.read()
                    if not ok: break
                    display = frame.copy()
                    cv2.putText(display, f"{label}: {collected}/{args.per_label}  SPACE=record  ESC=quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                    cv2.imshow("Sign dataset collector", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27: return
                    if key != 32: continue

                    frames=[]
                    for _ in range(args.seq_len):
                        ok, frame = cap.read()
                        if not ok: break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = hands.process(rgb)
                        if result.multi_hand_landmarks:
                            feat = features(result.multi_hand_landmarks[0])
                            if feat is not None: frames.append(feat)
                        cv2.putText(frame, f"RECORDING {label} {len(frames)}/{args.seq_len}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
                        cv2.imshow("Sign dataset collector", frame)
                        if cv2.waitKey(1) & 0xFF == 27: return
                    if len(frames) < max(8, args.seq_len // 2):
                        print("Skipped: hand was not visible for enough frames.")
                        continue
                    sequence_id += 1
                    # Preserve the actual observed frame sequence.
                    for feat in frames:
                        writer.writerow(list(feat) + [label, sequence_id])
                    f.flush(); collected += 1
                    print(f"Saved {label} sequence {collected}/{args.per_label}")
        finally:
            hands.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
