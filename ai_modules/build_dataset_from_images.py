"""Build a GCN-T training CSV from a folder of hand-sign images.

No webcam needed — just organise images into labelled sub-folders and run.

Folder layout (two supported structures)
-----------------------------------------
Structure A — one folder per label (static signs / alphabets):

    data/images/
    ├── HELLO/
    │   ├── img01.jpg
    │   ├── img02.png
    │   └── ...
    ├── YES/
    │   └── ...
    └── NO/
        └── ...

Structure B — one folder per *sequence* (motion signs, e.g. short GIFs as frames):

    data/images/
    ├── HELLO/
    │   ├── seq_001/        ← each sub-folder = one gesture clip
    │   │   ├── frame_01.jpg
    │   │   ├── frame_02.jpg
    │   │   └── ...
    │   └── seq_002/
    │       └── ...
    └── YES/
        └── ...

Usage
-----
  # Structure A (static photos):
  python ai_modules/build_dataset_from_images.py data/images --output data/sign_sequences.csv

  # Structure B (frame sequences):
  python ai_modules/build_dataset_from_images.py data/images --sequences --output data/sign_sequences.csv

  # Augment each image N times (flip + brightness + noise):
  python ai_modules/build_dataset_from_images.py data/images --augment 5 --output data/sign_sequences.csv

  # Preview detected landmarks before saving (requires display):
  python ai_modules/build_dataset_from_images.py data/images --preview

Supported image formats : jpg, jpeg, png, bmp, webp
Output CSV format       : x0,y0,...,x20,y20,label,sequence_id
  (same format as collect_sign_sequences.py - plug directly into train_sign_model.py)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Force UTF-8 output on Windows (avoids cp1252 errors with special chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

# ── constants ──────────────────────────────────────────────────────────────
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLS = [f"{ax}{i}" for i in range(21) for ax in ("x", "y")] + ["label", "sequence_id"]


# ── landmark extraction ────────────────────────────────────────────────────

def extract_landmarks(image: np.ndarray, hands) -> Optional[np.ndarray]:
    """Return (42,) normalised float32 array or None if no hand found."""
    rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm  = result.multi_hand_landmarks[0]
    pts = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
    pts -= pts[0]                                    # wrist at origin
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale <= 1e-6:
        return None
    return (pts / scale).reshape(-1).astype(np.float32)


# ── augmentation ───────────────────────────────────────────────────────────

def augment_image(img: np.ndarray, n: int) -> List[np.ndarray]:
    """Return `n` augmented copies of `img`."""
    variants = []
    for _ in range(n):
        out = img.copy()
        # Random horizontal flip
        if random.random() < 0.5:
            out = cv2.flip(out, 1)
        # Random brightness / contrast
        alpha = random.uniform(0.75, 1.25)   # contrast
        beta  = random.randint(-30, 30)       # brightness
        out   = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        # Gaussian noise
        noise = np.random.normal(0, 8, out.shape).astype(np.int16)
        out   = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Small random rotation (±10°)
        angle = random.uniform(-10, 10)
        h, w  = out.shape[:2]
        M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out   = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        variants.append(out)
    return variants


# ── preview helper ─────────────────────────────────────────────────────────

def draw_landmarks(img: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Draw 21 normalised joints back onto a copy of img (for preview)."""
    vis   = img.copy()
    h, w  = vis.shape[:2]
    pts   = feat.reshape(21, 2)
    # Denormalise back to pixel space (approx — wrist-relative, scaled)
    min_xy, max_xy = pts.min(axis=0), pts.max(axis=0)
    span = max_xy - min_xy + 1e-6
    px   = ((pts - min_xy) / span * np.array([w * 0.6, h * 0.6]) + np.array([w * 0.2, h * 0.2])).astype(int)
    for x, y in px:
        cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
    return vis


# ── core processing ────────────────────────────────────────────────────────

def process_static(
    root: Path,
    hands,
    augment_n: int,
    preview: bool,
    writer: csv.writer,
    counters: dict,
) -> Tuple[int, int]:
    """
    Structure A — one sub-folder per label, images inside.
    Returns (saved, skipped).
    """
    saved = skipped = 0
    seq_id = counters.get("seq_id", 0)

    label_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not label_dirs:
        print(f"[WARN] No sub-folders found in {root}. Check your folder structure.")
        return 0, 0

    for label_dir in label_dirs:
        label    = label_dir.name
        img_paths = sorted([p for p in label_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT])
        if not img_paths:
            print(f"  [{label}] No images found — skipping.")
            continue

        print(f"  [{label}] {len(img_paths)} image(s) found ...")
        label_saved = 0

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    SKIP (unreadable): {img_path.name}")
                skipped += 1
                continue

            candidates = [img] + augment_image(img, augment_n)
            for variant in candidates:
                feat = extract_landmarks(variant, hands)
                if feat is None:
                    skipped += 1
                    continue

                if preview:
                    vis = draw_landmarks(variant, feat)
                    cv2.imshow(f"Preview — {label}", vis)
                    key = cv2.waitKey(400) & 0xFF
                    if key == 27:           # ESC → stop preview
                        preview = False
                        cv2.destroyAllWindows()

                seq_id += 1
                writer.writerow(list(feat) + [label, seq_id])
                saved += 1
                label_saved += 1

        print(f"    → {label_saved} row(s) saved")

    counters["seq_id"] = seq_id
    if preview:
        cv2.destroyAllWindows()
    return saved, skipped


def process_sequences(
    root: Path,
    hands,
    augment_n: int,
    preview: bool,
    writer: csv.writer,
    counters: dict,
) -> Tuple[int, int]:
    """
    Structure B — label/seq_NNN/frame_*.jpg  (multi-frame gesture clips).
    Returns (saved_frames, skipped).
    """
    saved = skipped = 0
    seq_id = counters.get("seq_id", 0)

    label_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for label_dir in label_dirs:
        label    = label_dir.name
        seq_dirs = sorted([d for d in label_dir.iterdir() if d.is_dir()])
        if not seq_dirs:
            print(f"  [{label}] No sequence sub-folders — skipping.")
            continue

        print(f"  [{label}] {len(seq_dirs)} sequence(s) ...")
        label_saved = 0

        for seq_dir in seq_dirs:
            frames = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT])
            if not frames:
                continue

            # Build landmark list for this clip
            feats = []
            for fp in frames:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                feat = extract_landmarks(img, hands)
                if feat is not None:
                    feats.append(feat)

            if len(feats) < 2:
                print(f"    SKIP {seq_dir.name}: only {len(feats)} detectable frame(s)")
                skipped += len(frames)
                continue

            # Optionally augment: add jitter copies of entire clip
            all_clips = [feats]
            for _ in range(augment_n):
                noise = np.random.normal(0, 0.008, (len(feats), 42)).astype(np.float32)
                all_clips.append([f + noise[i] for i, f in enumerate(feats)])

            for clip in all_clips:
                seq_id += 1
                for feat in clip:
                    writer.writerow(list(feat) + [label, seq_id])
                    saved += 1
                    label_saved += 1

        print(f"    → {label_saved} frame row(s) saved")

    counters["seq_id"] = seq_id
    if preview:
        cv2.destroyAllWindows()
    return saved, skipped


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build sign-language training CSV from a folder of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "image_dir",
        help="Root folder containing labelled sub-folders of images.",
    )
    ap.add_argument(
        "--output", "-o",
        default="data/sign_sequences.csv",
        help="Output CSV path (default: data/sign_sequences.csv).",
    )
    ap.add_argument(
        "--sequences", "-s",
        action="store_true",
        help="Use Structure B: label/seq_NNN/frame_*.jpg (motion gestures).",
    )
    ap.add_argument(
        "--augment", "-a",
        type=int,
        default=0,
        metavar="N",
        help="Number of augmented copies per image/clip (default: 0).",
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Show detected landmarks on each image before saving.",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting.",
    )
    ap.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.50,
        dest="det_conf",
        help="MediaPipe hand detection confidence threshold (default: 0.50).",
    )
    args = ap.parse_args()

    # ── validate input dir ──────────────────────────────────────────────────
    root = Path(args.image_dir)
    if not root.is_dir():
        sys.exit(f"[ERROR] Image directory not found: {root}")

    # ── MediaPipe ───────────────────────────────────────────────────────────
    try:
        import mediapipe as mp
    except ImportError:
        sys.exit(
            "[ERROR] MediaPipe is required.\n"
            "Install: pip install mediapipe"
        )

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,      # ← static mode for images (no tracking)
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=args.det_conf,
    )

    # ── output CSV ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode   = "a" if args.append and out_path.exists() else "w"
    header = not (args.append and out_path.exists() and out_path.stat().st_size > 0)

    # Find highest existing sequence_id so we don't collide when appending
    seq_id_start = 0
    if mode == "a" and out_path.exists():
        try:
            import csv as _csv
            with out_path.open() as f:
                for row in _csv.DictReader(f):
                    try:
                        seq_id_start = max(seq_id_start, int(row.get("sequence_id", 0)))
                    except ValueError:
                        pass
            print(f"[INFO] Appending — continuing from sequence_id {seq_id_start + 1}")
        except Exception:
            pass

    counters = {"seq_id": seq_id_start}

    print(f"\nImage root  : {root.resolve()}")
    print(f"Output CSV  : {out_path.resolve()}")
    print(f"Mode        : {'append' if mode == 'a' else 'overwrite'}")
    print(f"Sequences   : {'yes (Structure B)' if args.sequences else 'no (Structure A)'}")
    print(f"Augment ×   : {args.augment}")
    print(f"Preview     : {args.preview}")
    print()

    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(COLS)

        if args.sequences:
            saved, skipped = process_sequences(root, hands, args.augment, args.preview, writer, counters)
        else:
            saved, skipped = process_static(root, hands, args.augment, args.preview, writer, counters)

    hands.close()

    print()
    print("=" * 50)
    print(f"Rows saved  : {saved}")
    print(f"Skipped     : {skipped}  (no hand detected)")
    print(f"Output      : {out_path.resolve()}")
    if saved == 0:
        print("\n[WARN] Nothing was saved — check that:")
        print("  • Folders are named after the sign labels (e.g. data/images/HELLO/)")
        print("  • Images clearly show a hand (try --min-detection-confidence 0.3)")
        print("  • Image format is jpg/png/bmp/webp")
    else:
        print(f"\nNext step -> train the model:")
        print(f"  python ai_modules/train_sign_model.py {args.output}")


if __name__ == "__main__":
    main()
