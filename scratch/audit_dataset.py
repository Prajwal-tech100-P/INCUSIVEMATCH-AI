import os
import json
from pathlib import Path
import numpy as np

DATA_ROOT = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")
CKPT_CLASSES_FILE = Path("sign_model_output/classes.json")

print("=" * 70)
print("PHASE 2: DATASET INTEGRITY & CLASS AUDIT")
print("=" * 70)

with open(CKPT_CLASSES_FILE, "r") as f:
    ckpt_classes = json.load(f)

print(f"Existing Checkpoint Classes ({len(ckpt_classes)}): {list(ckpt_classes.keys())}\n")

all_classes = []
class_sample_counts = {}
shape_counts = {}
dtype_counts = {}
corrupted_files = []
nan_inf_files = []
all_zero_files = []
single_hand_only = 0
both_hands_active = 0
no_hands_active = 0
total_files = 0

for folder in sorted(DATA_ROOT.iterdir()):
    if not folder.is_dir():
        continue
    all_classes.append(folder.name)
    npy_files = sorted(folder.glob("*.npy"))
    class_sample_counts[folder.name] = len(npy_files)
    
    for f in npy_files:
        total_files += 1
        try:
            arr = np.load(f)
            shape = arr.shape
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            dtype_counts[str(arr.dtype)] = dtype_counts.get(str(arr.dtype), 0) + 1
            
            if not np.isfinite(arr).all():
                nan_inf_files.append((str(f), "contains NaN or Inf"))
            
            # Check active hands (arr is T, 84 -> T, 42, 2)
            reshaped = arr.reshape(-1, 42, 2)
            h1 = reshaped[:, :21]
            h2 = reshaped[:, 21:]
            h1_act = np.any(h1 != 0)
            h2_act = np.any(h2 != 0)
            
            if h1_act and h2_act:
                both_hands_active += 1
            elif h1_act or h2_act:
                single_hand_only += 1
            else:
                no_hands_active += 1
                all_zero_files.append(str(f))
                
        except Exception as e:
            corrupted_files.append((str(f), str(e)))

print(f"Total Folders/Classes in processed_landmarks: {len(all_classes)}")
print(f"Total .npy Files: {total_files}")
print(f"Shapes Distribution: {shape_counts}")
print(f"Data Types: {dtype_counts}")
print(f"Corrupted (unreadable) Files: {len(corrupted_files)}")
print(f"Files with NaN/Inf: {len(nan_inf_files)}")
print(f"Files with All Zero (no landmarks at all): {len(all_zero_files)}")
print(f"Samples with Both Hands Detected: {both_hands_active} ({both_hands_active/total_files*100:.1f}%)")
print(f"Samples with Single Hand Only: {single_hand_only} ({single_hand_only/total_files*100:.1f}%)")
print(f"Samples with No Hands Active: {no_hands_active} ({no_hands_active/total_files*100:.1f}%)\n")

print("-" * 70)
print("CLASS SAMPLES BREAKDOWN (Sorted by count)")
print("-" * 70)
sorted_counts = sorted(class_sample_counts.items(), key=lambda x: x[1], reverse=True)
included_in_ckpt = []
dropped_from_ckpt = []

for c_name, count in sorted_counts:
    status = "INCLUDED (>=15)" if count >= 15 else f"DROPPED (<15 samples)"
    if count >= 15:
        included_in_ckpt.append((c_name, count))
    else:
        dropped_from_ckpt.append((c_name, count))
    in_ckpt = "Yes" if c_name in ckpt_classes else "NO"
    print(f"  {c_name:16s}: {count:3d} samples | Status: {status:22s} | In Checkpoint: {in_ckpt}")

print("\n" + "=" * 70)
print(f"SUMMARY: 40 Dataset Classes vs 29 Checkpoint Classes")
print("=" * 70)
print(f"Classes with >= 15 samples: {len(included_in_ckpt)}")
print(f"Classes with < 15 samples : {len(dropped_from_ckpt)}")
print(f"Dropped class names ({len(dropped_from_ckpt)}): {[c[0] for c in dropped_from_ckpt]}")
