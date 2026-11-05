"""
validate_dataset.py - Dataset Integrity and Validation Suite for HWGAT Landmarks

Scans all .npy landmark files and verifies:
- Strict shape check: (192, 64, 2)
- Zero NaN and Inf values
- Motion/landmark coverage threshold
- Per-class sample counts and split viability (>= 5 samples)

Strictly isolated within experimental/hwgat/
"""
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import dataset_utils as du


def audit_landmark_file(filepath: str) -> Tuple[bool, str]:
    """Inspect a single .npy file for corruption and dimension compliance."""
    if not os.path.exists(filepath):
        return False, "File does not exist"
    if os.path.getsize(filepath) == 0:
        return False, "File is 0 bytes (empty)"

    try:
        arr = np.load(filepath)
    except Exception as e:
        return False, f"Failed to load numpy array: {e}"

    if not isinstance(arr, np.ndarray):
        return False, f"Not a numpy array (type: {type(arr)})"

    if arr.shape != (192, 64, 2):
        return False, f"Incorrect shape {arr.shape}, expected (192, 64, 2)"

    if np.isnan(arr).any():
        return False, "Contains NaN values"

    if np.isinf(arr).any():
        return False, "Contains Infinite values"

    non_zero = np.count_nonzero(arr) / arr.size
    if non_zero < 0.01:
        return False, f"Insufficient detected landmarks (density: {non_zero:.4f})"

    return True, "OK"


def run_dataset_audit() -> Dict[str, Any]:
    """Execute complete dataset validation across custom_dataset/landmarks/."""
    du.init_dataset_structure()
    landmarks_root = du.LANDMARKS_DIR

    class_dirs = [
        d for d in os.listdir(landmarks_root)
        if os.path.isdir(os.path.join(landmarks_root, d))
    ]

    total_files = 0
    passed_files = 0
    failed_files: List[Tuple[str, str]] = []
    class_stats: Dict[str, Dict[str, int]] = {}

    for cname in sorted(class_dirs):
        cdir = os.path.join(landmarks_root, cname)
        npy_files = [f for f in os.listdir(cdir) if f.endswith(".npy")]
        class_stats[cname] = {"total": len(npy_files), "valid": 0, "invalid": 0}

        for nf in npy_files:
            total_files += 1
            fpath = os.path.join(cdir, nf)
            ok, reason = audit_landmark_file(fpath)
            if ok:
                passed_files += 1
                class_stats[cname]["valid"] += 1
            else:
                failed_files.append((fpath, reason))
                class_stats[cname]["invalid"] += 1

    # Check classes with < 5 and < 10 samples
    under_5 = [c for c, s in class_stats.items() if s["valid"] < 5]
    under_10 = [c for c, s in class_stats.items() if 5 <= s["valid"] < 10]

    return {
        "total_classes": len(class_dirs),
        "total_files": total_files,
        "passed_files": passed_files,
        "failed_files": failed_files,
        "class_stats": class_stats,
        "classes_under_5": under_5,
        "classes_under_10": under_10,
        "all_valid": (len(failed_files) == 0 and total_files > 0),
    }


def print_audit_report(results: Dict[str, Any]) -> bool:
    """Print formatted terminal report of validation results."""
    print("\n" + "=" * 70)
    print("           HWGAT DATASET INTEGRITY & VALIDATION REPORT")
    print("=" * 70)
    print(f"Total Sign Classes Discovered: {results['total_classes']}")
    print(f"Total Landmark (.npy) Files:   {results['total_files']}")
    print(f"Passed Verification:           {results['passed_files']}")
    print(f"Failed Verification:           {len(results['failed_files'])}")
    print("-" * 70)

    if results["class_stats"]:
        print(f"{'Class Name':<25} | {'Total':<8} | {'Valid':<8} | {'Status'}")
        print("-" * 70)
        for cname, s in results["class_stats"].items():
            if s["valid"] < 5:
                status = "CRITICAL (< 5 samples)"
            elif s["valid"] < 10:
                status = "WARNING (< 10 samples)"
            else:
                status = "READY (>= 10 samples)"
            print(f"{cname:<25} | {s['total']:<8} | {s['valid']:<8} | {status}")
        print("-" * 70)

    if results["failed_files"]:
        print("\n[FAILED FILES AUDIT]")
        for fpath, reason in results["failed_files"]:
            print(f"  - {os.path.basename(fpath)}: {reason}")
        print("-" * 70)

    if results["classes_under_5"]:
        print(f"\n[ALERT] Classes with < 5 samples (cannot be reliably split into train/val):")
        for c in results["classes_under_5"]:
            print(f"  * {c} ({results['class_stats'][c]['valid']} samples)")

    if results["classes_under_10"]:
        print(f"\n[NOTICE] Classes with 5-9 samples (recommended >= 10 for fine-tuning):")
        for c in results["classes_under_10"]:
            print(f"  * {c} ({results['class_stats'][c]['valid']} samples)")

    is_success = len(results["failed_files"]) == 0
    print("\n" + "=" * 70)
    if is_success:
        print("[AUDIT PASSED] All landmark files are intact, non-empty, and valid.")
    else:
        print("[AUDIT FAILED] One or more corrupted or invalid landmark files detected.")
    print("=" * 70 + "\n")

    return is_success


def main():
    results = run_dataset_audit()
    success = print_audit_report(results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

