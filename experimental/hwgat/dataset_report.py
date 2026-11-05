"""
dataset_report.py - Storage Efficiency and Dataset Statistics Generator

Generates empirical comparisons between raw video footprints and extracted .npy landmarks.
Saves markdown report to custom_dataset/reports/dataset_report.md.

Strictly isolated within experimental/hwgat/
"""
import os
import sys
import time
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import dataset_utils as du


def generate_storage_report() -> Dict[str, Any]:
    """Compile comprehensive storage and class distribution statistics."""
    du.init_dataset_structure()
    stats = du.get_storage_statistics()
    manifest = du.load_manifest()

    # Per-class breakdown
    raw_root = du.RAW_DIR
    npy_root = du.LANDMARKS_DIR

    all_classes = set()
    if os.path.exists(raw_root):
        all_classes.update([d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))])
    if os.path.exists(npy_root):
        all_classes.update([d for d in os.listdir(npy_root) if os.path.isdir(os.path.join(npy_root, d))])

    class_breakdown: List[Dict[str, Any]] = []

    for c in sorted(all_classes):
        c_raw_dir = os.path.join(raw_root, c)
        c_npy_dir = os.path.join(npy_root, c)

        v_count = 0
        v_bytes = 0
        if os.path.exists(c_raw_dir):
            for f in os.listdir(c_raw_dir):
                fp = os.path.join(c_raw_dir, f)
                if os.path.isfile(fp):
                    v_count += 1
                    v_bytes += os.path.getsize(fp)

        n_count = 0
        n_bytes = 0
        if os.path.exists(c_npy_dir):
            for f in os.listdir(c_npy_dir):
                fp = os.path.join(c_npy_dir, f)
                if os.path.isfile(fp) and f.endswith(".npy"):
                    n_count += 1
                    n_bytes += os.path.getsize(fp)

        v_mb = v_bytes / (1024 * 1024)
        n_mb = n_bytes / (1024 * 1024)
        pct = ((v_bytes - n_bytes) / v_bytes * 100) if v_bytes > 0 else 0.0

        class_breakdown.append({
            "class_name": c,
            "raw_count": v_count,
            "raw_mb": round(v_mb, 2),
            "npy_count": n_count,
            "npy_mb": round(n_mb, 2),
            "savings_pct": round(pct, 1),
        })

    avg_raw_kb = (stats["raw_videos_bytes"] / stats["raw_videos_count"] / 1024) if stats["raw_videos_count"] > 0 else 0.0
    avg_npy_kb = (stats["landmarks_bytes"] / stats["landmarks_count"] / 1024) if stats["landmarks_count"] > 0 else 0.0

    report_data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_manifest_entries": len(manifest),
        "storage": stats,
        "avg_raw_kb": round(avg_raw_kb, 1),
        "avg_npy_kb": round(avg_npy_kb, 1),
        "class_breakdown": class_breakdown,
    }

    # Save Markdown report
    md_content = f"""# HWGAT Sign Language Dataset Storage & Efficiency Report
**Generated:** {report_data['generated_at']}

## 1. Executive Storage Summary

| Metric | Raw Videos (`raw_videos/`) | Landmarks (`landmarks/`) | Difference / Savings |
| :--- | :--- | :--- | :--- |
| **Total Files** | {stats['raw_videos_count']} | {stats['landmarks_count']} | - |
| **Total Disk Footprint** | **{stats['raw_videos_mb']} MB** | **{stats['landmarks_mb']} MB** | **-{stats['savings_mb']} MB** |
| **Average File Size** | {round(avg_raw_kb, 1)} KB | {round(avg_npy_kb, 1)} KB | **{stats['savings_percentage']}% smaller** |
| **Storage Savings Ratio** | Baseline (1.0x) | Processed | **{stats['savings_percentage']}% Reduction** |

> **Key Architectural Insight:** Extracted `.npy` landmark arrays `(192, 64, 2)` eliminate pixel redundancy, background lighting, and encoder container overhead, retaining only essential skeletal coordinates. This enables storing thousands of sign samples on minimal disk quotas.

## 2. Per-Class Storage & Sample Distribution

| Class Name | Raw Videos | Raw Size (MB) | Landmarks (.npy) | Landmark Size (MB) | Savings (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for row in class_breakdown:
        md_content += f"| **{row['class_name']}** | {row['raw_count']} | {row['raw_mb']} MB | {row['npy_count']} | {row['npy_mb']} MB | {row['savings_pct']}% |\n"

    md_content += """
## 3. Data Integrity & Retention Policy

1. **Storage Optimization:** Once `.npy` landmarks are generated and verified via `validate_dataset.py`, raw MP4 video files can be safely deleted using `--delete_raw_confirm`.
2. **Deterministic Preprocessing:** All normalized coordinate geometry is permanently preserved in float32 precision, ensuring identical training convergence without needing to re-decode video codecs.
"""

    report_path = os.path.join(du.REPORTS_DIR, "dataset_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return report_data


def main():
    report_data = generate_storage_report()
    stats = report_data["storage"]

    print("\n" + "=" * 70)
    print("      HWGAT DATASET STORAGE & EFFICIENCY SUMMARY")
    print("=" * 70)
    print(f"Raw Videos Footprint:    {stats['raw_videos_mb']} MB ({stats['raw_videos_count']} files, avg {report_data['avg_raw_kb']} KB)")
    print(f"Landmark (.npy) Storage: {stats['landmarks_mb']} MB ({stats['landmarks_count']} files, avg {report_data['avg_npy_kb']} KB)")
    print(f"Net Disk Space Saved:    {stats['savings_mb']} MB ({stats['savings_percentage']}% compression)")
    print("-" * 70)
    print(f"Markdown report written to: {os.path.join(du.REPORTS_DIR, 'dataset_report.md')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

