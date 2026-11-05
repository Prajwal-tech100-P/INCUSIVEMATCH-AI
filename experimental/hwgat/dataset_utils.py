"""
dataset_utils.py - Storage, Metadata, and Class Mapping Manager for HWGAT Dataset System

Strictly isolated within experimental/hwgat/
"""
import os
import csv
import json
import time
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "custom_dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw_videos")
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")
METADATA_DIR = os.path.join(DATASET_DIR, "metadata")
REPORTS_DIR = os.path.join(DATASET_DIR, "reports")
CHECKPOINTS_DIR = os.path.join(DATASET_DIR, "checkpoints")

MANIFEST_PATH = os.path.join(METADATA_DIR, "dataset_manifest.csv")
CLASS_MAPPING_PATH = os.path.join(METADATA_DIR, "class_mapping.csv")
SUMMARY_PATH = os.path.join(METADATA_DIR, "collection_summary.json")
BASE_CLASS_MAP_PATH = os.path.join(BASE_DIR, "input", "class_map_FDMSE.csv")

MANIFEST_HEADERS = [
    "sample_id",
    "class_name",
    "class_id",
    "is_custom",
    "video_path",
    "npy_path",
    "frames",
    "duration_sec",
    "valid_landmarks",
    "created_at",
]

CLASS_MAP_HEADERS = ["class_id", "class_name", "is_custom", "sample_count"]


def init_dataset_structure() -> None:
    """Ensure all custom_dataset subdirectories and metadata files exist."""
    for d in [DATASET_DIR, RAW_DIR, LANDMARKS_DIR, METADATA_DIR, REPORTS_DIR, CHECKPOINTS_DIR]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(MANIFEST_HEADERS)

    if not os.path.exists(CLASS_MAPPING_PATH):
        _initialize_class_mapping()


def load_base_vocabulary() -> Dict[int, str]:
    """Load the original 2,002 vocabulary classes from class_map_FDMSE.csv."""
    vocab: Dict[int, str] = {}
    if not os.path.exists(BASE_CLASS_MAP_PATH):
        return vocab

    with open(BASE_CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    vocab[int(row[0])] = row[1].strip()
                except ValueError:
                    continue
    return vocab


def _initialize_class_mapping() -> None:
    """Initialize class_mapping.csv by copying base vocabulary and tracking sample counts."""
    base_vocab = load_base_vocabulary()
    with open(CLASS_MAPPING_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CLASS_MAP_HEADERS)
        for cid, cname in sorted(base_vocab.items(), key=lambda x: x[0]):
            writer.writerow([cid, cname, 0, 0])


def load_class_mapping() -> Dict[str, Dict[str, Any]]:
    """Load class mapping indexed by class_name."""
    init_dataset_structure()
    mapping: Dict[str, Dict[str, Any]] = {}
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cname = row["class_name"].strip()
            mapping[cname] = {
                "class_id": int(row["class_id"]),
                "class_name": cname,
                "is_custom": int(row.get("is_custom", 0)) == 1,
                "sample_count": int(row.get("sample_count", 0)),
            }
    return mapping


def register_or_get_class(class_name: str) -> Dict[str, Any]:
    """
    Get class info for a given class_name.
    If it is a new custom class, allocate a new unique ID without colliding with base classes.
    """
    init_dataset_structure()
    clean_name = class_name.strip()
    mapping = load_class_mapping()

    if clean_name in mapping:
        return mapping[clean_name]

    # Allocate new unique class_id beyond existing max ID
    all_ids = [info["class_id"] for info in mapping.values()]
    new_id = (max(all_ids) + 1) if all_ids else 2002
    if new_id < 2002:
        new_id = 2002

    new_info = {
        "class_id": new_id,
        "class_name": clean_name,
        "is_custom": True,
        "sample_count": 0,
    }
    mapping[clean_name] = new_info

    # Append to class_mapping.csv
    with open(CLASS_MAPPING_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([new_id, clean_name, 1, 0])

    return new_info


def increment_class_sample_count(class_name: str) -> None:
    """Increment sample_count for a class in class_mapping.csv."""
    mapping = load_class_mapping()
    clean_name = class_name.strip()
    if clean_name not in mapping:
        register_or_get_class(clean_name)
        mapping = load_class_mapping()

    mapping[clean_name]["sample_count"] += 1

    # Rewrite class_mapping.csv cleanly
    with open(CLASS_MAPPING_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CLASS_MAP_HEADERS)
        for cname, info in sorted(mapping.items(), key=lambda x: x[1]["class_id"]):
            writer.writerow([
                info["class_id"],
                info["class_name"],
                1 if info["is_custom"] else 0,
                info["sample_count"],
            ])


def record_sample_in_manifest(
    sample_id: str,
    class_name: str,
    class_id: int,
    is_custom: bool,
    video_path: str,
    npy_path: str,
    frames: int,
    duration_sec: float,
    valid_landmarks: bool,
) -> None:
    """Append a sample entry to dataset_manifest.csv."""
    init_dataset_structure()
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # Relative paths for portability
    rel_video = os.path.relpath(video_path, BASE_DIR) if os.path.exists(video_path) else video_path
    rel_npy = os.path.relpath(npy_path, BASE_DIR) if os.path.exists(npy_path) else npy_path

    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            sample_id,
            class_name,
            class_id,
            1 if is_custom else 0,
            rel_video,
            rel_npy,
            frames,
            f"{duration_sec:.2f}",
            1 if valid_landmarks else 0,
            created_at,
        ])


def load_manifest() -> List[Dict[str, Any]]:
    """Load all rows from dataset_manifest.csv as dictionaries."""
    init_dataset_structure()
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(MANIFEST_PATH):
        return rows

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def get_storage_statistics() -> Dict[str, Any]:
    """
    Compute disk usage in bytes and human-readable format for:
    - raw_videos/
    - landmarks/
    - total dataset
    """
    init_dataset_structure()

    def get_dir_size(path: str) -> Tuple[int, int]:
        total_size = 0
        file_count = 0
        if not os.path.exists(path):
            return 0, 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total_size += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass
        return total_size, file_count

    raw_bytes, raw_count = get_dir_size(RAW_DIR)
    npy_bytes, npy_count = get_dir_size(LANDMARKS_DIR)

    raw_mb = raw_bytes / (1024 * 1024)
    npy_mb = npy_bytes / (1024 * 1024)

    savings_mb = raw_mb - npy_mb
    savings_pct = ((raw_bytes - npy_bytes) / raw_bytes * 100) if raw_bytes > 0 else 0.0

    return {
        "raw_videos_count": raw_count,
        "raw_videos_bytes": raw_bytes,
        "raw_videos_mb": round(raw_mb, 2),
        "landmarks_count": npy_count,
        "landmarks_bytes": npy_bytes,
        "landmarks_mb": round(npy_mb, 2),
        "savings_mb": round(savings_mb, 2),
        "savings_percentage": round(savings_pct, 2),
    }


def update_collection_summary() -> Dict[str, Any]:
    """Generate or update collection_summary.json."""
    init_dataset_structure()
    manifest = load_manifest()
    mapping = load_class_mapping()
    stats = get_storage_statistics()

    class_distribution: Dict[str, int] = {}
    for row in manifest:
        c = row["class_name"]
        class_distribution[c] = class_distribution.get(c, 0) + 1

    summary = {
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(manifest),
        "total_classes": len(class_distribution),
        "storage": stats,
        "class_distribution": class_distribution,
        "classes_under_10_samples": [c for c, n in class_distribution.items() if n < 10],
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

