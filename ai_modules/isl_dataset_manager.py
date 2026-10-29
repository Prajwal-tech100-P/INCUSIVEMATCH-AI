"""
Indian Sign Language (ISL) Dataset Manager & Metadata Tracker
============================================================
MCA Major Project: InclusiveMatch AI

Manages dataset storage under datasets/isl_custom/:
- Structure: datasets/isl_custom/<class_name>/<recording_id>.npy
- Numerical format: NumPy float32 array of shape (T, 84) (42 nodes x 2 coordinates)
- Metadata file: datasets/isl_custom/metadata.json
- Preserves existing 29 classes and supports recording new examples or new classes.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Default dataset path inside the project
DEFAULT_DATASET_DIR = Path("datasets/isl_custom")
CLASSES_JSON_PATH = Path("models/sign_gcnt_best_classes.json")
EXTERNAL_LANDMARKS_DIR = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")


def get_default_classes() -> List[str]:
    """Returns the ordered list of default 29 classes from sign_gcnt_best_classes.json."""
    if CLASSES_JSON_PATH.exists():
        try:
            with open(CLASSES_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [str(c).lower().strip() for c in data.get("classes", [])]
        except Exception:
            pass
    return [
        "drink", "eat", "father", "food", "friend", "go", "he", "hello",
        "help", "hospital", "market", "mother", "no", "okay", "please",
        "school", "she", "sister", "sit", "student", "tea", "teacher",
        "thank_you", "today", "water", "what", "where", "yes", "you"
    ]


class ISLDatasetManager:
    """Manages the ISL custom landmark dataset and its metadata."""

    def __init__(self, dataset_dir: Path | str = DEFAULT_DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.dataset_dir / "metadata.json"
        self._ensure_metadata_file()

    def _ensure_metadata_file(self) -> None:
        """Creates metadata.json if it doesn't exist."""
        if not self.metadata_file.exists():
            initial_meta = {
                "version": "1.0.0",
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "num_classes": 0,
                "total_samples": 0,
                "classes": {},
                "recordings": {}
            }
            self._save_metadata(initial_meta)

    def load_metadata(self) -> Dict[str, Any]:
        """Loads metadata.json safely."""
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "version": "1.0.0",
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "num_classes": 0,
                "total_samples": 0,
                "classes": {},
                "recordings": {}
            }

    def _save_metadata(self, meta: Dict[str, Any]) -> None:
        """Writes metadata.json safely with atomic replacement."""
        meta["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp_file = self.dataset_dir / f"metadata_tmp_{uuid.uuid4().hex[:8]}.json"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        shutil.move(str(temp_file), str(self.metadata_file))

    def get_supported_classes(self) -> List[str]:
        """Returns union of default 29 classes and any custom classes created on disk."""
        classes = set(get_default_classes())
        # Check subdirectories
        for d in self.dataset_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                classes.add(d.name.lower())
        meta = self.load_metadata()
        for c in meta.get("classes", {}).keys():
            classes.add(c.lower())
        return sorted(list(classes))

    def add_class(self, class_name: str) -> str:
        """Creates a class directory if not present and registers it."""
        norm_name = class_name.lower().strip().replace(" ", "_")
        if not norm_name:
            raise ValueError("Class name cannot be empty.")
        class_dir = self.dataset_dir / norm_name
        class_dir.mkdir(parents=True, exist_ok=True)
        meta = self.load_metadata()
        if norm_name not in meta.get("classes", {}):
            meta.setdefault("classes", {})[norm_name] = {
                "sample_count": 0,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            meta["num_classes"] = len(meta["classes"])
            self._save_metadata(meta)
        return norm_name

    def save_recording(
        self,
        class_name: str,
        landmark_sequence: np.ndarray,
        signer_id: str = "webcam_user",
        source: str = "webcam",
        fps: float = 30.0,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates and saves a landmark sequence into datasets/isl_custom/<class_name>/<rec_id>.npy.

        Validation rules:
        - Sequence shape must be (T, 84) or (T, 42, 2)
        - Must have at least 10 frames
        - Must contain active hand landmarks in at least 5 frames
        """
        arr = np.asarray(landmark_sequence, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[1] == 42 and arr.shape[2] == 2:
            arr = arr.reshape(len(arr), 84)
        elif arr.ndim != 2 or arr.shape[1] != 84:
            return False, f"Invalid landmark shape {arr.shape}. Expected (T, 84).", {}

        total_frames = len(arr)
        if total_frames < 8:
            return False, f"Sequence too short ({total_frames} frames). Minimum 8 frames required.", {}

        # Count frames where at least one landmark is non-zero
        active_frames = int(np.sum([np.any(arr[t] != 0) for t in range(total_frames)]))
        if active_frames < 5:
            return False, f"Insufficient hand detection: only {active_frames} active frames.", {}

        norm_class = self.add_class(class_name)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:6]
        recording_id = f"{norm_class}__webcam__{timestamp}_{unique_token}"
        file_path = self.dataset_dir / norm_class / f"{recording_id}.npy"

        # Check collision
        if file_path.exists():
            recording_id = f"{norm_class}__webcam__{timestamp}_{uuid.uuid4().hex[:10]}"
            file_path = self.dataset_dir / norm_class / f"{recording_id}.npy"

        # Save numpy array
        np.save(str(file_path), arr)

        # Update metadata
        meta = self.load_metadata()
        meta.setdefault("classes", {}).setdefault(norm_class, {"sample_count": 0})
        meta["classes"][norm_class]["sample_count"] = len(list((self.dataset_dir / norm_class).glob("*.npy")))
        meta["num_classes"] = len(meta["classes"])
        meta.setdefault("recordings", {})[recording_id] = {
            "class_name": norm_class,
            "filename": file_path.name,
            "total_frames": total_frames,
            "active_frames": active_frames,
            "signer_id": signer_id,
            "source": source,
            "fps": fps,
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        meta["total_samples"] = sum(c.get("sample_count", 0) for c in meta["classes"].values())
        self._save_metadata(meta)

        info = {
            "recording_id": recording_id,
            "file_path": str(file_path),
            "total_frames": total_frames,
            "active_frames": active_frames,
            "class_name": norm_class,
            "class_total_samples": meta["classes"][norm_class]["sample_count"]
        }
        return True, f"Successfully saved recording for sign '{norm_class}'.", info

    def sync_from_external_landmarks(
        self,
        source_dir: Path | str = EXTERNAL_LANDMARKS_DIR,
        classes: Optional[List[str]] = None,
        copy_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Imports or links existing landmark files from C:\\Users\\ACER\\isl-40words\\processed_landmarks
        into datasets/isl_custom/ to preserve all valid existing classes and samples.
        """
        src = Path(source_dir)
        if not src.exists():
            return {"ok": False, "message": f"Source directory {src} does not exist.", "synced": 0}

        target_classes = set(c.lower() for c in (classes or get_default_classes()))
        synced_count = 0
        skipped_count = 0

        meta = self.load_metadata()

        for class_dir in sorted(src.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
            c_name = class_dir.name.lower()
            if target_classes and c_name not in target_classes:
                continue

            target_class_dir = self.dataset_dir / c_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            self.add_class(c_name)

            for npy_file in class_dir.glob("*.npy"):
                target_file = target_class_dir / npy_file.name
                if not target_file.exists():
                    if copy_mode:
                        shutil.copy2(str(npy_file), str(target_file))
                    else:
                        try:
                            os.symlink(str(npy_file), str(target_file))
                        except Exception:
                            shutil.copy2(str(npy_file), str(target_file))
                    synced_count += 1
                else:
                    skipped_count += 1

                rec_id = npy_file.stem
                if rec_id not in meta.get("recordings", {}):
                    meta.setdefault("recordings", {})[rec_id] = {
                        "class_name": c_name,
                        "filename": target_file.name,
                        "source": "benchmark_dataset",
                        "synced_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }

        # Update summary counts
        self.rebuild_metadata()
        summary = self.get_summary()
        return {
            "ok": True,
            "synced_count": synced_count,
            "skipped_count": skipped_count,
            "total_samples": summary["total_samples"],
            "classes_count": summary["num_classes"]
        }

    def rebuild_metadata(self) -> Dict[str, Any]:
        """Scans disk to ensure metadata.json is 100% consistent with directory state."""
        meta = self.load_metadata()
        classes_data = {}
        total = 0

        for d in sorted(self.dataset_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            c_name = d.name.lower()
            samples = list(d.glob("*.npy"))
            classes_data[c_name] = {
                "sample_count": len(samples),
                "last_scanned": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            total += len(samples)

        meta["classes"] = classes_data
        meta["num_classes"] = len(classes_data)
        meta["total_samples"] = total
        self._save_metadata(meta)
        return meta

    def get_summary(self) -> Dict[str, Any]:
        """Returns complete dataset summary including class sample counts and distribution."""
        meta = self.load_metadata()
        # Verify counts with disk
        classes_summary = {}
        for c in self.get_supported_classes():
            c_dir = self.dataset_dir / c
            count = len(list(c_dir.glob("*.npy"))) if c_dir.exists() else 0
            classes_summary[c] = count

        return {
            "dataset_dir": str(self.dataset_dir.resolve()),
            "num_classes": len(classes_summary),
            "total_samples": sum(classes_summary.values()),
            "classes": classes_summary,
            "min_samples": min(classes_summary.values()) if classes_summary else 0,
            "max_samples": max(classes_summary.values()) if classes_summary else 0,
            "last_updated": meta.get("updated_at")
        }

