# HWGAT Custom Sign Language Dataset & Model Training Guide

This guide describes the complete workflow for collecting sign-language datasets from a webcam, extracting MediaPipe Holistic landmarks, fine-tuning the HWGAT neural network, evaluating accuracy, and packaging a lightweight deployment bundle.

All scripts operate strictly inside `experimental/hwgat/` using the isolated Python environment `experimental/hwgat_env/`. The main InclusiveMatch AI platform is completely isolated and unaffected.

---

## 1. Directory Structure

```text
experimental/hwgat/
├── custom_dataset/
│   ├── raw_videos/            # Recorded MP4 video clips (per class)
│   ├── landmarks/             # Compact (192, 64, 2) .npy landmark tensors
│   ├── metadata/
│   │   ├── dataset_manifest.csv
│   │   ├── class_mapping.csv
│   │   ├── collection_summary.json
│   │   └── test_split.json
│   ├── reports/
│   │   ├── dataset_report.md
│   │   ├── training_history.json
│   │   └── test_evaluation_report.md
│   └── checkpoints/
│       └── hwgat_custom_best.pt
├── deployment_package/        # Lightweight production bundle (no raw datasets)
│   ├── checkpoint.pt
│   ├── class_mapping.json
│   ├── inference.py
│   ├── models/
│   └── pose_modules/
├── dataset_utils.py           # Metadata, storage, and manifest helper
├── model_utils.py             # Head surgery, model loading, checkpoint saving
├── extract_landmarks.py       # PyAV + MediaPipe Holistic landmark extractor
├── webcam_dataset_collector.py# DirectShow webcam dataset collector HUD
├── validate_dataset.py        # Dataset integrity & NaN audit suite
├── dataset_report.py          # Storage efficiency & compression report generator
├── train_custom.py            # PyTorch fine-tuning pipeline
├── test_custom_model.py       # Test evaluation, confusion matrix & latency
└── prepare_deployment.py      # Deployment packager & standalone tester
```

---

## 2. Environment Setup

Always use the isolated `hwgat_env` Python environment:

```powershell
# In PowerShell:
.\experimental\hwgat_env\Scripts\python.exe <script_name>.py
```

---

## 3. Step-by-Step Operational Workflow

### Step 1: Collect Sign Language Samples via Webcam

Launch the interactive collector:
```powershell
.\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\webcam_dataset_collector.py
```

- **Label Selector:** Search from the 2,002 base vocabulary classes or create a new custom label (e.g., `Namaste`, `Help`, `ThankYou`).
- **HUD Controls:**
  - Press `SPACE` to record a sign gesture (default 3.5 seconds).
  - Press `Q` to quit.
- **Storage Saving:** Automatically extracts MediaPipe landmarks into `.npy` format and indexes the clip in `dataset_manifest.csv`.
- Optional: Add `--delete_raw` to remove raw MP4 files after verified `.npy` generation:
  ```powershell
  .\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\webcam_dataset_collector.py --class_name "Namaste" --delete_raw
  ```

---

### Step 2: Batch Landmark Extraction (Offline Videos)

If you have existing raw MP4 videos in `custom_dataset/raw_videos/`:
```powershell
.\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\extract_landmarks.py --all
```

To extract and delete raw videos only after strict shape `(192, 64, 2)` verification:
```powershell
.\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\extract_landmarks.py --all --delete_raw_confirm
```

---

### Step 3: Validate Dataset Integrity

Before training, verify that all `.npy` files are non-empty, corruption-free, and meet the required shape:
```powershell
.\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\validate_dataset.py
```
- Verifies array shape: exactly `(192, 64, 2)`.
- Checks for `NaN` and `Inf` values.
- Warns if any class has $< 5$ samples (insufficient for train/val split).
- Warns if any class has $< 10$ samples (recommended minimum).

---

### Step 4: Storage Footprint & Efficiency Report

Generate an empirical comparison between raw MP4 video sizes and compact `.npy` landmark files:
```powershell
.\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\dataset_report.py
```
Outputs `custom_dataset/reports/dataset_report.md` showing compression percentage (typically **98% to 99% space savings**).

---

### Step 5: Fine-Tune the HWGAT Model

Train the model on your custom classes using transfer learning:
```powershell
.\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\train_custom.py --epochs 15 --batch_size 4 --lr 1e-4
```
- Replaces the 2,002-class classification head with an adapted head matching your custom classes.
- Automatically creates stratified Train (70%), Val (15%), and Test (15%) splits.
- Saves the best checkpoint to `custom_dataset/checkpoints/hwgat_custom_best.pt`.
- Saves the held-out test manifest to `custom_dataset/metadata/test_split.json`.

---

### Step 6: Test & Evaluate Model Performance

Evaluate the trained checkpoint on the held-out test set:
```powershell
.\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\test_custom_model.py
```
- Computes Top-1 Accuracy, Precision, Recall, and Macro F1-Score.
- Generates a multi-class Confusion Matrix.
- Benchmarks CPU single-sample inference latency (mean, median, p95).
- Diagnoses misclassified samples.
- Saves `custom_dataset/reports/test_evaluation_report.md`.

---

### Step 7: Package Lightweight Deployment Bundle

Create an isolated production package containing only what is required for inference:
```powershell
.\experimental\hwgat_env\Scripts\python.exe experimental\hwgat\prepare_deployment.py
```
- Creates `experimental/hwgat/deployment_package/`.
- Strictly excludes raw videos, training landmarks, and training scripts.
- Runs a standalone verification test with dummy input `(1, 192, 64, 2)`.
- Generates `DEPLOYMENT_REPORT.md`.

---

## 4. Production Integration Example

Using the packaged engine in an application:

```python
from experimental.hwgat.deployment_package.inference import SignRecognizer

# Initialize once
recognizer = SignRecognizer(device="cpu")

# Provide (192, 64, 2) landmark tensor from MediaPipe
result = recognizer.predict_landmarks(landmark_array)

print("Recognized Sign:", result["prediction"])
print("Confidence:", result["confidence"])
print("Top 5 Predictions:", result["top_predictions"])
```

---

## 5. Safety & Invariant Guarantees

1. **Isolation:** Zero changes to the root InclusiveMatch AI application, its database, or its existing 29-class ISL model.
2. **Storage Safety:** Raw videos are NEVER deleted unless explicitly confirmed via `--delete_raw_confirm` AND verified that the corresponding `.npy` file exists, has shape `(192, 64, 2)`, and contains valid float32 values.
3. **No Collision:** Custom class labels are mapped without overwriting base vocabulary IDs.

