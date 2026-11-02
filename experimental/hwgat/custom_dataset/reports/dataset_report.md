# HWGAT Sign Language Dataset Storage & Efficiency Report
**Generated:** 2026-11-02 15:19:52

## 1. Executive Storage Summary

| Metric | Raw Videos (`raw_videos/`) | Landmarks (`landmarks/`) | Difference / Savings |
| :--- | :--- | :--- | :--- |
| **Total Files** | 0 | 12 | - |
| **Total Disk Footprint** | **0.0 MB** | **1.13 MB** | **--1.13 MB** |
| **Average File Size** | 0.0 KB | 96.1 KB | **0.0% smaller** |
| **Storage Savings Ratio** | Baseline (1.0x) | Processed | **0.0% Reduction** |

> **Key Architectural Insight:** Extracted `.npy` landmark arrays `(192, 64, 2)` eliminate pixel redundancy, background lighting, and encoder container overhead, retaining only essential skeletal coordinates. This enables storing thousands of sign samples on minimal disk quotas.

## 2. Per-Class Storage & Sample Distribution

| Class Name | Raw Videos | Raw Size (MB) | Landmarks (.npy) | Landmark Size (MB) | Savings (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MockHello** | 0 | 0.0 MB | 4 | 0.38 MB | 0.0% |
| **MockNamaste** | 0 | 0.0 MB | 4 | 0.38 MB | 0.0% |
| **MockThankYou** | 0 | 0.0 MB | 4 | 0.38 MB | 0.0% |

## 3. Data Integrity & Retention Policy

1. **Storage Optimization:** Once `.npy` landmarks are generated and verified via `validate_dataset.py`, raw MP4 video files can be safely deleted using `--delete_raw_confirm`.
2. **Deterministic Preprocessing:** All normalized coordinate geometry is permanently preserved in float32 precision, ensuring identical training convergence without needing to re-decode video codecs.
