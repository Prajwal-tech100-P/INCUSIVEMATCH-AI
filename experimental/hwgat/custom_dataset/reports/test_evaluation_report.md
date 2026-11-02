# HWGAT Custom Model Test & Evaluation Report
**Evaluated:** 2026-11-02 15:20:51  
**Checkpoint:** `hwgat_custom_best.pt`  
**Test Set Size:** 3 samples across 3 classes

---

## 1. Overall Performance Summary

| Metric | Score | Status |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **33.33%** | ACCEPTABLE |
| **Macro Precision** | **11.11%** | - |
| **Macro Recall** | **33.33%** | - |
| **Macro F1-Score** | **16.67%** | - |
| **Misclassified Samples** | **2** | - |

---

## 2. CPU Inference Latency (Single-Sample)

| Metric | Latency (ms) | Target (< 3000 ms) |
| :--- | :--- | :--- |
| **Mean Latency** | **1860.08 ms** | COMPLIANT |
| **Median Latency** | **1860.42 ms** | COMPLIANT |
| **95th Percentile** | **1982.95 ms** | COMPLIANT |
| **Min / Max** | 1732.21 / 2018.52 ms | - |

---

## 3. Per-Class Performance Breakdown

| Class ID | Class Name | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | **MockHello** | 0.0% | 0.0% | 0.0% | 1 |
| 1 | **MockNamaste** | 33.3% | 100.0% | 50.0% | 1 |
| 2 | **MockThankYou** | 0.0% | 0.0% | 0.0% | 1 |

---

## 4. Confusion Matrix

```text
True \ Pred | MockHell MockNama MockThan
----------------------------------------
MockHello  |        0        1        0
MockNamast |        0        1        0
MockThankY |        0        1        0
```

---

## 5. Misclassified Sample Diagnostics

| Sample ID | True Class | Predicted As | Confidence |
| :--- | :--- | :--- | :--- |
| `MockHello_sample_0.npy` | MockHello | MockNamaste | 93.1% |
| `MockThankYou_sample_0.npy` | MockThankYou | MockNamaste | 93.1% |
