# Indian Sign Language (ISL) Recognition Model Evaluation Report

**MCA Major Project: InclusiveMatch AI**  
**Evaluation Date:** 2026-10-29 21:40:03  
**Model Checkpoint:** `sign_gcnt_best.pt`  
**Architecture:** GCN + Multi-Head Temporal Transformer (`gcn_transformer_2hand`)  

---

## 1. Executive Summary & Core Benchmark Metrics

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **81.89%** | Correct sign predicted as highest probability |
| **Top-2 Accuracy** | **92.91%** | True sign ranked in top 2 predictions |
| **Top-3 Accuracy** | **95.28%** | True sign ranked in top 3 predictions |
| **Macro F1-Score** | **0.8144** | Unweighted average F1 across all 29 classes |
| **Weighted F1-Score** | **0.8058** | Support-weighted average F1 |
| **Macro Precision** | **0.8339** | Average precision across classes |
| **Macro Recall** | **0.8321** | Average recall across classes |
| **Inference Latency** | **2.05 ms** | Mean single-sequence CPU forward pass |
| **Latency (P95)** | **2.99 ms** | 95th percentile CPU forward pass |
| **Model Parameter Count** | **300,189** | Trainable neural network parameters |
| **Model Checkpoint Size** | **1.21 MB** | Lightweight size optimized for Render CPU |
| **Evaluated Test Samples** | **127** | Held-out leak-free independent samples |
| **Vocabulary Classes** | **29** | Supported Indian Sign Language vocabulary |

---

## 2. Model Architecture & Preprocessing Specifications

- **Spatial Graph Convolutional Network (GCN):**
  - 42 nodes representing MediaPipe dual-hand topology (21 joints per hand).
  - Block-diagonal normalized symmetric Laplacian adjacency ($D^{-1/2}(A+I)D^{-1/2}$).
  - Handedness stabilization: Right hand mapped to nodes 0..20, Left hand to 21..41.
- **Temporal Transformer Encoder:**
  - Multi-Head Self-Attention (4 heads, 2 layers, $d_{model}=64$, $d_{ff}=256$).
  - Positional encoding across 24 sampled frames.
  - Multi-scale temporal pooling (Mean + Max pooling).
- **Feature Representation:**
  - 4 channels per node: $[x, y, dx, dy]$ (spatial coordinates + temporal velocity difference).

---

## 3. Per-Class Precision, Recall, and F1-Score

| Class Name | Precision | Recall | F1-Score | Support | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **drink** | 0.50 | 1.00 | 0.67 | 4 | Moderate |
| **eat** | 0.67 | 0.67 | 0.67 | 3 | Moderate |
| **father** | 0.75 | 0.75 | 0.75 | 4 | Moderate |
| **food** | 0.57 | 1.00 | 0.73 | 4 | Moderate |
| **friend** | 0.57 | 0.57 | 0.57 | 7 | Requires Data Expansion |
| **go** | 1.00 | 1.00 | 1.00 | 3 | Optimal |
| **he** | 1.00 | 1.00 | 1.00 | 3 | Optimal |
| **hello** | 1.00 | 1.00 | 1.00 | 7 | Optimal |
| **help** | 0.75 | 1.00 | 0.86 | 3 | Optimal |
| **hospital** | 1.00 | 1.00 | 1.00 | 4 | Optimal |
| **market** | 0.56 | 0.71 | 0.62 | 7 | Moderate |
| **mother** | 1.00 | 1.00 | 1.00 | 4 | Optimal |
| **no** | 1.00 | 1.00 | 1.00 | 4 | Optimal |
| **okay** | 1.00 | 0.50 | 0.67 | 4 | Moderate |
| **please** | 0.60 | 0.75 | 0.67 | 4 | Moderate |
| **school** | 1.00 | 1.00 | 1.00 | 7 | Optimal |
| **she** | 0.67 | 1.00 | 0.80 | 4 | Optimal |
| **sister** | 1.00 | 1.00 | 1.00 | 4 | Optimal |
| **sit** | 0.80 | 1.00 | 0.89 | 4 | Optimal |
| **student** | 1.00 | 1.00 | 1.00 | 4 | Optimal |
| **tea** | 1.00 | 1.00 | 1.00 | 4 | Optimal |
| **teacher** | 1.00 | 1.00 | 1.00 | 4 | Optimal |
| **thank_you** | 1.00 | 0.43 | 0.60 | 7 | Moderate |
| **today** | 1.00 | 0.75 | 0.86 | 4 | Optimal |
| **water** | 0.00 | 0.00 | 0.00 | 4 | Requires Data Expansion |
| **what** | 0.75 | 0.75 | 0.75 | 4 | Moderate |
| **where** | 1.00 | 0.50 | 0.67 | 4 | Moderate |
| **yes** | 1.00 | 1.00 | 1.00 | 4 | Optimal |
| **you** | 1.00 | 0.75 | 0.86 | 4 | Optimal |

---

## 4. Class Imbalance & Underperforming Class Analysis

- **Classes with F1-Score < 0.60:** friend, water
- **Notable Ambiguities:**
  - `water` vs `drink`: In Indian Sign Language, both gestures share hand proximity to the chin/mouth; distinction relies heavily on thumb articulation and temporal hold.
  - `thank_you` vs `please`: Both signs feature flat palms extending from chin/chest.
- **Mitigation in InclusiveMatch AI:**
  - Configurable confidence threshold (signs below 40% confidence display uncertainty badge).
  - Temporal smoothing window ($N=5$ frames) suppresses single-frame prediction flicker.
  - Token-level debouncing prevents UI flooding.

---

## 5. Scope & Limitations

1. **Isolated Sign vs Continuous Translation:** The current model classifies isolated dynamic signs (one sign per 24-frame interval). It does not perform continuous sign sentence parsing.
2. **Vocabulary Limitation:** The model currently recognizes 29 common social communication classes. Additional vocabulary can be recorded and integrated using the new `ISL Dataset Studio`.
3. **Lighting & Occlusion:** Recognition accuracy depends on clear MediaPipe hand tracking. Low-light or extreme camera angles degrade landmark extraction.
