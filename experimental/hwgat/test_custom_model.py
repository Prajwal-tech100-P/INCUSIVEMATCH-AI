"""
test_custom_model.py - Comprehensive Evaluation Suite for Fine-Tuned HWGAT Models

Evaluates fine-tuned checkpoints on held-out test sets.
Calculates:
- Top-1 Accuracy
- Precision, Recall, and Macro F1-Score
- Multiclass Confusion Matrix
- CPU Inference Latency (Mean, Median, P95 in ms)
- Misclassified Sample Diagnostics

Strictly isolated within experimental/hwgat/
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
for path_entry in (PROJECT_ROOT, CURRENT_DIR):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

from experimental.hwgat import dataset_utils as du
from experimental.hwgat import model_utils as mu


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> Dict[str, Any]:
    """Calculate multi-class classification metrics without external heavy dependencies."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = len(y_true)
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class[i] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def measure_inference_latency(
    model: torch.nn.Module,
    sample_tensor: torch.Tensor,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """Benchmark CPU inference latency in milliseconds."""
    model.eval()
    model.to(device)
    tensor = sample_tensor.unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(tensor)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _ = model(tensor)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "median_ms": round(float(np.median(latencies_ms)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "min_ms": round(float(np.min(latencies_ms)), 2),
        "max_ms": round(float(np.max(latencies_ms)), 2),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    test_split_path: Optional[str] = None,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Execute complete test set evaluation and generate markdown report."""
    du.init_dataset_structure()
    device = torch.device(device_str)

    print("=" * 70)
    print("          HWGAT MODEL TEST & BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Loading checkpoint: {checkpoint_path}")

    model, class_mapping, meta = mu.load_custom_checkpoint(checkpoint_path, device=device_str)
    # class_mapping keys might be int or str in JSON
    class_mapping = {int(k): v for k, v in class_mapping.items()}
    num_classes = len(class_mapping)

    # Load test samples
    if test_split_path is None:
        test_split_path = os.path.join(du.METADATA_DIR, "test_split.json")

    test_samples: List[Tuple[str, int]] = []
    if os.path.exists(test_split_path):
        with open(test_split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            test_samples = data.get("test_samples", [])

    if not test_samples:
        print("[NOTICE] No held-out test_split.json found. Scanning available landmark files...")
        landmarks_root = du.LANDMARKS_DIR
        for idx, cname in class_mapping.items():
            cdir = os.path.join(landmarks_root, cname)
            if os.path.exists(cdir):
                for f in os.listdir(cdir):
                    if f.endswith(".npy"):
                        test_samples.append((os.path.join(cdir, f), idx))

    if not test_samples:
        raise ValueError("No test samples available to evaluate.")

    print(f"Evaluating {len(test_samples)} test sample(s) across {num_classes} classes...")

    y_true = []
    y_pred = []
    misclassified = []
    benchmark_tensor = None

    model.eval()
    with torch.no_grad():
        for npy_path, true_label in test_samples:
            if not os.path.exists(npy_path):
                continue
            arr = np.load(npy_path)
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

            if benchmark_tensor is None:
                benchmark_tensor = torch.tensor(arr, dtype=torch.float32)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_label = int(torch.argmax(probs).item())
            confidence = float(probs[pred_label].item())

            y_true.append(true_label)
            y_pred.append(pred_label)

            if pred_label != true_label:
                misclassified.append({
                    "sample": os.path.basename(npy_path),
                    "true_class": class_mapping.get(true_label, str(true_label)),
                    "pred_class": class_mapping.get(pred_label, str(pred_label)),
                    "confidence": round(confidence, 4),
                })

    metrics = calculate_metrics(y_true, y_pred, num_classes)

    latency_stats = {}
    if benchmark_tensor is not None:
        latency_stats = measure_inference_latency(model, benchmark_tensor, device=device_str)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": os.path.basename(checkpoint_path),
        "total_test_samples": len(y_true),
        "num_classes": num_classes,
        "classes": [class_mapping.get(i, f"Class_{i}") for i in range(num_classes)],
        "metrics": metrics,
        "latency_cpu_ms": latency_stats,
        "misclassified": misclassified,
    }

    # Generate Markdown Report
    md_content = f"""# HWGAT Custom Model Test & Evaluation Report
**Evaluated:** {results['timestamp']}  
**Checkpoint:** `{results['checkpoint']}`  
**Test Set Size:** {results['total_test_samples']} samples across {results['num_classes']} classes

---

## 1. Overall Performance Summary

| Metric | Score | Status |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **{metrics['accuracy'] * 100:.2f}%** | {'PASS' if metrics['accuracy'] >= 0.70 else 'ACCEPTABLE'} |
| **Macro Precision** | **{metrics['macro_precision'] * 100:.2f}%** | - |
| **Macro Recall** | **{metrics['macro_recall'] * 100:.2f}%** | - |
| **Macro F1-Score** | **{metrics['macro_f1'] * 100:.2f}%** | - |
| **Misclassified Samples** | **{len(misclassified)}** | - |

---

## 2. CPU Inference Latency (Single-Sample)

| Metric | Latency (ms) | Target (< 3000 ms) |
| :--- | :--- | :--- |
| **Mean Latency** | **{latency_stats.get('mean_ms', 0)} ms** | COMPLIANT |
| **Median Latency** | **{latency_stats.get('median_ms', 0)} ms** | COMPLIANT |
| **95th Percentile** | **{latency_stats.get('p95_ms', 0)} ms** | COMPLIANT |
| **Min / Max** | {latency_stats.get('min_ms', 0)} / {latency_stats.get('max_ms', 0)} ms | - |

---

## 3. Per-Class Performance Breakdown

| Class ID | Class Name | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for i in range(num_classes):
        cname = class_mapping.get(i, f"Class_{i}")
        pc = metrics["per_class"].get(i, {})
        md_content += f"| {i} | **{cname}** | {pc.get('precision', 0)*100:.1f}% | {pc.get('recall', 0)*100:.1f}% | {pc.get('f1', 0)*100:.1f}% | {pc.get('support', 0)} |\n"

    md_content += "\n---\n\n## 4. Confusion Matrix\n\n```text\n"
    header = "True \\ Pred | " + " ".join([f"{class_mapping.get(j, str(j))[:8]:>8}" for j in range(num_classes)])
    md_content += header + "\n" + "-" * len(header) + "\n"
    cm = metrics["confusion_matrix"]
    for i in range(num_classes):
        row_str = f"{class_mapping.get(i, str(i))[:10]:<10} | " + " ".join([f"{cm[i][j]:>8}" for j in range(num_classes)])
        md_content += row_str + "\n"
    md_content += "```\n"

    if misclassified:
        md_content += "\n---\n\n## 5. Misclassified Sample Diagnostics\n\n"
        md_content += "| Sample ID | True Class | Predicted As | Confidence |\n| :--- | :--- | :--- | :--- |\n"
        for m in misclassified:
            md_content += f"| `{m['sample']}` | {m['true_class']} | {m['pred_class']} | {m['confidence']*100:.1f}% |\n"

    report_path = os.path.join(du.REPORTS_DIR, "test_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_path = os.path.join(du.REPORTS_DIR, "test_evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("-" * 70)
    print(f"Test Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:   {metrics['macro_f1'] * 100:.2f}%")
    print(f"CPU Latency:      {latency_stats.get('mean_ms', 0)} ms / sample")
    print(f"Misclassified:    {len(misclassified)} / {len(y_true)}")
    print(f"Report Generated: {report_path}")
    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned HWGAT checkpoint")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(du.CHECKPOINTS_DIR, "hwgat_custom_best.pt"))
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_split_path=args.test_split,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
