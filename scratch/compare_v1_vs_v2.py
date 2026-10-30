import os
import sys
import json
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_modules.nlp_safety import NLPModerator

# Paths
V1_PATH = os.path.join(PROJECT_ROOT, "models", "toxicity_classifier_v1_backup.joblib")
V2_PATH = os.path.join(PROJECT_ROOT, "models", "toxicity_classifier.joblib")
TEST_SPLIT_PATH = os.path.join(PROJECT_ROOT, "datasets", "toxicity", "test.json")

CATEGORIES = ["toxic", "severe_toxic", "insult", "threat", "harassment", "hate"]

# 1. Load artifacts
art_v1 = joblib.load(V1_PATH)
pipe_v1 = art_v1["pipeline"]

art_v2 = joblib.load(V2_PATH)
pipe_v2 = art_v2["pipeline"]
thresh_v2 = art_v2.get("tuned_thresholds", {
    "toxic": 0.40, "severe_toxic": 0.35, "insult": 0.45, "threat": 0.55, "harassment": 0.45, "hate": 0.45
})
thresh_v1 = {
    "severe_toxic": 0.65, "threat": 0.65, "harassment": 0.65, "hate": 0.65, "insult": 0.78, "toxic": 0.85
}

# 2. Load held-out test split (n=961)
with open(TEST_SPLIT_PATH, "r", encoding="utf-8") as f:
    test_records = json.load(f)

test_texts = [x["text"] for x in test_records]
y_test = np.array([x["labels"] for x in test_records], dtype=int)

print(f"Comparing v1 vs v2 on HELD-OUT TEST SPLIT ({len(test_texts)} samples)...")

probs_v1 = pipe_v1.predict_proba(test_texts)
probs_v2 = pipe_v2.predict_proba(test_texts)

def evaluate_raw_predictions(probs, thresholds, y_true):
    metrics = {}
    f1_list = []
    for idx, cat in enumerate(CATEGORIES):
        t = thresholds[cat]
        y_pred = (probs[:, idx] >= t).astype(int)
        p = precision_score(y_true[:, idx], y_pred, zero_division=0)
        r = recall_score(y_true[:, idx], y_pred, zero_division=0)
        f = f1_score(y_true[:, idx], y_pred, zero_division=0)
        f1_list.append(f)
        metrics[cat] = {"precision": p, "recall": r, "f1": f, "positives": int(y_true[:, idx].sum())}
    macro_f1 = np.mean(f1_list)
    return metrics, macro_f1

v1_test_metrics, v1_macro_f1 = evaluate_raw_predictions(probs_v1, thresh_v1, y_test)
v2_test_metrics, v2_macro_f1 = evaluate_raw_predictions(probs_v2, thresh_v2, y_test)

# 3. Load 127-sample audit benchmark
try:
    from scratch.run_nlp_audit import EVAL_DATASET
except (ImportError, ModuleNotFoundError):
    try:
        from run_nlp_audit import EVAL_DATASET
    except (ImportError, ModuleNotFoundError):
        _scratch_dir = os.path.dirname(os.path.abspath(__file__))
        if _scratch_dir not in sys.path:
            sys.path.insert(0, _scratch_dir)
        from run_nlp_audit import EVAL_DATASET

bench_texts = [x[0] for x in EVAL_DATASET]
bench_labels = np.array([x[1] for x in EVAL_DATASET], dtype=int)

bench_probs_v1 = pipe_v1.predict_proba(bench_texts)
bench_probs_v2 = pipe_v2.predict_proba(bench_texts)

v1_bench_metrics, v1_bench_macro = evaluate_raw_predictions(bench_probs_v1, thresh_v1, bench_labels)
v2_bench_metrics, v2_bench_macro = evaluate_raw_predictions(bench_probs_v2, thresh_v2, bench_labels)

# 4. Evaluate End-to-End Hybrid Moderator
# v1 Moderator (using V1 model artifact)
NLPModerator._cached_pipeline = None
NLPModerator._cached_categories = None
NLPModerator._cached_model_path = None
NLPModerator._cached_version = None
NLPModerator._inference_cache.clear()

mod_v1 = NLPModerator(model_path=V1_PATH, thresholds=thresh_v1)
mod_v1.keyword_thresholds = {
    "severe_toxic": 0.55, "threat": 0.55, "harassment": 0.55, "hate": 0.55, "insult": 0.65, "toxic": 0.50
}

# v2 Moderator (using V2 model artifact)
NLPModerator._cached_pipeline = None
NLPModerator._cached_categories = None
NLPModerator._cached_model_path = None
NLPModerator._cached_version = None
NLPModerator._inference_cache.clear()

mod_v2 = NLPModerator(model_path=V2_PATH, thresholds=thresh_v2)
mod_v2.keyword_thresholds = {
    "severe_toxic": 0.35, "threat": 0.50, "harassment": 0.40, "hate": 0.40, "insult": 0.40, "toxic": 0.35
}

def evaluate_moderator_on_benchmark(mod, benchmark):
    tp = fp = fn = tn = 0
    fps = []
    fns = []
    
    with mod._cache_lock:
        mod._inference_cache.clear()
        
    for text, gt_labels in benchmark:
        gt_toxic = (sum(gt_labels) > 0)
        res = mod.analyze_message(text)
        pred_toxic = (not res["is_safe"])
        
        if gt_toxic and pred_toxic:
            tp += 1
        elif not gt_toxic and pred_toxic:
            fp += 1
            fps.append((text, res["flagged_categories"], res["reason"]))
        elif gt_toxic and not pred_toxic:
            fn += 1
            fns.append((text, [CATEGORIES[i] for i, v in enumerate(gt_labels) if v == 1], res["scores"]))
        else:
            tn += 1
            
    acc = (tp + tn) / len(benchmark)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "false_positives": fps, "false_negatives": fns
    }

v1_mod_eval = evaluate_moderator_on_benchmark(mod_v1, EVAL_DATASET)
v2_mod_eval = evaluate_moderator_on_benchmark(mod_v2, EVAL_DATASET)

# 5. Evaluate 'You Are' Compliments
you_phrases = [
    "You are great!",
    "You are kind",
    "You are very beautiful",
    "You are very funny",
    "You are so smart",
    "You are wonderful",
    "You look stunning in your profile photo.",
    "You have such a warm smile.",
    "How are you?",
    "Are you free tonight?",
    "Where are you from?",
    "What do you do for work?",
    "Thank you so much",
    "I really enjoy talking with you.",
    "It is nice to meet you",
    "Can you help me with this?",
    "I agree with you completely.",
    "Do you like movies?",
    "You are amazing at what you do!",
    "You are an inspiring person."
]

def check_you_phrases(mod):
    fp_cnt = 0
    details = []
    with mod._cache_lock:
        mod._inference_cache.clear()
    for yp in you_phrases:
        res = mod.analyze_message(yp)
        if not res["is_safe"]:
            fp_cnt += 1
            details.append((yp, res["flagged_categories"], res["scores"]))
    return fp_cnt, len(you_phrases), fp_cnt / len(you_phrases), details

v1_you_fps, _, v1_you_rate, v1_you_details = check_you_phrases(mod_v1)
v2_you_fps, _, v2_you_rate, v2_you_details = check_you_phrases(mod_v2)

# Output Side-by-Side Comparison
print("\n" + "=" * 90)
print("SIDE-BY-SIDE MODEL COMPARISON REPORT")
print("=" * 90)
print(f"Metric / Category        | Model v1 (Seed 142 samples) | Model v2 (Improved 6,397 dataset)")
print("-" * 90)
print(f"Dataset Size             | 142 samples (synthetic seed) | 6,397 samples (Civil Comments + Davidson + Domain)")
print(f"Training Architecture    | TF-IDF + OvR Logistic Reg   | TF-IDF + OvR Logistic Reg (C=2.0)")
print("-" * 90)
print("PER-CATEGORY F1 ON HELD-OUT TEST SPLIT (n=961):")
for cat in CATEGORIES:
    f1_old = v1_test_metrics[cat]["f1"]
    f1_new = v2_test_metrics[cat]["f1"]
    print(f"  {cat:20s} | F1: {f1_old:6.4f}                 | F1: {f1_new:6.4f} (diff: {f1_new-f1_old:+6.4f})")
print(f"  {'Macro-F1 (Test)':20s} | {v1_macro_f1:6.4f}                 | {v2_macro_f1:6.4f} (diff: {v2_macro_f1-v1_macro_f1:+6.4f})")

print("-" * 90)
print("INDEPENDENT AUDIT BENCHMARK (n=127) - END-TO-END HYBRID MODERATION POLICY:")
print(f"  Overall Accuracy       | {v1_mod_eval['accuracy']*100:6.2f}% ({v1_mod_eval['tp']+v1_mod_eval['tn']}/127)       | {v2_mod_eval['accuracy']*100:6.2f}% ({v2_mod_eval['tp']+v2_mod_eval['tn']}/127)")
print(f"  Overall Precision      | {v1_mod_eval['precision']:6.4f}                 | {v2_mod_eval['precision']:6.4f}")
print(f"  Overall Recall         | {v1_mod_eval['recall']:6.4f}                 | {v2_mod_eval['recall']:6.4f} (diff: {v2_mod_eval['recall']-v1_mod_eval['recall']:+6.4f})")
print(f"  Overall F1-Score       | {v1_mod_eval['f1']:6.4f}                 | {v2_mod_eval['f1']:6.4f} (diff: {v2_mod_eval['f1']-v1_mod_eval['f1']:+6.4f})")
print(f"  False Positives (FP)   | {v1_mod_eval['fp']:2d} / 57 safe messages     | {v2_mod_eval['fp']:2d} / 57 safe messages")
print(f"  False Negatives (FN)   | {v1_mod_eval['fn']:2d} / 70 toxic messages    | {v2_mod_eval['fn']:2d} / 70 toxic messages")
print("-" * 90)
print(f"  'You Are' FP Rate      | {v1_you_fps:2d} / 20 ({v1_you_rate*100:5.1f}%)            | {v2_you_fps:2d} / 20 ({v2_you_rate*100:5.1f}%)")
print("=" * 90)

# Save structured comparison JSON
comp_results = {
    "test_split_n": len(test_texts),
    "v1_test_macro_f1": v1_macro_f1,
    "v2_test_macro_f1": v2_macro_f1,
    "v1_test_metrics": v1_test_metrics,
    "v2_test_metrics": v2_test_metrics,
    "benchmark_n": len(EVAL_DATASET),
    "v1_moderation": {
        "accuracy": v1_mod_eval["accuracy"],
        "precision": v1_mod_eval["precision"],
        "recall": v1_mod_eval["recall"],
        "f1": v1_mod_eval["f1"],
        "fp": v1_mod_eval["fp"],
        "fn": v1_mod_eval["fn"]
    },
    "v2_moderation": {
        "accuracy": v2_mod_eval["accuracy"],
        "precision": v2_mod_eval["precision"],
        "recall": v2_mod_eval["recall"],
        "f1": v2_mod_eval["f1"],
        "fp": v2_mod_eval["fp"],
        "fn": v2_mod_eval["fn"]
    },
    "v1_you_fp_rate": v1_you_rate,
    "v2_you_fp_rate": v2_you_rate,
    "v2_false_negatives": [x[0] for x in v2_mod_eval["false_negatives"]],
    "v2_false_positives": [x[0] for x in v2_mod_eval["false_positives"]]
}

with open(os.path.join(PROJECT_ROOT, "datasets", "toxicity", "model_comparison.json"), "w") as f:
    json.dump(comp_results, f, indent=2)

print(f"Comparison saved to datasets/toxicity/model_comparison.json")
