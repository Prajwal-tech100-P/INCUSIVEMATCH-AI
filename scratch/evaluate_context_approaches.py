"""
Evaluation script to measure and compare:
Approach 1: Existing model baseline (raw TF-IDF + OvR Logistic Regression + Keyword-assisted thresholds).
Approach 2: Existing model + Context-Aware Moderation & Disambiguation Rules.
Approach 3: Transformer feasibility assessment (latency, memory, dependencies).
Measures:
- Precision, Recall, F1 per category
- Macro-F1
- False Positive Rate on harmless jokes & idioms
- False Negative Rate on threats & harassment
- Confusion matrix
- Evaluation on datasets/toxicity/context_eval_dataset.json
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = r"c:\Users\ACER\Desktop\InclusiveMatch_AI_Secure_WebRTC_HTTPS_Ready 8"
sys.path.insert(0, PROJECT_ROOT)

from ai_modules.nlp_safety import NLPModerator

EVAL_PATH = os.path.join(PROJECT_ROOT, "datasets", "toxicity", "context_eval_dataset.json")
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_records = json.load(f)

CATEGORIES = ["toxic", "severe_toxic", "insult", "threat", "harassment", "hate"]

print(f"Loaded {len(eval_records)} context evaluation samples.")

# Evaluate Approach 1: Existing NLPModerator
mod_v2 = NLPModerator()

y_true_toxic = []
y_pred_action_v2 = []

joke_total = 0
joke_fp_v2 = 0

threat_total = 0
threat_fn_v2 = 0

ambig_total = 0
ambig_warned_v2 = 0

for item in eval_records:
    text = item["text"]
    cat_type = item["category_type"]
    exp_action = item["expected_action"]
    gt_labels = item["labels"]
    is_actually_toxic = (sum(gt_labels) > 0)
    y_true_toxic.append(1 if is_actually_toxic else 0)

    # Approach 1 moderation
    res1 = mod_v2.analyze_message(text)
    pred_blocked_v2 = (not res1["is_safe"])
    y_pred_action_v2.append(1 if pred_blocked_v2 else 0)

    # Track sub-groups
    if cat_type in ["friendly_joke", "idiom_hyperbole", "benign_discussion"]:
        joke_total += 1
        if pred_blocked_v2:
            joke_fp_v2 += 1
            print(f"  [Approach 1 FP] {cat_type}: '{text}' -> {res1['flagged_categories']}")

    if cat_type in ["clear_threat", "threat_no_profanity", "masked_threat_emoji"]:
        threat_total += 1
        if not pred_blocked_v2:
            threat_fn_v2 += 1
            sc = {k: round(v, 2) for k, v in res1.get('scores', {}).items()}
            print(f"  [Approach 1 FN] {cat_type}: '{text}' -> scores: {sc}")

    if cat_type == "ambiguous_phrase":
        ambig_total += 1
        if pred_blocked_v2:
            ambig_warned_v2 += 1

print("\n" + "=" * 80)
print("APPROACH 1 (CURRENT V2 BASELINE) EVALUATION RESULTS:")
print("=" * 80)

prec1 = precision_score(y_true_toxic, y_pred_action_v2, zero_division=0)
rec1 = recall_score(y_true_toxic, y_pred_action_v2, zero_division=0)
f1_1 = f1_score(y_true_toxic, y_pred_action_v2, zero_division=0)
cm1 = confusion_matrix(y_true_toxic, y_pred_action_v2)

print(f"Overall Precision : {prec1:6.4f}")
print(f"Overall Recall    : {rec1:6.4f}")
print(f"Overall F1 Score  : {f1_1:6.4f}")
print(f"Confusion Matrix  : TN={cm1[0,0]}, FP={cm1[0,1]}, FN={cm1[1,0]}, TP={cm1[1,1]}")
print(f"Jokes/Idioms FP Rate : {joke_fp_v2}/{joke_total} ({joke_fp_v2/joke_total*100:.1f}%)")
print(f"Threats/Abuse FN Rate: {threat_fn_v2}/{threat_total} ({threat_fn_v2/threat_total*100:.1f}%)")
print(f"Ambiguous Punitive Warning Rate: {ambig_warned_v2}/{ambig_total} ({ambig_warned_v2/ambig_total*100:.1f}%)")
