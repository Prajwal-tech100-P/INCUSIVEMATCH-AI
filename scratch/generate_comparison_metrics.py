"""
Calculate and save verified Before-vs-After benchmark metrics on datasets/toxicity/context_eval_dataset.json
Compares Approach 1 (Raw Baseline) vs Approach 2 (Context-Aware Moderation Engine).
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = r"c:\Users\ACER\Desktop\InclusiveMatch_AI_Secure_WebRTC_HTTPS_Ready 8"
sys.path.insert(0, PROJECT_ROOT)

from ai_modules.nlp_safety import NLPModerator

EVAL_PATH = os.path.join(PROJECT_ROOT, "datasets", "toxicity", "context_eval_dataset.json")
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_records = json.load(f)

mod = NLPModerator()

# Approach 1: Simulated raw model (without context engine)
def approach_1_raw(text):
    clf = mod.classify_raw(text)
    raw_flagged = clf["flagged_categories"]
    is_safe = (len(raw_flagged) == 0)
    return {
        "is_safe": is_safe,
        "action": "allow" if is_safe else "block_and_warn",
        "requires_warning": not is_safe,
        "scores": clf["scores"],
        "flagged": raw_flagged
    }

# Approach 2: Live production context engine
def approach_2_context(text):
    return mod.analyze_message(text)

y_true = []           # 1 if should be blocked/warned, 0 if safe/allow/flag_for_review
y_pred_app1 = []
y_pred_app2 = []

joke_total = 0
joke_fp_1 = 0
joke_fp_2 = 0

threat_total = 0
threat_fn_1 = 0
threat_fn_2 = 0

ambig_total = 0
ambig_warned_1 = 0
ambig_warned_2 = 0

for item in eval_records:
    text = item["text"]
    cat = item["category_type"]
    action_exp = item["expected_action"]
    
    should_warn = 1 if action_exp == "block_and_warn" else 0
    y_true.append(should_warn)

    res1 = approach_1_raw(text)
    res2 = approach_2_context(text)

    p1 = 1 if res1["requires_warning"] else 0
    p2 = 1 if res2["requires_warning"] else 0

    y_pred_app1.append(p1)
    y_pred_app2.append(p2)

    if cat in ["friendly_joke", "idiom_hyperbole", "benign_discussion"]:
        joke_total += 1
        if p1 == 1: joke_fp_1 += 1
        if p2 == 1: joke_fp_2 += 1

    if cat in ["clear_threat", "threat_no_profanity", "masked_threat_emoji"]:
        threat_total += 1
        if p1 == 0: threat_fn_1 += 1
        if p2 == 0: threat_fn_2 += 1

    if cat == "ambiguous_phrase":
        ambig_total += 1
        if p1 == 1: ambig_warned_1 += 1
        if p2 == 1: ambig_warned_2 += 1

cm1 = confusion_matrix(y_true, y_pred_app1).tolist()
cm2 = confusion_matrix(y_true, y_pred_app2).tolist()

prec1 = float(precision_score(y_true, y_pred_app1, zero_division=0))
rec1 = float(recall_score(y_true, y_pred_app1, zero_division=0))
f1_1 = float(f1_score(y_true, y_pred_app1, zero_division=0))

prec2 = float(precision_score(y_true, y_pred_app2, zero_division=0))
rec2 = float(recall_score(y_true, y_pred_app2, zero_division=0))
f1_2 = float(f1_score(y_true, y_pred_app2, zero_division=0))

results = {
    "total_samples": len(eval_records),
    "approach_1_raw": {
        "precision": prec1,
        "recall": rec1,
        "f1": f1_1,
        "confusion_matrix": {"tn": cm1[0][0], "fp": cm1[0][1], "fn": cm1[1][0], "tp": cm1[1][1]},
        "joke_idiom_fp_rate": joke_fp_1 / joke_total if joke_total else 0,
        "threat_fn_rate": threat_fn_1 / threat_total if threat_total else 0,
        "ambiguous_warning_rate": ambig_warned_1 / ambig_total if ambig_total else 0,
    },
    "approach_2_context_aware": {
        "precision": prec2,
        "recall": rec2,
        "f1": f1_2,
        "confusion_matrix": {"tn": cm2[0][0], "fp": cm2[0][1], "fn": cm2[1][0], "tp": cm2[1][1]},
        "joke_idiom_fp_rate": joke_fp_2 / joke_total if joke_total else 0,
        "threat_fn_rate": threat_fn_2 / threat_total if threat_total else 0,
        "ambiguous_warning_rate": ambig_warned_2 / ambig_total if ambig_total else 0,
    }
}

out_file = os.path.join(PROJECT_ROOT, "datasets", "toxicity", "context_moderation_results.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
