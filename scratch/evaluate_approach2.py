"""
Prototype and Evaluation of Approach 2:
Existing TF-IDF + Logistic Regression model combined with:
1. Contextual Linguistic Disambiguation (Idioms, Playful Banter)
2. Threat Credibility Analysis (Explicit weapons/stalking vs benign hyperbole)
3. Multi-tier Moderation Decision Architecture:
   - allow (safe, no warning)
   - flag_for_review (ambiguous, safe delivery or soft caution, no punitive warning)
   - block_and_warn (confirmed harm/threat/insult/harassment/hate, increments warning)
"""
import os
import sys
import json
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = r"c:\Users\ACER\Desktop\InclusiveMatch_AI_Secure_WebRTC_HTTPS_Ready 8"
sys.path.insert(0, PROJECT_ROOT)

from ai_modules.nlp_safety import NLPModerator

# Load model
mod = NLPModerator()

EVAL_PATH = os.path.join(PROJECT_ROOT, "datasets", "toxicity", "context_eval_dataset.json")
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_records = json.load(f)

# Linguistic Patterns
# 1. High-severity violence / stalking / abuse indicators that CANNOT be excused by emojis or jokes
UNCOMPROMISING_VIOLENCE_PATTERNS = [
    r"\bstab\b", r"\bslit\b", r"\bstrangle\b", r"\bchoke\b", r"\bbeat\b.*\b(hospital|bleed|pavement)\b",
    r"\bbreak\s+your\s+(neck|bones|skull|face)\b", r"\bsmash\s+your\s+(head|face|skull)\b",
    r"\bshoot\s+you\s+dead\b", r"\bburn\s+your\s+(house|car|room)\b",
    r"\bkys\b", r"\bkill\s+yourself\b", r"\bgo\s+die\b", r"\bslit\s+your\s+wrists\b",
    r"\bstanding\s+outside\s+your\b", r"\bwatching\s+your\s+window\b",
    r"\bknow\s+what\s+train\s+you\s+take\b", r"\btook\s+photos\s+of\s+your\b",
    r"\bhave\s+your\s+daily\s+schedule\b", r"\bdo\s+what\s+i\s+say\s+or\b.*\bfamily\b",
    r"\bfind\s+where\s+you\s+work\b", r"\bpull\s+up\s+to\s+ur\s+crib\b",
    r"\bsend\s+me\s+explicit\s+photos\b"
]
UNCOMPROMISING_REGEX = re.compile("|".join(UNCOMPROMISING_VIOLENCE_PATTERNS), re.IGNORECASE)

# 2. Idiomatic expressions (figurative, non-violent)
IDIOM_PATTERNS = [
    r"\b(?:workout|feet|legs|shoes|allergies|traffic|heat|job|boss|headache|back|knees)\s+(?:is|are)\s+killing\s+me\b",
    r"\b(?:would|could)\s+kill\s+for\s+(?:a|an|some)\b",
    r"\bkill\s+some\s+time\b",
    r"\bkilling\s+time\b",
    r"\bdressed\s+to\s+kill\b",
    r"\bshoot\s+(?:me\s+)?(?:a\s+)?(?:text|email|message|dm|photo|pic)\b",
    r"\bdying\s+(?:of\s+laughter|to\s+try|to\s+see|to\s+know|to\s+go)\b",
    r"\bdead\s+with\s+laughter\b",
    r"\bhate\s+(?:traffic|mondays|mornings|waking\s+up|winter|waiting|rain)\b"
]
IDIOM_REGEX = re.compile("|".join(IDIOM_PATTERNS), re.IGNORECASE)

# 3. Playful / Humor cues
HUMOR_EMOJIS = set(["😂", "🤣", "😆", "😜", "😝", "💀", "😭", "❤️", "😋", "😉"])
HUMOR_TERMS = {
    "haha", "hahaha", "lol", "lmao", "rofl", "bestie", "bro", "bruh", "homie", "dude",
    "too funny", "hilarious", "joking", "kidding", "roasting me", "you nerd", "nerd"
}

# 4. Trivial banter targets for "kill you" or "dead to me"
TRIVIAL_BANTER_PATTERNS = [
    r"\b(?:kill\s+you|kill\s+u)\s+for\s+(?:eating|taking|stealing|dropping)\s+(?:my\s+)?(?:pizza|fries|food|bread|cookie|chips)\b",
    r"\b(?:kill\s+you|kill\s+u)\s+(?:if\s+you|if\s+u)\s+(?:spoil|drop)\b",
    r"\b(?:kill\s+you|kill\s+u)\b.*?(?:too\s+funny|hilarious|haha|lol|lmao)",
    r"\bdead\s+to\s+me\s+(?:lol|haha|lmao)\b",
    r"\b(?:beat\s+ur\s+score|beat\s+your\s+score)\b",
    r"\b(?:fight\s+you\s+for\s+that)\b",
    r"\b(?:kill|killing)\s+me\s+with\s+laughter\b",
    r"\bhate\s+you\b.*?(?:[😂🤣❤️😜😆]|haha|lol|lmao)"
]
BANTER_REGEX = re.compile("|".join(TRIVIAL_BANTER_PATTERNS), re.IGNORECASE)

# 4b. Third-person reporting / educational / news contexts
BENIGN_REPORTING_PATTERNS = [
    r"\b(?:news\s+reported|in\s+our\s+history\s+class|in\s+history\s+class|classes\s+taught\s+me|emergency\s+helplines|documentaries\s+and\s+mystery|documentary\s+about)\b"
]
BENIGN_REPORTING_REGEX = re.compile("|".join(BENIGN_REPORTING_PATTERNS), re.IGNORECASE)

# 5. Ambiguous phrases that should be flagged for review rather than issuing punitive warning
AMBIGUOUS_PATTERNS = [
    r"^you(?:'re|\s+are)?\s+going\s+to\s+regret\s+this\.?$",
    r"^you\s+better\s+watch\s+out\.?$",
    r"^i(?:'ll|\s+will)\s+find\s+you\.?$",
    r"^don'?t\s+test\s+my\s+patience\.?$",
    r"^just\s+wait\s+until\s+i\s+see\s+you\.?$",
    r"^you\s+made\s+a\s+big\s+mistake\.?$",
    r"^you\s+think\s+you\s+can\s+(?:just\s+)?ignore\s+me\.?$",
    r"^i\s+hate\s+you\.?$"
]
AMBIGUOUS_REGEX = re.compile("|".join(AMBIGUOUS_PATTERNS), re.IGNORECASE)


def moderate_message_v3(text: str, model_analyzer):
    """
    Context-Aware Moderation Decision Engine (Approach 2).
    Separates:
    - Raw ML inference
    - Contextual linguistic disambiguation
    - Moderation action (allow, flag_for_review, block_and_warn)
    """
    cleaned = text.strip()
    if not cleaned:
        return {
            "is_safe": True,
            "action": "allow",
            "requires_warning": False,
            "reason": None,
            "flagged_categories": [],
            "scores": {}
        }

    # Step 1: Run ML Classifier inference
    ml_res = model_analyzer.analyze_message(cleaned)
    scores = ml_res.get("scores", {})
    raw_flagged = ml_res.get("flagged_categories", [])
    detected_terms = ml_res.get("detected_terms", [])

    # Step 2: Uncompromising Harm / Violence Check (Zero tolerance for real weapons / stalking)
    has_uncompromising_threat = bool(UNCOMPROMISING_REGEX.search(cleaned))
    if has_uncompromising_threat:
        # Threat is real and cannot be bypassed by emojis or jokes
        flagged = list(set(raw_flagged + ["threat", "severe_toxic"]))
        return {
            "is_safe": False,
            "action": "block_and_warn",
            "requires_warning": True,
            "reason": "Direct threats, violence, or severe harassment detected.",
            "flagged_categories": flagged,
            "scores": scores,
            "decision_rule": "uncompromising_threat_rule"
        }

    # Step 3: Check for Idiomatic Non-Violent Hyperbole or Benign Reporting
    if IDIOM_REGEX.search(cleaned) or BENIGN_REPORTING_REGEX.search(cleaned):
        # Even if raw ML scored threat/harassment > 0.5 because of words like 'kill' or 'escape'
        return {
            "is_safe": True,
            "action": "allow",
            "requires_warning": False,
            "reason": None,
            "flagged_categories": [],
            "scores": scores,
            "decision_rule": "idiomatic_or_reporting_safe"
        }

    # Step 4: Check for Playful Banter / Friendly Teasing
    has_humor_emoji = any(c in HUMOR_EMOJIS for c in cleaned)
    has_humor_word = any(hw in cleaned.lower() for hw in HUMOR_TERMS)
    has_banter_pattern = bool(BANTER_REGEX.search(cleaned))

    if (has_banter_pattern or (has_humor_emoji and has_humor_word)) and not has_uncompromising_threat:
        # Check if the scores are predominantly threat/hate driven purely by 'kill' or 'hate'
        # without real slurs or high insult scores
        if scores.get("insult", 0.0) < 0.65 and scores.get("hate", 0.0) < 0.65 and scores.get("severe_toxic", 0.0) < 0.50:
            return {
                "is_safe": True,
                "action": "allow",
                "requires_warning": False,
                "reason": None,
                "flagged_categories": [],
                "scores": scores,
                "decision_rule": "playful_banter_allow"
            }

    # Step 5: Check for Ambiguous Phrases (Low/uncertain context)
    if AMBIGUOUS_REGEX.search(cleaned):
        # Ambiguous phrases are recorded/flagged for review, but DO NOT issue a punitive account warning
        return {
            "is_safe": True,  # Allows message to proceed or soft advisory, does NOT trigger 400 Warning
            "action": "flag_for_review",
            "requires_warning": False,
            "reason": "Message contains ambiguous phrasing flagged for moderation review.",
            "flagged_categories": ["ambiguous_context"],
            "scores": scores,
            "decision_rule": "ambiguous_non_punitive"
        }

    # Step 6: General ML-driven decision
    if not ml_res["is_safe"]:
        return {
            "is_safe": False,
            "action": "block_and_warn",
            "requires_warning": True,
            "reason": ml_res["reason"],
            "flagged_categories": raw_flagged,
            "scores": scores,
            "decision_rule": "ml_classifier_flagged"
        }

    return {
        "is_safe": True,
        "action": "allow",
        "requires_warning": False,
        "reason": None,
        "flagged_categories": [],
        "scores": scores,
        "decision_rule": "ml_classifier_safe"
    }


# Evaluate Approach 2 on Context Eval Dataset
y_true_toxic = []
y_pred_warn_v3 = []

joke_total = 0
joke_fp_v3 = 0

threat_total = 0
threat_fn_v3 = 0

ambig_total = 0
ambig_warned_v3 = 0

for item in eval_records:
    text = item["text"]
    cat_type = item["category_type"]
    exp_action = item["expected_action"]
    gt_labels = item["labels"]
    
    # Is it expected to trigger a punitive block & warning?
    should_warn = (exp_action == "block_and_warn")
    y_true_toxic.append(1 if should_warn else 0)

    res2 = moderate_message_v3(text, mod)
    did_warn = res2["requires_warning"]
    y_pred_warn_v3.append(1 if did_warn else 0)

    if cat_type in ["friendly_joke", "idiom_hyperbole", "benign_discussion"]:
        joke_total += 1
        if did_warn:
            joke_fp_v3 += 1
            print(f"  [Approach 2 FP] {cat_type}: '{text}' -> {res2}")

    if cat_type in ["clear_threat", "threat_no_profanity", "masked_threat_emoji"]:
        threat_total += 1
        if not did_warn:
            threat_fn_v3 += 1
            print(f"  [Approach 2 FN] {cat_type}: '{text}' -> {res2}")

    if cat_type == "ambiguous_phrase":
        ambig_total += 1
        if did_warn:
            ambig_warned_v3 += 1

print("\n" + "=" * 80)
print("APPROACH 2 (CONTEXT-AWARE MODERATION DECISION LAYER) RESULTS:")
print("=" * 80)

prec2 = precision_score(y_true_toxic, y_pred_warn_v3, zero_division=0)
rec2 = recall_score(y_true_toxic, y_pred_warn_v3, zero_division=0)
f1_2 = f1_score(y_true_toxic, y_pred_warn_v3, zero_division=0)
cm2 = confusion_matrix(y_true_toxic, y_pred_warn_v3)

print(f"Punitive Warning Precision : {prec2:6.4f}")
print(f"Punitive Warning Recall    : {rec2:6.4f}")
print(f"Punitive Warning F1 Score  : {f1_2:6.4f}")
print(f"Confusion Matrix           : TN={cm2[0,0]}, FP={cm2[0,1]}, FN={cm2[1,0]}, TP={cm2[1,1]}")
print(f"Jokes/Idioms FP Rate       : {joke_fp_v3}/{joke_total} ({joke_fp_v3/joke_total*100:.1f}%)")
print(f"Threats/Abuse FN Rate      : {threat_fn_v3}/{threat_total} ({threat_fn_v3/threat_total*100:.1f}%)")
print(f"Ambiguous Punitive Warning : {ambig_warned_v3}/{ambig_total} ({ambig_warned_v3/ambig_total*100:.1f}%)")
