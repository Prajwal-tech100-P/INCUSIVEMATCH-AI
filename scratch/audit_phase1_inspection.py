import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")
from ai_modules.nlp_safety import NLPModerator

mod = NLPModerator()
print("Model Loaded:", mod.model_loaded, "Status:", mod.model_status, "Version:", mod._cached_version)
print("Thresholds:", mod.thresholds)
print("Keyword Thresholds:", mod.keyword_thresholds)

test_phrases = [
    # Friendly jokes / idioms
    "Bro, I will kill you for eating my pizza 😂",
    "I'll kill you 😂 you're too funny.",
    "I hate you 😂❤️",
    "You're so annoying, bestie 😂",
    "This workout is killing me.",
    "I would kill for a cup of coffee.",
    # Ambiguous messages
    "You're going to regret this.",
    "I hate you.",
    "You better watch out.",
    "I'll find you.",
    # Clear threats and abuse
    "I will hurt you tonight.",
    "I will attack you if you leave me.",
    "I am going to come to your house and hurt you.",
    "You are a disgusting ugly pig and nobody wants you"
]

print("\n=== ANALYSIS OF KEY EXAMPLES ===")
for p in test_phrases:
    res = mod.analyze_message(p)
    sc = {k: round(v, 3) for k, v in res.get("scores", {}).items()}
    print(f"Text: {p}")
    print(f"  is_safe: {res['is_safe']} | flagged: {res['flagged_categories']} | terms: {res['detected_terms']}")
    print(f"  reason: {res.get('reason')}")
    print(f"  scores: {sc}\n")
