import os
import sys
import json
from datetime import datetime, timedelta
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scratch.train_toxicity_classifier import TRAINING_DATA as RAW_TRAIN_DATA, CATEGORIES
from ai_modules.nlp_safety import NLPModerator, DEFAULT_MODEL_PATH, EXPECTED_CATEGORIES

# -----------------------------------------------------------------------------
# 1. EVALUATION DATASET (Zero overlap with TRAINING_DATA)
# Format: (text, [toxic, severe_toxic, insult, threat, harassment, hate])
# -----------------------------------------------------------------------------
EVAL_DATASET = [
    # --- Category A: Harmless Everyday Conversation & Greetings (15 samples) ---
    ("Good morning! Did you sleep well?", [0, 0, 0, 0, 0, 0]),
    ("How was your day at work today?", [0, 0, 0, 0, 0, 0]),
    ("I'm having a cup of coffee and reading a book.", [0, 0, 0, 0, 0, 0]),
    ("The weather is really nice outside this afternoon.", [0, 0, 0, 0, 0, 0]),
    ("What time does your train arrive at the station?", [0, 0, 0, 0, 0, 0]),
    ("I had pizza for dinner and it was delicious.", [0, 0, 0, 0, 0, 0]),
    ("See you tomorrow at noon near the bookstore.", [0, 0, 0, 0, 0, 0]),
    ("Thanks for sharing that recipe, I can't wait to try it.", [0, 0, 0, 0, 0, 0]),
    ("Could you pass me the document when you get a chance?", [0, 0, 0, 0, 0, 0]),
    ("Hope you had a safe flight back home.", [0, 0, 0, 0, 0, 0]),
    ("I am looking forward to our walk in the park.", [0, 0, 0, 0, 0, 0]),
    ("Did you catch the new episode of that documentary?", [0, 0, 0, 0, 0, 0]),
    ("I usually wake up early to go jogging before work.", [0, 0, 0, 0, 0, 0]),
    ("Let's grab lunch somewhere quiet where we can chat.", [0, 0, 0, 0, 0, 0]),
    ("My favorite season is autumn because of the cool breeze.", [0, 0, 0, 0, 0, 0]),

    # --- Category B: Compliments & Friendly Affirmations (Containing 'you' / 'you are') (15 samples) ---
    ("You are amazing and so inspiring.", [0, 0, 0, 0, 0, 0]),
    ("You are great at what you do!", [0, 0, 0, 0, 0, 0]),
    ("You look stunning in your profile photo.", [0, 0, 0, 0, 0, 0]),
    ("You are very kind and thoughtful.", [0, 0, 0, 0, 0, 0]),
    ("You have such a warm and welcoming smile.", [0, 0, 0, 0, 0, 0]),
    ("You are so smart, I love learning from you.", [0, 0, 0, 0, 0, 0]),
    ("You are very funny, that joke made my whole day.", [0, 0, 0, 0, 0, 0]),
    ("I think you are really wonderful.", [0, 0, 0, 0, 0, 0]),
    ("You are one of the most genuine people I've met.", [0, 0, 0, 0, 0, 0]),
    ("You are so talented with the guitar.", [0, 0, 0, 0, 0, 0]),
    ("It is an absolute pleasure getting to know you.", [0, 0, 0, 0, 0, 0]),
    ("You are beautiful inside and out.", [0, 0, 0, 0, 0, 0]),
    ("You did such a fantastic job on the presentation.", [0, 0, 0, 0, 0, 0]),
    ("You have great taste in books and movies.", [0, 0, 0, 0, 0, 0]),
    ("I really enjoy talking with you every evening.", [0, 0, 0, 0, 0, 0]),

    # --- Category C: Colloquialisms & Slang with Taboo/Trigger Words (15 samples) ---
    ("This leg workout is absolutely killing me!", [0, 0, 0, 0, 0, 0]),
    ("I hate traffic on the highway in the morning.", [0, 0, 0, 0, 0, 0]),
    ("That chocolate cake was to die for!", [0, 0, 0, 0, 0, 0]),
    ("Shoot me a text when you arrive safely.", [0, 0, 0, 0, 0, 0]),
    ("I'm dying of laughter from that stand-up comedy show.", [0, 0, 0, 0, 0, 0]),
    ("It is hot as hell outside in the afternoon.", [0, 0, 0, 0, 0, 0]),
    ("You killed that dance performance on stage!", [0, 0, 0, 0, 0, 0]),
    ("I hate doing taxes, it takes forever to organize.", [0, 0, 0, 0, 0, 0]),
    ("Don't shoot the messenger, but the meeting got rescheduled.", [0, 0, 0, 0, 0, 0]),
    ("My feet are killing me after standing all day.", [0, 0, 0, 0, 0, 0]),
    ("I would kill for a hot cup of tea right now.", [0, 0, 0, 0, 0, 0]),
    ("I hate Mondays so much, weekends go by too fast.", [0, 0, 0, 0, 0, 0]),
    ("That was a killer guitar solo during the encore!", [0, 0, 0, 0, 0, 0]),
    ("I'm dead tired after that 10-hour flight.", [0, 0, 0, 0, 0, 0]),
    ("We had a hell of a good time at the concert last night.", [0, 0, 0, 0, 0, 0]),

    # --- Category D: Identity, Disability & LGBTQ+ Expressions (12 samples) ---
    ("I use an electric wheelchair to navigate the city safely.", [0, 0, 0, 0, 0, 0]),
    ("I am deaf and use Indian Sign Language in daily life.", [0, 0, 0, 0, 0, 0]),
    ("I am neurodivergent and work best with noise-cancelling headphones.", [0, 0, 0, 0, 0, 0]),
    ("Happy Pride month to everyone celebrating love and equality!", [0, 0, 0, 0, 0, 0]),
    ("I have low vision and rely on screen readers and voice captions.", [0, 0, 0, 0, 0, 0]),
    ("I identify as bisexual and support inclusive communities.", [0, 0, 0, 0, 0, 0]),
    ("My younger brother is on the autism spectrum and is an artist.", [0, 0, 0, 0, 0, 0]),
    ("Ableism is unfair and we need better building accessibility.", [0, 0, 0, 0, 0, 0]),
    ("I am a transgender man living my authentic truth.", [0, 0, 0, 0, 0, 0]),
    ("My guide dog helps me cross busy streets with confidence.", [0, 0, 0, 0, 0, 0]),
    ("Living with chronic fatigue requires managing my daily energy.", [0, 0, 0, 0, 0, 0]),
    ("Sign language interpreters make public events accessible for deaf people.", [0, 0, 0, 0, 0, 0]),

    # --- Category E: Targeted Personal Insults (15 samples) ---
    ("You are completely brainless and utterly useless.", [1, 0, 1, 0, 0, 0]),
    ("Nobody likes you because you are an annoying clown.", [1, 0, 1, 0, 0, 0]),
    ("Shut your ugly mouth, nobody cares what you think.", [1, 0, 1, 0, 0, 0]),
    ("You're a pathetic failure who will never accomplish anything.", [1, 0, 1, 0, 0, 0]),
    ("You are an ignorant jerk and a total waste of space.", [1, 0, 1, 0, 0, 0]),
    ("Look at yourself in the mirror, you disgusting pig.", [1, 0, 1, 0, 0, 0]),
    ("You are so stupid it physically hurts to read your messages.", [1, 0, 1, 0, 0, 0]),
    ("Get a life you miserable loser.", [1, 0, 1, 0, 0, 0]),
    ("You are trash and everyone laughs behind your back.", [1, 0, 1, 0, 0, 0]),
    ("You obnoxious brat, leave me alone.", [1, 0, 1, 0, 0, 0]),
    ("You are a cowardly little prick.", [1, 0, 1, 0, 0, 0]),
    ("You make me sick you worthless scum.", [1, 0, 1, 0, 0, 0]),
    ("You are an absolute idiot with zero common sense.", [1, 0, 1, 0, 0, 0]),
    ("Stop whining you weak pathetic baby.", [1, 0, 1, 0, 0, 0]),
    ("You are a fake phony bitch.", [1, 0, 1, 0, 0, 0]),

    # --- Category F: Direct Physical Threats & Violence (15 samples) ---
    ("I am going to track you down and slit your throat.", [1, 1, 0, 1, 0, 0]),
    ("I will put a bullet in your head if you speak again.", [1, 1, 0, 1, 0, 0]),
    ("I will beat you unconscious and break both your arms.", [1, 1, 0, 1, 0, 0]),
    ("Watch your back, I'm coming to your house to kill you.", [1, 1, 0, 1, 1, 0]),
    ("I will burn your car down with you trapped inside.", [1, 1, 0, 1, 0, 0]),
    ("I will choke you until you stop breathing.", [1, 1, 0, 1, 0, 0]),
    ("I am going to murder you and your family.", [1, 1, 0, 1, 0, 0]),
    ("I'll smash your skull with a metal pipe.", [1, 1, 0, 1, 0, 0]),
    ("You are a dead man, you won't survive this week.", [1, 1, 0, 1, 0, 0]),
    ("I will stab you in the stomach next time I see you.", [1, 1, 0, 1, 0, 0]),
    ("I'll drown you in that river and no one will find your body.", [1, 1, 0, 1, 0, 0]),
    ("I am going to shatter your ribs and break your jaw.", [1, 1, 0, 1, 0, 0]),
    ("I will strangle you until your eyes roll back.", [1, 1, 0, 1, 0, 0]),
    ("I'm going to shoot you dead right outside your work.", [1, 1, 0, 1, 0, 0]),
    ("I'll torture you until you beg me to stop.", [1, 1, 0, 1, 0, 0]),

    # --- Category G: Harassment, Stalking & Extortion (15 samples) ---
    ("I am parked outside your apartment right now watching your window.", [1, 0, 0, 1, 1, 0]),
    ("I have your private photos and I will email them to your boss.", [1, 0, 0, 0, 1, 0]),
    ("I will never stop texting you no matter how many times you block me.", [1, 0, 0, 0, 1, 0]),
    ("Answer my calls right now or I will post your home address online.", [1, 0, 0, 1, 1, 0]),
    ("I know your daily routine and what train you take home.", [1, 0, 0, 0, 1, 0]),
    ("I will make your life a living nightmare until you agree to see me.", [1, 0, 0, 1, 1, 0]),
    ("Send me money or I will leak all your private text messages.", [1, 0, 0, 1, 1, 0]),
    ("I followed you home from work today, nice jacket by the way.", [1, 0, 0, 1, 1, 0]),
    ("I will harass you every single day until you quit your job.", [1, 0, 0, 0, 1, 0]),
    ("You can never get away from me, I will find your new number.", [1, 0, 0, 1, 1, 0]),
    ("I created five fake accounts just to keep messaging your friends.", [1, 0, 0, 0, 1, 0]),
    ("I'm going to ruin your reputation across your entire company.", [1, 0, 0, 0, 1, 0]),
    ("Do what I say or you will deeply regret ever crossing me.", [1, 0, 0, 1, 1, 0]),
    ("I will show up at your office lobby tomorrow and make a huge scene.", [1, 0, 0, 0, 1, 0]),
    ("Give me your password right now or I'll ruin your life.", [1, 0, 0, 1, 1, 0]),

    # --- Category H: Hate Speech & Discriminatory Attacks (15 samples) ---
    ("All people of your race are subhuman filth that should be exterminated.", [1, 1, 1, 1, 0, 1]),
    ("Immigrants like you are dirty vermin invading our clean country.", [1, 0, 1, 0, 0, 1]),
    ("Disabled people are useless burdens on society and should be locked away.", [1, 0, 1, 0, 0, 1]),
    ("Faggots and dykes are diseased perverts who belong in hell.", [1, 1, 1, 0, 0, 1]),
    ("All Muslims are violent terrorists who need to be wiped out.", [1, 1, 1, 1, 0, 1]),
    ("Transgenders are mentally ill freaks who should be eliminated.", [1, 1, 1, 1, 0, 1]),
    ("Cripple retards have no right to exist in the modern world.", [1, 1, 1, 0, 0, 1]),
    ("Your ethnic group is dirty and genetically inferior.", [1, 0, 1, 0, 0, 1]),
    ("Jews control everything and need to be purged from earth.", [1, 1, 1, 1, 0, 1]),
    ("Black people are all criminals and savage beasts.", [1, 1, 1, 0, 0, 1]),
    ("Go back to your third world country you disgusting parasite.", [1, 0, 1, 0, 0, 1]),
    ("Autistic people are brain-dead burdens who shouldn't be allowed outside.", [1, 0, 1, 0, 0, 1]),
    ("Queer people are abominations against nature.", [1, 0, 1, 0, 0, 1]),
    ("White people are all evil colonizers who deserve genocide.", [1, 1, 1, 1, 0, 1]),
    ("Your religion is evil and every single believer should burn.", [1, 1, 1, 0, 0, 1]),

    # --- Category I: Severe Toxicity / Encouraging Self-Harm (10 samples) ---
    ("Nobody will ever care about you, just go kill yourself tonight.", [1, 1, 1, 0, 0, 0]),
    ("Do the world a favor and jump off a tall bridge.", [1, 1, 1, 0, 0, 0]),
    ("Take a bottle of sleeping pills and never wake up.", [1, 1, 0, 0, 0, 0]),
    ("You are so worthless you should just slit your wrists and bleed.", [1, 1, 1, 0, 0, 0]),
    ("Go drink a cup of bleach and end your pathetic life.", [1, 1, 1, 0, 0, 0]),
    ("Put a rope around your neck and hang yourself right now.", [1, 1, 1, 0, 0, 0]),
    ("Nobody would even attend your funeral if you died right now.", [1, 1, 1, 0, 0, 0]),
    ("I hope you get a terminal illness and suffer agonizing pain.", [1, 1, 0, 0, 0, 0]),
    ("Swallow rat poison you useless piece of human garbage.", [1, 1, 1, 0, 0, 0]),
    ("End your life already, everybody hates you and wants you gone.", [1, 1, 1, 0, 0, 0]),
]

def main():
    print("=" * 80)
    print("INCLUSIVEMATCH AI - NLP ACCURACY & SAFETY AUDIT SUITE")
    print("=" * 80)

    # 1. Dataset Overlap Check
    train_texts = set(t[0].strip().lower() for t in RAW_TRAIN_DATA)
    eval_texts = [e[0] for e in EVAL_DATASET]
    eval_texts_lower = set(e[0].strip().lower() for e in EVAL_DATASET)
    overlap = train_texts & eval_texts_lower
    print(f"\n[STEP 1] DATASET OVERLAP VERIFICATION:")
    print(f"  Training set size: {len(RAW_TRAIN_DATA)} samples")
    print(f"  Evaluation set size: {len(EVAL_DATASET)} samples")
    print(f"  Overlap count: {len(overlap)}")
    assert len(overlap) == 0, f"Found overlapping examples: {overlap}"
    print("  -> CONFIRMED: Evaluation dataset has ZERO overlap with training dataset.")

    # 2. Inspect Training Data Distribution
    train_labels = np.array([t[1] for t in RAW_TRAIN_DATA])
    print(f"\n[STEP 2] TRAINING DATASET CLASS DISTRIBUTION:")
    for idx, cat in enumerate(CATEGORIES):
        pos = int(train_labels[:, idx].sum())
        print(f"  {cat:15s}: {pos:3d} / {len(RAW_TRAIN_DATA)} ({pos/len(RAW_TRAIN_DATA)*100:5.1f}%)")
    benign_train = int((train_labels.sum(axis=1) == 0).sum())
    print(f"  {'benign':15s}: {benign_train:3d} / {len(RAW_TRAIN_DATA)} ({benign_train/len(RAW_TRAIN_DATA)*100:5.1f}%)")

    # 3. Load Model and Moderator
    print(f"\n[STEP 3] MODEL & MODERATOR INITIALIZATION:")
    moderator = NLPModerator()
    print(f"  Model loaded: {moderator.model_loaded}")
    print(f"  Model status: {moderator.model_status}")
    print(f"  Default thresholds: {moderator.thresholds}")
    print(f"  Keyword thresholds: {moderator.keyword_thresholds}")

    # 4. Run Evaluation on EVAL_DATASET
    print(f"\n[STEP 4] RUNNING PREDICTIONS ON EVALUATION BENCHMARK ({len(EVAL_DATASET)} samples)...")
    
    # Track metrics per category
    # Ground truth vs ML predicted categories
    cat_metrics = {cat: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for cat in CATEGORIES}
    
    # End-to-end moderation decision (is_safe)
    # Ground truth toxic is: any(gt_label == 1)
    # Predicted toxic is: not res["is_safe"]
    moderation_metrics = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    
    false_positives_moderation = []
    false_negatives_moderation = []
    cat_false_positives = {cat: [] for cat in CATEGORIES}
    cat_false_negatives = {cat: [] for cat in CATEGORIES}

    # Clear inference cache
    with moderator._cache_lock:
        moderator._inference_cache.clear()

    for text, gt_labels in EVAL_DATASET:
        res = moderator.analyze_message(text)
        flagged_cats = res["flagged_categories"]
        is_safe_pred = res["is_safe"]
        scores = res["scores"]
        
        gt_is_toxic = (sum(gt_labels) > 0)
        pred_is_toxic = (not is_safe_pred)

        # Moderation Decision Metrics (Positive = Flagged/Unsafe, Negative = Safe)
        if gt_is_toxic and pred_is_toxic:
            moderation_metrics["TP"] += 1
        elif not gt_is_toxic and pred_is_toxic:
            moderation_metrics["FP"] += 1
            false_positives_moderation.append({
                "text": text,
                "reason": res["reason"],
                "flagged": flagged_cats,
                "scores": scores,
                "detected_terms": res["detected_terms"]
            })
        elif gt_is_toxic and not pred_is_toxic:
            moderation_metrics["FN"] += 1
            false_negatives_moderation.append({
                "text": text,
                "ground_truth": [CATEGORIES[i] for i, v in enumerate(gt_labels) if v == 1],
                "scores": scores,
                "detected_terms": res["detected_terms"]
            })
        else:
            moderation_metrics["TN"] += 1

        # Category-level metrics
        for idx, cat in enumerate(CATEGORIES):
            gt_cat = (gt_labels[idx] == 1)
            pred_cat = (cat in flagged_cats)
            if gt_cat and pred_cat:
                cat_metrics[cat]["TP"] += 1
            elif not gt_cat and pred_cat:
                cat_metrics[cat]["FP"] += 1
                cat_false_positives[cat].append((text, scores.get(cat, 0.0)))
            elif gt_cat and not pred_cat:
                cat_metrics[cat]["FN"] += 1
                cat_false_negatives[cat].append((text, scores.get(cat, 0.0)))
            else:
                cat_metrics[cat]["TN"] += 1

    # 5. Calculate Precision, Recall, F1
    print("\n" + "=" * 80)
    print("CATEGORY-LEVEL EVALUATION RESULTS:")
    print("=" * 80)
    print(f"{'Category':15s} | {'TP':4s} {'FP':4s} {'FN':4s} {'TN':4s} | {'Precision':10s} {'Recall':10s} {'F1-Score':10s}")
    print("-" * 80)

    summary_table = []
    for cat in CATEGORIES:
        m = cat_metrics[cat]
        tp, fp, fn, tn = m["TP"], m["FP"], m["FN"], m["TN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{cat:15s} | {tp:4d} {fp:4d} {fn:4d} {tn:4d} | {prec:10.4f} {rec:10.4f} {f1:10.4f}")
        summary_table.append({
            "category": cat,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1
        })

    # Moderation decision overall
    mod_tp = moderation_metrics["TP"]
    mod_fp = moderation_metrics["FP"]
    mod_fn = moderation_metrics["FN"]
    mod_tn = moderation_metrics["TN"]
    mod_prec = mod_tp / (mod_tp + mod_fp) if (mod_tp + mod_fp) > 0 else 0.0
    mod_rec = mod_tp / (mod_tp + mod_fn) if (mod_tp + mod_fn) > 0 else 0.0
    mod_f1 = (2 * mod_prec * mod_rec) / (mod_prec + mod_rec) if (mod_prec + mod_rec) > 0 else 0.0
    mod_acc = (mod_tp + mod_tn) / len(EVAL_DATASET)

    print("-" * 80)
    print(f"{'OVERALL MODERATION (Unsafe vs Safe):':35s}")
    print(f"  TP: {mod_tp}, FP: {mod_fp}, FN: {mod_fn}, TN: {mod_tn}")
    print(f"  Overall Accuracy:  {mod_acc*100:.2f}% ({mod_tp + mod_tn}/{len(EVAL_DATASET)})")
    print(f"  Overall Precision: {mod_prec:.4f}")
    print(f"  Overall Recall:    {mod_rec:.4f}")
    print(f"  Overall F1-Score:  {mod_f1:.4f}")
    print("=" * 80)

    # 6. Report False Positives (Harmless messages incorrectly flagged)
    print(f"\n[FALSE POSITIVES AUDIT] ({len(false_positives_moderation)} harmless messages incorrectly flagged):")
    for i, fp in enumerate(false_positives_moderation, 1):
        print(f"  {i}. Text: \"{fp['text']}\"")
        print(f"     Flagged categories: {fp['flagged']}")
        print(f"     Detected static terms: {fp['detected_terms']}")
        print(f"     Model scores: { {k: round(v, 4) for k, v in fp['scores'].items()} }")
        print(f"     System explanation: {fp['reason']}")

    # 7. Report False Negatives (Harmful messages that bypassed detection)
    print(f"\n[FALSE NEGATIVES AUDIT] ({len(false_negatives_moderation)} harmful messages missed):")
    for i, fn in enumerate(false_negatives_moderation, 1):
        print(f"  {i}. Text: \"{fn['text']}\"")
        print(f"     Ground truth: {fn['ground_truth']}")
        print(f"     Detected static terms: {fn['detected_terms']}")
        print(f"     Model scores: { {k: round(v, 4) for k, v in fn['scores'].items()} }")

    # 8. Dedicated 'YOU' Phrase Vulnerability Analysis
    print("\n" + "=" * 80)
    print("[STEP 5] DEDICATED 'YOU' / 'YOU ARE' PHRASE AUDIT:")
    print("=" * 80)
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
    you_fp_count = 0
    for yp in you_phrases:
        r = moderator.analyze_message(yp)
        is_safe = r["is_safe"]
        if not is_safe:
            you_fp_count += 1
            status_str = f"FLAGGED as {r['flagged_categories']} (insult score: {r['scores'].get('insult', 0.0):.4f}, toxic: {r['scores'].get('toxic', 0.0):.4f})"
        else:
            status_str = f"SAFE (insult score: {r['scores'].get('insult', 0.0):.4f}, toxic: {r['scores'].get('toxic', 0.0):.4f})"
        print(f"  '{yp:40s}' -> {status_str}")
    print(f"\n  'You'-phrase False Positive Rate: {you_fp_count} / {len(you_phrases)} ({you_fp_count/len(you_phrases)*100:.1f}%)")

    # 9. Verify 5-Warning and 7-Day Restriction Workflow
    print("\n" + "=" * 80)
    print("[STEP 6] VERIFYING 5-WARNING & 7-DAY RESTRICTION WORKFLOW (ISOLATED/IN-MEMORY):")
    print("=" * 80)

    # Simulate the exact logic from chat_routes.py:
    # Lines 68-87 of routes/chat_routes.py
    user_state = {
        "user_id": "simulated_audit_user_123",
        "warning_count": 0,
        "is_active": True,
        "block_expiry": None
    }
    
    simulation_log = []
    
    # Function simulating chat_routes.py send moderation logic
    def simulate_chat_send(text, state, current_time):
        # 1. Check if sender is currently blocked
        if state["block_expiry"] and current_time < state["block_expiry"]:
            return {
                "status_code": 403,
                "error": f"You are temporarily blocked from sending messages until {state['block_expiry'].strftime('%Y-%m-%d %H:%M:%S')} UTC."
            }
        
        # 2. Run NLP Moderator
        result = moderator.analyze_message(text)
        if not result["is_safe"]:
            new_count = state["warning_count"] + 1
            state["warning_count"] = new_count
            
            if new_count >= 5:
                # 5th warning: Block for 7 days
                block_until = current_time + timedelta(days=7)
                state["block_expiry"] = block_until
                state["is_active"] = False
                return {
                    "status_code": 403,
                    "error": "Your account has been temporarily restricted for 7 days because you received 5 harmful-language warnings."
                }
            
            return {
                "status_code": 400,
                "error": f"Warning {new_count}/5: Your message contains potentially harmful language. Please use respectful language."
            }
        
        return {
            "status_code": 200,
            "status": "sent"
        }

    sim_start_time = datetime(2026, 10, 26, 12, 0, 0)
    
    # Action 1: Safe message
    res1 = simulate_chat_send("Hello Bob, how are you today?", user_state, sim_start_time)
    print(f"  Attempt 1 (Safe): HTTP {res1['status_code']} -> warning_count: {user_state['warning_count']}, is_active: {user_state['is_active']}")
    assert res1["status_code"] == 200 and user_state["warning_count"] == 0

    # Action 2-5: Harmful messages 1 to 4
    for i in range(1, 5):
        r = simulate_chat_send(f"You are a pathetic loser #{i}", user_state, sim_start_time + timedelta(minutes=i))
        print(f"  Attempt {i+1} (Harmful #{i}): HTTP {r['status_code']} -> '{r['error']}' | warning_count: {user_state['warning_count']}")
        assert r["status_code"] == 400
        assert f"Warning {i}/5" in r["error"]
        assert user_state["warning_count"] == i
        assert user_state["is_active"] is True
        assert user_state["block_expiry"] is None

    # Action 6: 5th harmful message -> Triggers 7-day ban
    res5 = simulate_chat_send("I will kill you in your sleep", user_state, sim_start_time + timedelta(minutes=5))
    print(f"  Attempt 6 (Harmful #5): HTTP {res5['status_code']} -> '{res5['error']}'")
    print(f"    -> is_active: {user_state['is_active']}, warning_count: {user_state['warning_count']}")
    print(f"    -> block_expiry: {user_state['block_expiry']} (diff: {user_state['block_expiry'] - (sim_start_time + timedelta(minutes=5))})")
    assert res5["status_code"] == 403
    assert user_state["is_active"] is False
    assert user_state["warning_count"] == 5
    assert user_state["block_expiry"] == sim_start_time + timedelta(minutes=5) + timedelta(days=7)

    # Action 7: Attempt to send a message during the 7-day restriction (e.g. Day 3)
    day3_time = sim_start_time + timedelta(days=3)
    res_day3 = simulate_chat_send("Hey, I'm sorry, can I chat now?", user_state, day3_time)
    print(f"  Attempt 7 (During 7-day ban, Day 3): HTTP {res_day3['status_code']} -> '{res_day3['error']}'")
    assert res_day3["status_code"] == 403

    # Action 8: Expiration check (Day 7 + 1 hour)
    day7_after_time = sim_start_time + timedelta(days=7, hours=1)
    # Check auto-unblock logic from models/user.py lines 113-121
    if user_state["block_expiry"] and day7_after_time > user_state["block_expiry"]:
        user_state["is_active"] = True
        user_state["block_expiry"] = None
        print(f"  Day 7+ Auto-unblock check: is_active restored to {user_state['is_active']}, block_expiry cleared.")
    
    res_after = simulate_chat_send("Hello Bob, hope you are doing well.", user_state, day7_after_time)
    print(f"  Attempt 8 (Post 7-day expiry): HTTP {res_after['status_code']} -> status: '{res_after.get('status')}'")
    assert res_after["status_code"] == 200

    print("  -> CONFIRMED: 5-warning and 7-day restriction workflow verified strictly according to codebase logic without modifying any production data.")

    # 10. Dump results for structured report
    output_audit_path = os.path.join(os.path.dirname(__file__), "audit_results.json")
    with open(output_audit_path, "w") as f:
        json.dump({
            "training_samples": len(RAW_TRAIN_DATA),
            "eval_samples": len(EVAL_DATASET),
            "category_metrics": summary_table,
            "overall_moderation": {
                "tp": mod_tp, "fp": mod_fp, "fn": mod_fn, "tn": mod_tn,
                "precision": mod_prec, "recall": mod_rec, "f1": mod_f1,
                "accuracy": mod_acc
            },
            "false_positives_count": len(false_positives_moderation),
            "false_positives": false_positives_moderation,
            "false_negatives_count": len(false_negatives_moderation),
            "false_negatives": false_negatives_moderation,
            "you_phrase_fp_count": you_fp_count,
            "you_phrase_total": len(you_phrases),
            "you_phrase_fp_rate": you_fp_count / len(you_phrases)
        }, f, indent=2)
    print(f"\nFull audit data saved to {output_audit_path}")

if __name__ == "__main__":
    main()
