import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')
from ai_modules.nlp_safety import NLPModerator

mod = NLPModerator()

harmless_sentences = [
    "Bro, you're so dumb 😂",
    "I hate you… just kidding, love you ❤️",
    "You're getting on my nerves 😂",
    "I'll steal your fries if you don't share 🍟",
    "You're dead to me until you give me chocolate 🍫",
    "I'm going to kill you in Ludo 🎮",
    "Your cooking is a crime against humanity 🤣",
    "You're annoying, but you're my favourite person ❤️",
    "I'm dying from laughter 🤣",
    "You're such a drama queen 👑😂",
    "I would kill for a lemonade.",
    "My shoulder is killing me.",
    "This workout is killing me.",
    "Bro you are such a clown 😂",
]

threat_sentences = [
    "I will punch your face in bro 😂",
    "I will put you in the hospital bro 😂",
    "I will break your legs lol",
    "I will smash your face dude 😂",
    "I'm going to break your legs tonight.",
    "I'll find you and hurt you.",
    "I'm joking, but I'll actually hurt you tomorrow.",
]

ambiguous_sentences = [
    "I could slap you for eating my pizza 🍕😂",
    "You better watch your back.",
    "I hate you.",
    "Shut up, you idiot.",
]

print("=== 1. HARMLESS & FUNNY MESSAGES (Expected: ALLOW) ===")
for s in harmless_sentences:
    res = mod.analyze_message(s)
    print(f"[{res['action']}] safe={res['is_safe']} | scores={res.get('scores', {})} | raw_flagged={res.get('flagged_categories', [])} | class={res['decision'].get('context_classification')} | '{s}'")

print("\n=== 2. GENUINE THREATS (Expected: BLOCK_AND_WARN) ===")
for s in threat_sentences:
    res = mod.analyze_message(s)
    print(f"[{res['action']}] safe={res['is_safe']} | scores={res.get('scores', {})} | raw_flagged={res.get('flagged_categories', [])} | class={res['decision'].get('context_classification')} | '{s}'")

print("\n=== 3. AMBIGUOUS MESSAGES (Contextual Evaluation) ===")
for s in ambiguous_sentences:
    res = mod.analyze_message(s)
    print(f"[{res['action']}] safe={res['is_safe']} | scores={res.get('scores', {})} | raw_flagged={res.get('flagged_categories', [])} | class={res['decision'].get('context_classification')} | '{s}'")
