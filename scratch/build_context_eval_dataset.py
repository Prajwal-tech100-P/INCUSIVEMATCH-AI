"""
Build and validate the independent Context Evaluation Dataset for InclusiveMatch AI.
Ensures zero leakage against train.json, val.json, and test.json.
Covers:
1. Friendly teasing & jokes (with/without emojis)
2. Idiomatic expressions / colloquial hyperbole ("kill", "hate", "dead", "shoot")
3. Ambiguous messages (requiring cautious non-punitive moderation)
4. Clear direct threats
5. Threats without profanity (stalking, intimidation)
6. Direct insults
7. Targeted harassment
8. Hate speech
9. Masked threats containing emojis (proving emojis cannot bypass moderation)
10. Benign discussions of news, crime, safety, or academic topics
"""
import os
import sys
import json
import re

PROJECT_ROOT = r"c:\Users\ACER\Desktop\InclusiveMatch_AI_Secure_WebRTC_HTTPS_Ready 8"
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "toxicity")

# Define comprehensive, human-reviewed benchmark samples
CONTEXT_EVAL_SAMPLES = [
    # =========================================================================
    # 1. FRIENDLY TEASING & JOKES (SAFE / ALLOW)
    # =========================================================================
    {
        "id": "joke_01",
        "text": "Bro, I will kill you for eating my pizza 😂",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": "kill"
    },
    {
        "id": "joke_02",
        "text": "I'll kill you 😂 you're too funny.",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": "kill"
    },
    {
        "id": "joke_03",
        "text": "I hate you 😂❤️",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": "hate"
    },
    {
        "id": "joke_04",
        "text": "You're so annoying, bestie 😂",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": None
    },
    {
        "id": "joke_05",
        "text": "I will literally kill you if you spoil the movie ending haha 🤣",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": "kill"
    },
    {
        "id": "joke_06",
        "text": "Stop making me laugh so hard, you are trying to kill me with laughter 😆",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": "kill"
    },
    {
        "id": "joke_07",
        "text": "You stole my last piece of garlic bread, you are dead to me lol",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "dead"
    },
    {
        "id": "joke_08",
        "text": "Oh man I hate you for making me stay up this late playing video games haha",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "hate"
    },
    {
        "id": "joke_09",
        "text": "You're such a nerd, but that's why we get along so well!",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "joke_10",
        "text": "You beat my high score again? I'm gonna fight you for that 😜",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": "fight"
    },

    # =========================================================================
    # 2. IDIOMATIC EXPRESSIONS & HYPERBOLE (SAFE / ALLOW)
    # =========================================================================
    {
        "id": "idiom_01",
        "text": "This workout is killing me.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "killing"
    },
    {
        "id": "idiom_02",
        "text": "I would kill for a cup of coffee right now.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "kill"
    },
    {
        "id": "idiom_03",
        "text": "My feet are killing me after walking across the campus all day.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "killing"
    },
    {
        "id": "idiom_04",
        "text": "Shoot me a text when you arrive home safely tonight.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "shoot"
    },
    {
        "id": "idiom_05",
        "text": "I hate traffic on Mondays, it takes forever to commute.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "hate"
    },
    {
        "id": "idiom_06",
        "text": "We need to kill some time before the movie begins at eight.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "kill"
    },
    {
        "id": "idiom_07",
        "text": "You are dressed to kill in that formal outfit!",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "kill"
    },
    {
        "id": "idiom_08",
        "text": "I'm dying to try that new Italian restaurant downtown.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "dying"
    },
    {
        "id": "idiom_09",
        "text": "That stand-up comedian had the entire room dead with laughter.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "dead"
    },
    {
        "id": "idiom_10",
        "text": "Let's tackle this project deadline together tomorrow morning.",
        "category_type": "idiom_hyperbole",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": None
    },

    # =========================================================================
    # 3. AMBIGUOUS MESSAGES (FLAG FOR REVIEW / CAUTIOUS NON-PUNITIVE)
    # =========================================================================
    {
        "id": "ambig_01",
        "text": "You're going to regret this.",
        "category_type": "ambiguous_phrase",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "flag_for_review",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "ambig_02",
        "text": "I hate you.",
        "category_type": "ambiguous_phrase",
        "labels": [1, 0, 0, 0, 0, 0],
        "expected_action": "flag_for_review",
        "has_emoji": False,
        "trigger_word": "hate"
    },
    {
        "id": "ambig_03",
        "text": "You better watch out.",
        "category_type": "ambiguous_phrase",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "flag_for_review",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "ambig_04",
        "text": "I'll find you.",
        "category_type": "ambiguous_phrase",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "flag_for_review",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "ambig_05",
        "text": "Don't test my patience today, I've had enough.",
        "category_type": "ambiguous_phrase",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "flag_for_review",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "ambig_06",
        "text": "Just wait until I see you in person next week.",
        "category_type": "ambiguous_phrase",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "flag_for_review",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "ambig_07",
        "text": "You made a big mistake turning me down.",
        "category_type": "ambiguous_phrase",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "flag_for_review",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "ambig_08",
        "text": "You think you can just ignore me without consequences?",
        "category_type": "ambiguous_phrase",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "flag_for_review",
        "has_emoji": False,
        "trigger_word": None
    },

    # =========================================================================
    # 4. CLEAR DIRECT THREATS (HARMFUL / BLOCK & WARN)
    # =========================================================================
    {
        "id": "threat_01",
        "text": "I will hurt you tonight.",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "hurt"
    },
    {
        "id": "threat_02",
        "text": "I will attack you if you leave me.",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "attack"
    },
    {
        "id": "threat_03",
        "text": "I am going to come to your house and hurt you.",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "hurt"
    },
    {
        "id": "threat_04",
        "text": "I will find where you work and slit your throat.",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "slit"
    },
    {
        "id": "threat_05",
        "text": "I'm going to smash your head into the pavement next time I see you.",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "smash"
    },
    {
        "id": "threat_06",
        "text": "I'll shoot you dead if you ever talk to another guy.",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "shoot"
    },
    {
        "id": "threat_07",
        "text": "I will burn your car down with you inside.",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "burn"
    },
    {
        "id": "threat_08",
        "text": "I will beat you until you are in the emergency room.",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "beat"
    },

    # =========================================================================
    # 5. THREATS WITHOUT PROFANITY (STALKING / INTIMIDATION) (HARMFUL / BLOCK & WARN)
    # =========================================================================
    {
        "id": "threat_noprof_01",
        "text": "I know what train you take every evening and which car you sit in.",
        "category_type": "threat_no_profanity",
        "labels": [1, 0, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "threat_noprof_02",
        "text": "Do what I say or something terrible will happen to your family.",
        "category_type": "threat_no_profanity",
        "labels": [1, 1, 0, 1, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "threat_noprof_03",
        "text": "I am standing outside your apartment building right now watching your window.",
        "category_type": "threat_no_profanity",
        "labels": [1, 1, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "threat_noprof_04",
        "text": "You will never feel safe walking down your street again.",
        "category_type": "threat_no_profanity",
        "labels": [1, 0, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "threat_noprof_05",
        "text": "I took photos of your front door and I have your daily schedule.",
        "category_type": "threat_no_profanity",
        "labels": [1, 0, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },

    # =========================================================================
    # 6. MASKED THREATS / ABUSE WITH EMOJIS (MUST NOT BE BYPASSED) (HARMFUL / BLOCK & WARN)
    # =========================================================================
    {
        "id": "masked_01",
        "text": "I will stab you in your sleep 😂",
        "category_type": "masked_threat_emoji",
        "labels": [1, 1, 0, 1, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": True,
        "trigger_word": "stab"
    },
    {
        "id": "masked_02",
        "text": "I'm going to find where you live and break your neck 🤣",
        "category_type": "masked_threat_emoji",
        "labels": [1, 1, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": True,
        "trigger_word": "break"
    },
    {
        "id": "masked_03",
        "text": "You are a disgusting ugly pig and nobody wants you 😂",
        "category_type": "masked_threat_emoji",
        "labels": [1, 0, 1, 0, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": True,
        "trigger_word": "pig"
    },
    {
        "id": "masked_04",
        "text": "I will make your life a living hell 😈",
        "category_type": "masked_threat_emoji",
        "labels": [1, 0, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": True,
        "trigger_word": "hell"
    },
    {
        "id": "masked_05",
        "text": "I'll strangle you till you pass out 😅",
        "category_type": "masked_threat_emoji",
        "labels": [1, 1, 0, 1, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": True,
        "trigger_word": "strangle"
    },

    # =========================================================================
    # 7. DIRECT INSULTS & ABUSIVE LANGUAGE (HARMFUL / BLOCK & WARN)
    # =========================================================================
    {
        "id": "insult_01",
        "text": "You are a disgusting ugly pig and nobody wants you",
        "category_type": "direct_insult",
        "labels": [1, 0, 1, 0, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "pig"
    },
    {
        "id": "insult_02",
        "text": "Shut your lying mouth you filthy jerk, you make me sick.",
        "category_type": "direct_insult",
        "labels": [1, 0, 1, 0, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "jerk"
    },
    {
        "id": "insult_03",
        "text": "You are completely brainless, pathetic, and worthless.",
        "category_type": "direct_insult",
        "labels": [1, 0, 1, 0, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "pathetic"
    },
    {
        "id": "insult_04",
        "text": "Get lost you repulsive scumbag, nobody here respects you.",
        "category_type": "direct_insult",
        "labels": [1, 0, 1, 0, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "scumbag"
    },
    {
        "id": "insult_05",
        "text": "You're a miserable failure at life, just give up already.",
        "category_type": "direct_insult",
        "labels": [1, 0, 1, 0, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "failure"
    },

    # =========================================================================
    # 8. REPEATED TARGETED HARASSMENT (HARMFUL / BLOCK & WARN)
    # =========================================================================
    {
        "id": "harass_01",
        "text": "I created five new accounts just to keep messaging you after you blocked me.",
        "category_type": "repeated_harassment",
        "labels": [1, 0, 0, 0, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "harass_02",
        "text": "Answer me right now. Why are you ignoring me? Pick up the phone now!",
        "category_type": "repeated_harassment",
        "labels": [1, 0, 0, 0, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "harass_03",
        "text": "I'm going to message all your friends and family until you reply to me.",
        "category_type": "repeated_harassment",
        "labels": [1, 0, 0, 0, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "harass_04",
        "text": "Send me explicit photos right now or I will post your private info everywhere.",
        "category_type": "repeated_harassment",
        "labels": [1, 1, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },

    # =========================================================================
    # 9. HATE SPEECH & IDENTITY-BASED DISCRIMINATION (HARMFUL / BLOCK & WARN)
    # =========================================================================
    {
        "id": "hate_01",
        "text": "People with intellectual disabilities should never be allowed in our community.",
        "category_type": "hate_speech",
        "labels": [1, 0, 1, 0, 0, 1],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "hate_02",
        "text": "All members of that religious group are violent terrorists.",
        "category_type": "hate_speech",
        "labels": [1, 0, 1, 0, 0, 1],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "hate_03",
        "text": "You are a subhuman freak because of your sexual orientation.",
        "category_type": "hate_speech",
        "labels": [1, 0, 1, 0, 0, 1],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "hate_04",
        "text": "Deaf people cannot understand real communication and are inferior.",
        "category_type": "hate_speech",
        "labels": [1, 0, 1, 0, 0, 1],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": None
    },

    # =========================================================================
    # 10. BENIGN DISCUSSIONS OF VIOLENCE, SAFETY, OR NEWS (SAFE / ALLOW)
    # =========================================================================
    {
        "id": "benign_01",
        "text": "The news reported that there was a shooting downtown last night, please stay safe!",
        "category_type": "benign_discussion",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "shooting"
    },
    {
        "id": "benign_02",
        "text": "In our history class we studied how medieval kings executed traitors in castles.",
        "category_type": "benign_discussion",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "executed"
    },
    {
        "id": "benign_03",
        "text": "Domestic violence is a serious crime and emergency helplines are available 24/7.",
        "category_type": "benign_discussion",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "violence"
    },
    {
        "id": "benign_04",
        "text": "I really enjoy watching crime investigation documentaries and mystery thrillers on weekends.",
        "category_type": "benign_discussion",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "crime"
    },
    {
        "id": "benign_05",
        "text": "Self-defense classes taught me how to escape if someone grabs your arm.",
        "category_type": "benign_discussion",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": None
    },

    # =========================================================================
    # 11. SLANG & SPELLING VARIATIONS (FRIENDLY BANTER VS REAL ATTACKS)
    # =========================================================================
    {
        "id": "slang_safe_01",
        "text": "bruh ima kill u if u drop the pizza box 💀",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": "kill"
    },
    {
        "id": "slang_safe_02",
        "text": "im deadddd u are hilarious 😭💀",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": "dead"
    },
    {
        "id": "slang_safe_03",
        "text": "ur totally roasting me rn haha",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "slang_safe_04",
        "text": "gonna beat ur score in Mario Kart tonight, watch out!",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": "beat"
    },
    {
        "id": "slang_toxic_01",
        "text": "u stupid ugly bish kys",
        "category_type": "clear_threat",
        "labels": [1, 1, 1, 0, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "kys"
    },
    {
        "id": "slang_toxic_02",
        "text": "ima pull up to ur crib and smash ur face in",
        "category_type": "clear_threat",
        "labels": [1, 1, 0, 1, 1, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "smash"
    },
    {
        "id": "slang_toxic_03",
        "text": "ur a worthless piece of trash, go die",
        "category_type": "clear_threat",
        "labels": [1, 1, 1, 0, 0, 0],
        "expected_action": "block_and_warn",
        "has_emoji": False,
        "trigger_word": "die"
    },

    # =========================================================================
    # 12. SARCASM & MILD SNARK (NON-THREATENING BANTER)
    # =========================================================================
    {
        "id": "sarcasm_01",
        "text": "Oh wow, another genius remark. You really outdid yourself today.",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": None
    },
    {
        "id": "sarcasm_02",
        "text": "Yeah right, because you are always so punctual 🙄",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": True,
        "trigger_word": None
    },
    {
        "id": "sarcasm_03",
        "text": "Sure, blame the GPS for driving in the wrong direction for 20 miles haha.",
        "category_type": "friendly_joke",
        "labels": [0, 0, 0, 0, 0, 0],
        "expected_action": "allow",
        "has_emoji": False,
        "trigger_word": None
    }
]

def check_leakage(new_samples, split_name):
    filepath = os.path.join(DATA_DIR, f"{split_name}.json")
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} does not exist.")
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_texts = set(x["text"].strip().lower() for x in existing)
    
    leaks = 0
    for s in new_samples:
        t = s["text"].strip().lower()
        if t in existing_texts:
            print(f"LEAKAGE DETECTED in {split_name}: '{s['text']}'")
            leaks += 1
    return leaks

def main():
    print("Checking dataset leakage across train/val/test splits...")
    total_leaks = 0
    for split in ["train", "val", "test"]:
        leaks = check_leakage(CONTEXT_EVAL_SAMPLES, split)
        print(f"  Leakage against {split}.json: {leaks} duplicate matches")
        total_leaks += leaks
        
    assert total_leaks == 0, f"Critical: Found {total_leaks} data leakage matches across splits!"
    print("Zero leakage verified! Saving context evaluation dataset...")

    out_path = os.path.join(DATA_DIR, "context_eval_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(CONTEXT_EVAL_SAMPLES, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(CONTEXT_EVAL_SAMPLES)} context evaluation samples to {out_path}")

if __name__ == "__main__":
    main()
