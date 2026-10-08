import re


class NLPModerator:
    # Baseline moderation for an academic MVP.
    # Terms are grouped by harmful-language category.

    TERMS = {
        # Violence / killing / serious threats
        "kill",
        "killing",
        "murder",
        "murderer",
        "assassinate",
        "execute",
        "shoot",
        "stab",
        "strangle",
        "destroy",

        # Threats / intimidation
        "threat",
        "threaten",
        "terrorize",
        "blackmail",

        # Bullying / harassment
        "bully",
        "bullying",
        "harass",
        "harassment",
        "humiliate",
        "humiliation",

        # Strong offensive / abusive language
        "fuck",
        "fucking",
        "fucker",
        "motherfucker",
        "bitch",
        "bastard",
        "asshole",
        "dumbass",
        "dickhead",
        "dick",
        "prick",
        "shithead",
        "bullshit",
        "shit",
        "whore",
        "slut",
        "scumbag",
        "jackass",
        "dipshit",
        "sonofabitch",
        "twat",
        "wanker",
        "moron",
        "imbecile",
        "retard",

        # General insulting / abusive language
        "hate",
        "idiot",
        "stupid",
        "abuse",
        "abusive",
        "fool",
        "loser",
        "pathetic",

        # Sexual / explicit terminology
        "penis",
        "vagina",
        "vulva",
        "breast",
        "breasts",
        "boobs",
        "butt",
        "buttocks",
        "anus",
        "genitals",
        "genital",
        "nipple",
        "nipples",
        "testicle",
        "testicles",
        "scrotum",
        "erection",
        "semen",
        "porn",
        "pornography",
        "nude",
        "nudity",
        "sexual",
        "sexually",

        # Discrimination-related concepts
        "racist",
        "racism",
        "discrimination",
        "discriminatory",
        "bigot",
        "bigotry",
        "xenophobic",
        "xenophobia",
        "homophobic",
        "homophobia",
        "transphobic",
        "transphobia",
        "ableist",
        "ableism"
    }

    def analyze_message(self, text):
        tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
        found = tokens & self.TERMS

        if found:
            return {
                "is_safe": False,
                "reason": "Potentially abusive, harmful, explicit, or discriminatory language detected.",
                "detected_terms": list(found)
            }

        return {
            "is_safe": True,
            "reason": None,
            "detected_terms": []
        }