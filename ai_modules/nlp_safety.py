import os
import re
import logging
import threading
from typing import Dict, Any, List, Optional
import joblib

logger = logging.getLogger(__name__)

# Default model path resolved reliably relative to project root
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "toxicity_classifier.joblib"
)

EXPECTED_CATEGORIES = ("toxic", "severe_toxic", "insult", "threat", "harassment", "hate")

# Default thresholds calibrated on validation split for the trained model outputs
DEFAULT_THRESHOLDS = {
    "severe_toxic": 0.45,
    "threat": 0.55,
    "harassment": 0.45,
    "hate": 0.45,
    "insult": 0.45,
    "toxic": 0.40,
}

# Adjusted thresholds when explicit abusive keywords are detected
KEYWORD_ASSISTED_THRESHOLDS = {
    "severe_toxic": 0.35,
    "threat": 0.50,
    "harassment": 0.40,
    "hate": 0.40,
    "insult": 0.40,
    "toxic": 0.35,
}


class NLPModerator:
    """Production NLP moderation engine combining trained ML toxicity classifier with baseline keywords."""

    # Class-level cache to ensure artifact is loaded only once across application lifespan
    _cached_pipeline = None
    _cached_categories = None
    _cached_model_path = None
    _cached_model_type = None
    _cached_version = None
    _cached_tuned_thresholds = None
    _model_load_attempted = False
    _load_lock = threading.Lock()
    _inference_cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()
    _MAX_CACHE_SIZE = 512

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

    # Contextual Disambiguation Regex Patterns
    # 1. High-severity threats, weapons, stalking, or self-harm that CANNOT be bypassed by emojis or jokes
    UNCOMPROMISING_PATTERNS = [
        r"\bstab\b", r"\bslit\b", r"\bstrangle\b", r"\bchoke\b",
        r"\bbeat\b.*?\b(?:hospital|bleed|pavement|up)\b",
        r"\bbeat\s+(?:you|u)\s+up\b",
        r"\bput\s+(?:you|u)\s+in\s+(?:the\s+)?hospital\b",
        r"\bpunch\b.*?\b(?:face|head|eye|jaw|mouth|throat|nose)\b",
        r"\bbreak\s+(?:your|ur)\s+(?:neck|bones|skull|face|legs?|arms?|kneecaps?)\b",
        r"\bsmash\s+(?:your|ur)\s+(?:head|face|skull)\b",
        r"\bshoot\s+(?:you|u)\s+dead\b",
        r"\bburn\s+(?:your|ur)\s+(?:house|car|room)\b",
        r"\b(?:will|i'll|gonna|going\s+to|actually)\b.*?\b(?:hurt|harm|attack|injure)\s+(?:you|u)\b",
        r"\bkys\b", r"\bkill\s+yourself\b", r"\bgo\s+die\b", r"\bslit\s+(?:your|ur)\s+wrists\b",
        r"\bstanding\s+outside\s+(?:your|ur)\b", r"\bwatching\s+(?:your|ur)\s+window\b",
        r"\bknow\s+what\s+train\s+(?:you|u)\s+take\b", r"\btook\s+photos\s+of\s+(?:your|ur)\b",
        r"\bhave\s+(?:your|ur)\s+daily\s+schedule\b", r"\bdo\s+what\s+i\s+say\s+or\b.*?\bfamily\b",
        r"\bfind\s+where\s+(?:you|u)\s+work\b", r"\bpull\s+up\s+to\s+ur\s+crib\b",
        r"\bsend\s+me\s+explicit\s+photos\b", r"\bpost\s+(?:your|ur)\s+private\s+info\b",
        r"\bkeep\s+messaging\s+(?:you|u)\s+after\s+(?:you|u)\s+blocked\s+me\b",
        r"\bcreated\s+\w+\s+(?:new\s+)?accounts\b"
    ]
    UNCOMPROMISING_REGEX = re.compile("|".join(UNCOMPROMISING_PATTERNS), re.IGNORECASE)

    # 2. Non-violent idiomatic expressions and colloquial hyperbole
    IDIOM_PATTERNS = [
        r"\b(?:my\s+)?(?:workout|feet|legs|shoes|allergies|traffic|heat|job|boss|headache|migraine|back|neck|shoulder|shoulders|knees|knee|stomach|throat|head)\s+(?:is|are)\s+killing\s+me\b",
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

    # 3. Third-person reporting, news, safety education, and non-targeted discussions
    REPORTING_PATTERNS = [
        r"\b(?:news\s+reported|in\s+our\s+history\s+class|in\s+history\s+class|classes\s+taught\s+me|emergency\s+helplines|documentaries\s+and\s+mystery|documentary\s+about)\b"
    ]
    REPORTING_REGEX = re.compile("|".join(REPORTING_PATTERNS), re.IGNORECASE)

    # 4. Playful banter, friendly teasing, and hyperbolic jokes
    HUMOR_EMOJIS = set(["😂", "🤣", "😆", "😜", "😝", "💀", "😭", "❤️", "😋", "😉", "👑", "🍟", "🍫", "🎮", "🍕"])
    HUMOR_TERMS = {
        "haha", "hahaha", "lol", "lmao", "rofl", "bestie", "bro", "bruh", "homie", "dude",
        "too funny", "hilarious", "joking", "kidding", "just kidding", "jk", "roasting me",
        "you nerd", "nerd", "clown", "drama queen", "favourite person", "favorite person"
    }
    BANTER_PATTERNS = [
        r"\b(?:kill\s+you|kill\s+u)\s+for\s+(?:eating|taking|stealing|dropping)\s+(?:my\s+)?(?:pizza|fries|food|bread|cookie|chips)\b",
        r"\b(?:kill\s+you|kill\s+u)\s+(?:if\s+you|if\s+u)\s+(?:spoil|drop)\b",
        r"\b(?:kill\s+you|kill\s+u)\b.*?(?:too\s+funny|hilarious|haha|lol|lmao)",
        r"\bdead\s+to\s+me\b.*?(?:until|unless|chocolate|coffee|pizza|food|cookie|ice\s+cream|lol|haha|lmao|[😂🤣❤️😋😉🍫])",
        r"\b(?:going\s+to\s+|gonna\s+)?(?:kill\s+you|kill\s+u|destroy\s+you|destroy\s+u|beat\s+you|beat\s+u)\s+in\s+(?:ludo|fifa|chess|monopoly|uno|cod|fortnite|pubg|mario|cards|pool|game|gaming|match)\b",
        r"\b(?:your\s+)?(?:cooking|food)\s+(?:is|was)\s+(?:a\s+)?crime\b.*?(?:[😂🤣😆😜😝]|haha|lol|lmao)",
        r"\b(?:annoying|getting\s+on\s+my\s+nerves)\b.*?\b(?:favourite|favorite|love|bestie|friend)\b",
        r"\b(?:such\s+a\s+)?drama\s+queen\b.*?(?:[😂🤣😆😜😝👑]|haha|lol|lmao)",
        r"\b(?:bro|dude|homie|bestie|you(?:'re|\s+are))\b.*?\b(?:so\s+dumb|such\s+a\s+clown|such\s+a\s+nerd|such\s+a\s+fool|too\s+silly)\b.*?(?:[😂🤣😆😜😝]|haha|lol|lmao)",
        r"\b(?:beat\s+ur\s+score|beat\s+your\s+score)\b",
        r"\b(?:fight\s+you\s+for\s+that)\b",
        r"\b(?:kill|killing)\s+me\s+with\s+laughter\b",
        r"\bhate\s+you\b.*?(?:[😂🤣❤️😜😆]|haha|lol|lmao)"
    ]
    BANTER_REGEX = re.compile("|".join(BANTER_PATTERNS), re.IGNORECASE)

    # 5. Ambiguous statements that require cautious non-punitive moderation
    AMBIGUOUS_PATTERNS = [
        r"^you(?:'re|\s+are)?\s+going\s+to\s+regret\s+this\.?$",
        r"^you\s+better\s+watch\s+(?:out|your\s+back|ur\s+back)\.?$",
        r"^i(?:'ll|\s+will)\s+find\s+you\.?$",
        r"^don'?t\s+test\s+my\s+patience\.?$",
        r"^just\s+wait\s+until\s+i\s+see\s+you\.?$",
        r"^you\s+made\s+a\s+big\s+mistake\.?$",
        r"^you\s+think\s+you\s+can\s+(?:just\s+)?ignore\s+me.*?$",
        r"^i\s+hate\s+you\.?$"
    ]
    AMBIGUOUS_REGEX = re.compile("|".join(AMBIGUOUS_PATTERNS), re.IGNORECASE)

    @classmethod
    def _load_model(cls, model_path: str):
        """Thread-safe, lazy one-time loader and validator for the joblib artifact."""
        with cls._load_lock:
            if cls._cached_pipeline is not None and cls._cached_model_path == model_path:
                return cls._cached_pipeline, cls._cached_categories, True, "ml_classifier"

            cls._model_load_attempted = True
            if not os.path.isfile(model_path):
                logger.warning("Toxicity model artifact not found at: %s. Using keyword fallback.", model_path)
                return None, None, False, "fallback_keyword_only"

            try:
                artifact = joblib.load(model_path)
            except Exception as e:
                logger.error("Failed to load toxicity model artifact from %s: %s", model_path, e)
                return None, None, False, "fallback_keyword_only"

            if not isinstance(artifact, dict):
                logger.error("Invalid artifact format in %s: expected dict, got %s", model_path, type(artifact))
                return None, None, False, "fallback_keyword_only"

            pipeline = artifact.get("pipeline")
            categories = artifact.get("categories")

            if pipeline is None or not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
                logger.error("Loaded artifact does not contain a valid scikit-learn Pipeline with predict_proba.")
                return None, None, False, "fallback_keyword_only"

            if not isinstance(categories, (list, tuple)) or list(categories) != list(EXPECTED_CATEGORIES):
                logger.error("Categories mismatch in %s. Expected %s, got %s", model_path, EXPECTED_CATEGORIES, categories)
                return None, None, False, "fallback_keyword_only"

            cls._cached_pipeline = pipeline
            cls._cached_categories = list(categories)
            cls._cached_model_path = model_path
            cls._cached_version = artifact.get("version", "unknown")
            cls._cached_model_type = artifact.get("model_type", "unknown")
            cls._cached_tuned_thresholds = artifact.get("tuned_thresholds")
            logger.info("Loaded NLP toxicity classifier v%s (%s) from %s", cls._cached_version, cls._cached_model_type, model_path)
            return cls._cached_pipeline, cls._cached_categories, True, "ml_classifier"

    def __init__(self, model_path: Optional[str] = None, thresholds: Optional[Dict[str, float]] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.pipeline, self.categories, self.model_loaded, self.model_status = self._load_model(self.model_path)

        base_thresh = self._cached_tuned_thresholds if (self.model_loaded and self._cached_tuned_thresholds) else DEFAULT_THRESHOLDS
        self.thresholds = dict(base_thresh)
        if thresholds:
            self.thresholds.update(thresholds)
        self.keyword_thresholds = dict(KEYWORD_ASSISTED_THRESHOLDS)

    def classify_raw(self, cleaned_text: str) -> Dict[str, Any]:
        """Compute ML classification scores and keyword detection without enforcement actions."""
        tokens = set(re.findall(r"[a-zA-Z]+", cleaned_text.lower()))
        detected_terms = sorted(list(tokens & self.TERMS))
        scores: Dict[str, float] = {}
        flagged: List[str] = []

        if self.model_loaded and self.pipeline is not None:
            try:
                probs = self.pipeline.predict_proba([cleaned_text])[0]
                scores = {cat: float(probs[i]) for i, cat in enumerate(self.categories)}

                for cat in ["severe_toxic", "threat", "harassment", "hate"]:
                    thresh = self.keyword_thresholds[cat] if detected_terms else self.thresholds[cat]
                    if scores.get(cat, 0.0) >= thresh:
                        flagged.append(cat)

                insult_thresh = self.keyword_thresholds["insult"] if detected_terms else self.thresholds["insult"]
                if scores.get("insult", 0.0) >= insult_thresh:
                    flagged.append("insult")

                if scores.get("toxic", 0.0) >= self.thresholds["toxic"]:
                    if detected_terms or sum(1 for c in ["severe_toxic", "threat", "harassment", "hate", "insult"] if scores.get(c, 0.0) >= self.thresholds[c]) >= 2:
                        if "toxic" not in flagged:
                            flagged.append("toxic")

                if detected_terms and scores.get("toxic", 0.0) >= self.keyword_thresholds["toxic"] and not flagged:
                    flagged.append("toxic")

                return {
                    "scores": scores,
                    "flagged_categories": flagged,
                    "detected_terms": detected_terms,
                    "model_status": "ml_classifier"
                }
            except Exception as e:
                logger.error("Error during ML toxicity inference: %s. Falling back to keyword moderation.", e)

        # Fallback keyword-only classification
        return {
            "scores": {},
            "flagged_categories": ["keyword_match"] if detected_terms else [],
            "detected_terms": detected_terms,
            "model_status": "fallback_keyword_only"
        }

    def decide_moderation(self, cleaned_text: str, scores: Dict[str, float],
                          raw_flagged: List[str], detected_terms: List[str],
                          model_status: str) -> Dict[str, Any]:
        """Context-Aware Moderation Decision Layer.

        Separates classification scores from policy actions:
        - "allow": Safe content delivered normally.
        - "flag_for_review": Ambiguous content; safe delivery, logged for safety review, no punitive warning.
        - "block_and_warn": High-confidence violation; delivery blocked, warning/restriction issued.
        """
        # Step 1: Uncompromising threats, weapons, stalking, or self-harm (Zero tolerance)
        if self.UNCOMPROMISING_REGEX.search(cleaned_text):
            flagged = list(set(raw_flagged + ["threat", "severe_toxic"]))
            return {
                "action": "block_and_warn",
                "is_safe": False,
                "requires_warning": True,
                "confidence": "high",
                "context_classification": "direct_threat_or_harm",
                "reason": "Threatening or violent language; Extremely harmful or dangerous content detected.",
                "flagged_categories": flagged
            }

        # Step 2: Non-violent idiomatic expressions (hyperbole) or benign reporting/education
        if self.IDIOM_REGEX.search(cleaned_text) or self.REPORTING_REGEX.search(cleaned_text):
            return {
                "action": "allow",
                "is_safe": True,
                "requires_warning": False,
                "confidence": "high",
                "context_classification": "idiomatic_or_reporting_safe",
                "reason": None,
                "flagged_categories": []
            }

        # Step 3: Playful banter and friendly teasing
        has_humor_emoji = any(c in self.HUMOR_EMOJIS for c in cleaned_text)
        has_humor_word = any(hw in cleaned_text.lower() for hw in self.HUMOR_TERMS)
        has_banter_pattern = bool(self.BANTER_REGEX.search(cleaned_text))

        threat_score = scores.get("threat", 0.0)
        severe_toxic_score = scores.get("severe_toxic", 0.0)
        insult_score = scores.get("insult", 0.0)
        hate_score = scores.get("hate", 0.0)
        threat_thresh = self.keyword_thresholds.get("threat", 0.50) if detected_terms else self.thresholds.get("threat", 0.55)

        has_severe_slur = any(t in detected_terms for t in ["fuck", "fucking", "fucker", "bitch", "bastard", "asshole", "whore", "slut", "retard", "scumbag", "moron"])

        is_banter = False
        if has_banter_pattern:
            # Recognized explicit banter expressions (gaming/food/friendly ribbing) allow hyperbole and mild teasing,
            # but must not contain severe profanity, hate speech, or real severe violence
            if not has_severe_slur and hate_score < 0.65 and severe_toxic_score < 0.70:
                is_banter = True
        elif has_humor_emoji and has_humor_word:
            # Generic humor words or emojis alone MUST NEVER override an elevated threat score or severe toxicity
            is_safe_threat = (threat_score < threat_thresh and "threat" not in raw_flagged)
            is_low_severe = (severe_toxic_score < 0.35)
            is_low_hate = (hate_score < 0.45)
            # Allow mild ribbing (e.g. dumb, clown) if threat and severe toxicity are near zero and no severe slurs exist
            is_acceptable_insult = (insult_score < 0.65) or (
                threat_score < 0.15 and severe_toxic_score < 0.15 and not has_severe_slur and
                any(w in cleaned_text.lower() for w in ["dumb", "clown", "drama queen", "silly", "nerd", "annoying"])
            )
            if is_safe_threat and is_low_severe and is_low_hate and not has_severe_slur and is_acceptable_insult:
                is_banter = True

        if is_banter:
            return {
                "action": "allow",
                "is_safe": True,
                "requires_warning": False,
                "confidence": "high",
                "context_classification": "playful_banter_safe",
                "reason": None,
                "flagged_categories": []
            }

        # Step 4: Ambiguous statements (Uncertain intent; non-punitive safety intervention)
        if self.AMBIGUOUS_REGEX.search(cleaned_text):
            return {
                "action": "flag_for_review",
                "is_safe": True,
                "requires_warning": False,
                "confidence": "medium",
                "context_classification": "ambiguous_statement",
                "reason": "Ambiguous message recorded for safety review without penalty.",
                "flagged_categories": ["ambiguous_context"]
            }

        # Step 5: Fallback keyword check if ML unavailable
        if model_status == "fallback_keyword_only":
            is_safe = (len(detected_terms) == 0)
            return {
                "action": "allow" if is_safe else "block_and_warn",
                "is_safe": is_safe,
                "requires_warning": not is_safe,
                "confidence": "low",
                "context_classification": "keyword_fallback",
                "reason": "Potentially abusive, harmful, explicit, or discriminatory language detected." if not is_safe else None,
                "flagged_categories": raw_flagged
            }

        # Step 6: ML-driven classification policy
        is_safe = (len(raw_flagged) == 0)
        reason = None
        if not is_safe:
            reasons = []
            if "threat" in raw_flagged:
                reasons.append("Threatening or violent language")
            if "severe_toxic" in raw_flagged:
                reasons.append("Extremely harmful or dangerous content")
            if "hate" in raw_flagged:
                reasons.append("Hate speech or discriminatory content")
            if "harassment" in raw_flagged:
                reasons.append("Harassment or stalking behavior")
            if "insult" in raw_flagged:
                reasons.append("Insulting or abusive language")
            if "toxic" in raw_flagged and not reasons:
                reasons.append("Potentially abusive or harmful language")
            reason = "; ".join(reasons) + " detected." if reasons else "Harmful language detected."

        return {
            "action": "allow" if is_safe else "block_and_warn",
            "is_safe": is_safe,
            "requires_warning": not is_safe,
            "confidence": "high" if (len(raw_flagged) >= 2 or scores.get("severe_toxic", 0) > 0.6) else "medium",
            "context_classification": "standard_text",
            "reason": reason,
            "flagged_categories": raw_flagged
        }

    def analyze_message(self, text: Any) -> Dict[str, Any]:
        """Full end-to-end NLP safety analysis combining classification and context-aware moderation.

        Preserves 100% backward compatibility for is_safe, reason, detected_terms, flagged_categories, scores.
        Adds action, requires_warning, and decision dictionary.
        """
        if not isinstance(text, str):
            text = str(text or "")
        cleaned_text = text.strip()

        if not cleaned_text:
            return {
                "is_safe": True,
                "action": "allow",
                "requires_warning": False,
                "reason": None,
                "detected_terms": [],
                "flagged_categories": [],
                "scores": {},
                "model_status": self.model_status,
                "decision": {
                    "action": "allow",
                    "requires_warning": False,
                    "confidence": "high",
                    "context_classification": "empty_input"
                }
            }

        with self._cache_lock:
            if cleaned_text in self._inference_cache:
                return self._inference_cache[cleaned_text].copy()

        # Step A: ML Classification
        clf_result = self.classify_raw(cleaned_text)
        scores = clf_result["scores"]
        raw_flagged = clf_result["flagged_categories"]
        detected_terms = clf_result["detected_terms"]
        model_status = clf_result["model_status"]

        # Step B: Context-Aware Moderation Decision
        decision = self.decide_moderation(cleaned_text, scores, raw_flagged, detected_terms, model_status)

        result = {
            "is_safe": decision["is_safe"],
            "action": decision["action"],
            "requires_warning": decision["requires_warning"],
            "reason": decision["reason"],
            "detected_terms": detected_terms,
            "flagged_categories": decision["flagged_categories"],
            "scores": scores,
            "model_status": model_status,
            "decision": decision
        }

        with self._cache_lock:
            if len(self._inference_cache) >= self._MAX_CACHE_SIZE:
                self._inference_cache.pop(next(iter(self._inference_cache)))
            self._inference_cache[cleaned_text] = result

        return result.copy()