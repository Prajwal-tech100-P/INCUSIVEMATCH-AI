import re

class NLPModerator:
    # Baseline moderation for an academic MVP; replace with a trained transformer
    # classifier for production-quality toxicity detection.
    TERMS = {"kill", "hate", "idiot", "stupid", "abuse"}

    def analyze_message(self, text):
        tokens=set(re.findall(r"[a-zA-Z]+", text.lower()))
        found=tokens & self.TERMS
        if found:
            return {"is_safe":False,"reason":"Potentially abusive or harmful language detected."}
        return {"is_safe":True,"reason":None}
