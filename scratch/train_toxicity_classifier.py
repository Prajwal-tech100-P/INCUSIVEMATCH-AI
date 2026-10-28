"""
Production training script for the InclusiveMatch AI NLP Toxicity & Safety Classifier.
Produces a lightweight, calibrated multi-label text classifier saved to models/toxicity_classifier.joblib.
Includes word and character n-gram feature union and calibrated linear classifiers for:
- toxic
- severe_toxic
- insult
- threat
- harassment
- hate
"""
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Target categories
CATEGORIES = ["toxic", "severe_toxic", "insult", "threat", "harassment", "hate"]

# Curated high-precision training dataset
# Format: (text, [toxic, severe_toxic, insult, threat, harassment, hate])
TRAINING_DATA = [
    # ==================== 1. THREATS & PHYSICAL VIOLENCE ====================
    ("I will kill you if you don't answer me", [1, 1, 0, 1, 1, 0]),
    ("I am going to murder you", [1, 1, 0, 1, 0, 0]),
    ("I will find where you live and shoot you dead", [1, 1, 0, 1, 1, 0]),
    ("I will slit your throat in your sleep", [1, 1, 0, 1, 0, 0]),
    ("watch your back because I'm going to hunt you down", [1, 0, 0, 1, 1, 0]),
    ("I will break your neck and shatter your bones", [1, 1, 0, 1, 0, 0]),
    ("I will track you down and beat you until you bleed", [1, 1, 0, 1, 1, 0]),
    ("you are going to die tonight", [1, 1, 0, 1, 0, 0]),
    ("I will stab you in the chest", [1, 1, 0, 1, 0, 0]),
    ("I'm going to strangle you until you stop breathing", [1, 1, 0, 1, 0, 0]),
    ("I will burn your house down with you inside", [1, 1, 0, 1, 0, 0]),
    ("I will destroy you and everyone you love", [1, 1, 0, 1, 0, 0]),
    ("I'm gonna punch your face in", [1, 0, 0, 1, 0, 0]),
    ("You better watch out or you will end up in the hospital", [1, 0, 0, 1, 0, 0]),
    ("I will choke you to death", [1, 1, 0, 1, 0, 0]),
    ("I'll come over and smash your head", [1, 1, 0, 1, 0, 0]),
    ("I'm going to beat the shit out of you", [1, 0, 0, 1, 0, 0]),
    ("I'll put a bullet through your skull", [1, 1, 0, 1, 0, 0]),
    ("I will make you disappear forever", [1, 1, 0, 1, 1, 0]),
    ("Your days are numbered, watch yourself", [1, 0, 0, 1, 0, 0]),

    # ==================== 2. SEVERE TOXICITY / SELF-HARM ENCOURAGEMENT ====================
    ("go kill yourself right now you piece of trash", [1, 1, 1, 0, 1, 0]),
    ("kys nobody will ever love you", [1, 1, 1, 0, 0, 0]),
    ("kill yourself already", [1, 1, 0, 0, 0, 0]),
    ("I hope you die a painful and agonizing death", [1, 1, 0, 0, 0, 0]),
    ("you should hang yourself and do the world a favor", [1, 1, 1, 0, 0, 0]),
    ("fucking die already you waste of oxygen", [1, 1, 1, 0, 0, 0]),
    ("drink bleach and kill yourself", [1, 1, 0, 0, 0, 0]),
    ("you deserve to suffer and rot in hell", [1, 1, 0, 0, 0, 0]),
    ("jump off a bridge and end your pathetic life", [1, 1, 1, 0, 0, 0]),
    ("slit your wrists and bleed out", [1, 1, 0, 0, 0, 0]),
    ("the world would be so much better if you just died", [1, 1, 0, 0, 0, 0]),
    ("swallow some pills and never wake up", [1, 1, 0, 0, 0, 0]),

    # ==================== 3. PERSONAL INSULTS & DEROGATORY ATTACKS ====================
    ("you are an idiot and completely brainless", [1, 0, 1, 0, 0, 0]),
    ("shut up you stupid bitch", [1, 0, 1, 0, 0, 0]),
    ("you are a pathetic loser with no future", [1, 0, 1, 0, 0, 0]),
    ("you ugly moron, look in the mirror", [1, 0, 1, 0, 0, 0]),
    ("you are a disgusting piece of shit", [1, 0, 1, 0, 0, 0]),
    ("nobody wants you, you dumbass", [1, 0, 1, 0, 0, 0]),
    ("you are a worthless scumbag", [1, 0, 1, 0, 0, 0]),
    ("get lost you retarded clown", [1, 0, 1, 0, 0, 1]),
    ("you're a fat pig and you make me sick", [1, 0, 1, 0, 0, 0]),
    ("you imbecile, can't you do anything right", [1, 0, 1, 0, 0, 0]),
    ("you are pure garbage and filthy", [1, 0, 1, 0, 0, 0]),
    ("fuck you, you arrogant prick", [1, 0, 1, 0, 0, 0]),
    ("you're such a dickhead", [1, 0, 1, 0, 0, 0]),
    ("get out of my sight you useless fool", [1, 0, 1, 0, 0, 0]),
    ("you twat, stop talking to me", [1, 0, 1, 0, 0, 0]),
    ("you're so ugly nobody would ever date you", [1, 0, 1, 0, 0, 0]),
    ("stupid whore looking for attention", [1, 0, 1, 0, 0, 0]),
    ("you bastard, go fuck yourself", [1, 0, 1, 0, 0, 0]),
    ("you are an asshole and a freak", [1, 0, 1, 0, 0, 0]),
    ("you dumb slut", [1, 0, 1, 0, 0, 0]),
    ("you're disgusting and repulsive", [1, 0, 1, 0, 0, 0]),
    ("eat shit asshole", [1, 0, 1, 0, 0, 0]),

    # ==================== 4. HARASSMENT & STALKING ====================
    ("I know where you live and I'm watching you", [1, 0, 0, 1, 1, 0]),
    ("you can't escape from me, I will find you anywhere", [1, 0, 0, 1, 1, 0]),
    ("I'm going to leak all your private photos online", [1, 0, 0, 0, 1, 0]),
    ("answer me right now or I will ruin your life", [1, 0, 0, 1, 1, 0]),
    ("I am standing outside your house right now", [1, 0, 0, 1, 1, 0]),
    ("I will message your family and tell them lies about you", [1, 0, 0, 0, 1, 0]),
    ("stop ignoring me you bitch or else", [1, 0, 1, 1, 1, 0]),
    ("I'll make sure nobody in this city ever hires you", [1, 0, 0, 0, 1, 0]),
    ("I will follow you until you give me what I want", [1, 0, 0, 1, 1, 0]),
    ("I have your phone number and address and I won't stop calling", [1, 0, 0, 0, 1, 0]),
    ("You can't hide from me, I will keep stalking you", [1, 0, 0, 1, 1, 0]),
    ("Send me your naked pictures or I will expose you", [1, 0, 0, 1, 1, 0]),

    # ==================== 5. HATE SPEECH, DISCRIMINATORY SLURS & ATTACKS ====================
    ("all people of your race are filthy subhumans", [1, 1, 1, 0, 0, 1]),
    ("go back to your country you dirty immigrant", [1, 0, 1, 0, 0, 1]),
    ("your kind should all be eliminated from earth", [1, 1, 0, 1, 0, 1]),
    ("faggot, you don't belong in society", [1, 1, 1, 0, 0, 1]),
    ("tranny freak, nobody wants your kind", [1, 1, 1, 0, 0, 1]),
    ("you cripple retard, you are a burden on taxpayers", [1, 1, 1, 0, 0, 1]),
    ("dirty nigger get out of here", [1, 1, 1, 0, 0, 1]),
    ("all fags should burn in hell", [1, 1, 0, 0, 0, 1]),
    ("I hate all members of your religion, you are terrorists", [1, 1, 0, 0, 0, 1]),
    ("disabled people like you are subhuman trash", [1, 1, 1, 0, 0, 1]),
    ("chink go back to your country", [1, 1, 1, 0, 0, 1]),
    ("spic trash like you doesn't belong here", [1, 1, 1, 0, 0, 1]),
    ("kike scum running the world", [1, 1, 1, 0, 0, 1]),

    # ==================== 6. GENERAL TOXIC / FOUL-MOUTHED ABUSE ====================
    ("fuck this shit and fuck you", [1, 0, 1, 0, 0, 0]),
    ("eat shit and choke on it", [1, 0, 1, 0, 0, 0]),
    ("shut the fuck up", [1, 0, 0, 0, 0, 0]),
    ("you are full of bullshit and lies", [1, 0, 1, 0, 0, 0]),
    ("motherfucker, leave me alone", [1, 0, 1, 0, 0, 0]),
    ("go suck a dick you dipshit", [1, 0, 1, 0, 0, 0]),
    ("fucking piece of trash", [1, 0, 1, 0, 0, 0]),
    ("bitch ass motherfucker", [1, 0, 1, 0, 0, 0]),

    # ==================== 7. BENIGN: DISABILITY & ACCESSIBILITY (MUST BE SAFE) ====================
    ("I have cerebral palsy and use an electric wheelchair.", [0, 0, 0, 0, 0, 0]),
    ("I am deaf and Indian Sign Language is my native language.", [0, 0, 0, 0, 0, 0]),
    ("I am hard of hearing and prefer video calls with captions.", [0, 0, 0, 0, 0, 0]),
    ("Being neurodivergent and having ADHD gives me unique perspectives.", [0, 0, 0, 0, 0, 0]),
    ("I have been blind since birth and use a guide dog.", [0, 0, 0, 0, 0, 0]),
    ("I live with chronic illness and take things one day at a time.", [0, 0, 0, 0, 0, 0]),
    ("We need more accessibility and disability rights in our community.", [0, 0, 0, 0, 0, 0]),
    ("I faced ableism in the past, but I am proud of who I am.", [0, 0, 0, 0, 0, 0]),
    ("Discrimination against disabled people is wrong and needs to stop.", [0, 0, 0, 0, 0, 0]),
    ("I am learning sign language so I can communicate with deaf friends.", [0, 0, 0, 0, 0, 0]),
    ("My visual impairment doesn't stop me from enjoying art and audiobooks.", [0, 0, 0, 0, 0, 0]),
    ("Can we do a sign language video call tonight?", [0, 0, 0, 0, 0, 0]),
    ("InclusiveMatch AI makes dating accessible for disabled people.", [0, 0, 0, 0, 0, 0]),
    ("I am an autism advocate and work with special education children.", [0, 0, 0, 0, 0, 0]),
    ("I use speech-to-text assistive technology on my phone.", [0, 0, 0, 0, 0, 0]),
    ("I have mobility limitations and use crutches when walking outdoors.", [0, 0, 0, 0, 0, 0]),
    ("Mental health matters, therapy helped me manage my depression.", [0, 0, 0, 0, 0, 0]),
    ("Deaf culture has a rich and vibrant history.", [0, 0, 0, 0, 0, 0]),
    ("I had to deal with accessibility barriers at the train station today.", [0, 0, 0, 0, 0, 0]),
    ("My hearing aid batteries died today, glad I had a backup pair.", [0, 0, 0, 0, 0, 0]),

    # ==================== 8. BENIGN: LGBTQ+ & IDENTITIES (MUST BE SAFE) ====================
    ("I am gay and proud of my identity.", [0, 0, 0, 0, 0, 0]),
    ("Happy Pride month! Looking forward to the parade.", [0, 0, 0, 0, 0, 0]),
    ("I am a transgender woman looking for a kind partner.", [0, 0, 0, 0, 0, 0]),
    ("I identify as bisexual and support queer youth.", [0, 0, 0, 0, 0, 0]),
    ("Homophobic discrimination should be challenged everywhere.", [0, 0, 0, 0, 0, 0]),
    ("Transphobia has no place in our society.", [0, 0, 0, 0, 0, 0]),
    ("My pronouns are they/them, what are yours?", [0, 0, 0, 0, 0, 0]),
    ("I am lesbian and excited to meet open-minded people here.", [0, 0, 0, 0, 0, 0]),
    ("Being queer is just one part of who I am.", [0, 0, 0, 0, 0, 0]),

    # ==================== 9. BENIGN: CASUAL DATING, SLANG & COLLOQUIALISMS (MUST BE SAFE) ====================
    ("Hey there, how is your weekend going?", [0, 0, 0, 0, 0, 0]),
    ("Would you like to meet for coffee or tea sometime?", [0, 0, 0, 0, 0, 0]),
    ("I really loved your profile pictures, especially the hiking one!", [0, 0, 0, 0, 0, 0]),
    ("That movie was so good, the plot twist blew my mind!", [0, 0, 0, 0, 0, 0]),
    ("I hate getting stuck in traffic on Monday mornings.", [0, 0, 0, 0, 0, 0]),
    ("I hate rainy weather, I much prefer warm sunny days.", [0, 0, 0, 0, 0, 0]),
    ("I hate broccoli, but I love spinach and carrots.", [0, 0, 0, 0, 0, 0]),
    ("This leg workout at the gym is killing me today haha!", [0, 0, 0, 0, 0, 0]),
    ("That joke was hilarious, I was laughing so hard!", [0, 0, 0, 0, 0, 0]),
    ("Damn, that sounds like an amazing trip to the mountains!", [0, 0, 0, 0, 0, 0]),
    ("No way, that is so cool! Tell me more about your hobby.", [0, 0, 0, 0, 0, 0]),
    ("I'm so exhausted after work, ready to relax and listen to music.", [0, 0, 0, 0, 0, 0]),
    ("You seem like a genuine, kind person. Glad we matched!", [0, 0, 0, 0, 0, 0]),
    ("Good morning! Hope you have a wonderful day ahead.", [0, 0, 0, 0, 0, 0]),
    ("Good night, talk to you tomorrow!", [0, 0, 0, 0, 0, 0]),
    ("What kind of food do you like to cook?", [0, 0, 0, 0, 0, 0]),
    ("I am studying software engineering and love building web apps.", [0, 0, 0, 0, 0, 0]),
    ("I love dogs and volunteering at the animal shelter.", [0, 0, 0, 0, 0, 0]),
    ("That's awesome, congratulations on your graduation!", [0, 0, 0, 0, 0, 0]),
    ("I struggled with anxiety during the pandemic, but therapy helped.", [0, 0, 0, 0, 0, 0]),
    ("I'm looking for a serious long-term relationship based on mutual respect.", [0, 0, 0, 0, 0, 0]),
    ("I hate seeing people get hurt or treated unfairly.", [0, 0, 0, 0, 0, 0]),
    ("Can't wait to see you at the cafe later!", [0, 0, 0, 0, 0, 0]),
    ("Are you free for a phone call or chat this evening?", [0, 0, 0, 0, 0, 0]),
    ("Let's plan something fun for next weekend.", [0, 0, 0, 0, 0, 0]),
    ("That was such a killer guitar solo!", [0, 0, 0, 0, 0, 0]),
]

def train_and_save_model(output_path="models/toxicity_classifier.joblib"):
    print("Preparing training dataset...")
    dataset_train_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets", "toxicity", "train.json"
    )
    tuned_thresh_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets", "toxicity", "training_eval_results.json"
    )

    if os.path.exists(dataset_train_path):
        import json
        with open(dataset_train_path, "r", encoding="utf-8") as f:
            train_records = json.load(f)
        texts = [x["text"] for x in train_records]
        y = np.array([x["labels"] for x in train_records], dtype=np.float32)
        print(f"Loaded expanded training set from {dataset_train_path} ({len(texts)} samples)")
        version = "2.0.0"
        
        tuned_thresholds = {
            "toxic": 0.40,
            "severe_toxic": 0.35,
            "insult": 0.45,
            "threat": 0.55,
            "harassment": 0.45,
            "hate": 0.45,
        }
        if os.path.exists(tuned_thresh_path):
            try:
                with open(tuned_thresh_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if "tuned_thresholds" in meta:
                        tuned_thresholds.update(meta["tuned_thresholds"])
            except Exception:
                pass
    else:
        texts = [row[0] for row in TRAINING_DATA]
        y = np.array([row[1] for row in TRAINING_DATA], dtype=np.float32)
        print(f"Fallback to seed training samples: {len(texts)}")
        version = "1.1.0"
        tuned_thresholds = None

    print(f"Total training samples: {len(texts)}")
    print(f"Categories: {CATEGORIES}")

    # Build hybrid word & char n-gram pipeline
    feature_union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            sublinear_tf=True,
            min_df=2,
            max_features=25000,
            strip_accents="unicode"
        )),
        ("char_tfidf", TfidfVectorizer(
            ngram_range=(2, 5),
            analyzer="char_wb",
            sublinear_tf=True,
            min_df=2,
            max_features=35000,
            strip_accents="unicode"
        )),
    ])

    # Multi-label OneVsRest Logistic Regression with balanced weights
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=500,
            random_state=42
        )
    )

    pipeline = Pipeline([
        ("features", feature_union),
        ("clf", classifier)
    ])

    print("Fitting model...")
    pipeline.fit(texts, y)

    # Save model artifact
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dict = {
        "pipeline": pipeline,
        "categories": CATEGORIES,
        "version": version,
        "model_type": "Tfidf_CharWb_OneVsRest_LogisticRegression"
    }
    if tuned_thresholds:
        save_dict["tuned_thresholds"] = tuned_thresholds

    joblib.dump(save_dict, output_path, compress=3)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved model to {output_path} ({size_kb:.1f} KB)")
    return output_path

if __name__ == "__main__":
    train_and_save_model()
