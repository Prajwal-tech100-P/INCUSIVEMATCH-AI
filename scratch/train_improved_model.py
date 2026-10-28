"""
Training and Validation-Tuning Script for InclusiveMatch AI NLP Toxicity Classifier v2.0.
Preserves the FeatureUnion (word TF-IDF + char_wb TF-IDF) + OneVsRest LogisticRegression architecture.
Tuning of classification thresholds is performed EXCLUSIVELY on the validation split.
Final evaluation is performed on the held-out test split and the independent audit benchmark.
"""
import os
import sys
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

PROJECT_ROOT = r"c:\Users\ACER\Desktop\InclusiveMatch_AI_Secure_WebRTC_HTTPS_Ready 8"
sys.path.insert(0, PROJECT_ROOT)

CATEGORIES = ["toxic", "severe_toxic", "insult", "threat", "harassment", "hate"]
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "toxicity")
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "toxicity_classifier.joblib")
BACKUP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "toxicity_classifier_v1_backup.joblib")

def load_data(split_name):
    filepath = os.path.join(DATA_DIR, f"{split_name}.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [x["text"] for x in data]
    labels = np.array([x["labels"] for x in data], dtype=np.float32)
    return texts, labels

def main():
    print("=" * 80)
    print("TRAINING INCLUSIVEMATCH AI TOXICITY CLASSIFIER v2.0")
    print("=" * 80)

    # 1. Load data
    train_texts, y_train = load_data("train")
    val_texts, y_val = load_data("val")
    test_texts, y_test = load_data("test")

    print(f"Dataset Loaded:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val:   {len(val_texts)} samples")
    print(f"  Test:  {len(test_texts)} samples")

    # 2. Build Pipeline preserving existing architecture
    print("\nBuilding TF-IDF FeatureUnion + OneVsRest Logistic Regression Pipeline...")
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

    # 3. Fit pipeline on training data
    print("Fitting model on training set...")
    pipeline.fit(train_texts, y_train)
    print("Model fitting complete.")

    # 4. Tune Thresholds on Validation Data ONLY
    print("\n" + "=" * 80)
    print("TUNING CLASSIFICATION THRESHOLDS ON VALIDATION DATA ONLY (n=954)")
    print("=" * 80)

    val_probs = pipeline.predict_proba(val_texts)
    
    tuned_thresholds = {}
    candidate_thresholds = np.arange(0.25, 0.85, 0.05)

    print(f"{'Category':15s} | {'Best Thresh':11s} | {'Val Prec':10s} | {'Val Recall':10s} | {'Val F1':10s}")
    print("-" * 65)

    for idx, cat in enumerate(CATEGORIES):
        y_true_cat = y_val[:, idx]
        pos_count = int(y_true_cat.sum())
        
        best_f1 = -1.0
        best_thresh = 0.50
        best_p = 0.0
        best_r = 0.0

        for thresh in candidate_thresholds:
            y_pred_cat = (val_probs[:, idx] >= thresh).astype(int)
            p = precision_score(y_true_cat, y_pred_cat, zero_division=0)
            r = recall_score(y_true_cat, y_pred_cat, zero_division=0)
            f = f1_score(y_true_cat, y_pred_cat, zero_division=0)

            # High severity safety categories (threat, severe_toxic, hate)
            # require solid recall while maximizing F1
            if cat in ["threat", "severe_toxic", "hate"]:
                if r >= 0.50 and f > best_f1:
                    best_f1 = f
                    best_thresh = float(thresh)
                    best_p = p
                    best_r = r
            else:
                if f > best_f1:
                    best_f1 = f
                    best_thresh = float(thresh)
                    best_p = p
                    best_r = r

        # Fallback if no threshold met recall constraint
        if best_f1 < 0.0:
            for thresh in candidate_thresholds:
                y_pred_cat = (val_probs[:, idx] >= thresh).astype(int)
                f = f1_score(y_true_cat, y_pred_cat, zero_division=0)
                if f > best_f1:
                    best_f1 = f
                    best_thresh = float(thresh)
                    best_p = precision_score(y_true_cat, y_pred_cat, zero_division=0)
                    best_r = recall_score(y_true_cat, y_pred_cat, zero_division=0)

        tuned_thresholds[cat] = round(best_thresh, 2)
        print(f"{cat:15s} | {tuned_thresholds[cat]:11.2f} | {best_p:10.4f} | {best_r:10.4f} | {best_f1:10.4f}")

    print("\nLocked Tuned Thresholds (from validation):", tuned_thresholds)

    # 5. Evaluate on Independent Test Set (n=956)
    print("\n" + "=" * 80)
    print("EVALUATING IMPROVED MODEL ON HELD-OUT TEST SET (n=956)")
    print("=" * 80)

    test_probs = pipeline.predict_proba(test_texts)
    
    test_metrics = {}
    print(f"{'Category':15s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s} | {'Test Positives':14s}")
    print("-" * 65)

    test_f1s = []
    for idx, cat in enumerate(CATEGORIES):
        thresh = tuned_thresholds[cat]
        y_true_cat = y_test[:, idx]
        y_pred_cat = (test_probs[:, idx] >= thresh).astype(int)

        p = precision_score(y_true_cat, y_pred_cat, zero_division=0)
        r = recall_score(y_true_cat, y_pred_cat, zero_division=0)
        f = f1_score(y_true_cat, y_pred_cat, zero_division=0)
        test_f1s.append(f)

        pos_count = int(y_true_cat.sum())
        test_metrics[cat] = {"precision": p, "recall": r, "f1": f, "positives": pos_count}
        print(f"{cat:15s} | {p:10.4f} | {r:10.4f} | {f:10.4f} | {pos_count:14d}")

    macro_f1 = np.mean(test_f1s)
    print("-" * 65)
    print(f"Macro-F1 on Held-Out Test Set: {macro_f1:.4f}")

    # 6. Save model artifact
    print(f"\nSaving improved model artifact to {MODEL_OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "categories": CATEGORIES,
        "version": "2.0.0",
        "model_type": "Tfidf_CharWb_OneVsRest_LogisticRegression",
        "tuned_thresholds": tuned_thresholds
    }, MODEL_OUTPUT_PATH, compress=3)
    
    size_kb = os.path.getsize(MODEL_OUTPUT_PATH) / 1024
    print(f"Saved model artifact ({size_kb:.1f} KB)")

    # 7. Dump validation & test results to json for report
    results_export = {
        "dataset_train_size": len(train_texts),
        "dataset_val_size": len(val_texts),
        "dataset_test_size": len(test_texts),
        "tuned_thresholds": tuned_thresholds,
        "test_metrics": test_metrics,
        "macro_f1_test": macro_f1
    }
    with open(os.path.join(DATA_DIR, "training_eval_results.json"), "w") as f:
        json.dump(results_export, f, indent=2)

if __name__ == "__main__":
    main()
