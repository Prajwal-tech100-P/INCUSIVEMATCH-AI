import os
import re
import html
import io
import json
import random
import pandas as pd
import numpy as np
import requests

random.seed(42)
np.random.seed(42)

import sys
PROJECT_ROOT = r"c:\Users\ACER\Desktop\InclusiveMatch_AI_Secure_WebRTC_HTTPS_Ready 8"
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "toxicity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = ["toxic", "severe_toxic", "insult", "threat", "harassment", "hate"]

print("1. Loading external source: Google Civil Comments...")
parquet_path = r"C:\Users\ACER\.cache\huggingface\hub\datasets--google--civil_comments\snapshots\f2970eb3a55777454c94069077cc8d9b5866312d\data\validation-00000-of-00001.parquet"
df_civil = pd.read_parquet(parquet_path)
print(f"   Loaded Civil Comments: {len(df_civil)} rows")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract high-value slices from Civil Comments:
# - Threats
cc_threat_df = df_civil[df_civil['threat'] >= 0.4].copy()
# - Hate speech / Identity attacks
cc_hate_df = df_civil[df_civil['identity_attack'] >= 0.4].copy()
# - Severe toxic
cc_sev_df = df_civil[df_civil['severe_toxicity'] >= 0.2].copy()
# - Insults
sub_insult = df_civil[df_civil['insult'] >= 0.5]
cc_insult_df = sub_insult.sample(n=min(1200, len(sub_insult)), random_state=42).copy()
# - General toxic
sub_toxic = df_civil[(df_civil['toxicity'] >= 0.5) & (df_civil['insult'] < 0.4) & (df_civil['threat'] < 0.3) & (df_civil['identity_attack'] < 0.3)]
cc_toxic_df = sub_toxic.sample(n=min(600, len(sub_toxic)), random_state=42).copy()
# - Benign comments
sub_benign = df_civil[(df_civil['toxicity'] == 0.0) & (df_civil['insult'] == 0.0) & (df_civil['threat'] == 0.0) & (df_civil['identity_attack'] == 0.0)]
cc_benign_df = sub_benign.sample(n=min(1800, len(sub_benign)), random_state=42).copy()

def map_civil_row(row):
    text = clean_text(row['text'])
    if len(text) < 10 or len(text) > 800:
        return None
    
    toxic = 1 if row['toxicity'] >= 0.5 else 0
    severe = 1 if row['severe_toxicity'] >= 0.25 else 0
    insult = 1 if row['insult'] >= 0.5 else 0
    threat = 1 if row['threat'] >= 0.4 else 0
    hate = 1 if row['identity_attack'] >= 0.4 else 0
    # Harassment: sexual explicit or combined threat + insult
    harassment = 1 if (row['sexual_explicit'] >= 0.4 or (row['threat'] >= 0.35 and row['insult'] >= 0.35)) else 0
    
    # If any specific subcategory is 1, toxic should also be 1
    if severe or insult or threat or hate or harassment:
        toxic = 1
        
    return (text, [toxic, severe, insult, threat, harassment, hate], "google_civil_comments")

civil_samples = []
for df_slice in [cc_threat_df, cc_hate_df, cc_sev_df, cc_insult_df, cc_toxic_df, cc_benign_df]:
    for _, row in df_slice.iterrows():
        mapped = map_civil_row(row)
        if mapped:
            civil_samples.append(mapped)

print(f"   Sampled {len(civil_samples)} Civil Comments rows")

# 2. Loading Davidson et al. (MIT)
print("2. Loading external source: Davidson et al. Hate Speech...")
davidson_url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
r_dav = requests.get(davidson_url, timeout=15)
df_dav = pd.read_csv(io.StringIO(r_dav.text))

def clean_tweet(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'RT\s*:\s*', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

dav_samples = []
# Hate speech: class == 0 -> [1, 0, 1, 0, 0, 1]
df_dav_hate = df_dav[df_dav['class'] == 0].sample(n=min(500, (df_dav['class'] == 0).sum()), random_state=42)
for _, row in df_dav_hate.iterrows():
    t = clean_tweet(row['tweet'])
    if len(t) >= 15:
        dav_samples.append((t, [1, 0, 1, 0, 0, 1], "davidson_hate_speech"))

# Neither / benign: class == 2 -> [0, 0, 0, 0, 0, 0]
df_dav_benign = df_dav[df_dav['class'] == 2].sample(n=min(800, (df_dav['class'] == 2).sum()), random_state=42)
for _, row in df_dav_benign.iterrows():
    t = clean_tweet(row['tweet'])
    if len(t) >= 15:
        dav_samples.append((t, [0, 0, 0, 0, 0, 0], "davidson_benign"))

print(f"   Sampled {len(dav_samples)} Davidson rows")

# Helper for robust dynamic loading across different script locations
import importlib.util

def _load_module_attribute(module_name, filenames, attr_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        current_dir,
        os.path.join(current_dir, "scratch"),
        os.path.join(PROJECT_ROOT, "scratch"),
        PROJECT_ROOT
    ]
    for d in search_dirs:
        for fname in filenames:
            fpath = os.path.join(d, fname)
            if os.path.isfile(fpath):
                spec = importlib.util.spec_from_file_location(module_name, fpath)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    val = getattr(mod, attr_name, None)
                    if val is not None:
                        return val
    for mod_path in [module_name, f"scratch.{module_name}"]:
        try:
            mod = __import__(mod_path, fromlist=[attr_name])
            return getattr(mod, attr_name)
        except (ImportError, ModuleNotFoundError):
            continue
    raise ImportError(f"Could not load '{attr_name}' from {module_name} in candidate paths.")

# 3. Existing 142 samples from train_toxicity_classifier.py
print("3. Loading existing project seed samples...")
SEED_DATA = _load_module_attribute("train_toxicity_classifier", ["train_toxicity_classifier.py"], "TRAINING_DATA")
seed_samples = [(t[0], t[1], "project_seed_v1") for t in SEED_DATA]
print(f"   Loaded {len(seed_samples)} project seed samples")

# 4. Curated Domain-Specific Samples (Dating, Inclusive, Figurative Negatives, Compliments)
print("4. Loading curated domain-specific expansion samples...")
CURATED_DOMAIN_SAMPLES = _load_module_attribute("curated_domain_data", ["curated_domain_data.py"], "CURATED_DOMAIN_SAMPLES")
print(f"   Loaded {len(CURATED_DOMAIN_SAMPLES)} domain-specific curated samples")

# Combine all datasets
all_records = civil_samples + dav_samples + seed_samples + CURATED_DOMAIN_SAMPLES

# Deduplication by lowercase stripped text
unique_records = {}
for text, labels, source in all_records:
    norm_text = text.strip().lower()
    norm_text = re.sub(r'\s+', ' ', norm_text)
    if norm_text not in unique_records:
        unique_records[norm_text] = (text.strip(), labels, source)

dataset = list(unique_records.values())
random.shuffle(dataset)
print(f"\nTOTAL UNIQUE SAMPLES: {len(dataset)}")

# Label statistics
labels_arr = np.array([item[1] for item in dataset])
print("\nLABEL DISTRIBUTION:")
for idx, cat in enumerate(CATEGORIES):
    cnt = int(labels_arr[:, idx].sum())
    print(f"  {cat:15s}: {cnt:4d} ({cnt/len(dataset)*100:5.1f}%)")
benign_cnt = int((labels_arr.sum(axis=1) == 0).sum())
print(f"  {'benign':15s}: {benign_cnt:4d} ({benign_cnt/len(dataset)*100:5.1f}%)")

# Split: 70% Train, 15% Val, 15% Test
n_total = len(dataset)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)
n_test = n_total - n_train - n_val

train_data = dataset[:n_train]
val_data = dataset[n_train:n_train+n_val]
test_data = dataset[n_train+n_val:]

print(f"\nSPLITS:")
print(f"  Train: {len(train_data)} samples")
print(f"  Val:   {len(val_data)} samples")
print(f"  Test:  {len(test_data)} samples")

# Verify zero overlap between splits
train_texts = set(x[0].strip().lower() for x in train_data)
val_texts = set(x[0].strip().lower() for x in val_data)
test_texts = set(x[0].strip().lower() for x in test_data)

assert len(train_texts & val_texts) == 0, "Train-Val overlap detected!"
assert len(train_texts & test_texts) == 0, "Train-Test overlap detected!"
assert len(val_texts & test_texts) == 0, "Val-Test overlap detected!"
print("  Zero overlap across all 3 splits verified!")

# Save to datasets/toxicity/
def save_split(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    formatted = [{"text": x[0], "labels": x[1], "source": x[2]} for x in data]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)
    print(f"Saved {filepath} ({len(formatted)} records)")

save_split(train_data, "train.json")
save_split(val_data, "val.json")
save_split(test_data, "test.json")

# Save metadata
metadata = {
    "dataset_name": "InclusiveMatch AI Toxicity & Safety Dataset v2.0",
    "categories": CATEGORIES,
    "total_samples": len(dataset),
    "split_counts": {
        "train": len(train_data),
        "validation": len(val_data),
        "test": len(test_data)
    },
    "category_counts": {cat: int(labels_arr[:, i].sum()) for i, cat in enumerate(CATEGORIES)},
    "benign_count": benign_cnt,
    "sources": {
        "google_civil_comments": {
            "license": "CC0 (Public Domain)",
            "description": "Jigsaw Unintended Bias in Toxicity Classification (real-world forum comments)"
        },
        "davidson_hate_speech": {
            "license": "MIT License",
            "description": "Davidson et al. (2017) Automated Hate Speech and Offensive Language"
        },
        "project_seed_v1": {
            "license": "Project Internal",
            "description": "Original handcrafted seed examples"
        },
        "curated_domain_data": {
            "license": "Project Internal",
            "description": "Targeted inclusive dating, compliments with 'you/you are', figurative negatives with 'shoot/kill/hate', disability/LGBTQ+ expressions"
        }
    }
}

with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nDataset preparation script completed successfully!")
