from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")
MIN_SAMPLES = 15

X = []
y = []
classes = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    files = sorted(folder.glob("*.npy"))
    if len(files) < MIN_SAMPLES:
        continue

    classes.append(folder.name)

    for file in files:
        arr = np.load(file)

        if arr.shape != (60, 84):
            print("SKIPPING unexpected shape:", file, arr.shape)
            continue

        if not np.isfinite(arr).all():
            print("SKIPPING non-finite data:", file)
            continue

        X.append(str(file))
        y.append(folder.name)

print("Valid samples:", len(X))
print("Classes:", len(set(y)))

counts = Counter(y)
print("\nClass counts:")
for name, count in sorted(counts.items()):
    print(f"{name:15} {count:3}")

train_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(train_idx))
print("Testing samples:", len(test_idx))
print("Training classes:", len(set(y[i] for i in train_idx)))
print("Testing classes:", len(set(y[i] for i in test_idx)))
print("\nDataset check completed. No model trained yet.")
