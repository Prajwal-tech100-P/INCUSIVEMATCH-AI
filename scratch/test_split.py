import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from ai_modules.train_sign_gcnt import extract_source_video_id

DATA_ROOT = Path(r"C:\Users\ACER\isl-40words\processed_landmarks")
MIN_SAMPLES = 15

classes = []
file_list = []
label_list = []
group_list = []

for folder in sorted(DATA_ROOT.iterdir()):
    if not folder.is_dir():
        continue
    valid_files = sorted(folder.glob("*.npy"))
    if len(valid_files) < MIN_SAMPLES:
        continue
    classes.append(folder.name)
    for f in valid_files:
        file_list.append(f)
        label_list.append(folder.name)
        group_list.append(extract_source_video_id(f.stem))

class_to_idx = {name: i for i, name in enumerate(classes)}
y = np.array([class_to_idx[l] for l in label_list])
groups = np.array(group_list)
X_dummy = np.zeros(len(y))

print(f"Total samples: {len(y)}, Classes: {len(classes)}, Unique video groups: {len(set(groups))}")

# Check 5-fold split
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X_dummy, y, groups)):
    tr_classes = set(y[tr_idx])
    te_classes = set(y[te_idx])
    tr_groups = set(groups[tr_idx])
    te_groups = set(groups[te_idx])
    overlap = tr_groups & te_groups
    print(f"Fold {fold}: Train={len(tr_idx)} ({len(tr_classes)} classes), Test={len(te_idx)} ({len(te_classes)} classes), Group overlap={len(overlap)}")
