import sys
sys.path.insert(0, ".")
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from ai_modules.sign_model_gcnt import GCNTransformerClassifier, preprocess_landmark_sequence

ckpt = torch.load('models/sign_gcnt_best.pt', map_location='cpu', weights_only=False)
model = GCNTransformerClassifier(
    num_classes=ckpt['num_classes'],
    seq_len=ckpt['seq_len'],
    node_dim=ckpt['node_dim'],
    gcn_dim=ckpt['gcn_dim'],
    tf_heads=ckpt['tf_heads'],
    tf_layers=ckpt['tf_layers'],
    dropout=ckpt['dropout']
)
model.load_state_dict(ckpt['state_dict'])
model.eval()

data_root = Path(r'C:\Users\ACER\isl-40words\processed_landmarks')
class_to_idx = ckpt['class_to_idx']
idx_to_class = {i: c for c, i in class_to_idx.items()}
test_stems = set(ckpt['test_file_stems'])

misclassifications = []
total_test = 0

for folder in sorted(data_root.iterdir()):
    if not folder.is_dir() or folder.name not in class_to_idx:
        continue
    c_idx = class_to_idx[folder.name]
    for f in folder.glob('*.npy'):
        if f.stem in test_stems:
            total_test += 1
            raw = np.load(f)
            feat = preprocess_landmark_sequence(raw, seq_len=ckpt['seq_len'])
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)[0].numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            true_conf = float(probs[c_idx])
            
            reshaped = raw.reshape(-1, 42, 2)
            h1_act = bool(np.any(reshaped[:, :21] != 0))
            h2_act = bool(np.any(reshaped[:, 21:] != 0))
            
            if pred_idx != c_idx:
                misclassifications.append({
                    'file': f.name,
                    'true_class': folder.name,
                    'pred_class': idx_to_class[pred_idx],
                    'conf': conf,
                    'true_conf': true_conf,
                    'h1': h1_act,
                    'h2': h2_act,
                })

print(f"Total test samples: {total_test}")
print(f"Correct: {total_test - len(misclassifications)} ({(total_test - len(misclassifications))/total_test*100:.2f}%)")
print(f"Incorrect: {len(misclassifications)} ({len(misclassifications)/total_test*100:.2f}%)\n")

by_class = defaultdict(list)
for m in misclassifications:
    by_class[m['true_class']].append(m)

print("=== MISCLASSIFICATIONS BY TRUE CLASS ===")
for c_name, items in sorted(by_class.items(), key=lambda x: len(x[1]), reverse=True):
    preds_str = ", ".join([f"{i['pred_class']} (conf={i['conf']:.2f}, true_conf={i['true_conf']:.2f})" for i in items])
    print(f"  {c_name:12s} ({len(items)} errors): {preds_str}")
