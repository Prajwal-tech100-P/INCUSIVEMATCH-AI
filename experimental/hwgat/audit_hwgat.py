import os
import sys
import csv
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from demo_utils import CFG
import importlib

cfg = CFG()
ckpt_path = cfg.save_model_path
class_map_path = cfg.class_map_path

print("=" * 70)
print("HWGAT MODEL & DATASET AUDIT REPORT")
print("=" * 70)
print(f"Checkpoint Path : {ckpt_path}")
print(f"Checkpoint Size : {os.path.getsize(ckpt_path) / (1024 * 1024):.2f} MB")
print(f"Class Map Path  : {class_map_path}")
print(f"Class Map Size  : {os.path.getsize(class_map_path) / 1024:.2f} KB")

class_map = {}
with open(class_map_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) >= 2:
            class_map[int(row[0])] = row[1].strip()

print(f"Total Sign Classes : {len(class_map)}")
print(f"Index Range        : 0 to {len(class_map) - 1}")
print(f"First 10 Classes   : {[class_map[i] for i in range(10)]}")
print(f"Last 10 Classes    : {[class_map[len(class_map) - 10 + i] for i in range(10)]}")

# Sample specific categories
indian_terms = [w for w in class_map.values() if w.lower() in ["dhoti", "sari", "surat", "namaste", "delhi", "mumbai", "india", "tea", "rupee"]]
print(f"Identified Indian Specific Terms: {indian_terms}")

common_actions = [w for w in class_map.values() if w.lower() in ["drink", "eat", "walk", "read", "write", "help", "hello", "thank you", "thanks"]]
print(f"Common Daily Life Actions       : {common_actions}")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"Checkpoint Keys: {list(ckpt.keys())}")
state_dict = ckpt["model_state_dict"]
print(f"State Dict Layers Count: {len(state_dict)}")
total_params = sum(v.numel() for v in state_dict.values())
print(f"Total Parameter Count  : {total_params:,}")
print(f"Head Weights Shape     : {tuple(state_dict['head.weight'].shape)} (num_classes, embed_dim)")
print(f"B Buffer Shape         : {tuple(state_dict['B'].shape)}")

# Architecture Instantiation Test
module = importlib.import_module("models." + cfg.model_type)
model = getattr(module, "Model")(*cfg.model_params.get_model_params())
model.load_state_dict(state_dict, strict=True)
model.eval()
print("Strict Model Weight Loading: PASSED (100% Parameter Match)")

# Forward Pass Check
dummy_in = torch.randn(1, 192, 64, 2)
with torch.no_grad():
    dummy_out = model(dummy_in)
print(f"Dummy Input Shape  : {tuple(dummy_in.shape)} (batch, temporal_frames, keypoints, coordinates)")
print(f"Dummy Output Shape : {tuple(dummy_out.shape)} (batch, num_classes)")
print(f"Min Logit: {dummy_out.min().item():.3f}, Max Logit: {dummy_out.max().item():.3f}")
print("=" * 70)

