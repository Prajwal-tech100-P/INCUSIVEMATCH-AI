import torch

ckpt = torch.load("models/sign_gcnt.pt", map_location="cpu", weights_only=False)
print("=== sign_gcnt.pt ===")
for k in ckpt:
    if k not in ("state_dict", "model_state_dict", "normalization_mean", "normalization_std"):
        print(f"{k}: {ckpt[k]}")

if "best_test_accuracy" in ckpt:
    print(f"Recorded Accuracy: {ckpt['best_test_accuracy'] * 100:.2f}%")
