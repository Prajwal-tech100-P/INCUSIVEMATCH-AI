import torch
try:
    ckpt2 = torch.load("models/gcn_transformer_match.pt", map_location="cpu", weights_only=False)
    print("=== gcn_transformer_match.pt ===")
    if isinstance(ckpt2, dict):
        for k in ckpt2:
            if not k.endswith("weight") and not k.endswith("bias"):
                print(f"{k}: {ckpt2[k]}")
    else:
        print("Type:", type(ckpt2))
except Exception as e:
    print("Error:", e)
