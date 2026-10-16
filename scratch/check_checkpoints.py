import torch
from pathlib import Path

for path in ['sign_model_output/sign_gcn_transformer.pth', 'models/sign_gcnt.pt', 'models/matching_gcn_transformer.pt']:
    p = Path(path)
    if p.exists():
        print(f"=== File: {path} ===")
        data = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            for k, v in data.items():
                if k not in ('state_dict', 'model_state_dict', 'normalization_mean', 'normalization_std'):
                    if isinstance(v, list) and len(v) > 10:
                        print(f"  {k}: list of length {len(v)}")
                    else:
                        print(f"  {k}: {v}")
        else:
            print("  Raw state dict keys:", len(data) if hasattr(data, '__len__') else type(data))
