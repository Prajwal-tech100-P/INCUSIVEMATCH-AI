"""
model_utils.py - Model Loading, Head Surgery, and Checkpoint Management for HWGAT

Strictly isolated within experimental/hwgat/
"""
import os
import sys
import importlib
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from demo_utils import CFG


def build_base_hwgat_model() -> nn.Module:
    """Instantiate the raw HWGAT model architecture without loading weights."""
    cfg = CFG()
    module = importlib.import_module("models." + cfg.model_type)
    model = getattr(module, "Model")(*cfg.model_params.get_model_params())
    return model


def load_pretrained_hwgat(device: str = "cpu") -> nn.Module:
    """Load the pretrained HWGAT checkpoint with original 2,002 classes."""
    cfg = CFG()
    model = build_base_hwgat_model()
    ckpt_path = cfg.save_model_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(CURRENT_DIR, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Original HWGAT checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def adapt_classifier_head(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Replace the 2,002-class output linear layer with a custom K-class classification head.
    Preserves all learned backbone weights (GCN + Window Graph Attention layers).
    """
    if not hasattr(model, "head"):
        raise AttributeError("The model does not contain a 'head' attribute.")

    in_features = model.head.in_features
    new_head = nn.Linear(in_features, num_classes)

    # Initialize new weights appropriately
    nn.init.xavier_uniform_(new_head.weight)
    if new_head.bias is not None:
        nn.init.constant_(new_head.bias, 0.0)

    model.head = new_head
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool = False) -> None:
    """
    Freeze or unfreeze backbone parameters.
    If trainable is False, only model.head parameters have requires_grad=True.
    """
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def save_custom_checkpoint(
    model: nn.Module,
    class_mapping: Dict[int, str],
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save fine-tuned model checkpoint with full metadata, class mapping, and metrics.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    num_classes = len(class_mapping)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    payload = {
        "model_state_dict": model.state_dict(),
        "class_mapping": class_mapping,
        "num_classes": num_classes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "metadata": extra_metadata or {},
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, save_path)


def load_custom_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[int, str], Dict[str, Any]]:
    """
    Load a fine-tuned custom HWGAT checkpoint, reconstructing its head to match num_classes.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_mapping = checkpoint.get("class_mapping", {})
    num_classes = checkpoint.get("num_classes", len(class_mapping))

    model = build_base_hwgat_model()
    if num_classes != 2002:
        model = adapt_classifier_head(model, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "val_acc": checkpoint.get("val_acc"),
        "total_parameters": checkpoint.get("total_parameters"),
        "trainable_parameters": checkpoint.get("trainable_parameters"),
        "extra": checkpoint.get("metadata", {}),
    }

    return model, class_mapping, metadata
