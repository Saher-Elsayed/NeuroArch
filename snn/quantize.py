"""INT8 quantization and magnitude pruning for FPGA deployment."""
import torch, torch.nn as nn
from .model import NeuroArchSNN, SNNConfig

def magnitude_prune(model: NeuroArchSNN, sparsity: float = 0.79) -> NeuroArchSNN:
    masks = []
    for layer in model.lif_layers:
        w = layer.fc.weight.data.abs()
        threshold = w.flatten().kthvalue(int(sparsity * w.numel())).values
        mask = (w >= threshold).float()
        layer.fc.weight.data *= mask
        masks.append(mask)
    model._pruning_masks = masks
    return model

def quantize_model(model: NeuroArchSNN) -> NeuroArchSNN:
    """Static INT8 quantization via PyTorch post-training quantization."""
    try:
        model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
    except Exception as e:
        print(f"Quantization fallback: {e}")
    return model
