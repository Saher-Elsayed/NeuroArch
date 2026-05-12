"""
SNN Explainability — spike-timing attribution, GradCAM-SNN, and SHAP wrappers.
All methods return channel-level importance scores matching the 14-sensor layout.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional


SENSOR_NAMES = [
    "air_temp", "radiant_temp", "humidity", "air_speed",
    "co2_ppm", "lux", "sound_db", "occupancy",
    "outdoor_temp", "solar_rad", "time_sin", "time_cos",
    "dow_sin", "dow_cos",
]


class SNNGradCAM:
    """Gradient-weighted Class Activation Mapping for LIF-SNN.

    Adapted from standard CNN GradCAM — uses surrogate gradients
    accumulated over the temporal dimension.
    """

    def __init__(self, model, target_layer_idx: int = 1):
        self.model = model
        self.target_layer = model.lif_layers[target_layer_idx]
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self._activations = out[0].detach()  # spikes
        def bwd_hook(module, grad_inp, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def compute(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return (n_neurons,) activation map."""
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()

        weights = self._gradients.mean(dim=0)           # (hidden,)
        cam = (weights * self._activations.mean(0)).relu()
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class SpikeTimingAttribution:
    """Attribute predictions using first-spike timing per channel.

    Earlier spikes in the input encoding indicate higher-salience channels.
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def attribute(self, x: torch.Tensor) -> np.ndarray:
        """x: (1, T, C) -> importance: (C,)"""
        B, T, C = x.shape
        importances = np.zeros(C)
        for c in range(C):
            x_masked = x.clone()
            x_masked[:, :, c] = 0.0
            base_logits = self.model(x).softmax(-1)
            masked_logits = self.model(x_masked).softmax(-1)
            importances[c] = (base_logits - masked_logits).abs().max().item()
        return importances / (importances.sum() + 1e-8)


class IntegratedGradientsSNN:
    """Integrated Gradients adapted for surrogate-gradient SNN."""

    def __init__(self, model, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def attribute(self, x: torch.Tensor, class_idx: int,
                  baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """x: (1, T, C) -> attribution: (T, C)"""
        if baseline is None:
            baseline = torch.zeros_like(x)

        grads = []
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            x_interp = baseline + alpha * (x - baseline)
            x_interp = x_interp.requires_grad_(True)
            logits = self.model(x_interp)
            score = logits[0, class_idx]
            score.backward()
            grads.append(x_interp.grad.detach().cpu().numpy())

        avg_grads = np.mean(grads, axis=0)  # (1, T, C)
        attribution = (x - baseline).cpu().numpy() * avg_grads
        return attribution[0]  # (T, C)

    def channel_importance(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Aggregate IG attribution to per-channel importance."""
        attr = self.attribute(x, class_idx)  # (T, C)
        importance = np.abs(attr).mean(0)    # (C,)
        return importance / (importance.sum() + 1e-8)


def sensor_importance_report(model, test_loader, device, n_batches: int = 10) -> dict:
    """Run all attribution methods and aggregate across test batches."""
    ig = IntegratedGradientsSNN(model)
    sta = SpikeTimingAttribution(model)

    ig_scores  = np.zeros(14)
    sta_scores = np.zeros(14)
    count = 0

    for i, (x, y) in enumerate(test_loader):
        if i >= n_batches:
            break
        x = x[:1].to(device)  # single example
        ci = y[0].item()
        ig_scores  += ig.channel_importance(x, ci)
        sta_scores += sta.attribute(x)
        count += 1

    ig_scores  /= count
    sta_scores /= count
    ensemble = 0.6 * ig_scores + 0.4 * sta_scores

    return {
        "sensors":   SENSOR_NAMES,
        "ig":        ig_scores.tolist(),
        "sta":       sta_scores.tolist(),
        "ensemble":  ensemble.tolist(),
    }
