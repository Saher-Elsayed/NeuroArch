"""
Temperature scaling calibration for SNN output logits.
Ensures that the confidence of the comfort classifier matches empirical accuracy.
Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr


class TemperatureScaling(nn.Module):
    """Learnable scalar temperature T that scales logits: logits / T."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 50):
        """Optimize temperature on validation logits/labels."""
        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(self(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies  = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Return data for reliability diagram (confidence vs accuracy per bin)."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies  = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bins.append({"mid": (lo+hi)/2, "acc": 0.0, "conf": 0.0, "count": 0})
        else:
            bins.append({
                "mid":   float((lo + hi) / 2),
                "acc":   float(accuracies[mask].mean()),
                "conf":  float(confidences[mask].mean()),
                "count": int(mask.sum()),
            })
    return {"bins": bins, "ece": expected_calibration_error(probs, labels, n_bins)}
