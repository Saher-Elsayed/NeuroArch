"""Poisson rate encoder: converts normalized float inputs to binary spike trains."""
import torch

class PoissonRateEncoder:
    def __init__(self, T: int = 100, dt: float = 1.0):
        self.T  = T
        self.dt = dt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C) normalized [0,1] -> spikes: (B, T, C)"""
        x = x.clamp(0, 1)
        probs = x.unsqueeze(1).expand(-1, self.T, -1)
        return torch.bernoulli(probs)
