"""Advanced data augmentation for SNN sensor training."""
from __future__ import annotations
import numpy as np
import torch


class SensorAugmentor:
    """Collection of augmentation strategies for multi-channel sensor time series.

    All methods operate on (T, C) float tensors.
    """

    def __init__(self, noise_std=0.01, warp_sigma=0.2, dropout_p=0.10,
                 magnitude_scale=0.1, seed=None):
        self.noise_std = noise_std
        self.warp_sigma = warp_sigma
        self.dropout_p = dropout_p
        self.magnitude_scale = magnitude_scale
        self.rng = np.random.default_rng(seed)

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.noise_std

    def channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand(x.shape[-1]) > self.dropout_p).float()
        return x * mask

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        T, C = x.shape
        t_orig = np.linspace(0, 1, T)
        # Random smooth warp field
        n_knots = 4
        knots = np.sort(self.rng.uniform(0, 1, n_knots))
        warp = np.interp(t_orig, np.linspace(0, 1, n_knots), knots)
        warp = np.clip(warp, 0, 1) * (T - 1)

        warped = np.zeros_like(x.numpy())
        for c in range(C):
            warped[:, c] = np.interp(warp, t_orig * (T-1), x[:, c].numpy())
        return torch.from_numpy(warped.astype(np.float32))

    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.rng.uniform(-self.magnitude_scale, self.magnitude_scale, x.shape[-1])
        return x * torch.from_numpy(scale.astype(np.float32))

    def window_slice(self, x: torch.Tensor, reduce_ratio: float = 0.9) -> torch.Tensor:
        """Randomly crop and resize back to original length."""
        T, C = x.shape
        crop_len = int(T * reduce_ratio)
        start = self.rng.integers(0, T - crop_len + 1)
        sliced = x[start : start + crop_len, :]
        # Upsample back
        sliced_np = sliced.numpy().T  # (C, crop_len)
        t_orig = np.linspace(0, 1, crop_len)
        t_new  = np.linspace(0, 1, T)
        out = np.stack([np.interp(t_new, t_orig, sliced_np[c]) for c in range(C)], axis=-1)
        return torch.from_numpy(out.astype(np.float32))

    def mixup(self, x1: torch.Tensor, x2: torch.Tensor,
              y1: int, y2: int, alpha: float = 0.2):
        """Mixup augmentation (returns mixed x and soft label vector)."""
        lam = float(self.rng.beta(alpha, alpha))
        x_mix = lam * x1 + (1 - lam) * x2
        n_cls = 5
        y_mix = torch.zeros(n_cls)
        y_mix[y1] += lam
        y_mix[y2] += (1 - lam)
        return x_mix, y_mix

    def apply_all(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random subset of augmentations."""
        ops = [self.gaussian_noise, self.channel_dropout,
               self.magnitude_warp, self.window_slice]
        chosen = self.rng.choice(ops, size=2, replace=False)
        for op in chosen:
            x = op(x)
        return x
