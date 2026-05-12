"""Comfort sensor dataset — loads CSV sensor logs and returns spike-encoded windows."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import Dataset


SENSOR_CHANNELS = [
    "air_temp", "radiant_temp", "humidity", "air_speed",
    "co2_ppm", "lux", "sound_db", "occupancy",
    "outdoor_temp", "solar_rad", "time_of_day_sin", "time_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
]

COMFORT_LABELS = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}  # ISO 7730 -> index


class ComfortDataset(Dataset):
    """Sliding-window comfort dataset.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory containing sensor_logs/ subdirectory.
    buildings : list of str
        Building IDs to include (e.g. ["medium_office", "residential"]).
    split : str
        "train", "val", or "test".
    window_size : int
        Number of timesteps per window (T=100).
    stride : int
        Sliding window stride.
    normalize : bool
        Z-score normalize each channel.
    augment : bool
        Apply random augmentation (jitter, time warp, channel dropout).
    noise_std : float
        Gaussian noise std for augmentation.
    """

    def __init__(self, data_dir: str, buildings: List[str], split: str = "train",
                 window_size: int = 100, stride: int = 50, normalize: bool = True,
                 augment: bool = False, noise_std: float = 0.01):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.noise_std = noise_std

        data_dir = Path(data_dir)
        all_windows, all_labels = [], []

        for bldg in buildings:
            log_path = data_dir / "sensor_logs" / f"{bldg}_sensor_log_sample.csv"
            label_path = data_dir / "sensor_logs" / "comfort_labels.csv"
            if not log_path.exists():
                continue

            df = pd.read_csv(log_path)
            labels_df = pd.read_csv(label_path)
            bldg_labels = labels_df[labels_df["building"] == bldg].copy()

            sensor_data = df[SENSOR_CHANNELS].values.astype(np.float32)

            if normalize:
                mu = sensor_data.mean(0)
                std = sensor_data.std(0) + 1e-8
                sensor_data = (sensor_data - mu) / std

            # Create sliding windows
            N = len(sensor_data)
            for start in range(0, N - window_size + 1, stride):
                window = sensor_data[start : start + window_size]
                # Find label for this window (majority vote)
                mid = start + window_size // 2
                label_row = bldg_labels.iloc[min(mid, len(bldg_labels)-1)]
                label_iso = int(label_row.get("comfort_label", 0))
                label = COMFORT_LABELS.get(label_iso, 2)
                all_windows.append(window)
                all_labels.append(label)

        windows = np.array(all_windows, dtype=np.float32)
        labels  = np.array(all_labels, dtype=np.int64)

        # Train/val/test split (70/15/15)
        n = len(windows)
        idx = np.random.RandomState(42).permutation(n)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        splits = {
            "train": idx[:n_train],
            "val":   idx[n_train : n_train + n_val],
            "test":  idx[n_train + n_val:],
        }
        sel = splits[split]
        self.windows = torch.from_numpy(windows[sel])
        self.labels  = torch.from_numpy(labels[sel])

    def class_weights(self) -> torch.Tensor:
        counts = torch.bincount(self.labels, minlength=5).float()
        weights_per_class = 1.0 / (counts + 1e-6)
        return weights_per_class[self.labels]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].clone()  # (T, C)
        y = self.labels[idx]
        if self.augment:
            x = self._augment(x)
        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        x = x + torch.randn_like(x) * self.noise_std
        # Channel dropout (10% of channels zeroed)
        mask = (torch.rand(x.shape[-1]) > 0.10).float()
        x = x * mask
        # Time shift: circular roll ±5 timesteps
        shift = np.random.randint(-5, 6)
        x = torch.roll(x, shift, dims=0)
        return x
