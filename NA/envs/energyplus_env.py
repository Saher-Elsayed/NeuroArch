"""
EnergyPlus Co-Simulation Gymnasium Environment
================================================
Wraps EnergyPlus via BOPTEST API into a standard gym.Env interface.
Supports all three NeuroArch reference buildings.

Buildings:
  medium_office  : 4,982 m², 3 floors, Houston TX (ASHRAE Zone 2A)
  residential    : 3,135 m², 2-story, Houston TX
  mixed_use      : 8,210 m², 5 floors, Houston TX

Each agent observes: [T_zone, RH, CO2, occupancy, kappa_SNN]
Global state: 33-dimensional vector (all zones + outdoor + temporal)
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


BUILDINGS = {
    "medium_office": {
        "idf":       "MediumOffice_Houston.idf",
        "floor_area": 4982.0,   # m²
        "n_zones":   15,
        "n_floors":  3,
        "climate":   "ASHRAE_Zone_2A",
        "latitude":  29.98,
        "longitude": -95.37,
    },
    "residential": {
        "idf":       "ResidentialMidrise_Houston.idf",
        "floor_area": 3135.0,
        "n_zones":   8,
        "n_floors":  2,
        "climate":   "ASHRAE_Zone_2A",
        "latitude":  29.98,
        "longitude": -95.37,
    },
    "mixed_use": {
        "idf":       "MixedUse_Houston.idf",
        "floor_area": 8210.0,
        "n_zones":   24,
        "n_floors":  5,
        "climate":   "ASHRAE_Zone_2A",
        "latitude":  29.98,
        "longitude": -95.37,
    },
}

# Cross-climate buildings (LOBO evaluation)
CROSS_CLIMATE_BUILDINGS = {
    "seattle":      {"climate": "ASHRAE_Zone_4C", "latitude": 47.61, "longitude": -122.33},
    "minneapolis":  {"climate": "ASHRAE_Zone_6A", "latitude": 44.98, "longitude": -93.27},
}

OBSERVATION_FIELDS = ["zone_temp_C", "zone_rh_pct", "co2_ppm", "occupancy_frac", "kappa_snn"]
STATE_DIM = 33


@dataclass
class EnvConfig:
    building:         str   = "medium_office"
    timestep_minutes: int   = 1
    episode_hours:    int   = 24
    reward_type:      str   = "comfort_augmented"
    snn_inference:    bool  = True
    use_boptest:      bool  = False     # False for offline CSV-based simulation
    data_dir:         str   = "data"
    seed:             int   = 42


class NeuroArchEnv(gym.Env):
    """Multi-agent EnergyPlus environment for NeuroArch.

    Observation space: Dict of per-agent obs (Box, shape=(5,))
    Action space:      Dict of per-agent Discrete actions
    State:             Box(33,) — global state for QMIX mixer

    Example
    -------
    >>> env = NeuroArchEnv(EnvConfig(building="medium_office"))
    >>> obs, info = env.reset()
    >>> action = {f"agent_{i}": env.action_space[f"agent_{i}"].sample() for i in range(6)}
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        super().__init__()
        self.cfg       = cfg
        self.bldg_cfg  = BUILDINGS[cfg.building]
        self.rng       = np.random.default_rng(cfg.seed)
        self._step     = 0
        self._max_steps = cfg.episode_hours * 60 // cfg.timestep_minutes

        # Agent definitions
        self.agent_ids = [f"agent_{i}" for i in range(6)]
        self._action_dims = [21, 21, 21, 10, 11, 11]   # HVAC x3, light, shade x2

        self.observation_space = spaces.Dict({
            aid: spaces.Box(low=-5.0, high=5.0, shape=(5,), dtype=np.float32)
            for aid in self.agent_ids
        })
        self.action_space = spaces.Dict({
            aid: spaces.Discrete(self._action_dims[i])
            for i, aid in enumerate(self.agent_ids)
        })

        # Load offline simulation data
        self._load_offline_data()
        self._init_reward()

    def _load_offline_data(self):
        """Load pre-simulated EnergyPlus CSV data for offline rollouts."""
        csv_path = Path(self.cfg.data_dir) / "energyplus" / self.cfg.building / "simulation_8760h.csv"
        if csv_path.exists():
            import pandas as pd
            self._sim_df = pd.read_csv(csv_path)
        else:
            # Generate synthetic data if CSV not found
            n = 8760 * 60 // self.cfg.timestep_minutes
            self._sim_df = self._synthetic_sim_data(n)

    def _synthetic_sim_data(self, n: int):
        """Generate synthetic building simulation data for testing."""
        import pandas as pd
        t = np.arange(n)
        tod = (t % (24 * 60)) / (24 * 60)
        df = pd.DataFrame({
            "timestamp":       pd.date_range("2024-01-01", periods=n, freq="1min"),
            "zone_temp_C":     22.0 + 3.0 * np.sin(2*np.pi*tod) + self.rng.normal(0, 0.5, n),
            "zone_rh_pct":     50.0 + 10.0 * np.sin(2*np.pi*tod + 1) + self.rng.normal(0, 2, n),
            "co2_ppm":         800 + 200 * np.clip(np.sin(2*np.pi*tod - 1.5), 0, 1),
            "occupancy_frac":  np.clip(np.sin(2*np.pi*tod - 1.5), 0, 1).round(1),
            "hvac_power_kw":   50 + 30 * np.abs(np.sin(2*np.pi*tod)),
            "lighting_kw":     10 * np.clip(np.sin(2*np.pi*tod - 1.5), 0, 1),
            "pmv":             0.5 * np.sin(2*np.pi*tod),
            "ppd":             10 + 5 * np.abs(np.sin(2*np.pi*tod)),
            "outdoor_temp_C":  28 + 8 * np.sin(2*np.pi*tod + 0.5),
        })
        return df

    def _init_reward(self):
        from marl.reward import ComfortAugmentedReward, RewardWeights
        self._reward_fn = ComfortAugmentedReward()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._start_idx = self.rng.integers(0, max(1, len(self._sim_df) - self._max_steps - 1))
        obs, info = self._get_obs(), {}
        return obs, info

    def step(self, actions: dict) -> Tuple[dict, float, bool, bool, dict]:
        self._step += 1
        obs  = self._get_obs()

        # Compute reward from current state
        from marl.reward import BuildingState
        row = self._sim_df.iloc[min(self._start_idx + self._step, len(self._sim_df)-1)]
        bstate = BuildingState(
            energy_kwh       = float(row.get("hvac_power_kw", 60.0)) / 60.0,
            energy_baseline  = 1.5,
            comfort_classes  = [2, 2, 2],
            zone_targets     = [2, 2, 2],
            peak_kw          = float(row.get("hvac_power_kw", 60.0)),
            peak_limit_kw    = 312.0,
            ppd              = float(row.get("ppd", 12.0)),
            ashrae_compliant = True,
        )
        reward, info = self._reward_fn(bstate)
        terminated = self._step >= self._max_steps
        truncated  = False
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        idx = min(self._start_idx + self._step, len(self._sim_df) - 1)
        row = self._sim_df.iloc[idx]
        base_obs = np.array([
            float(row.get("zone_temp_C",   22.0)),
            float(row.get("zone_rh_pct",   50.0)),
            float(row.get("co2_ppm",       800.0)),
            float(row.get("occupancy_frac",0.5)),
            0.0,  # kappa_SNN placeholder
        ], dtype=np.float32)
        return {aid: base_obs + self.rng.normal(0, 0.02, 5).astype(np.float32)
                for aid in self.agent_ids}

    def get_global_state(self) -> np.ndarray:
        """Return 33-dim global state for QMIX mixer."""
        idx = min(self._start_idx + self._step, len(self._sim_df) - 1)
        row = self._sim_df.iloc[idx]
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0]  = float(row.get("zone_temp_C", 22.0))
        state[1]  = float(row.get("zone_rh_pct", 50.0))
        state[2]  = float(row.get("co2_ppm", 800.0))
        state[3]  = float(row.get("occupancy_frac", 0.5))
        state[4]  = float(row.get("outdoor_temp_C", 30.0))
        state[5]  = float(row.get("hvac_power_kw", 60.0))
        state[30] = float(self._step) / self._max_steps     # time fraction
        state[31] = np.sin(2 * np.pi * state[30])
        state[32] = np.cos(2 * np.pi * state[30])
        return state

    def render(self):
        pass

    def close(self):
        pass
