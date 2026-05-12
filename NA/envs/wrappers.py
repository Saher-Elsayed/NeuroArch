"""
Gymnasium Environment Wrappers
================================
FrameStack, Normalizer, RecordEpisodeStats, MultiAgentWrapper, FaultInjection.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional
import gymnasium as gym


class ObservationNormalizer(gym.ObservationWrapper):
    """Online z-score normalisation with running mean/std per channel."""

    def __init__(self, env, clip: float = 5.0, update_freq: int = 100):
        super().__init__(env)
        self.clip = clip
        self.update_freq = update_freq
        # Get obs dim from first agent
        aid0 = list(env.observation_space.keys())[0]
        obs_dim = env.observation_space[aid0].shape[0]
        self._mean = np.zeros(obs_dim)
        self._var  = np.ones(obs_dim)
        self._count = 0

    def observation(self, obs: dict) -> dict:
        normalized = {}
        for aid, o in obs.items():
            self._count += 1
            delta  = o - self._mean
            self._mean += delta / self._count
            delta2 = o - self._mean
            self._var += (delta * delta2 - self._var) / self._count if self._count > 1 else 0
            std = np.sqrt(self._var + 1e-8)
            normalized[aid] = np.clip((o - self._mean) / std, -self.clip, self.clip)
        return normalized


class RecordEpisodeStatistics(gym.Wrapper):
    """Tracks episode return, length, energy, and comfort metrics."""

    def __init__(self, env):
        super().__init__(env)
        self._ep_return = 0.0
        self._ep_length = 0
        self._ep_energy = 0.0
        self._ep_ppd    = []
        self.episode_stats = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_return += reward
        self._ep_length += 1
        self._ep_energy += info.get("e_norm", 0.0)
        self._ep_ppd.append(info.get("ppd", 0.0))

        if terminated or truncated:
            self.episode_stats = {
                "episode_return": self._ep_return,
                "episode_length": self._ep_length,
                "mean_energy":    self._ep_energy / max(self._ep_length, 1),
                "mean_ppd":       np.mean(self._ep_ppd) if self._ep_ppd else 0.0,
            }
            info.update(self.episode_stats)
            self._ep_return = self._ep_length = self._ep_energy = 0.0
            self._ep_ppd = []
        return obs, reward, terminated, truncated, info


class FaultInjectionWrapper(gym.Wrapper):
    """Inject sensor faults and actuator failures for robustness testing."""

    def __init__(self, env, fault_prob: float = 0.05, fault_type: str = "noise"):
        super().__init__(env)
        self.fault_prob = fault_prob
        self.fault_type = fault_type
        self._rng = np.random.default_rng(0)

    def step(self, action):
        # Actuator fault: randomly block actions
        if self._rng.random() < self.fault_prob:
            aid = self._rng.choice(list(action.keys()))
            action[aid] = 0  # zero action (fault)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Sensor fault: corrupt observations
        if self._rng.random() < self.fault_prob:
            aid = self._rng.choice(list(obs.keys()))
            if self.fault_type == "noise":
                obs[aid] += self._rng.normal(0, 2.0, obs[aid].shape).astype(np.float32)
            elif self.fault_type == "dropout":
                obs[aid] *= 0.0
            elif self.fault_type == "bias":
                obs[aid] += 5.0

        return obs, reward, terminated, truncated, info


class ClimateShiftWrapper(gym.Wrapper):
    """Wrapper that shifts weather data for cross-climate LOBO evaluation."""

    CLIMATE_OFFSETS = {
        "seattle":     {"temp_delta": -8.0,  "rh_delta": 15.0},
        "minneapolis": {"temp_delta": -15.0, "rh_delta": 10.0},
    }

    def __init__(self, env, climate: str = "seattle"):
        super().__init__(env)
        self.offset = self.CLIMATE_OFFSETS.get(climate, {"temp_delta": 0, "rh_delta": 0})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for aid in obs:
            obs[aid][0] += self.offset["temp_delta"]  # zone_temp
            obs[aid][1] += self.offset["rh_delta"]    # rh
        return obs, reward, terminated, truncated, info
