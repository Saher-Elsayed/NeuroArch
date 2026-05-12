"""
Prioritised Experience Replay Buffer for QMIX
==============================================
Supports standard uniform sampling and proportional PER (Schaul et al. 2015).
"""
from __future__ import annotations
import numpy as np
from typing import Optional


class EpisodeBuffer:
    """Fixed-size episode replay buffer (uniform sampling)."""

    def __init__(self, capacity: int, episode_limit: int,
                 n_agents: int, obs_dim: int, state_dim: int,
                 action_dim: int):
        self.capacity      = capacity
        self.episode_limit = episode_limit
        self.n_agents      = n_agents
        self.obs_dim       = obs_dim
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self._ptr = 0
        self._size = 0
        self._init_storage()

    def _init_storage(self):
        EL, N, O, S, A = (self.episode_limit, self.n_agents,
                          self.obs_dim, self.state_dim, self.action_dim)
        C = self.capacity
        self.obs       = np.zeros((C, EL+1, N, O), dtype=np.float32)
        self.actions   = np.zeros((C, EL,   N),    dtype=np.int64)
        self.rewards   = np.zeros((C, EL,   N),    dtype=np.float32)
        self.states    = np.zeros((C, EL+1, S),    dtype=np.float32)
        self.dones     = np.zeros((C, EL),         dtype=np.float32)
        self.filled    = np.zeros((C, EL),         dtype=np.float32)

    def add(self, episode: dict):
        """Store one episode dictionary (obs, actions, rewards, states, dones, filled)."""
        T = episode["filled"].sum()
        p = self._ptr
        self.obs[p]     = episode["obs"]
        self.actions[p] = episode["actions"]
        self.rewards[p] = episode["rewards"]
        self.states[p]  = episode["states"]
        self.dones[p]   = episode["dones"]
        self.filled[p]  = episode["filled"]
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.choice(self._size, batch_size, replace=False)
        return {
            "obs":     self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "states":  self.states[idx],
            "dones":   self.dones[idx],
            "filled":  self.filled[idx],
        }

    def __len__(self):
        return self._size


class PrioritisedEpisodeBuffer(EpisodeBuffer):
    """Proportional PER with importance-sampling weights."""

    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha    = alpha
        self.beta     = beta
        self._priorities = np.ones(self.capacity, dtype=np.float32)

    def add(self, episode: dict, priority: Optional[float] = None):
        if priority is None:
            priority = self._priorities[:self._size].max() if self._size > 0 else 1.0
        self._priorities[self._ptr] = priority ** self.alpha
        super().add(episode)

    def sample(self, batch_size: int) -> dict:
        probs = self._priorities[:self._size]
        probs = probs / probs.sum()
        idx = np.random.choice(self._size, batch_size, replace=False, p=probs)

        # Importance-sampling weights
        N = self._size
        weights = (N * probs[idx]) ** (-self.beta)
        weights = weights / weights.max()

        batch = {
            "obs":     self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "states":  self.states[idx],
            "dones":   self.dones[idx],
            "filled":  self.filled[idx],
            "weights": weights.astype(np.float32),
            "indices": idx,
        }
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        self._priorities[indices] = (np.abs(td_errors) + 1e-6) ** self.alpha
