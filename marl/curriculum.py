"""
Curriculum Learning for QMIX
==============================
Progressively increases task difficulty: start with a single building and
simple setpoint ranges, then unlock more agents, wider action spaces, and
cross-climate calibration challenges.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class CurriculumStage:
    name: str
    n_agents_active: int
    action_ranges: List[tuple]       # (lo, hi, step) per active agent
    episode_reward_threshold: float  # mean reward to advance
    min_episodes: int = 500
    buildings: List[str] = field(default_factory=lambda: ["medium_office"])


CURRICULUM = [
    CurriculumStage(
        name="Stage-1: HVAC only, 1 building",
        n_agents_active=3,
        action_ranges=[(18, 24, 1.0)] * 3,
        episode_reward_threshold=-0.6,
        min_episodes=500,
    ),
    CurriculumStage(
        name="Stage-2: Add lighting agent",
        n_agents_active=4,
        action_ranges=[(15, 25, 0.5)] * 3 + [(300, 750, 50)],
        episode_reward_threshold=-0.45,
        min_episodes=500,
        buildings=["medium_office"],
    ),
    CurriculumStage(
        name="Stage-3: Full 6 agents, 2 buildings",
        n_agents_active=6,
        action_ranges=[(15, 25, 0.5)] * 3 + [(300, 750, 50)] + [(0, 1, 0.1)] * 2,
        episode_reward_threshold=-0.30,
        min_episodes=1000,
        buildings=["medium_office", "residential"],
    ),
    CurriculumStage(
        name="Stage-4: Cross-climate generalization",
        n_agents_active=6,
        action_ranges=[(15, 25, 0.5)] * 3 + [(300, 750, 50)] + [(0, 1, 0.1)] * 2,
        episode_reward_threshold=-0.20,
        min_episodes=2000,
        buildings=["medium_office", "residential", "mixed_use"],
    ),
]


class CurriculumScheduler:
    """Manages stage progression based on rolling reward window."""

    def __init__(self, window: int = 50):
        self.window = window
        self.stage_idx = 0
        self._rewards = []

    @property
    def current_stage(self) -> CurriculumStage:
        return CURRICULUM[min(self.stage_idx, len(CURRICULUM) - 1)]

    def step(self, episode_reward: float) -> bool:
        """Returns True if stage advanced."""
        self._rewards.append(episode_reward)
        if len(self._rewards) > self.window:
            self._rewards.pop(0)

        stage = self.current_stage
        if (len(self._rewards) >= self.window and
                np.mean(self._rewards) >= stage.episode_reward_threshold and
                len(self._rewards) >= stage.min_episodes // 10 and
                self.stage_idx < len(CURRICULUM) - 1):
            self.stage_idx += 1
            self._rewards.clear()
            return True
        return False
