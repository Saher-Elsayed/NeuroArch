"""
Comfort-Augmented Reward Function (Eq. 12 in paper)
=====================================================
R = -w_e * E_norm - w_c * (1 - C_score) - w_p * P_penalty + w_b * Bonus

where:
  E_norm    = normalised energy consumption [0,1]
  C_score   = ASHRAE-55 comfort satisfaction rate [0,1]
  P_penalty = peak demand violation indicator
  Bonus     = bonus for sustained comfort + energy co-optimisation
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class RewardWeights:
    energy:  float = 0.40
    comfort: float = 0.45
    peak:    float = 0.10
    bonus:   float = 0.05


@dataclass
class BuildingState:
    """Current building state for reward computation."""
    energy_kwh:      float   # current interval energy (kWh)
    energy_baseline: float   # baseline energy for normalisation (kWh)
    comfort_classes: list    # per-zone SNN comfort prediction [0..4]
    zone_targets:    list    # target comfort class per zone (2 = Neutral)
    peak_kw:         float   # current demand
    peak_limit_kw:   float   # contractual demand limit
    ppd:             float   # Predicted Percentage Dissatisfied [0,100]
    ashrae_compliant: bool   # True if all zones satisfy ASHRAE-55


class ComfortAugmentedReward:
    """Compute the NeuroArch multi-objective reward signal.

    All outputs are in [-1, 1] by design for stable QMIX training.
    """

    def __init__(self, weights: RewardWeights = None,
                 energy_norm_max: float = 500.0,
                 comfort_window: int = 4,
                 ppd_threshold: float = 20.0):
        self.w = weights or RewardWeights()
        self.energy_norm_max = energy_norm_max
        self.comfort_window  = comfort_window
        self.ppd_threshold   = ppd_threshold
        self._comfort_history = []

    def __call__(self, state: BuildingState) -> tuple[float, dict]:
        """Compute reward and diagnostic breakdown.

        Returns
        -------
        reward : float in [-1.5, 1.5]
        info   : dict with component breakdowns
        """
        # 1. Energy term
        e_norm  = min(state.energy_kwh / self.energy_norm_max, 1.0)
        r_energy = -self.w.energy * e_norm

        # 2. Comfort term — penalise deviation from neutral (class 2)
        deviations = [abs(c - 2) for c in state.comfort_classes]
        max_dev    = 2.0
        c_score    = 1.0 - (np.mean(deviations) / max_dev)
        c_score    = float(np.clip(c_score, 0.0, 1.0))
        r_comfort  = -self.w.comfort * (1.0 - c_score)

        # 3. PPD penalty (steeper for PPD > threshold)
        ppd_excess = max(state.ppd - self.ppd_threshold, 0.0)
        r_ppd      = -0.005 * ppd_excess  # small auxiliary term

        # 4. Peak demand penalty
        if state.peak_kw > state.peak_limit_kw:
            overshoot   = (state.peak_kw - state.peak_limit_kw) / state.peak_limit_kw
            r_peak      = -self.w.peak * min(overshoot, 1.0)
        else:
            r_peak = 0.0

        # 5. Co-optimisation bonus: ASHRAE-55 compliant AND energy saved
        self._comfort_history.append(state.ashrae_compliant)
        if len(self._comfort_history) > self.comfort_window:
            self._comfort_history.pop(0)

        sustained_comfort = all(self._comfort_history)
        energy_saved = state.energy_kwh < state.energy_baseline
        r_bonus = self.w.bonus if (sustained_comfort and energy_saved) else 0.0

        reward = r_energy + r_comfort + r_ppd + r_peak + r_bonus

        info = {
            "reward_total":   reward,
            "r_energy":       r_energy,
            "r_comfort":      r_comfort,
            "r_ppd":          r_ppd,
            "r_peak":         r_peak,
            "r_bonus":        r_bonus,
            "c_score":        c_score,
            "e_norm":         e_norm,
            "ashrae_ok":      state.ashrae_compliant,
        }
        return reward, info

    def reset(self):
        self._comfort_history.clear()


class SparseComfortReward:
    """Sparse episode-level reward baseline for ablation studies."""

    def __call__(self, state: BuildingState) -> tuple[float, dict]:
        if state.ashrae_compliant:
            r = 1.0 - min(state.energy_kwh / 500.0, 1.0)
        else:
            r = -1.0
        return r, {"reward_total": r}


class EnergyOnlyReward:
    """Energy-only reward baseline (no comfort) for ablation studies."""

    def __call__(self, state: BuildingState) -> tuple[float, dict]:
        r = -min(state.energy_kwh / 500.0, 1.0)
        return r, {"reward_total": r}
