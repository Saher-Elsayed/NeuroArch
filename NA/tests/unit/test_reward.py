"""Unit tests for comfort-augmented reward function."""
import pytest, sys; sys.path.insert(0, "/home/claude/NA")
from marl.reward import ComfortAugmentedReward, BuildingState, RewardWeights, EnergyOnlyReward


def make_state(**kwargs):
    defaults = dict(
        energy_kwh=1.0, energy_baseline=1.5,
        comfort_classes=[2, 2, 2], zone_targets=[2, 2, 2],
        peak_kw=200, peak_limit_kw=312, ppd=10.0, ashrae_compliant=True
    )
    defaults.update(kwargs)
    return BuildingState(**defaults)


class TestComfortAugmentedReward:
    def test_positive_reward_neutral_comfort(self):
        fn = ComfortAugmentedReward()
        r, _ = fn(make_state())
        assert r > -0.5  # should not be terrible

    def test_peak_violation_penalised(self):
        fn = ComfortAugmentedReward()
        r_ok, _ = fn(make_state(peak_kw=200))
        r_bad,_ = fn(make_state(peak_kw=400))
        assert r_ok > r_bad

    def test_discomfort_penalised(self):
        fn = ComfortAugmentedReward()
        r_ok, _ = fn(make_state(comfort_classes=[2, 2, 2]))
        r_bad,_ = fn(make_state(comfort_classes=[0, 4, 0]))
        assert r_ok > r_bad

    def test_info_keys_present(self):
        fn = ComfortAugmentedReward()
        _, info = fn(make_state())
        for key in ["reward_total","r_energy","r_comfort","r_peak","r_bonus","c_score"]:
            assert key in info

    def test_bonus_after_sustained_comfort(self):
        fn = ComfortAugmentedReward(comfort_window=3)
        for _ in range(4):
            r, info = fn(make_state(ashrae_compliant=True, energy_kwh=1.0))
        assert info["r_bonus"] > 0.0

    def test_reward_in_range(self):
        fn = ComfortAugmentedReward()
        for _ in range(100):
            import random
            r, _ = fn(make_state(
                energy_kwh=random.uniform(0, 3),
                comfort_classes=[random.randint(0,4)]*3,
                peak_kw=random.uniform(100, 500),
                ppd=random.uniform(5, 50),
            ))
            assert -2.0 <= r <= 1.0

    def test_reset_clears_history(self):
        fn = ComfortAugmentedReward(comfort_window=2)
        fn(make_state()); fn(make_state()); fn(make_state())
        fn.reset()
        assert len(fn._comfort_history) == 0


class TestEnergyOnlyReward:
    def test_energy_only(self):
        fn = EnergyOnlyReward()
        r, _ = fn(make_state(energy_kwh=0.0))
        assert r == 0.0
        r2, _ = fn(make_state(energy_kwh=500.0))
        assert r2 < 0.0
