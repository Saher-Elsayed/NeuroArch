"""Tests for baseline controllers."""
import pytest
from marl.baselines.rule_based_g36 import RuleBasedG36
from marl.baselines.mpc_rom import MPCReducedOrder


def test_g36_occupied_setpoints():
    ctrl = RuleBasedG36()
    out = ctrl.act({"T_zone": 25.0, "hour_of_day": 12, "occupancy": 1})
    assert out["T_supply_C"] == 24.0  # cooling
    assert out["lighting_lux"] > 0


def test_g36_unoccupied():
    ctrl = RuleBasedG36()
    out = ctrl.act({"T_zone": 22.0, "hour_of_day": 2, "occupancy": 0})
    assert out["lighting_lux"] == 0.0


def test_mpc_predict_length():
    ctrl = MPCReducedOrder(horizon=4)
    traj = ctrl.predict(22.0, 30.0, 4)
    assert len(traj) == 5  # T_now + 4 steps


def test_mpc_supply_temp_in_range():
    ctrl = MPCReducedOrder()
    out = ctrl.act({"T_zone": 26.0, "T_outdoor": 32.0})
    assert 15.0 <= out["T_supply_C"] <= 25.0
