"""Unit tests for NeuroArchEnv."""
import numpy as np
import pytest
from envs.energyplus_env import NeuroArchEnv


def test_env_reset():
    env = NeuroArchEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (6, 5)
    assert "building" in info


def test_env_step():
    env = NeuroArchEnv()
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    assert obs.shape == (6, 5)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_env_action_space():
    env = NeuroArchEnv()
    assert env.action_space.nvec.tolist() == [21, 21, 21, 10, 11, 11]


def test_env_state_dim():
    env = NeuroArchEnv()
    env.reset()
    state = env.get_state()
    assert state.shape == (33,)


def test_decode_action():
    env = NeuroArchEnv()
    action = np.array([0, 10, 20, 5, 0, 11-1])
    setpoints = env._decode_action(action)
    assert setpoints["HVAC_1"] == pytest.approx(15.0)
    assert setpoints["HVAC_2"] == pytest.approx(20.0)
    assert setpoints["HVAC_3"] == pytest.approx(25.0)
    assert setpoints["Light"]  == pytest.approx(550.0)
