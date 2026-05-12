"""
Integration tests: SNN -> reward -> QMIX pipeline.
These run end-to-end with mock data; ~30 seconds.
"""
import torch
import numpy as np
import pytest
from snn.model import LIFComfortClassifier
from snn.rate_encoder import rate_encode
from marl.agent import QAgent
from marl.qmix_network import QMIXMixer
from marl.reward import ComfortAugmentedReward
from marl.replay_buffer import ReplayBuffer
from envs.energyplus_env import NeuroArchEnv


def test_snn_to_reward_pipeline():
    """SNN confidence -> MARL reward computation."""
    model = LIFComfortClassifier(T=20)
    sensor_batch = torch.rand(6, 14)  # 6 zones
    spikes = rate_encode(sensor_batch, T=20)
    kappa  = model.confidence(spikes)          # (6,)
    # Mock PMV and energy
    pmv    = torch.randn(6) * 0.3
    ppd    = torch.clamp(5 + pmv.abs() * 20, 5, 100)
    energy = torch.ones(6) * 0.1
    rf     = ComfortAugmentedReward()
    reward = rf(energy, kappa, pmv, ppd)
    assert isinstance(reward, float)
    assert -100 < reward < 100


def test_qmix_forward_pass():
    """QMIX agent + mixer forward pass with realistic dims."""
    agents = [QAgent(5, na) for na in [21,21,21,10,11,11]]
    mixer  = QMIXMixer(n_agents=6, state_dim=33)
    obs    = torch.rand(32, 6, 5)   # batch=32, agents=6, obs=5
    state  = torch.rand(32, 33)
    qs = torch.stack([ag(obs[:,i,:]) for i,ag in enumerate(agents)], dim=1)
    # Take max Q per agent
    qs_max = qs.max(dim=2).values                 # (32, 6)
    q_tot  = mixer(qs_max, state)
    assert q_tot.shape == (32, 1)


def test_full_episode_mock():
    """Run one mock episode in the environment."""
    env = NeuroArchEnv(building="medium_office")
    obs, _ = env.reset(seed=42)
    total_r = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        total_r += r
        if done or trunc:
            break
    assert obs.shape == (6, 5)
    assert isinstance(total_r, float)


def test_replay_buffer_integration():
    """Fill buffer from mock env and sample batch."""
    env = NeuroArchEnv()
    buf = ReplayBuffer(capacity=200)
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        nobs, r, done, trunc, info = env.step(action)
        state = env.get_state()
        buf.push(obs.flatten(), action, r, nobs.flatten(),
                 int(done), state, state)
        obs = nobs
    batch = buf.sample(32)
    assert batch[0].shape[0] == 32
