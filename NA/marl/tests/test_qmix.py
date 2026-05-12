"""Unit tests for QMIX components."""
import pytest
import torch
from marl.agent import QAgent
from marl.qmix_network import QMIXMixer
from marl.replay_buffer import ReplayBuffer
from marl.reward import ComfortAugmentedReward


def test_qagent_output_shape():
    agent = QAgent(obs_dim=5, n_actions=21)
    obs = torch.rand(8, 5)
    q = agent(obs)
    assert q.shape == (8, 21)


def test_qmix_monotonicity():
    mixer = QMIXMixer(n_agents=6, state_dim=33)
    B = 16
    qs = torch.rand(B, 6)
    state = torch.rand(B, 33)
    q_tot = mixer(qs, state)
    assert q_tot.shape == (B, 1)
    # Check monotonicity: increasing one agent Q should not decrease Q_tot
    qs2 = qs.clone(); qs2[:, 0] += 1.0
    q_tot2 = mixer(qs2, state)
    assert (q_tot2 >= q_tot - 1e-4).all(), "Monotonicity violated"


def test_replay_buffer_push_sample():
    buf = ReplayBuffer(capacity=100)
    for _ in range(50):
        buf.push(
            obs=[0.1]*5, actions=[1]*6, reward=1.0,
            next_obs=[0.2]*5, done=0, state=[0.0]*33, next_state=[0.1]*33
        )
    assert len(buf) == 50
    batch = buf.sample(16)
    assert len(batch) == 7


def test_reward_positive_in_comfort():
    rf = ComfortAugmentedReward()
    kappa = torch.ones(6) * 0.9
    pmv   = torch.zeros(6)
    energy = torch.ones(6) * 0.01
    ppd    = torch.ones(6) * 5.0
    r = rf(energy, kappa, pmv, ppd)
    assert r > 0, "Should be positive when all zones comfortable"


def test_reward_penalises_energy():
    rf = ComfortAugmentedReward()
    kappa = torch.ones(6) * 0.9; pmv = torch.zeros(6); ppd = torch.ones(6)*5
    r_low  = rf(torch.ones(6)*0.01, kappa, pmv, ppd)
    r_high = rf(torch.ones(6)*10.0,  kappa, pmv, ppd)
    assert r_low > r_high


def test_reward_components_sum():
    rf = ComfortAugmentedReward()
    kappa = torch.rand(6); pmv = torch.randn(6)*0.5; ppd = torch.rand(6)*20
    energy = torch.rand(6)
    comps = ComfortAugmentedReward.components(energy, kappa, pmv, ppd)
    assert set(comps.keys()) == {"r_energy", "r_comfort", "r_ppd"}
