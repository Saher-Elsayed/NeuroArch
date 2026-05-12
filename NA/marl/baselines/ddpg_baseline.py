"""
DDPG (Deep Deterministic Policy Gradient) single-agent baseline.
Paper: Table 4 row "DDPG" — 118.6 kWh/m², -16.7%, 88.3% compliance.
Treats all 6 zones as one flat observation/action space.
"""
import torch
import torch.nn as nn


class DDPGActor(nn.Module):
    def __init__(self, obs_dim=30, act_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh()
        )
        # Act dim: 3 supply temps + 1 lighting + 2 blinds (all continuous)
    def forward(self, obs): return self.net(obs)


class DDPGCritic(nn.Module):
    def __init__(self, obs_dim=30, act_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+act_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),             nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, obs, act): return self.net(torch.cat([obs, act], dim=-1))
