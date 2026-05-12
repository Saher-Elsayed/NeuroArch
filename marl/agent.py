"""Individual Q-network agent for QMIX."""
import torch, torch.nn as nn

class QAgent(nn.Module):
    """2-layer MLP Q-network, obs_dim=5 (or 7 with comfort-neighbour)."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, obs):
        return self.net(obs)
