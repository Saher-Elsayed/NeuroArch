"""
QMIX Individual Agent Q-Network
================================
Each of the 6 NeuroArch agents maintains an independent Q-network with GRU
memory for partial observability. Observation dim: 5  |  Actions: 10-21 depending on agent.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AgentQNetwork(nn.Module):
    """Individual agent recurrent Q-network (DRQN-style).

    Architecture:
        obs -> Linear(64) -> GRU(64) -> Linear(64) -> Q-values
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 n_gru_layers: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc_in = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=n_gru_layers,
                          batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name and p.dim() >= 2:
                nn.init.kaiming_normal_(p)

    def forward(self, obs: torch.Tensor,
                h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : (B, T, obs_dim)  or  (B, obs_dim) for single step
        h   : (n_layers, B, hidden_dim)

        Returns
        -------
        q_values : (B, T, action_dim)  or  (B, action_dim)
        h_new    : (n_layers, B, hidden_dim)
        """
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)  # (B, 1, obs_dim)

        x = self.fc_in(obs)               # (B, T, hidden)
        x, h_new = self.gru(x, h)         # (B, T, hidden)
        q = self.fc_out(x)                # (B, T, action_dim)

        if single_step:
            q = q.squeeze(1)              # (B, action_dim)
        return q, h_new

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)


class DuelingAgentQNetwork(AgentQNetwork):
    """Dueling DRQN variant — separate value and advantage streams."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(obs_dim, action_dim, hidden_dim)
        # Override fc_out with dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.fc_out = None  # replaced

    def forward(self, obs: torch.Tensor,
                h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)

        x = self.fc_in(obs)
        x, h_new = self.gru(x, h)

        V = self.value_stream(x)                       # (B, T, 1)
        A = self.adv_stream(x)                         # (B, T, action_dim)
        q = V + A - A.mean(dim=-1, keepdim=True)       # dueling combination

        if single_step:
            q = q.squeeze(1)
        return q, h_new
