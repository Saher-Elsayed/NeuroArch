"""
QMIX Mixing Network
====================
Hypernetwork-based monotonic mixing of individual Q-values into Q_total.
Ensures IGM (Individual-Global-Max) property via absolute-value weight constraint.

Paper: Rashid et al. (2018) QMIX: Monotonic Value Function Factorisation
       for Deep Multi-Agent Reinforcement Learning
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixNetwork(nn.Module):
    """Monotonic mixing network.

    Parameters
    ----------
    n_agents     : number of MARL agents (6)
    state_dim    : dimension of global state (33)
    embed_dim    : embedding dimension for hypernetworks (64)
    hypernet_layers : depth of hypernetworks
    """

    def __init__(self, n_agents: int = 6, state_dim: int = 33,
                 embed_dim: int = 64, hypernet_layers: int = 2):
        super().__init__()
        self.n_agents  = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork 1 -> weights for mixing layer 1
        self.hyper_w1 = self._make_hypernet(state_dim, n_agents * embed_dim, hypernet_layers)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork 2 -> weights for mixing layer 2
        self.hyper_w2 = self._make_hypernet(state_dim, embed_dim, hypernet_layers)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    @staticmethod
    def _make_hypernet(in_dim: int, out_dim: int, n_layers: int) -> nn.Module:
        layers = [nn.Linear(in_dim, 64), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(64, 64), nn.ReLU()]
        layers.append(nn.Linear(64, out_dim))
        return nn.Sequential(*layers)

    def forward(self, q_vals: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        q_vals : (B, T, n_agents)  individual Q-values
        states : (B, T, state_dim) global state

        Returns
        -------
        q_total : (B, T, 1)
        """
        B, T, n = q_vals.shape
        q_flat = q_vals.view(B * T, 1, n)          # (BT, 1, n_agents)
        s_flat = states.view(B * T, self.state_dim) # (BT, state_dim)

        # Layer 1
        w1 = self.hyper_w1(s_flat).abs()           # monotonic: abs constraint
        w1 = w1.view(B * T, n, self.embed_dim)
        b1 = self.hyper_b1(s_flat).unsqueeze(1)    # (BT, 1, embed)
        h = F.elu(torch.bmm(q_flat, w1) + b1)      # (BT, 1, embed)

        # Layer 2
        w2 = self.hyper_w2(s_flat).abs()
        w2 = w2.view(B * T, self.embed_dim, 1)
        b2 = self.hyper_b2(s_flat).unsqueeze(1)    # (BT, 1, 1)
        q_total = torch.bmm(h, w2) + b2            # (BT, 1, 1)

        return q_total.view(B, T, 1)


class VDNNetwork(nn.Module):
    """Value Decomposition Networks baseline (linear sum)."""

    def forward(self, q_vals: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return q_vals.sum(dim=-1, keepdim=True)


class QTRANNetwork(nn.Module):
    """Simplified QTRAN joint action-value network baseline."""

    def __init__(self, n_agents: int = 6, state_dim: int = 33,
                 action_dim: int = 21, hidden_dim: int = 64):
        super().__init__()
        self.joint_net = nn.Sequential(
            nn.Linear(state_dim + n_agents * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q_vals: torch.Tensor, states: torch.Tensor,
                actions_onehot: torch.Tensor) -> torch.Tensor:
        B, T, _ = states.shape
        inp = torch.cat([states, actions_onehot.view(B, T, -1)], dim=-1)
        return self.joint_net(inp)
