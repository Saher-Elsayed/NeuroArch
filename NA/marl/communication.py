"""
CommNet-style communication module for NeuroArch MARL.
Agents share compressed hidden states via a differentiable communication channel.
Used in ablation study: CommNet vs. no communication.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CommunicationChannel(nn.Module):
    """Mean-field agent communication (CommNet).

    Each agent broadcasts a compressed embedding to all others,
    then receives the mean of all embeddings.
    """

    def __init__(self, n_agents: int, hidden_dim: int, comm_dim: int = 16):
        super().__init__()
        self.n_agents = n_agents
        self.encode = nn.Linear(hidden_dim, comm_dim)
        self.decode = nn.Linear(comm_dim, hidden_dim)

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        """
        hiddens : (B, n_agents, hidden_dim)
        returns : (B, n_agents, hidden_dim)  — comm-augmented hiddens
        """
        msgs = self.encode(hiddens)              # (B, N, comm_dim)
        mean_msg = msgs.mean(dim=1, keepdim=True)  # (B, 1, comm_dim)
        # Each agent receives mean of others (exclude self)
        sum_all = msgs.sum(dim=1, keepdim=True)
        comm    = (sum_all - msgs) / (self.n_agents - 1)
        return hiddens + self.decode(comm)       # residual connection
