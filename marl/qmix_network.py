"""
QMIX monotonic mixing network.
Ensures dQ_tot/dQ_i >= 0 for all i via absolute-value weight constraint.
Paper: Section XI-A, Rashid et al. ICML 2018.
"""
import torch, torch.nn as nn


class QMIXMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        # Hypernetwork: state -> mixing weights (absolute value enforced)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents * hidden)
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.elu = nn.ELU()

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor):
        """
        agent_qs: (B, n_agents)
        state:    (B, state_dim)
        returns:  Q_tot (B, 1)
        """
        B = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, -1)
        b1 = self.hyper_b1(state).view(B, 1, -1)
        h  = self.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # (B,1,H)
        w2 = torch.abs(self.hyper_w2(state)).view(B, -1, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        return (torch.bmm(h, w2) + b2).squeeze(-1)                 # (B,1)
