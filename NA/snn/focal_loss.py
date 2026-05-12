"""Multi-class Focal Loss for imbalanced comfort class training."""
import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p     = log_p.exp()
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t     = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal   = (1 - p_t) ** self.gamma * (-log_p_t)
        return focal.mean() if self.reduction == "mean" else focal.sum()
