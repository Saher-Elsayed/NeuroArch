"""
NeuroArch LIF-SNN Comfort Classifier
4-layer fully-connected LIF network, surrogate gradient training.
Input(14) -> Hidden1(64,tau=10ms) -> Hidden2(32,tau=20ms) -> Output(5,tau=5ms)
Paper: Table 2, Section V
"""
import torch
import torch.nn as nn


class LIFNeuron(nn.Module):
    """Leaky integrate-and-fire neuron with fast-sigmoid surrogate gradient."""
    def __init__(self, tau_m: float = 10.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0, dt: float = 1.0):
        super().__init__()
        self.alpha = 1.0 - dt / tau_m   # decay factor
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v = None                    # membrane potential

    def reset(self):
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x)
        # Leaky integration: V[n+1] = alpha*V[n] + I[n]
        self.v = self.alpha * self.v + x
        # Spike generation with surrogate gradient
        spike = _FastSigmoid.apply(self.v - self.v_threshold)
        # Reset
        self.v = self.v * (1.0 - spike.detach()) + self.v_reset * spike.detach()
        return spike


class _FastSigmoid(torch.autograd.Function):
    """Fast-sigmoid surrogate, k=25 (Neftci et al. IEEE SPM 2019)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad / (25.0 * x.abs() + 1.0) ** 2


class LIFComfortClassifier(nn.Module):
    """
    4-layer LIF-SNN for 5-class ISO 7730 comfort classification.
    Args:
        T: inference window length in timesteps (default 100 = 100 ms)
    Input:  (T, N, 14)  spike train tensor
    Output: (N, 5)      spike counts (winner-take-all for class decision)
    """
    def __init__(self, T: int = 100):
        super().__init__()
        self.T = T
        self.fc1  = nn.Linear(14, 64, bias=False)
        self.lif1 = LIFNeuron(tau_m=10.0)
        self.fc2  = nn.Linear(64, 32, bias=False)
        self.lif2 = LIFNeuron(tau_m=20.0)
        self.fc3  = nn.Linear(32, 5,  bias=False)
        self.lif3 = LIFNeuron(tau_m=5.0)

    def _reset(self):
        for m in [self.lif1, self.lif2, self.lif3]:
            m.reset()

    def forward(self, x: torch.Tensor):
        """
        x: (T, N, 14) spike trains
        Returns: spike_counts (N,5), all_spikes (T, N, 5)
        """
        self._reset()
        all_spikes = []
        for t in range(x.shape[0]):
            h = self.lif1(self.fc1(x[t]))
            h = self.lif2(self.fc2(h))
            s = self.lif3(self.fc3(h))
            all_spikes.append(s)
        all_spikes = torch.stack(all_spikes, dim=0)  # (T, N, 5)
        return all_spikes.sum(0), all_spikes          # (N,5), (T,N,5)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        counts, _ = self.forward(x)
        return counts.argmax(dim=1)

    def confidence(self, x: torch.Tensor) -> torch.Tensor:
        """kappa_SNN: max softmax over spike counts."""
        counts, _ = self.forward(x)
        return torch.softmax(counts.float(), dim=1).max(dim=1).values
