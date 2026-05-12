"""
NeuroArch LIF-SNN Comfort Classifier
Architecture: Input(14) -> LIF(64,tau=10ms) -> LIF(32,tau=20ms) -> Output(5,tau=5ms)
Synapses: 14*64 + 64*32 + 32*5 = 896 + 2048 + 160 = 3,104
Paper: Section V, Table 2
"""
import torch
import torch.nn as nn


class _FastSigmoid(torch.autograd.Function):
    """Fast-sigmoid surrogate gradient (Neftci et al. IEEE SPM 2019, k=25)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad / (25.0 * x.abs() + 1.0) ** 2


class LIFLayer(nn.Module):
    def __init__(self, n_in, n_out, tau_m=10.0, dt=1.0):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out, bias=False)
        self.alpha = 1.0 - dt / tau_m
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(self.fc(x))
        self.v = self.alpha * self.v + self.fc(x)
        spike = _FastSigmoid.apply(self.v - 1.0)
        self.v = self.v * (1.0 - spike.detach())
        return spike


class LIFComfortClassifier(nn.Module):
    """
    4-layer LIF-SNN for 5-class ISO 7730 comfort classification.
    Input : (T, N, 14)  -- Poisson spike trains, T timesteps
    Output: (N, 5)      -- spike counts over window (WTA decision)
    """
    CLASSES = ["Cold", "Cool", "Neutral", "Warm", "Hot"]

    def __init__(self, T: int = 100):
        super().__init__()
        self.T = T
        self.lif1 = LIFLayer(14, 64, tau_m=10.0)
        self.lif2 = LIFLayer(64, 32, tau_m=20.0)
        self.lif3 = LIFLayer(32,  5, tau_m=5.0)
        self._n_params = sum(p.numel() for p in self.parameters())

    def reset(self):
        for lif in [self.lif1, self.lif2, self.lif3]:
            lif.reset()

    def forward(self, x: torch.Tensor):
        """
        x: (T, N, 14)
        Returns: spike_counts (N,5), all_spikes (T,N,5)
        """
        self.reset()
        all_spikes = []
        for t in range(x.shape[0]):
            h = self.lif1(x[t])
            h = self.lif2(h)
            s = self.lif3(h)
            all_spikes.append(s)
        spikes = torch.stack(all_spikes)    # (T, N, 5)
        return spikes.sum(0), spikes         # (N,5), (T,N,5)

    def predict(self, x):
        counts, _ = self.forward(x)
        return counts.argmax(dim=1)

    def confidence(self, x):
        """kappa_SNN: max softmax probability over spike counts."""
        counts, _ = self.forward(x)
        return counts.float().softmax(dim=1).max(dim=1).values

    def sparsity(self, x):
        """Mean fraction of zero spikes (proxy for energy efficiency)."""
        _, spikes = self.forward(x)
        return 1.0 - spikes.mean().item()

    @property
    def n_synapses(self):
        return self._n_params
