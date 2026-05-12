"""
Poisson rate encoder.
s_i in [0,1]  ->  spike train of length T
r_i(t) = r_max * s_i,  r_max = 100 Hz,  dt = 1 ms
Paper: Eq. (10)-(11), Section IV-B
"""
import torch


def rate_encode(s: torch.Tensor, T: int = 100,
                r_max: float = 100.0, dt: float = 0.001) -> torch.Tensor:
    """
    Args:
        s: (N, 14) normalised sensor values in [0, 1]
        T: window length in timesteps
        r_max: maximum firing rate (Hz)
        dt: timestep duration (s)
    Returns:
        spikes: (T, N, 14) binary Bernoulli spike tensor
    """
    p = (r_max * dt * s).clamp(0.0, 1.0)          # spike probability per step
    return torch.bernoulli(p.unsqueeze(0).expand(T, -1, -1))
