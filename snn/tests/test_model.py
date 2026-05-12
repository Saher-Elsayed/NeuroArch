"""Unit tests for LIF-SNN model."""
import pytest
import torch
from snn.model import LIFComfortClassifier
from snn.rate_encoder import rate_encode
from snn.focal_loss import FocalLoss


def test_model_output_shape():
    model = LIFComfortClassifier(T=10)
    x = rate_encode(torch.rand(4, 14), T=10)
    counts, spikes = model(x)
    assert counts.shape == (4, 5)
    assert spikes.shape == (10, 4, 5)


def test_model_synapses():
    model = LIFComfortClassifier()
    # 14*64 + 64*32 + 32*5 = 896 + 2048 + 160 = 3104
    assert model.n_synapses == 3104, f"Got {model.n_synapses}"


def test_spike_values_binary():
    model = LIFComfortClassifier(T=20)
    x = rate_encode(torch.rand(8, 14), T=20)
    _, spikes = model(x)
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})


def test_predict_returns_valid_class():
    model = LIFComfortClassifier(T=10)
    x = rate_encode(torch.rand(16, 14), T=10)
    preds = model.predict(x)
    assert preds.shape == (16,)
    assert preds.min() >= 0 and preds.max() <= 4


def test_confidence_in_range():
    model = LIFComfortClassifier(T=10)
    x = rate_encode(torch.rand(8, 14), T=10)
    kappa = model.confidence(x)
    assert (kappa >= 0).all() and (kappa <= 1).all()


def test_sparsity_in_range():
    model = LIFComfortClassifier(T=50)
    x = rate_encode(torch.rand(16, 14), T=50)
    spar = model.sparsity(x)
    assert 0.0 <= spar <= 1.0


def test_reset_clears_state():
    model = LIFComfortClassifier(T=10)
    x = rate_encode(torch.rand(4, 14), T=10)
    model(x)
    model.reset()
    for lif in [model.lif1, model.lif2, model.lif3]:
        assert lif.v is None


def test_focal_loss_class_weights():
    y = torch.tensor([0]*10 + [1]*20 + [2]*100 + [3]*30 + [4]*15)
    w = FocalLoss.class_weights(y)
    assert w.shape == (5,)
    # Majority class should have lower weight
    assert w[2] < w[0]


def test_rate_encode_shape():
    s = torch.rand(8, 14)
    spikes = rate_encode(s, T=100)
    assert spikes.shape == (100, 8, 14)
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})


def test_rate_encode_mean_rate():
    # Mean firing rate should be close to r_max * dt * mean(s) = 0.1 * 0.5 = 0.05
    torch.manual_seed(0)
    s = torch.full((1000, 14), 0.5)
    spikes = rate_encode(s, T=100, r_max=100.0, dt=0.001)
    assert abs(spikes.mean().item() - 0.05) < 0.01
