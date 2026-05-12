"""Pytest configuration and shared fixtures."""
import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_seed():
    """Ensure reproducibility across all tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def mock_sensor_batch():
    return torch.rand(8, 14)


@pytest.fixture
def mock_spike_batch(mock_sensor_batch):
    from snn.rate_encoder import rate_encode
    return rate_encode(mock_sensor_batch, T=20)


@pytest.fixture
def lif_model():
    from snn.model import LIFComfortClassifier
    return LIFComfortClassifier(T=20)
