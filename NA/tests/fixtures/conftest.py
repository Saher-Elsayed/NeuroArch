"""Shared fixtures for all tests."""
import pytest, torch, numpy as np
import sys; sys.path.insert(0, "/home/claude/NA")


@pytest.fixture(scope="session")
def snn_model():
    from snn.model import NeuroArchSNN, SNNConfig
    m = NeuroArchSNN(SNNConfig(T=20)); m.eval(); return m

@pytest.fixture
def sensor_window():
    """Realistic 14-channel, 100-timestep sensor window."""
    rng = np.random.default_rng(0)
    t = np.arange(100) / 100
    w = np.stack([
        22 + 2*np.sin(2*np.pi*t) + rng.normal(0, 0.1, 100),  # air_temp
        23 + 1.5*np.sin(2*np.pi*t) + rng.normal(0, 0.1, 100), # radiant_temp
        50 + 5*np.sin(2*np.pi*t),     # humidity
        rng.exponential(0.1, 100),    # air_speed
        800 + 100*np.ones(100),        # co2
        500*np.ones(100),              # lux
        40*np.ones(100),               # sound_db
        0.8*np.ones(100),              # occupancy
        30*np.ones(100),               # outdoor_temp
        400*np.sin(np.pi*t)**2,        # solar_rad
        np.sin(2*np.pi*t),            # time_sin
        np.cos(2*np.pi*t),            # time_cos
        np.sin(2*np.pi*t*7/100),      # dow_sin
        np.cos(2*np.pi*t*7/100),      # dow_cos
    ], axis=-1).astype(np.float32)
    return w  # (100, 14)

@pytest.fixture
def comfort_state():
    from marl.reward import BuildingState
    return BuildingState(
        energy_kwh=1.2, energy_baseline=1.5,
        comfort_classes=[2,2,2], zone_targets=[2,2,2],
        peak_kw=250, peak_limit_kw=312, ppd=10.4, ashrae_compliant=True
    )
