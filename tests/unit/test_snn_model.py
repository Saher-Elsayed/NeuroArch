"""Unit tests for NeuroArchSNN model."""
import pytest, torch
import sys; sys.path.insert(0, "/home/claude/NA")
from snn.model import NeuroArchSNN, SNNConfig, LIFLayer, FastSigmoid


class TestFastSigmoid:
    def test_forward_above_threshold(self):
        u = torch.tensor([1.5, 2.0, 0.5])
        out = FastSigmoid.apply(u, 1.0, 25.0)
        assert out[0] == 1.0 and out[1] == 1.0 and out[2] == 0.0

    def test_forward_at_threshold(self):
        u = torch.tensor([1.0])
        out = FastSigmoid.apply(u, 1.0, 25.0)
        assert out[0] == 1.0

    def test_gradient_nonzero(self):
        u = torch.tensor([0.9], requires_grad=True)
        out = FastSigmoid.apply(u, 1.0, 25.0)
        out.sum().backward()
        assert u.grad is not None


class TestLIFLayer:
    def test_output_shape(self):
        layer = LIFLayer(14, 64)
        x = torch.randn(4, 14)
        spikes, v_mem = layer(x)
        assert spikes.shape == (4, 64)
        assert v_mem.shape == (4, 64)

    def test_binary_spikes(self):
        layer = LIFLayer(10, 32)
        x = torch.randn(8, 10)
        spikes, _ = layer(x)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_membrane_reset(self):
        layer = LIFLayer(5, 8, v_th=0.1)  # low threshold
        x = torch.ones(2, 5) * 10  # large input
        spikes, v_mem = layer(x)
        # Where spikes fired, v_mem should be reset
        assert (v_mem[spikes == 1] == layer.v_reset).all()

    def test_state_persistence(self):
        layer = LIFLayer(10, 20)
        x = torch.randn(4, 10)
        _, v1 = layer(x)
        _, v2 = layer(x, v1)
        assert not torch.allclose(v1, v2)


class TestNeuroArchSNN:
    @pytest.fixture
    def model(self):
        return NeuroArchSNN(SNNConfig(T=10))  # shorter T for testing

    def test_output_shape(self, model):
        x = torch.randn(8, 10, 14)
        out = model(x)
        assert out.shape == (8, 5)

    def test_batch_size_1(self, model):
        x = torch.randn(1, 10, 14)
        out = model(x)
        assert out.shape == (1, 5)

    def test_rate_reg_loss(self, model):
        x = torch.randn(4, 10, 14)
        _ = model(x)
        reg = model.rate_regularization_loss()
        assert reg.item() >= 0.0

    def test_spike_rates_populated(self, model):
        x = torch.randn(4, 10, 14)
        model(x)
        assert hasattr(model, "_last_spike_rates")
        assert len(model._last_spike_rates) == 3  # 3 hidden layers

    def test_synapse_count(self, model):
        stats = model.count_synapses()
        assert stats["total"] > 0
        assert 0 <= stats["sparsity"] <= 1.0

    def test_membrane_traces_shape(self, model):
        x = torch.randn(2, 10, 14)
        traces = model.get_membrane_traces(x)
        assert len(traces) == 3
        assert traces[0].shape == (2, 10, 64)
        assert traces[1].shape == (2, 10, 32)

    def test_gradient_flow(self, model):
        x = torch.randn(4, 10, 14)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_wrong_channel_count_raises(self, model):
        x = torch.randn(4, 10, 7)  # wrong C
        with pytest.raises(AssertionError):
            model(x)

    def test_n_parameters(self, model):
        n = model.n_parameters
        assert n > 0 and n < 1_000_000

    def test_deterministic_eval(self, model):
        model.eval()
        x = torch.randn(2, 10, 14)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)
