"""Performance: verify SNN inference latency."""
import time, torch, numpy as np, pytest, sys
sys.path.insert(0, "/home/claude/NA")


class TestInferenceLatency:
    @pytest.fixture(scope="class")
    def model(self):
        from snn.model import NeuroArchSNN, SNNConfig
        m = NeuroArchSNN(SNNConfig(T=100)); m.eval(); return m

    def test_mean_latency_under_budget(self, model):
        x = torch.randn(1, 100, 14)
        for _ in range(5):
            with torch.no_grad(): model(x)
        lats = []
        for _ in range(50):
            t0 = time.perf_counter()
            with torch.no_grad(): model(x)
            lats.append((time.perf_counter() - t0) * 1000)
        mean_ms = np.mean(lats)
        # generous 3x budget for CPU CI runner
        assert mean_ms < 15.0, f"Mean {mean_ms:.2f}ms too high"

    def test_batch_throughput(self, model):
        x = torch.randn(16, 100, 14)
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.no_grad(): model(x)
        tput = (20 * 16) / (time.perf_counter() - t0)
        assert tput > 10, f"Throughput {tput:.0f} w/s too low"
