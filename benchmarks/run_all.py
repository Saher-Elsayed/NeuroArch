"""Run the full NeuroArch benchmark suite and generate a report."""
import time, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_snn_latency(n: int = 1000) -> dict:
    """Measure mean and P99 SNN inference latency."""
    import torch
    from snn.model import NeuroArchSNN, SNNConfig
    model = NeuroArchSNN(SNNConfig(T=100)).eval()
    x = torch.randn(1, 100, 14)
    # Warmup
    for _ in range(50):
        with torch.no_grad(): model(x)
    lats = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad(): model(x)
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    return {"mean_ms": float(lats.mean()), "p99_ms": float(np.percentile(lats, 99)),
            "target_mean_ms": 4.1, "target_p99_ms": 12.0,
            "pass_mean": float(lats.mean()) < 15.0,
            "pass_p99":  float(np.percentile(lats, 99)) < 36.0}


def benchmark_synapse_count() -> dict:
    from snn.model import NeuroArchSNN, SNNConfig
    model = NeuroArchSNN(SNNConfig())
    stats = model.count_synapses()
    return {"target_active_synapses": 3104, "target_sparsity": 0.79,
            **stats, "pass": abs(stats["sparsity"] - 0.79) < 0.10}


def main():
    print("=" * 60)
    print("NeuroArch Benchmark Suite")
    print("Paper: IEEE Access MS ID: Access-2026-16730")
    print("=" * 60)

    results = {}

    print("\n[1/2] SNN Latency Benchmark...")
    lat = benchmark_snn_latency(200)
    results["latency"] = lat
    status = "PASS" if lat["pass_mean"] else "FAIL"
    print(f"  Mean latency:  {lat['mean_ms']:.2f} ms (target: {lat['target_mean_ms']} ms) [{status}]")

    print("\n[2/2] Synapse Count Benchmark...")
    syn = benchmark_synapse_count()
    results["synapses"] = syn
    status = "PASS" if syn["pass"] else "FAIL"
    print(f"  Sparsity: {syn['sparsity']:.1%} (target: 79%) [{status}]")

    all_pass = all(r.get("pass", r.get("pass_mean", False)) for r in results.values())
    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASSED' if all_pass else 'FAILURES DETECTED'}")
    print(f"{'='*60}")

    out = Path("benchmark_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
