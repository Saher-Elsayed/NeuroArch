"""Validate all data files match paper numbers."""
import csv
import pytest
from pathlib import Path


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_controller_comparison_neuroarch():
    rows = load("data/energyplus/medium_office/controller_comparison.csv")
    na = next(r for r in rows if r["controller"] == "NeuroArch_QMIX")
    assert float(na["annual_kWh_m2"])             == pytest.approx(108.6, abs=0.1)
    assert float(na["energy_saving_pct"])          == pytest.approx(-23.7, abs=0.1)
    assert float(na["ashrae55_compliance_pct"])    == pytest.approx(91.3,  abs=0.1)
    assert float(na["mean_ppd_pct"])               == pytest.approx(10.4,  abs=0.1)
    assert int(na["peak_demand_kW"])               == 248


def test_simulation_8760h_length():
    rows = load("data/energyplus/medium_office/simulation_8760h.csv")
    assert len(rows) == 8760, f"Expected 8760 rows, got {len(rows)}"


def test_comfort_labels_count():
    rows = load("data/sensor_logs/comfort_labels.csv")
    assert len(rows) == 32000


def test_comfort_labels_class_distribution():
    rows = load("data/sensor_logs/comfort_labels.csv")
    from collections import Counter
    counts = Counter(int(r["comfort_class_int"]) for r in rows)
    total = len(rows)
    # Neutral should be ~61.4% of windows
    assert 55 < counts[2]/total*100 < 68


def test_user_study_n():
    rows = load("data/user_study/task_times.csv")
    assert len(rows) == 32


def test_user_study_vr_faster():
    rows = load("data/user_study/task_times.csv")
    t1_b = [float(r["T1_baseline_s"]) for r in rows]
    t1_v = [float(r["T1_vr_s"]) for r in rows]
    import numpy as np
    assert np.mean(t1_v) < np.mean(t1_b)


def test_pareto_frontier_count():
    rows = load("data/pareto/pareto_frontier.csv")
    assert len(rows) == 5


def test_lobo_results_count():
    rows = load("data/cross_climate/lobo_results.csv")
    assert len(rows) == 5


def test_snn_ablation_neuroarch_best_power():
    rows = load("data/ablations/snn_arch_ablation.csv")
    na   = next(r for r in rows if "NeuroArch" in r["config"])
    assert float(na["acc_pct"])    == pytest.approx(94.3, abs=0.1)
    assert float(na["power_mW"])   == pytest.approx(0.31, abs=0.01)
    assert int(na["sparsity_pct"]) == 79
    assert int(na["synapses"])     == 3104


def test_latency_budget_met():
    rows = load("data/latency/frame_latency_10000.csv")
    assert len(rows) == 10000
    met = sum(int(r["budget_met"]) for r in rows)
    assert met / len(rows) > 0.95  # >95% frames within 33.3ms budget


def test_sensor_importance_top():
    rows = load("data/ablations/sensor_importance.csv")
    top = sorted(rows, key=lambda r: float(r["acc_drop_pct"]), reverse=True)
    assert top[0]["name"] == "Temperature"
    assert float(top[0]["acc_drop_pct"]) == pytest.approx(12.3, abs=0.1)
