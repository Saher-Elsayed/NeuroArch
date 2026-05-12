"""
Validate all data files against expected paper values.
Prints PASS/FAIL with tolerances. Exit code 0 = all pass.

Usage:
    python scripts/validate_data.py
"""
import csv, sys, os
import numpy as np

PASS = []; FAIL = []
def check(name, val, expected, tol=0.2):
    ok = abs(float(val) - float(expected)) <= tol
    (PASS if ok else FAIL).append(f"{'[PASS]' if ok else '[FAIL]'} {name}: got {val:.3f}, expected {expected} +/-{tol}")

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

# Table 4: NeuroArch controller
t4 = load("data/energyplus/medium_office/controller_comparison.csv")
na = next(r for r in t4 if r["controller"]=="NeuroArch_QMIX")
check("Table4 energy kWh/m2",  float(na["annual_kWh_m2"]),           108.6, 0.1)
check("Table4 saving pct",     float(na["energy_saving_pct"]),        -23.7, 0.1)
check("Table4 ASHRAE compl.",  float(na["ashrae55_compliance_pct"]),   91.3, 0.1)
check("Table4 PPD pct",        float(na["mean_ppd_pct"]),              10.4, 0.1)
check("Table4 peak kW",        int(na["peak_demand_kW"]),              248,  0.5)

# 8760 rows
rows_8760 = load("data/energyplus/medium_office/simulation_8760h.csv")
if len(rows_8760)==8760: PASS.append("[PASS] 8760h: 8760 rows")
else: FAIL.append(f"[FAIL] 8760h: {len(rows_8760)} rows (expected 8760)")

# User study N=32
us = load("data/user_study/task_times.csv")
if len(us)==32: PASS.append("[PASS] User study N=32")
else: FAIL.append(f"[FAIL] User study N={len(us)}")

# VR faster than baseline
t1b = np.mean([float(r["T1_baseline_s"]) for r in us])
t1v = np.mean([float(r["T1_vr_s"]) for r in us])
if t1v < t1b: PASS.append(f"[PASS] VR T1 faster ({t1v:.1f}s < {t1b:.1f}s)")
else: FAIL.append(f"[FAIL] VR T1 NOT faster: {t1v:.1f} vs {t1b:.1f}")

# Comfort labels 32000
cl = load("data/sensor_logs/comfort_labels.csv")
if len(cl)==32000: PASS.append("[PASS] Comfort labels: 32000 rows")
else: FAIL.append(f"[FAIL] Comfort labels: {len(cl)} rows")

# Ablation: NeuroArch row
ab = load("data/ablations/snn_arch_ablation.csv")
na_ab = next(r for r in ab if "NeuroArch" in r["config"])
check("Table7 NeuroArch acc",  float(na_ab["acc_pct"]),   94.3, 0.1)
check("Table7 NeuroArch mW",   float(na_ab["power_mW"]),   0.31, 0.01)
check("Table7 synapses",       int(na_ab["synapses"]),      3104, 0.5)

# Latency budget
lat = load("data/latency/frame_latency_10000.csv")
met = sum(int(r["budget_met"]) for r in lat) / len(lat)
if met > 0.95: PASS.append(f"[PASS] Latency budget met: {met*100:.1f}%")
else: FAIL.append(f"[FAIL] Latency budget met: {met*100:.1f}% (expected >95%)")

# Summary
print("\n" + "="*50)
print(f"Data validation: {len(PASS)} PASS, {len(FAIL)} FAIL")
print("="*50)
for r in PASS: print(r)
for r in FAIL: print(r)
sys.exit(0 if not FAIL else 1)
