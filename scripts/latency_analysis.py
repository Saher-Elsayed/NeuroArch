"""
Analyse VR pipeline latency distribution — matches Paper Section IX-E.
Computes: mean, std, p50, p95, p99 per component and total.
Usage: python scripts/latency_analysis.py
"""
import csv
import numpy as np

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

rows = load("data/latency/frame_latency_10000.csv")
print(f"Frames analysed: {len(rows):,}")
print(f"{'Component':<22} {'Mean':>6} {'Std':>6} {'p50':>6} {'p95':>6} {'p99':>6} ms")
print("-" * 55)
for col, label in [("T_SNN_ms","SNN inference"),("T_BIM_ms","BIM update"),
                    ("T_WS_ms","WebSocket"), ("T_UE5_ms","UE5 render"),
                    ("total_ms","TOTAL")]:
    vals = np.array([float(r[col]) for r in rows])
    line = (f"{label:<22} {vals.mean():>6.1f} {vals.std():>6.1f} "
            f"{np.percentile(vals,50):>6.1f} {np.percentile(vals,95):>6.1f} "
            f"{np.percentile(vals,99):>6.1f}")
    if col == "total_ms":
        met = 100*sum(float(r["budget_met"]) for r in rows)/len(rows)
        line += f"  (budget 33.3ms: {met:.1f}% met)"
    print(line)
print()
print("Paper: mean 20.3ms, SNN 4.1ms, VR budget 33.3ms (30Hz)")
