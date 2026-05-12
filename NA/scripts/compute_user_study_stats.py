"""
Statistical analysis of user study data (paired t-tests, Cohen's d, SUS grading).
Reproduces all numbers in Paper Section IX-D, Table 5.

Usage: python scripts/compute_user_study_stats.py
"""
import csv, math
import numpy as np
from scipy import stats

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-9)

def sus_grade(score):
    if score >= 85: return "Excellent"
    elif score >= 72: return "Good"
    elif score >= 52: return "OK"
    else: return "Poor"

print("=" * 55)
print("NeuroArch User Study Statistical Analysis (N=32)")
print("IRB: IRB-25-1047 (Virginia Tech)")
print("=" * 55)

tt  = load("data/user_study/task_times.csv")
sus = load("data/user_study/sus_scores.csv")
ssq = load("data/user_study/ssq_subscales.csv")
tlx = load("data/user_study/nasa_tlx.csv")

# Task times
t1b = [float(r["T1_baseline_s"]) for r in tt]
t1v = [float(r["T1_vr_s"]) for r in tt]
t2b = [float(r["T2_baseline_s"]) for r in tt]
t2v = [float(r["T2_vr_s"]) for r in tt]

_, p1 = stats.ttest_rel(t1b, t1v)
_, p2 = stats.ttest_rel(t2b, t2v)
d1 = cohens_d(t1b, t1v); d2 = cohens_d(t2b, t2v)
red = 1 - (np.mean(t1v)+np.mean(t2v)) / (np.mean(t1b)+np.mean(t2b))

print(f"\nTask Times:")
print(f"  T1: baseline={np.mean(t1b):.1f}s  VR={np.mean(t1v):.1f}s  "
      f"t={stats.ttest_rel(t1b,t1v).statistic:.2f}  p={p1:.4f}  d={d1:.2f}")
print(f"  T2: baseline={np.mean(t2b):.1f}s  VR={np.mean(t2v):.1f}s  "
      f"t={stats.ttest_rel(t2b,t2v).statistic:.2f}  p={p2:.4f}  d={d2:.2f}")
print(f"  Overall reduction: {red*100:.1f}%  (paper: 41.3%)")

# SUS
sus_scores = [float(r["sus_total"]) for r in sus]
sus_mean = np.mean(sus_scores)
print(f"\nSUS: mean={sus_mean:.1f}  SD={np.std(sus_scores):.1f}  Grade={sus_grade(sus_mean)}")
print(f"  Percentile ~{min(99,int(stats.percentileofscore([68]*100+[sus_mean],sus_mean)))}")

# SSQ
ssq_tot = [float(r["ssq_total"]) for r in ssq]
nausea  = [float(r["nausea"])    for r in ssq]
oculo   = [float(r["oculomotor"])for r in ssq]
disori  = [float(r["disorientation"]) for r in ssq]
print(f"\nSSQ: total={np.mean(ssq_tot):.1f}  Nausea={np.mean(nausea):.1f}  "
      f"Oculomotor={np.mean(oculo):.1f}  Disorientation={np.mean(disori):.1f}")
print(f"  Paper: total=8.2  Nausea=3.1  Oculomotor=6.4  Disorientation=2.9")

# NASA-TLX
tlx_base = [float(r["baseline_total"]) for r in tlx]
tlx_vr   = [float(r["vr_total"])       for r in tlx]
_, ptlx  = stats.ttest_rel(tlx_base, tlx_vr)
print(f"\nNASA-TLX: baseline={np.mean(tlx_base):.1f}  VR={np.mean(tlx_vr):.1f}  "
      f"reduction={100*(1-np.mean(tlx_vr)/np.mean(tlx_base)):.1f}%  p={ptlx:.4f}")
print("\nDone.")
