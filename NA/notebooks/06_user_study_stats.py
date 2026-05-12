# ---
# jupyter:
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# # Notebook 06: User Study Statistics
# Reproduces Paper Table 5, Section IX-D

# %%
import sys; sys.path.insert(0,'.')
import csv, numpy as np
from scipy import stats

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-9)

tt  = load("data/user_study/task_times.csv")
sus = load("data/user_study/sus_scores.csv")
ssq = load("data/user_study/ssq_subscales.csv")

t1b = [float(r["T1_baseline_s"]) for r in tt]
t1v = [float(r["T1_vr_s"]) for r in tt]
t2b = [float(r["T2_baseline_s"]) for r in tt]
t2v = [float(r["T2_vr_s"]) for r in tt]

_, p1 = stats.ttest_rel(t1b, t1v); _, p2 = stats.ttest_rel(t2b, t2v)
print("N=32 participants  |  IRB-25-1047 (Virginia Tech)")
print(f"T1: baseline={np.mean(t1b):.1f}s  VR={np.mean(t1v):.1f}s  "
      f"p={p1:.4f}  d={cohens_d(t1b,t1v):.2f}")
print(f"T2: baseline={np.mean(t2b):.1f}s  VR={np.mean(t2v):.1f}s  "
      f"p={p2:.4f}  d={cohens_d(t2b,t2v):.2f}")
red = 1-(np.mean(t1v)+np.mean(t2v))/(np.mean(t1b)+np.mean(t2b))
print(f"Task-time reduction: {red*100:.1f}%  (paper: 41.3%)")
sus_scores = [float(r["sus_total"]) for r in sus]
print(f"SUS: {np.mean(sus_scores):.1f} SD={np.std(sus_scores):.1f}  (paper: 81.3)")
ssq_t = [float(r["ssq_total"]) for r in ssq]
print(f"SSQ: {np.mean(ssq_t):.1f}  Nausea={np.mean([float(r['nausea']) for r in ssq]):.1f}  (paper: 8.2)")
