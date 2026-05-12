#!/bin/bash
# Table 5: User study statistics
set -e; echo "=== Table 5: User Study ==="
python3 -c "
import csv, numpy as np
from scipy import stats
tt=list(csv.DictReader(open('data/user_study/task_times.csv')))
t1b=[float(r['T1_baseline_s']) for r in tt]; t1v=[float(r['T1_vr_s']) for r in tt]
t2b=[float(r['T2_baseline_s']) for r in tt]; t2v=[float(r['T2_vr_s']) for r in tt]
_,p1=stats.ttest_rel(t1b,t1v); _,p2=stats.ttest_rel(t2b,t2v)
d1=(np.mean(t1b)-np.mean(t1v))/np.std(np.array(t1b)-np.array(t1v),ddof=1)
d2=(np.mean(t2b)-np.mean(t2v))/np.std(np.array(t2b)-np.array(t2v),ddof=1)
red=1-(np.mean(t1v)+np.mean(t2v))/(np.mean(t1b)+np.mean(t2b))
sus=list(csv.DictReader(open('data/user_study/sus_scores.csv')))
ssq=list(csv.DictReader(open('data/user_study/ssq_subscales.csv')))
print(f'T1: Baseline={np.mean(t1b):.1f}s  VR={np.mean(t1v):.1f}s  p={p1:.4f}  d={d1:.2f}')
print(f'T2: Baseline={np.mean(t2b):.1f}s  VR={np.mean(t2v):.1f}s  p={p2:.4f}  d={d2:.2f}')
print(f'Task-time reduction: {red*100:.1f}%  (paper: 41.3%)')
print(f'SUS mean: {np.mean([float(r["sus_total"]) for r in sus]):.1f}  (paper: 81.3)')
print(f'SSQ total: {np.mean([float(r["ssq_total"]) for r in ssq]):.1f}  (paper: 8.2)')
"
