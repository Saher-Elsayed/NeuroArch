#!/bin/bash
# Reproduce Table 5: User study statistics
set -e
echo "=== Table 5: User Study ==="
python3 -c "
import csv, numpy as np
tt = list(csv.DictReader(open('data/user_study/task_times.csv')))
t1b=[float(r['T1_baseline_s']) for r in tt]; t1v=[float(r['T1_vr_s']) for r in tt]
t2b=[float(r['T2_baseline_s']) for r in tt]; t2v=[float(r['T2_vr_s']) for r in tt]
d1=(np.mean(t1b)-np.mean(t1v))/np.std([a-b for a,b in zip(t1b,t1v)],ddof=1)
d2=(np.mean(t2b)-np.mean(t2v))/np.std([a-b for a,b in zip(t2b,t2v)],ddof=1)
print(f'T1: Baseline={np.mean(t1b):.1f}s VR={np.mean(t1v):.1f}s d={d1:.2f}')
print(f'T2: Baseline={np.mean(t2b):.1f}s VR={np.mean(t2v):.1f}s d={d2:.2f}')
red=1-(np.mean(t1v)+np.mean(t2v))/(np.mean(t1b)+np.mean(t2b))
print(f'Mean task-time reduction: {red*100:.1f}%')
"
