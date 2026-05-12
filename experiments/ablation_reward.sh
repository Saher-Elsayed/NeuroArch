#!/bin/bash
# Table 8: Reward shaping ablation
set -e; echo "=== Table 8: Reward Shaping Ablation ==="
python3 -c "
import csv
rows=list(csv.DictReader(open('data/ablations/reward_shaping.csv')))
print(f'{"Config":<30} {"lE":>5} {"lC":>5} {"lP":>5} {"kWh/m2":>8} {"Compl%":>8} {"PPD%":>6}')
print('-'*70)
for r in rows:
    m=' *' if 'Full' in r['config'] else ''
    print(f'{r["config"]:<30} {float(r["lambda_E"]):>5.1f} {float(r["lambda_C"]):>5.1f} {float(r["lambda_P"]):>5.2f} {float(r["kWh_m2"]):>8.1f} {float(r["compliance_pct"]):>8.1f} {float(r["ppd_pct"]):>6.1f}{m}')
"
