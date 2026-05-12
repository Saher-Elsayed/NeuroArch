#!/bin/bash
# Table 7: SNN architecture ablation
set -e; echo "=== Table 7: SNN Architecture Ablation ==="
python3 -c "
import csv
rows=list(csv.DictReader(open('data/ablations/snn_arch_ablation.csv')))
print(f'{"Config":<35} {"Acc%":>6} {"mW":>6} {"Spar%":>7} {"Syn":>6}')
print('-'*65)
for r in rows:
    m=' *' if 'NeuroArch' in r['config'] else ''
    print(f'{r["config"]:<35} {float(r["acc_pct"]):>6.1f} {float(r["power_mW"]):>6.2f} {int(r["sparsity_pct"]):>7} {int(r["synapses"]):>6}{m}')
"
