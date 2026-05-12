#!/bin/bash
# Table 9: Cross-climate validation
set -e; echo "=== Table 9: Cross-Climate Validation ==="
python3 -c "
import csv
rows=list(csv.DictReader(open('data/cross_climate/lobo_results.csv')))
print(f'{"Climate":<20} {"Building":<18} {"Cal wks":>8} {"LOBO acc%":>10} {"kWh/m2":>8} {"Save%":>7}')
print('-'*75)
for r in rows:
    print(f'{r["climate"]:<20} {r["building"]:<18} {int(r["n_cal_weeks"]):>8} {float(r["lobo_acc_pct"]):>10.1f} {float(r["annual_kWh_m2"]):>8.1f} {float(r["energy_saving_pct"]):>7.1f}')
"
