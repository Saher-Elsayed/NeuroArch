#!/bin/bash
# Reproduce Table 4: Annual energy and comfort comparison
set -e
echo "=== Table 4: Annual Energy and Comfort ==="
python3 -c "
import csv
rows = list(csv.DictReader(open('data/energyplus/medium_office/controller_comparison.csv')))
print(f'{chr(10)}{"Controller":<25} {"kWh/m2":>8} {"Saving%":>8} {"Compl%":>8} {"PPD%":>6} {"Peak kW":>8}')
print('-'*70)
for r in rows:
    print(f'{r["controller"]:<25} {float(r["annual_kWh_m2"]):>8.1f} {float(r["energy_saving_pct"]):>8.1f} {float(r["ashrae55_compliance_pct"]):>8.1f} {float(r["mean_ppd_pct"]):>6.1f} {int(r["peak_demand_kW"]):>8}')
"
