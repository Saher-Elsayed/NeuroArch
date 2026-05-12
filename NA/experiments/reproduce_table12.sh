#!/bin/bash
# Table 12: Pareto frontier reward weight sweep
set -e; echo "=== Table 12: Pareto Frontier ==="
python3 -c "
import csv
rows=list(csv.DictReader(open('data/pareto/pareto_frontier.csv')))
print(f'{"lambda_C":>10} {"lambda_P":>10} {"kWh/m2":>8} {"Compl%":>8} {"PPD%":>6} {"Peak kW":>8}')
print('-'*55)
for r in rows:
    marker=' <- NeuroArch' if float(r['lambda_C'])==2.0 else ''
    print(f'{float(r["lambda_C"]):>10.1f} {float(r["lambda_P"]):>10.2f} {float(r["kWh_m2"]):>8.1f} {float(r["compliance_pct"]):>8.1f} {float(r["ppd_pct"]):>6.1f} {int(r["peak_kW"]):>8}{marker}')
"
