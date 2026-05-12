#!/bin/bash
# Table 8: Reward shaping ablation
echo "=== Table 8: Reward Shaping Ablation ==="
python3 -c "
rows=[('Energy only',104.1,84.2,21.3),
      ('+PMV',114.2,89.1,12.8),
      ('+PMV +kappa_SNN',109.8,90.4,11.6),
      ('Full NeuroArch',108.6,91.3,10.4)]
print(f'{chr(10)}{"Reward Config":<25} {"kWh/m2":>8} {"Compl%":>8} {"PPD%":>6}')
print('-'*52)
for r in rows:
    print(f'{r[0]:<25} {r[1]:>8.1f} {r[2]:>8.1f} {r[3]:>6.1f}')
"
