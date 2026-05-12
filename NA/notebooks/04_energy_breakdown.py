# ---
# jupyter:
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# # Notebook 04: Energy Breakdown Analysis
# Reproduces Paper Fig. 12 (Monthly End-Use)

# %%
import sys; sys.path.insert(0,'.')
import csv, numpy as np, matplotlib.pyplot as plt

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

rows = load("data/energyplus/medium_office/end_use_monthly.csv")
months=[r["month"] for r in rows]
hvac  =[float(r["HVAC_thermal_kWh_m2"]) for r in rows]
fans  =[float(r["fans_pumps_kWh_m2"])   for r in rows]
light =[float(r["interior_lighting_kWh_m2"]) for r in rows]
other =[float(r["other_kWh_m2"])        for r in rows]
total = sum(float(r["total_kWh_m2"]) for r in rows)

print(f"Annual total: {total:.1f} kWh/m²  (paper: 108.6)")
print(f"HVAC fraction: {sum(hvac)/total*100:.1f}%  (paper: 57.0%)")

x = range(len(months))
fig, ax = plt.subplots(figsize=(10,4))
ax.bar(x, hvac, label="HVAC Thermal", color="#1565C0")
ax.bar(x, fans, bottom=hvac, label="Fans+Pumps", color="#42A5F5")
b2=[a+b for a,b in zip(hvac,fans)]
ax.bar(x, light, bottom=b2, label="Lighting", color="#FDD835")
b3=[a+b for a,b in zip(b2,light)]
ax.bar(x, other, bottom=b3, label="Other",   color="#BDBDBD")
ax.set_xticks(x); ax.set_xticklabels(months, rotation=30, ha="right")
ax.set_ylabel("kWh/m²/month"); ax.set_title("Monthly End-Use (NeuroArch, Fig. 12)")
ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
