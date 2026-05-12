# %% [markdown]
# # Energy Consumption Analysis — NeuroArch vs Baselines
# Reproduces Table 4 and Figure 12.

# %% Load data
import sys; sys.path.insert(0, "..")
import pandas as pd, numpy as np, matplotlib.pyplot as plt

building = "medium_office"
ctrl = pd.read_csv(f"../data/energyplus/{building}/controller_comparison.csv")
monthly = pd.read_csv(f"../data/energyplus/{building}/end_use_monthly.csv")
sim = pd.read_csv(f"../data/energyplus/{building}/simulation_8760h.csv")

# Table 4 reproduction
print("Table 4 — Controller Comparison (Medium Office):")
print(ctrl.to_string(index=False))

# Figure 12 - Monthly energy breakdown
fig, ax = plt.subplots(figsize=(12, 6))
months = monthly.month
x = np.arange(len(months))
w = 0.15
ax.bar(x - 2*w, monthly.heating_kwh_m2,  w, label="Heating",  color='#d62728')
ax.bar(x - 1*w, monthly.cooling_kwh_m2,  w, label="Cooling",  color='#1f77b4')
ax.bar(x + 0*w, monthly.fans_kwh_m2,     w, label="Fans",     color='#aec7e8')
ax.bar(x + 1*w, monthly.lighting_kwh_m2, w, label="Lighting", color='#ffbb78')
ax.bar(x + 2*w, monthly.other_kwh_m2,    w, label="Other",    color='#98df8a')
ax.set_xticks(x); ax.set_xticklabels(months)
ax.set_ylabel("Energy (kWh/m²)"); ax.legend()
ax.set_title("Monthly End-Use Breakdown — Medium Office (Figure 12)")
plt.tight_layout(); plt.savefig("../data/energyplus/medium_office/monthly_breakdown.png", dpi=150)
plt.show()

# Demand profile: Rule-Based vs NeuroArch
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(sim.hour[:168*4], sim.hvac_power_kw[:168*4],    alpha=0.6, label="Rule-Based Baseline")
ax.plot(sim.hour[:168*4], sim.hvac_power_kw_na[:168*4], alpha=0.8, label="NeuroArch (QMIX)", linewidth=1.5)
ax.axhline(312, color='red', linestyle='--', linewidth=1, label="Peak Limit (312 kW)")
ax.set_xlabel("Hour"); ax.set_ylabel("HVAC Power (kW)")
ax.set_title("4-Week Demand Profile: Baseline vs NeuroArch"); ax.legend()
plt.tight_layout(); plt.show()
