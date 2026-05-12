# %% [markdown]
# # NeuroArch MARL Evaluation — QMIX Training Convergence
# Reproduces Figure 6 from IEEE Access MS ID: Access-2026-16730

# %% Load convergence data
import sys; sys.path.insert(0, "..")
import pandas as pd, numpy as np, matplotlib.pyplot as plt

seeds = [42, 123, 456]
dfs = [pd.read_csv(f"../data/ablations/marl_convergence_seed{s}.csv") for s in seeds]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#1f77b4","#ff7f0e","#2ca02c"]

for df, seed, col in zip(dfs, seeds, colors):
    axes[0].plot(df.episode, df.reward.rolling(20).mean(), alpha=0.8, color=col, label=f"Seed {seed}")
    axes[1].plot(df.episode, df.energy_saving_pct.rolling(20).mean(), alpha=0.8, color=col, label=f"Seed {seed}")

axes[0].axhline(-0.20, color='r', linestyle='--', label="Paper target")
axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Mean Episode Reward")
axes[0].set_title("QMIX Convergence (all seeds)"); axes[0].legend()

axes[1].axhline(23.7, color='r', linestyle='--', label="Paper (23.7%)")
axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Energy Saving (%)")
axes[1].set_title("Energy Saving During Training"); axes[1].legend()

plt.tight_layout(); plt.savefig("../data/ablations/marl_convergence_plot.png", dpi=150)
plt.show()

# %% Ablation table
reward_abl = pd.read_csv("../data/ablations/reward_shaping.csv")
print("Table 8 - Reward Shaping Ablation:")
print(reward_abl.to_string(index=False))

# %% Sensor importance
sensors = pd.read_csv("../data/ablations/sensor_importance.csv")
sensors = sensors.sort_values("ensemble", ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(sensors.sensor, sensors.ensemble, color='steelblue')
plt.xlabel("Ensemble Importance"); plt.title("Sensor Channel Importance (Fig. 13)")
plt.tight_layout(); plt.savefig("../data/ablations/sensor_importance_plot.png", dpi=150)
plt.show()
