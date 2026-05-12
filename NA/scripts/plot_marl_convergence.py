"""Plot MARL convergence across 3 seeds (reproduces Figure 6)."""
import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np
from pathlib import Path

def main(data_dir, out_dir):
    seeds = [42, 123, 456]; colors = ["#1f77b4","#ff7f0e","#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    all_rewards = []
    for seed, col in zip(seeds, colors):
        df = pd.read_csv(f"{data_dir}/marl_convergence_seed{seed}.csv")
        smoothed = df.reward.rolling(30, min_periods=1).mean()
        axes[0].plot(df.episode, smoothed, alpha=0.85, color=col, label=f"Seed {seed}")
        all_rewards.append(smoothed.values)
        axes[1].plot(df.episode, df.energy_saving_pct.rolling(30, min_periods=1).mean(),
                     alpha=0.85, color=col, label=f"Seed {seed}")

    # Confidence band
    all_r = np.stack(all_rewards)
    ep = np.arange(1, 501)
    axes[0].fill_between(ep, all_r.min(0), all_r.max(0), alpha=0.12, color='gray')
    axes[0].axhline(-0.20, color='red', linestyle='--', linewidth=2, label="Paper target")
    axes[0].set_xlabel("Training Episode"); axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title("(a) QMIX Convergence"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].axhline(23.7, color='red', linestyle='--', linewidth=2, label="Paper 23.7%")
    axes[1].set_xlabel("Training Episode"); axes[1].set_ylabel("Energy Saving (%)")
    axes[1].set_title("(b) Energy Saving During Training"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("NeuroArch MARL Training — Figure 6", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = Path(out_dir) / "fig6_marl_convergence.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}"); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ablations"); p.add_argument("--out", default=".")
    args = p.parse_args(); main(args.data, args.out)
