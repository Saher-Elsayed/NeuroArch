"""Plot Pareto frontier (reproduces Table 12 / Figure 11)."""
import pandas as pd, matplotlib.pyplot as plt, numpy as np, argparse
from pathlib import Path

def main(data_path, out_dir):
    df = pd.read_csv(data_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['gray','gray','red','gray','gray']
    sizes  = [80, 80, 180, 80, 80]
    for _, row in df.iterrows():
        col = 'red' if 'NeuroArch' in row.configuration else 'steelblue'
        sz  = 180 if 'NeuroArch' in row.configuration else 80
        ax.scatter(row.energy_saving_pct, row.ashrae55_pct, c=col, s=sz, zorder=3,
                   edgecolors='black', linewidths=0.8)
        ax.annotate(row.configuration.split("(")[0].strip(),
                    (row.energy_saving_pct, row.ashrae55_pct),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Energy Saving (%)"); ax.set_ylabel("ASHRAE-55 Compliance (%)")
    ax.set_title("Pareto Frontier: Energy vs. Comfort (Figure 11)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path(out_dir) / "fig11_pareto.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}"); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pareto/pareto_frontier.csv")
    p.add_argument("--out", default=".")
    args = p.parse_args(); main(args.data, args.out)
