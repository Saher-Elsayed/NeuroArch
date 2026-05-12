"""Plot latency distribution (reproduces Figure 9)."""
import pandas as pd, matplotlib.pyplot as plt, numpy as np, argparse
from pathlib import Path

def main(data_path, out_dir):
    df = pd.read_csv(data_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, col, title, budget in zip(
        axes,
        ["snn_inference_ms", "websocket_ms", "e2e_latency_ms"],
        ["SNN Inference", "WebSocket Tx", "End-to-End"],
        [4.5, 3.5, 33.3]
    ):
        ax.hist(df[col], bins=80, color='steelblue', alpha=0.75, edgecolor='none')
        ax.axvline(df[col].mean(), color='navy',  linestyle='--', label=f"Mean={df[col].mean():.1f}ms")
        ax.axvline(df[col].quantile(0.99), color='darkorange', linestyle='--',
                   label=f"P99={df[col].quantile(0.99):.1f}ms")
        ax.axvline(budget, color='red', linestyle=':', linewidth=2, label=f"Budget={budget}ms")
        ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Frame Count")
        ax.set_title(f"({chr(97+list(axes).index(ax))}) {title} Latency")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    drop_pct = df.frame_dropped.mean() * 100
    fig.suptitle(f"NeuroArch Latency Distribution (N={len(df):,} frames, drop rate={drop_pct:.2f}%)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = Path(out_dir) / "fig9_latency_distribution.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}  |  Frame drop rate: {drop_pct:.2f}%")
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/latency/frame_latency_10000.csv")
    p.add_argument("--out",  default=".")
    args = p.parse_args(); main(args.data, args.out)
