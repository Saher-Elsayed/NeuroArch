"""Plot SNN training curves (reproduces Figure 3 of paper)."""
import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np
from pathlib import Path

def main(data_path, out_dir):
    df = pd.read_csv(data_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df.epoch, df.train_accuracy, label="Train", color='navy')
    axes[0].plot(df.epoch, df.val_accuracy, label="Val", color='steelblue', linestyle='--')
    axes[0].axhline(0.918, color='red', linestyle=':', label="Paper 91.8%", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("(a) Classification Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df.epoch, df.train_loss, label="Train", color='darkgreen')
    axes[1].plot(df.epoch, df.val_loss, label="Val", color='mediumseagreen', linestyle='--')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Focal Loss")
    axes[1].set_title("(b) Focal Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].axvline(100, color='gray', linestyle='--', label="Phase 2 start")

    axes[2].plot(df.epoch, df.sparsity, color='darkorange', linewidth=2)
    axes[2].axhline(0.79, color='red', linestyle=':', label="Target 79%", linewidth=2)
    axes[2].fill_between(df.epoch, df.sparsity-0.01, df.sparsity+0.01, alpha=0.2, color='orange')
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Weight Sparsity")
    axes[2].set_title("(c) Network Sparsity"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle("NeuroArch SNN Training — Figure 3", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = Path(out_dir) / "fig3_training_curves.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ablations/training_curves.csv")
    p.add_argument("--out",  default=".")
    args = p.parse_args()
    main(args.data, args.out)
