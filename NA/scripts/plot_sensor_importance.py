"""Reproduce Fig. 13: Sensor modality ablation."""
import csv, os
import matplotlib.pyplot as plt

def main():
    with open("data/ablations/sensor_importance.csv") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["acc_drop_pct"]))
    names=[r["name"]          for r in rows]
    vals =[float(r["acc_drop_pct"]) for r in rows]

    fig,ax=plt.subplots(figsize=(8,5))
    colors=["#e53935" if v>=10 else "#ef9a9a" if v>=5 else "#ffcdd2" for v in vals]
    bars=ax.barh(names,vals,color=colors,edgecolor="gray",lw=0.5)
    for b,v in zip(bars,vals):
        ax.text(v+0.1,b.get_y()+b.get_height()/2,f"{v:.1f}%",va="center",fontsize=9)
    ax.set_xlabel("Accuracy Drop when Channel Masked (%)")
    ax.set_title("Sensor Modality Ablation (Fig. 13)")
    ax.axvline(0,color="black",lw=0.8); ax.grid(axis="x",alpha=0.3)
    plt.tight_layout()
    os.makedirs("results",exist_ok=True)
    plt.savefig("results/fig13_sensor_importance.pdf",dpi=150)
    print(f"Top channel: {rows[-1]['name']} ({vals[-1]:.1f}%)")

if __name__ == "__main__":
    main()
