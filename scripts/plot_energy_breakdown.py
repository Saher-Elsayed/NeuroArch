"""Reproduce Fig. 12: End-use energy saving breakdown."""
import csv, os
import matplotlib.pyplot as plt
import numpy as np

def main():
    with open("data/energyplus/medium_office/end_use_monthly.csv") as f:
        rows = list(csv.DictReader(f))
    months=[r["month"] for r in rows]
    hvac  =[float(r["HVAC_thermal_kWh_m2"])  for r in rows]
    fans  =[float(r["fans_pumps_kWh_m2"])     for r in rows]
    light =[float(r["interior_lighting_kWh_m2"]) for r in rows]
    other =[float(r["other_kWh_m2"])          for r in rows]

    x=range(len(months))
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(x,hvac,label="HVAC Thermal",color="steelblue")
    ax.bar(x,fans,bottom=hvac,label="Fans+Pumps",color="skyblue")
    b2=[h+f for h,f in zip(hvac,fans)]
    ax.bar(x,light,bottom=b2,label="Lighting",color="gold")
    b3=[a+l for a,l in zip(b2,light)]
    ax.bar(x,other,bottom=b3,label="Other",color="lightgray")
    ax.set_xticks(x); ax.set_xticklabels(months,rotation=30,ha="right")
    ax.set_ylabel("kWh/m²/month"); ax.set_title("Monthly End-Use Energy (NeuroArch, Fig. 12)")
    ax.legend(loc="upper left"); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout()
    os.makedirs("results",exist_ok=True)
    plt.savefig("results/fig12_energy_breakdown.pdf",dpi=150)
    print(f"Annual total: {sum(float(r['total_kWh_m2']) for r in rows):.1f} kWh/m² (paper: 108.6)")

if __name__ == "__main__":
    main()
