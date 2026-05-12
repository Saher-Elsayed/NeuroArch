"""
Evaluate QMIX policy — loads pre-computed results matching paper Table 4.
Usage:
    python -m marl.evaluate_qmix --building medium_office
    python -m marl.evaluate_qmix --building all
"""
import argparse, csv
from pathlib import Path

BUILDINGS = ["medium_office", "residential", "mixed_use"]


def evaluate(building="medium_office"):
    p = Path(f"data/energyplus/{building}/controller_comparison.csv")
    if not p.exists():
        print(f"Data not found: {p}"); return
    with open(p) as f:
        rows = list(csv.DictReader(f))

    print(f"
{'='*65}")
    print(f"Building: {building}")
    print(f"{'Controller':<25} {'kWh/m2':>8} {'Saving%':>8} {'Compl%':>8} {'PPD%':>6} {'Peak kW':>8}")
    print("-" * 65)
    for r in rows:
        marker = " ✓" if r["controller"] == "NeuroArch_QMIX" else ""
        print(f"{r['controller']:<25} {float(r['annual_kWh_m2']):>8.1f} "
              f"{float(r['energy_saving_pct']):>8.1f} "
              f"{float(r['ashrae55_compliance_pct']):>8.1f} "
              f"{float(r['mean_ppd_pct']):>6.1f} "
              f"{int(r['peak_demand_kW']):>8}{marker}")

    na = next(r for r in rows if r["controller"] == "NeuroArch_QMIX")
    print(f"
NeuroArch: {float(na['energy_saving_pct']):.1f}% energy saving "
          f"| PPD {float(na['mean_ppd_pct']):.1f}% "
          f"({'✓ meets' if float(na['mean_ppd_pct'])<=10 else '✗ misses'} 10% target)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="medium_office",
                    choices=BUILDINGS + ["all"])
    args = ap.parse_args()
    buildings = BUILDINGS if args.building == "all" else [args.building]
    for b in buildings:
        evaluate(b)
