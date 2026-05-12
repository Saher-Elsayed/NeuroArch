# ---
# jupyter:
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# # Notebook 05: Pareto Frontier
# Reproduces Paper Table 12 — Energy-Comfort trade-off

# %%
import sys; sys.path.insert(0,'.')
import csv, matplotlib.pyplot as plt

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

rows = load("data/pareto/pareto_frontier.csv")
lc = [float(r["lambda_C"]) for r in rows]
e  = [float(r["kWh_m2"])   for r in rows]
c  = [float(r["compliance_pct"]) for r in rows]
p  = [float(r["ppd_pct"])  for r in rows]
labels = ["E-only", "λC=1", "λC=2
(NeuroArch)", "λC=4", "C-only"]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sc = ax1.scatter(e, c, c=lc, cmap="coolwarm_r", s=150, zorder=5, edgecolors="k", lw=0.5)
ax1.plot(e, c, "--", alpha=0.4, color="gray")
for i,(ei,ci,lb) in enumerate(zip(e,c,labels)):
    ax1.annotate(lb,(ei,ci), xytext=(6,4), textcoords="offset points", fontsize=8)
plt.colorbar(sc, ax=ax1, label="λ_C")
ax1.set_xlabel("Annual Energy (kWh/m²)"); ax1.set_ylabel("ASHRAE-55 Compliance (%)")
ax1.set_title("Energy-Comfort Pareto Frontier"); ax1.grid(True, alpha=0.3)

sc2 = ax2.scatter(e, p, c=lc, cmap="coolwarm_r", s=150, zorder=5, edgecolors="k", lw=0.5)
ax2.plot(e, p, "--", alpha=0.4, color="gray")
ax2.axhline(10, ls="--", color="red", lw=1, label="PPD 10% target")
for i,(ei,pi,lb) in enumerate(zip(e,p,labels)):
    ax2.annotate(lb,(ei,pi), xytext=(6,4), textcoords="offset points", fontsize=8)
ax2.set_xlabel("Annual Energy (kWh/m²)"); ax2.set_ylabel("Mean PPD (%)")
ax2.set_title("Energy-PPD Pareto Frontier"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
print("NeuroArch (λC=2): 108.6 kWh/m², 91.3% compliance, 10.4% PPD")
