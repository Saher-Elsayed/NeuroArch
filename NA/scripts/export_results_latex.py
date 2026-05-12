"""
Export all results as LaTeX table snippets for paper revision.
Usage: python scripts/export_results_latex.py > results/all_tables.tex
"""
import csv, sys

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

# Table 4
print("% ---- Table 4: Controller Comparison ----")
print(r"\begin{tabular}{lrrrrr}")
print(r"\hline")
print(r"Controller & kWh/m$^2$ & Saving & ASHRAE-55 & PPD & Peak kW \\\hline")
for r in load("data/energyplus/medium_office/controller_comparison.csv"):
    bold = "\\textbf{" if r["controller"]=="NeuroArch_QMIX" else ""
    endb = "}" if bold else ""
    name = r["controller"].replace("_"," ")
    print(f"{bold}{name}{endb} & {r['annual_kWh_m2']} & "
          f"{r['energy_saving_pct']}\\% & {r['ashrae55_compliance_pct']}\\% & "
          f"{r['mean_ppd_pct']}\\% & {r['peak_demand_kW']} \\\\")
print(r"\hline\end{tabular}")
print()

# Table 7
print("% ---- Table 7: SNN Ablation ----")
print(r"\begin{tabular}{lrrrr}")
print(r"\hline")
print(r"Config & Acc & mW & Sparsity & Syn \\\hline")
for r in load("data/ablations/snn_arch_ablation.csv"):
    bold = "\\textbf{" if "NeuroArch" in r["config"] else ""
    endb = "}" if bold else ""
    print(f"{bold}{r['config']}{endb} & {r['acc_pct']}\\% & "
          f"{r['power_mW']} & {r['sparsity_pct']}\\% & {r['synapses']} \\\\")
print(r"\hline\end{tabular}")
print()

# Table 12 Pareto
print("% ---- Table 12: Pareto Frontier ----")
print(r"\begin{tabular}{rrrrrr}")
print(r"\hline")
print(r"$\lambda_C$ & $\lambda_P$ & kWh/m$^2$ & Compl\% & PPD\% & Peak kW \\\hline")
for r in load("data/pareto/pareto_frontier.csv"):
    marker = r" $\leftarrow$ NeuroArch" if float(r["lambda_C"])==2.0 else ""
    print(f"{r['lambda_C']} & {r['lambda_P']} & {r['kWh_m2']} & "
          f"{r['compliance_pct']} & {r['ppd_pct']} & {r['peak_kW']}{marker} \\\\")
print(r"\hline\end{tabular}")
