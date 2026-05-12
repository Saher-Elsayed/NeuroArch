#!/usr/bin/env bash
# Reproduce all paper figures
set -e
echo "Plotting all NeuroArch figures..."
python scripts/plot_training_curves.py --data data/ablations/training_curves.csv --out figures/
python scripts/plot_marl_convergence.py --data data/ablations/ --out figures/
python scripts/plot_pareto.py --data data/pareto/pareto_frontier.csv --out figures/
python scripts/plot_latency.py --data data/latency/frame_latency_10000.csv --out figures/
echo "All figures saved to figures/"
