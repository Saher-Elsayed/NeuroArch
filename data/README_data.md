# NeuroArch Dataset

All datasets — nothing on request.

## Files and Sources

| File | Rows | Source | Notes |
|------|------|--------|-------|
| energyplus/*/simulation_8760h.csv | 8,760 | EnergyPlus v23.1 | 5 controllers |
| energyplus/*/controller_comparison.csv | 5 | — | Matches Table 4 |
| energyplus/medium_office/end_use_monthly.csv | 12 | — | Matches Fig. 12 |
| sensor_logs/*/sensor_log_sample.csv | 4,000 | Texas A&M RELLIS BMS | 14-channel, 1s |
| sensor_logs/comfort_labels.csv | 32,000 | 38 occupants | κ=0.74, 5 classes |
| sensor_logs/class_distribution.csv | 5 | — | Table 1 |
| cross_climate/*/simulation_results.csv | 8,760 | EnergyPlus | Seattle/Minneapolis |
| cross_climate/lobo_results.csv | 5 | — | Table 9 |
| user_study/task_times.csv | 32 | IRB-25-1047 | T1/T2/T3 |
| user_study/nasa_tlx.csv | 32 | IRB-25-1047 | 6 subscales |
| user_study/sus_scores.csv | 32 | IRB-25-1047 | SUS=81.3 |
| user_study/ssq_subscales.csv | 32 | IRB-25-1047 | Nausea/Oculo/Disori |
| ablations/snn_arch_ablation.csv | 7 | — | Table 7 |
| ablations/reward_shaping.csv | 4 | — | Table 8 |
| ablations/sensor_importance.csv | 14 | — | Fig. 13 |
| ablations/window_length_sensitivity.csv | 4 | — | Appendix |
| ablations/marl_convergence_multiseed.csv | 243 | — | Fig. 8 (3 seeds) |
| ablations/training_curves.csv | 150 | — | Fig. 3 |
| pareto/pareto_frontier.csv | 5 | — | Table 12 |
| latency/frame_latency_10000.csv | 10,000 | Profiler | Fig. 11 |
| latency/latency_summary.csv | 1 | — | Paper Section IX-E |

## Label Mapping

| Class | Int | Fraction | ASHRAE range |
|-------|-----|----------|--------------|
| Cold | 0 | 4.2% | < -1.0 |
| Cool | 1 | 11.8% | [-1.0, -0.5) |
| Neutral | 2 | 61.4% | [-0.5, +0.5] |
| Warm | 3 | 17.3% | (+0.5, +1.0] |
| Hot | 4 | 5.3% | > +1.0 |

## Citation
Ali, Elsayed, Aziz (2025). NeuroArch. IEEE ACCESS. doi:10.1109/ACCESS.2025.XXXXXXX
