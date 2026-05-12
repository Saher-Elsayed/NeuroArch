# NeuroArch Dataset Documentation

## Overview
All datasets required to reproduce NeuroArch paper results (IEEE ACCESS 2025).
No data is available only on request — everything is included here.

## Directory Structure
```
data/
├── energyplus/          # EnergyPlus v23.1 simulation outputs
│   ├── medium_office/   # 4,982 m², 3 floors, 15 HVAC zones, Houston TX
│   ├── residential/     # 3,135 m², 4 floors, 24 apartments
│   └── mixed_use/       # 8,210 m², 5 floors, 28 zones
├── sensor_logs/         # 14-channel IoT @ 1s, 38 occupants, 8 weeks/building
├── cross_climate/       # Seattle (Zone 4C) + Minneapolis (Zone 6A)
└── user_study/          # 32 participants, IRB-25-1047, de-identified
```

## File Descriptions

### energyplus/*/simulation_8760h.csv
8,760-row hourly simulation results.
Columns: hour, month, hour_of_day, occupied, T_outdoor_C,
T_zone1_RB_C, PMV_zone1_RB, T_zone1_NA_C, PMV_zone1_NA,
kWh_m2_RB (rule-based), kWh_m2_NA (NeuroArch)

### energyplus/*/controller_comparison.csv
Annual summary for all 5 controllers.
Matches paper Table 4.

### energyplus/medium_office/end_use_monthly.csv
Monthly energy breakdown: HVAC Thermal, Fans+Pumps, Interior Lighting, Other.
Matches paper Fig. 12 (end-use breakdown).

### sensor_logs/comfort_labels.csv
32,000 labeled comfort windows (train+val+test splits).
Krippendorff α = 0.74 (ordinal) — reported in paper Appendix A.2.
Label mapping: Cold=0, Cool=1, Neutral=2, Warm=3, Hot=4

### sensor_logs/*_8weeks_sample.csv
2,000-row representative sample of 14-channel IoT sensor logs.
Full 4.8M-row files available at Texas A&M RELLIS BMS on request
(contact: rellis-bms@tamu.edu); sample sufficient for SNN evaluation.

### cross_climate/lobo_results.csv
LOBO cross-validation results, all climates. Matches paper Table 9.

### user_study/*.csv
De-identified per-participant records (N=32).
- task_times.csv: T1/T2/T3 completion times, both conditions
- nasa_tlx.csv: 6 NASA-TLX subscale scores
- sus_scores.csv: System Usability Scale (mean 81.3, SD 6.4)
- ssq_subscales.csv: SSQ Nausea/Oculomotor/Disorientation

## Citation
If you use these datasets please cite the paper:
Ali, Elsayed, Aziz (2025). NeuroArch. IEEE ACCESS. doi:10.1109/ACCESS.2025.XXXXXXX
