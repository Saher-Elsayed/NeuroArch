# Baseline Controllers

| File | Controller | kWh/m² | Saving | Compliance | PPD |
|------|-----------|--------|--------|------------|-----|
| `rule_based_g36.py` | ASHRAE G36 | 142.3 | 0.0% | 83.4% | 18.2% |
| `mpc_rom.py` | MPC-ROM | 124.8 | -12.3% | 87.1% | 14.6% |
| `ddpg_baseline.py` | DDPG | 118.6 | -16.7% | 88.3% | 13.1% |
| *(marl/evaluate_qmix.py)* | QMIX no SNN | 112.4 | -21.0% | 90.2% | 11.8% |
| *(marl/evaluate_qmix.py)* | **NeuroArch** | **108.6** | **-23.7%** | **91.3%** | **10.4%** |
