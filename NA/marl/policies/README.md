# Pre-trained QMIX Policies

| File | Building | Episodes | Return |
|------|----------|----------|--------|
| `neuroarch_medium_office.pkl` | Medium Office | 2,000 | 352 |
| `neuroarch_residential.pkl` | Residential | 2,000 | 318 |
| `neuroarch_mixed_use.pkl` | Mixed-Use | 2,000 | 389 |

To retrain: `python -m marl.train_qmix --config marl/configs/<building>.yaml`
Full training requires EnergyPlus co-simulation (~600h on A100).
