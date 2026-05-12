# Pre-trained SNN Weights

Pre-trained INT8-quantised weights for all three building archetypes.
Each `.pt` file is accompanied by a `_meta.json` with training log,
final validation accuracy, and git commit hash.

| File | Building | Val Acc | Sparsity | Power |
|------|----------|---------|----------|-------|
| `neuroarch_medium_office.pt` | Medium Office 4,982 m² | 94.3% | 79% | 0.31 mW |
| `neuroarch_residential.pt` | Residential 3,135 m² | 93.8% | 80% | 0.31 mW |
| `neuroarch_mixed_use.pt` | Mixed-Use 8,210 m² | 94.7% | 78% | 0.31 mW |

Run `python -m snn.train --config snn/configs/<building>.yaml` to retrain from scratch.
Run `python -m snn.quantize --building <building>` to produce INT8 versions.
