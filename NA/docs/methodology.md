# Extended Methodology

## SNN Comfort Classification

### Dataset Collection
- 38 occupants across 3 buildings, 8 weeks each
- ISO 7730 PMV labels at 1-minute resolution
- Krippendorff's alpha = 0.74 (acceptable inter-rater reliability)
- Class distribution: Cold 4.2%, Cool 11.8%, Neutral 61.4%, Warm 17.3%, Hot 5.3%

### Preprocessing Pipeline
```
raw sensor (14-ch, 1s) -> IQR outlier removal -> min-max normalisation
-> 100ms sliding window -> Poisson rate encoding -> LIF-SNN
```

### Two-Phase Training
- **Phase 1 (100 epochs)**: Rate-code pretraining with Adam (lr=5e-4), cosine schedule
- **Phase 2 (50 epochs)**: BPTT surrogate fine-tuning (lr=5e-5) + rate regulariser

### Quantisation
Post-training INT8 quantisation via PyTorch dynamic quantisation.
Power: FP32=0.31mW → INT8=0.23mW; accuracy drop <0.4%.

## MARL Formulation

### Dec-POMDP
- Agents: 6 (3x HVAC, 1x Lighting, 2x Shading)
- Global state: 33 scalars (6 agents × 5 obs + T_oa + GHI + time)
- Episode length: 8,760 steps (1 hour × 8,760 hours/year)

### Reward (Eq. 12)
```
r_t = -λ_E · ΣE_z + λ_C · Σ(κ_SNN · 1[PMV∈[-0.5,+0.5]]) - λ_P · ΣPPD
```
Optimal: λ_E=1.0, λ_C=2.0, λ_P=0.5 (Table 12 Pareto frontier)

### QMIX Training
- 2,000 episodes × 8,760 EnergyPlus timesteps
- Adam lr=1e-4, buffer=10K, batch=32, γ=0.99
- Target network: hard update every 10 episodes
- Epsilon-greedy: 1.0→0.05 over 500 episodes

## BIM Integration

### Differential Protocol
Only changed property values are transmitted:
- Mean delta payload: 340 bytes/tick
- Full IFC resend: 4.2 MB
- Compression ratio: 99.99%

### IFC Pset Schema
| Pset | Property | Type | Update |
|------|----------|------|--------|
| Pset_ThermalComfort | ComfortClass | IfcLabel | 1s |
| Pset_ThermalComfort | SNNConfidence | IfcReal | 1s |
| Pset_HVACConfig | SupplyTempSet | IfcReal | 1s |
