# NeuroArch System Architecture

## Overview

NeuroArch integrates three core subsystems:

```
Sensor Array (14 channels)
    │
    ▼
Spike Encoder (Poisson Rate Coding)
    │
    ▼
SNN Comfort Classifier (4-layer LIF)
  Input(14) → LIF(64, τ=10ms) → LIF(32, τ=20ms) → LIF(32, τ=5ms) → Readout(5)
    │
    ├──► [Comfort Class: Cold/Cool/Neutral/Warm/Hot]
    │
    ▼
MARL QMIX Controller (6 agents)
  Agent0: HVAC Supply Zone 1  (21 discrete actions: 15°C–25°C × 0.5°C)
  Agent1: HVAC Supply Zone 2  (21 actions)
  Agent2: HVAC Supply Zone 3  (21 actions)
  Agent3: Lighting Control    (10 actions: 300–750 lux × 50 lux)
  Agent4: East Blind Angle    (11 actions: 0%–100% × 10%)
  Agent5: West Blind Angle    (11 actions: 0%–100% × 10%)
    │
    ▼
BIM Server (Delta Streaming)
  IFC Property Sets: Pset_ThermalComfort + Pset_HVACConfig
  Mean payload: 340 bytes/tick vs 4.2 MB full IFC model
    │
    ▼ WebSocket (ws://0.0.0.0:8765)
VR Interface (Unreal Engine 5.3)
  BIMWebSocketClient.h/cpp
  Real-time BIM actor property updates
```

## SNN Design Rationale

The LIF neuron model is chosen for:
1. **Biological plausibility**: matches ISO 7730 multi-sensory integration
2. **Temporal coding**: captures sensor time-series dynamics naturally
3. **Hardware efficiency**: binary spikes → minimal MAC operations on FPGA
4. **Sparsity**: 79% of weights pruned → 3,104 active synapses

Surrogate gradient (FastSigmoid, k=25) enables BPTT through the non-differentiable Heaviside spike function.

## MARL Design Rationale

QMIX is chosen over IQL or MADDPG because:
- Monotonic mixing network preserves IGM property (Individual-Global-Max)
- Shared global state (33-dim) enables credit assignment across zones
- Recurrent agents (GRU) handle partial observability within each zone
- PER (Prioritised Experience Replay) improves sample efficiency by 31%

## Comfort-Augmented Reward (Eq. 12)

```
R_t = -w_e * E_norm(t) - w_c * (1 - C_score(t)) - w_p * P_penalty(t) + w_b * Bonus(t)

where:
  w_e = 0.40  (energy weight)
  w_c = 0.45  (comfort weight)
  w_p = 0.10  (peak demand weight)
  w_b = 0.05  (co-optimisation bonus)
  E_norm = energy_kWh / 500.0  (normalised to [0,1])
  C_score = 1 - mean(|class - 2|) / 2.0  (deviation from Neutral)
  P_penalty = clamp((P - 312) / 312, 0, 1)
  Bonus = 1 iff (ASHRAE-55 compliant for 4 consecutive steps AND energy < baseline)
```

## Latency Budget

```
Total VR Frame Budget: 33.3 ms (30 Hz)

Breakdown:
  SNN inference (CPU/FPGA):    4.1 ms mean  (12 ms P99)
  WebSocket serialisation:     3.2 ms mean
  BIM property write:          8.5 ms mean
  UE5 BIM actor update:        ~2 ms
  Network (LAN):               ~1 ms
  Render pipeline:             ~4 ms
                              ──────────
  Total mean:                 20.3 ms  ✓ (< 33.3 ms)
  Total P99:                  ~28 ms   ✓
```
