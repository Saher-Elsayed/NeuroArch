<div align="center">

# NeuroArch 🧠🏢

**Spiking Neural Networks and Virtual Reality for Building Energy Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)
[![IEEE Access](https://img.shields.io/badge/IEEE_Access-2026-red)](https://doi.org/ACCESS-2026-16730)
[![CI](https://github.com/NeuroArch-Lab/NeuroArch/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuroArch-Lab/NeuroArch/actions)

[Paper](https://ieee-access.org) • [Data](#data) • [Quickstart](#quickstart) • [Reproduce](#reproducing-paper-results) • [API Docs](#api)

</div>

---

## Abstract

NeuroArch integrates a **spiking neural network (SNN)** ISO 7730 comfort classifier with **multi-agent reinforcement learning (QMIX)** HVAC control, streamed to an **Unreal Engine 5 VR BIM interface** in real time. Deployed on a Xilinx Artix-7 FPGA, the SNN achieves **91.8% comfort accuracy at 0.31 mW** with **79% weight sparsity**. The 6-agent QMIX framework delivers **23.7% energy savings** and **91.3% ASHRAE-55 compliance** across three Houston commercial buildings, streaming IFC property-set updates at **20.3 ms mean E2E latency**.

---

## Key Results

| Metric | Value | Baseline |
|--------|-------|----------|
| SNN Accuracy | **91.8%** | 83.2% (1-layer) |
| Energy Saving | **23.7%** | 0% (rule-based) |
| ASHRAE-55 Compliance | **91.3%** | 78.2% |
| Peak Demand Reduction | **20.5%** (312→248 kW) | — |
| SNN Inference Latency | **4.1 ms mean** / 12 ms P99 | — |
| E2E VR Latency | **20.3 ms mean** | — |
| Power (Artix-7) | **0.31 mW** | 2.1 mW (GRU) |
| Weight Sparsity | **79%** | 0% (dense) |
| Active Synapses | **3,104** | 14,753 (dense) |
| User Study (SUS) | **81.3/100** | 61.2 (Desktop) |

---

## Repository Structure

```
NeuroArch/
├── snn/                   # Spiking Neural Network (SNN) comfort classifier
│   ├── model.py           # NeuroArchSNN: 4-layer LIF, FastSigmoid surrogate gradient
│   ├── train.py           # Two-phase training (Phase 1: 100ep / Phase 2: 50ep BPTT)
│   ├── dataset.py         # ComfortDataset: sliding-window, WeightedRandomSampler
│   ├── augmentation.py    # SensorAugmentor: Gaussian noise, time warp, mixup
│   ├── explainability.py  # SNNGradCAM, SpikeTimingAttribution, IntegratedGradients
│   ├── calibration.py     # Temperature scaling, ECE, reliability diagram
│   ├── focal_loss.py      # Multi-class focal loss (gamma=2.0)
│   ├── rate_encoder.py    # Poisson rate encoding for sensor inputs
│   ├── evaluate.py        # Per-class F1, confusion matrix, ASHRAE alignment
│   └── quantize.py        # INT8 quantization + magnitude pruning
│
├── marl/                  # Multi-Agent Reinforcement Learning (QMIX)
│   ├── agent.py           # AgentQNetwork (DRQN) + DuelingAgentQNetwork
│   ├── qmix_network.py    # QMixNetwork (hypernetwork) + VDN + QTRAN baselines
│   ├── reward.py          # ComfortAugmentedReward (Eq. 12) + ablation variants
│   ├── replay_buffer.py   # EpisodeBuffer + PrioritisedEpisodeBuffer (PER)
│   ├── train_qmix.py      # Full 6-agent QMIX training loop
│   ├── curriculum.py      # CurriculumScheduler: 4-stage progressive training
│   └── communication.py  # CommNet mean-field agent communication
│
├── envs/                  # Gymnasium environments
│   ├── energyplus_env.py  # NeuroArchEnv: 3 buildings, 6 agents, EnergyPlus API
│   └── wrappers.py        # Normalizer, RecordStats, FaultInjection, ClimateShift
│
├── bim_server/            # Real-time IFC BIM streaming server
│   ├── bim_server.py      # AsyncIO WebSocket server + delta encoder (340 bytes/tick)
│   └── ifc_pset_schema.json  # Pset_ThermalComfort + Pset_HVACConfig schema
│
├── deployment/            # Production deployment
│   ├── inference_server.py  # FastAPI: POST /comfort, WS /stream, GET /health
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── fpga/rtl/              # Verilog RTL for Artix-7 XC7A35T
│   ├── lif_pe.v           # LIF processing element (18 LUTs, 28 FFs)
│   ├── spike_encoder.v    # 14-channel rate encoder
│   └── neuroarch_top.v   # Top-level with AXI-Lite control
│
├── vr/Source/             # Unreal Engine 5.3 C++ BIM client
│   ├── BIMWebSocketClient.h
│   └── BIMWebSocketClient.cpp
│
├── data/                  # All data referenced in the paper
│   ├── energyplus/        # 8760-hour EnergyPlus simulations (3 buildings)
│   ├── sensor_logs/       # 32,000 labeled sensor windows (κ=0.74)
│   ├── weather/           # 3 climate zones (Houston TX, Seattle WA, Minneapolis MN)
│   ├── ablations/         # Training curves, MARL convergence, sensor importance
│   ├── latency/           # 10,000-frame E2E latency log
│   ├── pareto/            # Pareto frontier (Table 12)
│   ├── cross_climate/     # LOBO cross-climate results (Table 9)
│   └── user_study/        # 32-participant VR study 
│
├── scripts/               # Figure reproduction scripts
├── notebooks/             # Jupyter notebooks (training demo, MARL eval, energy analysis)
├── tests/                 # Unit, integration, and performance tests
├── monitoring/            # Prometheus metrics
├── benchmarks/            # Reproducibility benchmark suite
└── configs/               # YAML experiment configurations
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/NeuroArch-Lab/NeuroArch.git
cd NeuroArch
conda env create -f environment.yml
conda activate neuroarch

# Run unit tests
pytest tests/ -v --tb=short

# Run reproducibility benchmarks
python benchmarks/run_all.py

# Train SNN (medium office)
python -m snn.train --config configs/snn_medium_office.yaml

# Train QMIX
python -m marl.train_qmix --config configs/qmix_medium_office.yaml

# Start inference server
uvicorn deployment.inference_server:create_app --factory --port 8000

# Start BIM server
python -m bim_server.bim_server --port 8765

# Or: Docker Compose (inference + BIM + Prometheus + Grafana)
docker-compose -f deployment/docker-compose.yml up
```

---

## Reproducing Paper Results

All results can be reproduced from pre-computed data in `data/`:

```bash
# Figure 3: SNN Training Curves
python scripts/plot_training_curves.py --data data/ablations/training_curves.csv

# Figure 6: MARL Convergence
python scripts/plot_marl_convergence.py --data data/ablations/

# Figure 9: Latency Distribution
python scripts/plot_latency.py --data data/latency/frame_latency_10000.csv

# Figure 11: Pareto Frontier
python scripts/plot_pareto.py --data data/pareto/pareto_frontier.csv

# Table 4: Controller Comparison
python experiments/reproduce_table4.sh

# Run all paper figures at once
bash experiments/plot_all_figures.sh
```

---

## Data

| File | Description | Rows | Size |
|------|-------------|------|------|
| `sensor_logs/{building}_sensor_log_sample.csv` | 14-channel sensor time series | 4,000/building | ~1.5 MB |
| `sensor_logs/comfort_labels.csv` | Window labels with inter-rater agreement (κ=0.74) | 32,000 | 2.8 MB |
| `energyplus/{building}/simulation_8760h.csv` | Annual EnergyPlus simulation | 8,760/building | ~600 KB |
| `energyplus/{building}/controller_comparison.csv` | Table 4 data | 5 controllers | 1 KB |
| `latency/frame_latency_10000.csv` | E2E latency per frame | 10,000 | 400 KB |
| `user_study/participants.csv` | VR user study | 32 | 5 KB |

---

## Citation

```bibtex
@article{elsayed2026neuroarch,
  title     = {{NeuroArch}: Spiking Neural Networks and Virtual Reality
               for Building Energy Optimization},
  author    = {Elsayed, Saher and Ali, Mohamed and Aziz, Khairi Azhar},
  journal   = {IEEE Access},
  year      = {2026},
  note      = {MS ID: Access-2026-16730},
  url       = {https://github.com/NeuroArch-Lab/NeuroArch}
}
```

---

## Authors

| Author | Affiliation | Contact |
|--------|-------------|---------|
| **Saher Elsayed** | University of Pennsylvania | selsayed@seas.upenn.edu |
| **Mohamed Ali** | Virginia Tech | |
| **Ts. Dr. Khairi Azhar Aziz** | UNITEN | |

---

## License

MIT License. See [LICENSE](LICENSE).
