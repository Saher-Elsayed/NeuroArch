# NeuroArch: Neuromorphic–VR Co-Design for Real-Time Building Energy and Comfort Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![EnergyPlus 23.1](https://img.shields.io/badge/EnergyPlus-23.1-green.svg)](https://energyplus.net)

Official repository for the IEEE ACCESS paper:

> **NeuroArch: Neuromorphic–VR Co-Design for Real-Time Building Energy and Comfort Optimization**  
> Mohamed Ali, Saher Elsayed, Ts. Dr. Khairi Azhar Aziz  
> *IEEE ACCESS*, 2025. doi: 10.1109/ACCESS.2025.XXXXXXX

---

## Key Results

| Metric | Value |
|--------|-------|
| SNN comfort classification accuracy | **94.3%** (5-class ISO 7730) |
| SNN power consumption | **0.31 mW** (Artix-7 XC7A35T) |
| Annual energy saving vs. rule-based | **23.7%** (EnergyPlus simulation) |
| Peak demand reduction | **20.5%** |
| ASHRAE 55 compliance | **91.3%** (only controller achieving PPD ≤ 10%) |
| VR design-decision task-time reduction | **41.3%** (N=32 user study) |

---

## Repository Structure

```
NeuroArch/
├── snn/                    # LIF-SNN comfort classifier (PyTorch)
├── marl/                   # QMIX multi-agent RL controller
├── fpga/                   # Artix-7 RTL + Vivado reports
├── vr/                     # Unreal Engine 5 VR digital twin
├── bim_server/             # IfcOpenShell BIM server + WebSocket
├── data/                   # All datasets (CSV + IFC models)
├── experiments/            # Reproduce-all shell scripts
└── notebooks/              # Jupyter walkthroughs
```

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/NeuroArch-Lab/NeuroArch.git
cd NeuroArch
conda env create -f environment.yml
conda activate neuroarch
```

### 2. Evaluate Pre-Trained SNN

```bash
cd snn
python evaluate.py --building medium_office --weights weights/neuroarch_office.pt
# Expected: Accuracy 94.3%, Power 0.31 mW (post-quant INT8)
```

### 3. Reproduce Table 4 (Annual Energy Comparison)

```bash
cd experiments
bash reproduce_table4.sh
# Runs all 5 controllers on 3 archetypes, ~4 hours on CPU
```

### 4. Run MARL Training (GPU required)

```bash
cd marl
python train_qmix.py --config configs/medium_office.yaml --seed 42
# ~600 wall-clock hours on NVIDIA A100; pre-trained policies in marl/policies/
```

### 5. Launch BIM Server + VR

```bash
cd bim_server
python bim_server.py --ifc ifc_models/medium_office.ifc --port 8765
# Then open NeuroArchVR.uproject in Unreal Engine 5.3+
```

---

## Datasets

All datasets are included in `data/` — nothing is available only on request.

| Dataset | File | Rows | Description |
|---------|------|------|-------------|
| Medium Office EnergyPlus | `data/energyplus/medium_office/simulation_8760h.csv` | 8,760 | Hourly simulation results, 5 controllers |
| Residential EnergyPlus | `data/energyplus/residential/simulation_8760h.csv` | 8,760 | Same |
| Mixed-Use EnergyPlus | `data/energyplus/mixed_use/simulation_8760h.csv` | 8,760 | Same |
| IoT sensor logs – Office | `data/sensor_logs/medium_office_8weeks.csv` | 4,838,400 | 14-channel @ 1 s, 8 weeks |
| IoT sensor logs – Residential | `data/sensor_logs/residential_8weeks.csv` | 4,838,400 | Same |
| IoT sensor logs – Mixed-Use | `data/sensor_logs/mixed_use_8weeks.csv` | 4,838,400 | Same |
| Comfort labels | `data/sensor_logs/comfort_labels.csv` | 32,000 | ISO 7730, 38 occupants, κ=0.74 |
| Seattle cross-climate | `data/cross_climate/seattle_zone4c.csv` | 2,419,200 | 4-week calibration |
| Minneapolis cross-climate | `data/cross_climate/minneapolis_zone6a.csv` | 2,419,200 | 4-week calibration |
| User study – task times | `data/user_study/task_times.csv` | 32 | T1/T2/T3, Baseline + VR |
| User study – NASA-TLX | `data/user_study/nasa_tlx.csv` | 32 | 6 subscales |
| User study – SUS | `data/user_study/sus_scores.csv` | 32 | System Usability Scale |
| User study – SSQ | `data/user_study/ssq_subscales.csv` | 32 | Nausea/Oculomotor/Disorientation |
| End-use monthly | `data/energyplus/medium_office/end_use_monthly.csv` | 12 | kWh/m²/month by end-use |
| Cross-climate LOBO | `data/cross_climate/lobo_results.csv` | 6 | Leave-one-building-out results |

---

## Hardware

- **Edge node**: Digilent Arty A7-35T (Artix-7 XC7A35T), Cortex-M0, BLE 5.0
- **Sensors**: SHT45 (temp/RH), SCD41 (CO₂), BH1750 (lux), HC-SR501 (PIR)
- **Server**: NVIDIA Jetson AGX Orin
- **VR headset**: Meta Quest 3 (Wi-Fi 6)
- **FPGA toolchain**: Vivado 2023.2

---

## Citation

```bibtex
@article{ali2025neuroarch,
  title   = {{NeuroArch}: Neuromorphic--{VR} Co-Design for Real-Time
             Building Energy and Comfort Optimization},
  author  = {Ali, Mohamed and Elsayed, Saher and Aziz, {Ts. Dr. Khairi Azhar}},
  journal = {IEEE ACCESS},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.XXXXXXX}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
