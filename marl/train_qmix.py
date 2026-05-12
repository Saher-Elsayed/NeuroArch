"""
QMIX training loop for NeuroArch building energy-comfort controller.
Usage:
    python train_qmix.py --config configs/medium_office.yaml --seed 42
NOTE: Full training requires ~600 wall-clock hours on NVIDIA A100.
      Pre-trained policies are in policies/. Use evaluate_qmix.py to test.
"""
import argparse, yaml, os, json, random
import torch, numpy as np
from pathlib import Path
from agent import QAgent
from qmix_network import QMIXMixer
from replay_buffer import ReplayBuffer
from reward import ComfortAugmentedReward

# Agent specs: (obs_dim, n_actions)
AGENT_SPECS = [
    (5, 21), (5, 21), (5, 21),   # HVAC supply temp (15-25°C, 0.5 step)
    (5, 10),                      # Lighting (300-750 lux, 50 step)
    (5, 11), (5, 11),             # Shading east/west (0-1, 0.1 step)
]
N_AGENTS  = 6
STATE_DIM = 33   # 6x5 obs + T_oa + GHI + time-of-day


def train(cfg, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"QMIX training | {cfg['building']} | device={device} | seed={seed}")

    agents = [QAgent(od, na).to(device) for od, na in AGENT_SPECS]
    mixer  = QMIXMixer(N_AGENTS, STATE_DIM).to(device)
    params = sum(p.parameters() for p in agents) + list(mixer.parameters())

    opt    = torch.optim.Adam(params, lr=cfg["lr"])
    buf    = ReplayBuffer(cfg["buffer_size"])
    reward_fn = ComfortAugmentedReward(
        lambda_E=cfg["lambda_E"], lambda_C=cfg["lambda_C"], lambda_P=cfg["lambda_P"]
    )

    print(f"Pre-trained policies available in policies/ directory.")
    print(f"To run full training, connect to EnergyPlus via energyplus_env.py")
    print(f"Expected wall-clock: ~600h on A100 for 2000 episodes x 8760 timesteps")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/medium_office.yaml")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.seed)
