"""
NeuroArch SNN Training — Two-Phase Surrogate Gradient Training
Phase 1: 100 epochs, Adam lr=5e-4, rate regularisation
Phase 2: 50 epochs, BPTT, Adam lr=5e-5, fine-tune
"""

from __future__ import annotations
import argparse, json, logging, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import yaml

from .model import NeuroArchSNN, SNNConfig
from .focal_loss import FocalLoss
from .rate_encoder import PoissonRateEncoder
from .evaluate import evaluate_model, per_class_metrics
from .augmentation import SensorAugmentor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("neuroarch.train")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloader(cfg: dict, split: str, augment: bool = False):
    """Load sensor data from CSV, encode to spike trains, return DataLoader."""
    from .dataset import ComfortDataset
    ds = ComfortDataset(
        data_dir=cfg["data"]["root"],
        buildings=cfg["data"]["buildings"],
        split=split,
        window_size=cfg["snn"]["T"],
        stride=cfg["data"]["stride"],
        normalize=True,
        augment=augment and split == "train",
        noise_std=cfg["data"].get("noise_std", 0.01),
    )
    if split == "train":
        # Weighted sampling to handle class imbalance
        weights = ds.class_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                          sampler=sampler, num_workers=4, pin_memory=True)
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                      shuffle=False, num_workers=4, pin_memory=True)


def run_epoch(model, loader, optimizer, criterion, device, phase: int,
              grad_clip: float = 1.0, rate_reg: bool = True):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if rate_reg:
            loss = loss + model.rate_regularization_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        model.apply_pruning_masks()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def run_val(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train(cfg_path: str):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    snn_cfg = SNNConfig(
        n_inputs=cfg["snn"]["n_inputs"],
        hidden_sizes=cfg["snn"]["hidden_sizes"],
        n_classes=cfg["snn"]["n_classes"],
        tau_mem=cfg["snn"]["tau_mem"],
        T=cfg["snn"]["T"],
        rate_reg_lambda=cfg["snn"]["rate_reg_lambda"],
        target_sparsity=cfg["snn"]["target_sparsity"],
        dropout=cfg["snn"]["dropout"],
    )
    model = NeuroArchSNN(snn_cfg).to(device)
    log.info(f"Model: {model.n_parameters:,} parameters")

    criterion = FocalLoss(gamma=cfg["training"]["focal_gamma"], reduction="mean")
    train_loader = build_dataloader(cfg, "train", augment=True)
    val_loader   = build_dataloader(cfg, "val",   augment=False)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    # ---- Phase 1: Standard Adam + cosine LR --------------------------------
    log.info("Phase 1: 100 epochs, lr=5e-4, rate regularization ON")
    opt1 = torch.optim.Adam(model.parameters(), lr=5e-4,
                            betas=(0.9, 0.999), weight_decay=1e-5)
    sched1 = CosineAnnealingLR(opt1, T_max=100, eta_min=1e-6)

    for epoch in range(1, 101):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, opt1, criterion,
                                    device, phase=1, rate_reg=True)
        val_loss, val_acc, preds, labels = run_val(model, val_loader, criterion, device)
        sched1.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % 10 == 0:
            stats = model.count_synapses()
            log.info(
                f"P1 Epoch {epoch:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
                f"| sparsity={stats['sparsity']:.2%} | {time.time()-t0:.1f}s"
            )

    # ---- Phase 2: BPTT fine-tune with lower LR ----------------------------
    log.info("Phase 2: 50 epochs, lr=5e-5, BPTT fine-tune")
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    opt2 = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999))
    sched2 = ReduceLROnPlateau(opt2, patience=5, factor=0.5, min_lr=1e-7)

    for epoch in range(1, 51):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, opt2, criterion,
                                    device, phase=2, rate_reg=False)
        val_loss, val_acc, preds, labels = run_val(model, val_loader, criterion, device)
        sched2.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % 10 == 0:
            log.info(
                f"P2 Epoch {epoch:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
                f"| best={best_val_acc:.3f} | {time.time()-t0:.1f}s"
            )

    # Save final artifacts
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    log.info(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    train(args.config)
