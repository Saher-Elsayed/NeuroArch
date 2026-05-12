"""
NeuroArch SNN two-phase training.
Phase 1: rate-code pretraining  (100 epochs, cosine LR)
Phase 2: BPTT surrogate fine-tuning (50 epochs) + rate regulariser

Usage:
    python train.py --config configs/medium_office.yaml --seed 42
"""
import argparse, yaml, random, time, json
import torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from model import LIFComfortClassifier
from focal_loss import FocalLoss
from rate_encoder import rate_encode


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train(cfg, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Building: {cfg['building']} | Seed: {seed}")

    data = Path(cfg["data_dir"])
    X  = torch.tensor(np.load(data/"X_train.npy"), dtype=torch.float32)
    y  = torch.tensor(np.load(data/"y_train.npy"), dtype=torch.long)
    Xv = torch.tensor(np.load(data/"X_val.npy"),   dtype=torch.float32)
    yv = torch.tensor(np.load(data/"y_val.npy"),   dtype=torch.long)

    counts = torch.bincount(y, minlength=5).float()
    alpha  = (1.0 / counts) / (1.0 / counts).sum()
    loader = DataLoader(TensorDataset(X, y), batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=0)

    model = LIFComfortClassifier(T=cfg["T"]).to(device)
    crit  = FocalLoss(gamma=2.0, alpha=alpha.to(device))
    log   = []

    # Phase 1
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr_phase1"], betas=(0.9,0.999))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs_phase1"])
    for ep in range(1, cfg["epochs_phase1"]+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            sp = rate_encode(xb, T=cfg["T"]).to(device)
            cnt, _ = model(sp); loss = crit(cnt, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 10 == 0:
            acc = _eval(model, Xv, yv, cfg["T"], device)
            log.append({"phase":1,"epoch":ep,"val_acc":acc})
            print(f"  P1 ep {ep:3d}/{cfg['epochs_phase1']} | val {acc:.1f}%")

    # Phase 2
    opt2 = torch.optim.Adam(model.parameters(), lr=cfg["lr_phase2"])
    lam  = cfg.get("rate_reg", 0.01)
    for ep in range(1, cfg["epochs_phase2"]+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            sp = rate_encode(xb, T=cfg["T"]).to(device)
            cnt, all_sp = model(sp)
            loss = crit(cnt, yb) + lam * all_sp.mean()
            opt2.zero_grad(); loss.backward(); opt2.step()
        if ep % 10 == 0:
            acc = _eval(model, Xv, yv, cfg["T"], device)
            log.append({"phase":2,"epoch":ep,"val_acc":acc})
            print(f"  P2 ep {ep:3d}/{cfg['epochs_phase2']} | val {acc:.1f}%")

    out = Path(cfg["output_dir"])
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out/f"neuroarch_{cfg['building']}.pt")
    meta = {"building":cfg["building"],"seed":seed,"final_val_acc":log[-1]["val_acc"],"log":log}
    (out/f"neuroarch_{cfg['building']}_meta.json").write_text(json.dumps(meta,indent=2))
    print(f"Saved to {out}")


def _eval(model, X, y, T, device):
    model.eval()
    with torch.no_grad():
        preds = model.predict(rate_encode(X.to(device), T=T))
    return 100.0*(preds.cpu()==y).float().mean().item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/medium_office.yaml")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.seed)
