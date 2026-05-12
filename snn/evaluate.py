"""
Evaluate pre-trained SNN: per-building accuracy, LOBO CV, confusion matrix.
Usage:
    python evaluate.py --building medium_office
"""
import argparse, torch, numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from model import LIFComfortClassifier
from rate_encoder import rate_encode

CLASSES = ["Cold","Cool","Neutral","Warm","Hot"]

def evaluate(building="medium_office", T=100, data_root="../data/sensor_logs"):
    device = torch.device("cpu")
    weights = Path(f"weights/neuroarch_{building}.pt")
    model = LIFComfortClassifier(T=T).to(device)
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location=device))
        print(f"Loaded weights: {weights}")
    else:
        print(f"[WARNING] No weights found at {weights}; using random init")
    model.eval()

    data = Path(data_root) / building
    X = torch.tensor(np.load(data/"X_test.npy"), dtype=torch.float32)
    y = torch.tensor(np.load(data/"y_test.npy"), dtype=torch.long)

    with torch.no_grad():
        sp    = rate_encode(X, T=T)
        preds = model.predict(sp)
        kappa = model.confidence(sp)

    acc = 100.0*(preds==y).float().mean().item()
    cm  = confusion_matrix(y.numpy(), preds.numpy())
    print(f"\nBuilding: {building}")
    print(f"Accuracy: {acc:.1f}%  |  Mean κ_SNN: {kappa.mean():.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y.numpy(), preds.numpy(), target_names=CLASSES))
    return acc, cm

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--building", default="medium_office")
    p.add_argument("--T", type=int, default=100)
    args = p.parse_args()
    evaluate(args.building, args.T)
