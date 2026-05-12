"""SNN evaluation utilities: per-class metrics, confusion matrix, ASHRAE alignment."""
import torch, numpy as np
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ["Cold", "Cool", "Neutral", "Warm", "Hot"]

def evaluate_model(model, loader, device):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(-1).cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)

def per_class_metrics(preds, labels):
    return classification_report(labels, preds, target_names=CLASSES, output_dict=True)

def ashrae_alignment(preds, labels):
    """ASHRAE-55 allows PMV ∈ [-0.5, 0.5] (classes 1,2,3 = Cool/Neutral/Warm)."""
    ok_classes = {1, 2, 3}
    ashrae_preds  = np.array([1 if p in ok_classes else 0 for p in preds])
    ashrae_labels = np.array([1 if l in ok_classes else 0 for l in labels])
    return (ashrae_preds == ashrae_labels).mean()
