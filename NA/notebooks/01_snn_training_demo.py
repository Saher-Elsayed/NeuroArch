# %% [markdown]
# # NeuroArch SNN Training Demo
# Demonstrates the two-phase surrogate gradient training pipeline.
# Paper: NeuroArch — IEEE Access MS ID: Access-2026-16730

# %% Setup
import sys; sys.path.insert(0, "..")
import torch, numpy as np, matplotlib.pyplot as plt, pandas as pd
from snn.model import NeuroArchSNN, SNNConfig

cfg = SNNConfig(T=100)
model = NeuroArchSNN(cfg)
print(f"Model: {model.n_parameters:,} parameters")
print(model.count_synapses())

# %% Load training curves (pre-computed)
curves = pd.read_csv("../data/ablations/training_curves.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(curves.epoch, curves.train_accuracy, label="Train")
axes[0].plot(curves.epoch, curves.val_accuracy,   label="Val")
axes[0].axhline(0.918, color='r', linestyle='--', label="Paper (91.8%)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()
axes[0].set_title("Classification Accuracy")

axes[1].plot(curves.epoch, curves.train_loss, label="Train")
axes[1].plot(curves.epoch, curves.val_loss,   label="Val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Focal Loss"); axes[1].legend()
axes[1].set_title("Loss Curves")

axes[2].plot(curves.epoch, curves.sparsity, color='green')
axes[2].axhline(0.79, color='r', linestyle='--', label="Target (79%)")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Weight Sparsity"); axes[2].legend()
axes[2].set_title("Network Sparsity")
plt.tight_layout(); plt.savefig("../data/ablations/training_curves_plot.png", dpi=150)
plt.show()

# %% Spike raster visualization
T = 50
x = torch.randn(1, T, 14)
with torch.no_grad():
    traces = model.get_membrane_traces(x)
    logits = model(x)

# Plot spike raster for layer 1
layer0_trace = traces[0][0].numpy()  # (T, 64)
spike_raster = (layer0_trace > 1.0).astype(int)
plt.figure(figsize=(12, 5))
for neuron_idx in range(min(32, spike_raster.shape[1])):
    times = np.where(spike_raster[:, neuron_idx])[0]
    plt.vlines(times, neuron_idx - 0.4, neuron_idx + 0.4, color='navy', linewidth=1.5)
plt.xlabel("Timestep"); plt.ylabel("Neuron Index")
plt.title("Spike Raster Plot — Layer 1 (first 32 neurons)")
plt.tight_layout(); plt.savefig("../data/ablations/spike_raster.png", dpi=150)
plt.show()

# %% Comfort class prediction example
probs = logits.softmax(-1).squeeze().numpy()
classes = ["Cold", "Cool", "Neutral", "Warm", "Hot"]
plt.bar(classes, probs, color=["#1f77b4","#aec7e8","#2ca02c","#ffbb78","#d62728"])
plt.ylabel("Probability"); plt.title("SNN Comfort Classification Output")
plt.tight_layout(); plt.show()
