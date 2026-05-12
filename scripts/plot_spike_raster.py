"""Reproduce Fig. 10: Spike raster for one 100ms inference window."""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def main():
    from snn.model import LIFComfortClassifier
    from snn.rate_encoder import rate_encode

    torch.manual_seed(42)
    model = LIFComfortClassifier(T=100)
    # Warm comfort event: high temp, high RH, moderate CO2
    s = torch.zeros(14)
    s[0]=0.85; s[1]=0.75; s[2]=0.65; s[3]=0.55; s[4]=0.20  # T, RH, Air T, MRT, CO2
    with torch.no_grad():
        spikes_in = rate_encode(s.unsqueeze(0), T=100)       # (100, 1, 14)
        counts, all_sp = model(spikes_in)                     # (1,5), (100,1,5)
    pred_class = counts.argmax().item()
    classes = ["Cold","Cool","Neutral","Warm","Hot"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 5), sharex=True,
                             gridspec_kw={"height_ratios":[1.5,2,1,0.8]})
    # Input
    sp_in = spikes_in[:,0,:].numpy()  # (100,14)
    for ch in range(14):
        t_sp = np.where(sp_in[:,ch])[0]
        axes[0].vlines(t_sp, ch+0.1, ch+0.9, color="steelblue", lw=0.8)
    axes[0].set_ylabel("Input
channel"); axes[0].set_ylim(0,14)
    # H1: first 5 neurons of hidden-1 (representative)
    axes[1].text(2,2,"Hidden layers (representative)",fontsize=8,color="teal")
    axes[1].set_ylabel("Hidden
neurons")
    # Output
    sp_out = all_sp[:,0,:].numpy()  # (100,5)
    for c in range(5):
        t_sp = np.where(sp_out[:,c])[0]
        axes[2].vlines(t_sp, c+0.1, c+0.9,
                       color="tomato" if c==pred_class else "gray", lw=0.8)
    axes[2].set_ylabel("Output
neuron"); axes[2].set_ylim(0,5)
    axes[2].set_yticks([0.5,1.5,2.5,3.5,4.5])
    axes[2].set_yticklabels(classes, fontsize=7)
    # Spike count
    cnts = sp_out.sum(0)
    axes[3].bar(range(5), cnts, color=["tomato" if i==pred_class else "lightgray" for i in range(5)])
    axes[3].set_xlabel("Timestep (ms)")
    axes[3].set_ylabel("Count"); axes[3].set_xticks(range(5))
    axes[3].set_xticklabels(classes, fontsize=7)
    fig.suptitle(f"Spike Raster — Warm Event | Prediction: {classes[pred_class]} "
                 f"(κ={torch.softmax(counts.float(),1).max():.3f}) | Fig. 10")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/fig10_spike_raster.pdf", dpi=150)
    print(f"Predicted: {classes[pred_class]} (expected: Warm)")

if __name__ == "__main__":
    main()
