# ---
# jupyter:
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---

# # Notebook 03: Confusion Matrix Analysis
# Reproduces Paper Table 6

# %%
import sys; sys.path.insert(0,'.')
import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors

# Paper confusion matrix (row=true, col=pred, values in %)
cm = np.array([
    [93.6, 6.4,  0.0,  0.0,  0.0],   # Cold
    [ 0.8,91.2,  8.0,  0.0,  0.0],   # Cool
    [ 0.0, 1.6, 96.3,  2.1,  0.0],   # Neutral
    [ 0.0, 0.0,  6.9, 92.4,  0.7],   # Warm
    [ 0.0, 0.0,  0.0,  7.8, 92.2],   # Hot
])
classes = ["Cold","Cool","Neutral","Warm","Hot"]
print(f"Mean accuracy: {np.diag(cm).mean():.1f}%  (paper: 94.3%)")
print(f"Health-critical (Hot->Neut): {cm[4,2]:.1f}%  (Cold->Neut): {cm[0,2]:.1f}%")

# %%
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("SNN Confusion Matrix (%, Table 6)")
plt.colorbar(im, ax=ax, label="Recall (%)")
for i in range(5):
    for j in range(5):
        color = "white" if cm[i,j] > 60 else "black"
        ax.text(j, i, f"{cm[i,j]:.1f}", ha="center", va="center",
                color=color, fontsize=9)
plt.tight_layout(); plt.show()
