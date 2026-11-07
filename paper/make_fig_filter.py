"""Figure: the frozen filter pulls the generator into what it filters (illustrative rerun, src/fig_s5_snapshots.py, feature filter, seed 0)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); S = np.load(os.path.join(HERE, "..", "results", "fig_s5", "feature_s0.npz"))
steps, idx = (0, 1000, 2000), list(range(8)); INK = "#1f1e1c"
shares = {0: 14, 1000: 36, 2000: 53}
fig, axs = plt.subplots(len(steps) * 2, 1, figsize=(12, 9.6))
for i, st in enumerate(steps):
    for j, (kind, lab) in enumerate((("unfiltered", "what the generator makes"), ("filtered", "after the frozen filter"))):
        ax = axs[2 * i + j]; ax.imshow(np.concatenate([S[f"{kind}_{st}"][k] for k in idx], 1)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(lab, fontsize=8.5, color=INK, rotation=0, ha="right", va="center")
        if j == 0: ax.set_title(f"step {st:,}: {shares[st]}% of the generator's cats are orange", fontsize=10.5, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig_filter_pull.png"), dpi=110); print("ok")
