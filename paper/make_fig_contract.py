"""Figure: what a frozen contract is, with the real maps of trained networks (renders: src/fig_contract_maps.py, src/fig_gan_renders.py)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results", "fig")
Cm = np.load(os.path.join(RES, "contract_maps.npz")); G = np.load(os.path.join(RES, "gan_renders.npz"))
INK, ORANGE, BLUE, GREY = "#1f1e1c", "#eb6834", "#2a78d6", "#d9d6cf"
TRAIN = dict(boxstyle="round,pad=0.35", fc="#fde4d8", ec=ORANGE, lw=1.6); FROZ = dict(boxstyle="round,pad=0.35", fc="#dce9f8", ec=BLUE, lw=1.6)
NEUT = dict(boxstyle="round,pad=0.35", fc="white", ec="#9a978f", lw=1.0)
wirecmap = ListedColormap(["white", GREY]); sig = lambda z: 1 / (1 + np.exp(-z))

fig = plt.figure(figsize=(13, 12.2)); gs = fig.add_gridspec(6, 5, height_ratios=[0.42, 1, 1, 1, 0.42, 1.12], hspace=0.12, wspace=0.08)
heads = [("input image", NEUT), ("trainable detector:\nconv layers → score", TRAIN), ("frozen wire\n(grey = closed)", FROZ),
         ("frozen clamp + gate:\nopen passes, closed = 0", FROZ), ("frozen average\n→ answer", FROZ)]
def header(row, labels, title):
    ax = fig.add_subplot(gs[row, :]); ax.axis("off"); ax.set_xlim(0, 5); ax.set_ylim(0, 1)
    ax.text(0.0, 0.98, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    for j, (t, st) in enumerate(labels):
        ax.text(j + 0.5, 0.26, t, ha="center", va="center", fontsize=9.6, color=INK, bbox=st)
        if j < 4: ax.annotate("", xy=(j + 1.12, 0.26), xytext=(j + 0.88, 0.26), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4), zorder=0)
header(0, heads, "(a)  A contract on a classifier that cannot tell left from right by itself (P1, L2)")
fig.text(0.5, 0.905, "Orange: trained by gradient descent.   Blue: frozen, never changed by training.   Score maps: red positive, blue negative.", ha="center", fontsize=9.5, color="#555")
rows = [("p1", 0, "synthetic, blob left"), ("p1", 1, "synthetic, blob right"), ("l2", 0, "cat photo, disc left")]
for i, (pre, k, lab) in enumerate(rows):
    x = Cm[pre + "_x"][k].transpose(1, 2, 0).clip(0, 1); u = Cm[pre + "_u"][k]; g = Cm[pre + "_gated"][k]; off = Cm[pre + "_off"]; lg = float(Cm[pre + "_logit"][k])
    lim = max(np.abs(u).max(), 1e-6)
    panels = [(x, None), (u, "RdBu_r"), (off, wirecmap), (g, "RdBu_r")]
    for j, (img, cm) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i, j]); ax.set_xticks([]); ax.set_yticks([])
        if cm is None: ax.imshow(img, interpolation="nearest")
        elif cm == "RdBu_r": ax.imshow(img, cmap=cm, vmin=-lim, vmax=lim, interpolation="nearest")
        else: ax.imshow(img, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        if j == 0: ax.set_ylabel(lab, fontsize=9, color=INK)
        if j == 3:
            h, w = img.shape; ax.text(w * 0.75, h * 0.5, "exactly 0", ha="center", va="center", fontsize=9, color=INK)
    ax = fig.add_subplot(gs[1 + i, 4]); ax.axis("off")
    ax.text(0.5, 0.55, f"P(left) = {sig(lg):.2f}", ha="center", va="center", fontsize=13, color=INK)
    ax.text(0.5, 0.33, "correct" if (sig(lg) > 0.5) == (k == 0) else "wrong", ha="center", fontsize=10, color="#3a7d44" if (sig(lg) > 0.5) == (k == 0) else "#b3261e")
heads_b = [("generator's image\n(before the circuit)", NEUT), ("trainable detector:\n3×3 conv, block 5", TRAIN), ("frozen wire\n(right half closed)", FROZ),
           ("frozen AND gate:\nReLU(score − M·wire)", FROZ), ("frozen tint added\n→ output image", FROZ)]
header(4, heads_b, "(b)  The same pattern inside the cat generator: odd eyes (C1–C1d)")
r = 1
for j, (img, cm) in enumerate([(G["room"][r].transpose(1, 2, 0), None), (G["detector_score"][r], "RdBu_r"), (G["wire_right"], wirecmap),
                               (G["gated"][r], "magma"), (G["trained_detector"][r].transpose(1, 2, 0), None)]):
    ax = fig.add_subplot(gs[5, j]); ax.set_xticks([]); ax.set_yticks([])
    if cm is None: ax.imshow(img, interpolation="lanczos")
    elif cm == "RdBu_r":
        lim = np.abs(img).max(); ax.imshow(img, cmap=cm, vmin=-lim, vmax=lim)
    elif cm == "magma": ax.imshow(img, cmap=cm, vmin=0, vmax=max(img.max(), 1e-6)); ax.text(96, 64, "exactly 0", ha="center", color="white", fontsize=9)
    else: ax.imshow(img, cmap=cm, vmin=0, vmax=1)
fig.savefig(os.path.join(HERE, "figures", "fig_contract_diagram.png"), dpi=130, bbox_inches="tight"); print("ok")
