"""Figure: odd-eyed cats inside frozen wiring vs a student without it (renders from src/fig_gan_renders.py)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = np.load(os.path.join(HERE, "..", "results", "fig", "gan_renders.npz"))
rows = (1, 7, 8); BLUE, INK = "#1060c8", "#1f1e1c"
cols = [("Base generator", "base", None), ("Closed-form patch\n(companion paper)", "closed_form", None),
        ("Detector trained inside\nthe frozen wire and gate", "trained_detector", None), ("What it changed", None, "trained_detector"),
        ("Student trained to copy,\nwith no wiring (D5)", "student_a", None), ("What the student changed", None, "student_a")]
fig, axs = plt.subplots(len(rows), len(cols), figsize=(13.2, 6.9))
for j, (title, key, diffkey) in enumerate(cols):
    axs[0, j].set_title(title, fontsize=11, color=BLUE if j in (2, 4) else INK, fontweight="bold" if j in (2, 4) else None)
    for i, r in enumerate(rows):
        ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if key: ax.imshow(R[key][r].transpose(1, 2, 0), interpolation="lanczos")
        else:
            d = np.abs(R[diffkey][r] - R["base"][r]).max(0); ax.imshow(d, cmap="magma", vmin=0, vmax=0.5); ax.axvline(63.5, color="white", ls="--", lw=1)
            lm, rm = d[:, :64].max(), d[:, 64:].max()
            rtxt = f"{rm:.1e}".replace("e-0", "×10$^{-").replace("e-", "×10$^{-") + "}$" if rm < 1e-3 else f"{rm:.2f}"
            ax.text(31, 118, f"left\nmax {lm:.2f}", color="white", ha="center", fontsize=8.5)
            ax.text(96, 118, f"right\nmax {rtxt}", color="white", ha="center", fontsize=8.5)
# fig.text(0.5, 0.01, "The detector inside the frozen wiring changes only the left half (right half unchanged to float rounding); the student learned the odd eyes from the patched teacher but changes the right half too (D5).",
#         ha="center", fontsize=9.5, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig_odd_eyes.png"), dpi=130); print("ok")
