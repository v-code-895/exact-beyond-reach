"""Figure 1 of the follow-up paper: the reach law on P1's task, drawn for reach r = 4 (32 columns, gate passes columns 0-15)."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
INK, BLUE, ORANGE, AQUA, GREY = "#1f1e1c", "#2a78d6", "#eb6834", "#1baf7a", "#e6e4df"
r, S = 4, 32
fig, ax = plt.subplots(figsize=(11, 3.6)); ax.set_xlim(-1, 33); ax.set_ylim(-2.6, 4.2); ax.axis("off")

def strip(y, spans, label):
    for x0, x1, col, txt in spans:
        ax.add_patch(Rectangle((x0, y), x1 - x0, 1, facecolor=col, edgecolor="white", lw=2))
        if txt: ax.text((x0 + x1) / 2, y + 0.5, txt, ha="center", va="center", fontsize=9, color="white" if col != GREY else INK)
    ax.text(-0.8, y + 0.5, label, ha="right", va="center", fontsize=9.5, color=INK)

ax.text(24, 2.9, "gate sets the score to exactly 0", ha="center", va="center", fontsize=9, color=INK, zorder=5)
strip(2.4, [(0, 16, BLUE, "gate passes the detector's score (columns 0–15)"), (16, 32, GREY, "")], "frozen contract")
strip(0.6, [(0, 16, BLUE, "evidence"), (16, 16 + r, ORANGE, "within reach"), (16 + r, S - r, AQUA, "no effect, exactly"),
            (S - r, S, ORANGE, "within reach")], "input columns")
for x in (16, 16 + r):
    ax.plot([x, x], [0.55, 1.65], color=INK, lw=0.8, ls=":")
ax.annotate("", xy=(16, 0.35), xytext=(16 + r, 0.35), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
ax.text(16 + r / 2, 0.05, "reach r of the\ntrainable detector", ha="center", va="top", fontsize=8.5, color=INK)
ax.text(16, -1.35, "Guarantee (a theorem): no input change beyond the reach can move the output, for any weights.\n"
        "Renegotiation (measured): training moves a misplaced boundary back by about r + 1.4 × object width,\n"
        "less on cluttered images; anything within reach, the boundary included, is open to training.",
        ha="center", va="top", fontsize=9, color=INK)
ax.text(16, 3.85, "P1's task: is the red blob in the left half? A shift-equivariant network cannot answer it; a frozen gate makes it learnable.", ha="center", fontsize=9.5, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig_reach_schematic.png"), dpi=150); print("ok")
