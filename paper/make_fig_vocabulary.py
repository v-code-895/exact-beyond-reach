"""Figure: a visual vocabulary of what the experiments do to a network (stylized; no data)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
INK, GREY, LGREY = "#1f1e1c", "#8f8b82", "#cfcbc2"
ORANGE, BLUE, GREEN, RED, PURPLE = "#eb6834", "#2a78d6", "#1baf7a", "#c0392b", "#8a5cc2"

def node(ax, p, ec=GREY, fc="white", r=0.16, lw=1.6, z=3):
    ax.add_patch(Circle(p, r, ec=ec, fc=fc, lw=lw, zorder=z))
def edge(ax, p, q, c=LGREY, lw=1.0, ls="-", z=1, a=1.0):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, ls=ls, zorder=z, alpha=a, solid_capstyle="round")
def layer_pos(x, n, yc=1.5, gap=0.55):
    return [(x, yc + (i - (n - 1) / 2) * gap) for i in range(n)]
def net(ax, xs, ns, ec=LGREY, nc=GREY, lw=0.9, skip=()):
    P = [layer_pos(x, n) for x, n in zip(xs, ns)]
    for L in range(len(P) - 1):
        for i, p in enumerate(P[L]):
            for j, q in enumerate(P[L + 1]):
                if (L, i, j) not in skip: edge(ax, p, q, ec, lw)
    for L in P:
        for p in L: node(ax, p, ec=nc)
    return P
def frame(ax, title, caption):
    ax.set_xlim(-0.3, 4.3); ax.set_ylim(-0.75, 3.35); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11.5, color=INK, fontweight="bold", loc="left")
    ax.text(2.0, -0.55, caption, ha="center", va="top", fontsize=8.6, color="#444", wrap=True)
def tag(ax, p, text, c, dx=0.0, dy=0.33, fs=8):
    ax.text(p[0] + dx, p[1] + dy, text, ha="center", va="center", fontsize=fs, color=c, fontweight="bold")

XS, NS = [0.2, 1.4, 2.6, 3.8], [3, 4, 4, 2]
fig, axs = plt.subplots(3, 3, figsize=(13.5, 13.6)); A = axs.ravel()

# 1 trained network
ax = A[0]; frame(ax, "1  A trained network", "Weights learned by gradient descent, inside an\narchitecture fixed before training.")
net(ax, XS, NS); ax.text(0.2, 2.75, "input", ha="center", fontsize=8.5, color=GREY); ax.text(3.8, 2.75, "output", ha="center", fontsize=8.5, color=GREY)

# 2 make room
ax = A[1]; frame(ax, "2  Make room, exactly", "New units (green) get small incoming weights and\nexactly zero outgoing weights (dashed): the network's\nfunction is unchanged, but the units can now learn.")
P = net(ax, XS, NS)
for L in (1, 2):
    g = (XS[L], P[L][-1][1] + 0.55)
    for p in P[L - 1]: edge(ax, p, g, GREEN, 1.1, z=2)
    for q in P[L + 1]: edge(ax, g, q, GREEN, 1.1, "--", z=2)
    node(ax, g, ec=GREEN, fc="#d8f3e8"); tag(ax, g, "new", GREEN)

# 3 install a frozen circuit
ax = A[2]; frame(ax, "3  Install a frozen circuit", "A wire (blue) carries a fact the network cannot compute,\nsuch as 'this pixel is in the right half', exactly through\nevery layer; an AND gate combines it with a detector.")
P = net(ax, XS, NS)
w = [(x, 2.95) for x in XS[:3]]
for a, b in zip(w[:-1], w[1:]): edge(ax, a, b, BLUE, 2.6, z=2)
for p in w: node(ax, p, ec=BLUE, fc="#dce9f8")
tag(ax, w[0], "wire", BLUE, dy=0.3)
gate = (3.15, 2.25); edge(ax, w[2], gate, BLUE, 2.6, z=2)
for p in P[2]: edge(ax, p, gate, ORANGE, 1.6, z=2)
node(ax, gate, ec=BLUE, fc="#dce9f8", r=0.2); ax.text(gate[0], gate[1], "AND", ha="center", va="center", fontsize=7, color=BLUE, fontweight="bold", zorder=4)
for q in P[3]: edge(ax, gate, q, BLUE, 2.6, z=2)

# 4 train around it
ax = A[3]; frame(ax, "4  Train around it", "Gradient descent keeps updating every orange weight;\nthe blue ones never change. The question of this paper:\nwhat does the orange part do with the blue part?")
P = net(ax, XS, NS, ec=ORANGE, nc=ORANGE, lw=1.1)
for a, b in zip(w[:-1], w[1:]): edge(ax, a, b, BLUE, 2.6, z=2)
for p in w: node(ax, p, ec=BLUE, fc="#dce9f8")
edge(ax, w[2], gate, BLUE, 2.6, z=2)
for p in P[2]: edge(ax, p, gate, ORANGE, 1.6, z=2)
node(ax, gate, ec=BLUE, fc="#dce9f8", r=0.2); ax.text(gate[0], gate[1], "AND", ha="center", va="center", fontsize=7, color=BLUE, fontweight="bold", zorder=4)
for q in P[3]: edge(ax, gate, q, BLUE, 2.6, z=2)
ax.text(gate[0] + 0.05, gate[1] + 0.42, "frozen", ha="center", fontsize=8, color=BLUE, fontweight="bold")

# 5 the reach law
ax = A[4]; frame(ax, "5  The reach law", "Each trained layer looks one cell further. Inputs farther\nfrom the gate than the total reach cannot change the\noutput, ever (aqua); within reach (orange), training can.")
n, cw, rows = 16, 0.25, [0.0, 0.62, 1.24, 1.86]
gate_cols = range(0, 8)
for r_i, y in enumerate(rows):
    for c in range(n):
        x = c * cw + 0.05
        if r_i == 3: fc = "#dce9f8" if c in gate_cols else "#eeeeee"
        else:
            reach = 3 - r_i; d = c - 7                                      # distance past the gate's last open column
            fc = "#fde4d8" if 1 <= d <= reach else ("#d2f1e5" if d > reach else "white")
        ax.add_patch(Rectangle((x, y), cw * 0.92, 0.42, fc=fc, ec=LGREY, lw=0.6, zorder=2))
for r_i in range(3):                                                        # the receptive field of the gate's last open cell
    y0, y1 = rows[r_i] + 0.42, rows[r_i + 1]
    lo, hi = 7 - (3 - r_i), 7 + (3 - r_i)
    if False: ax.add_patch(Polygon([[lo * cw + 0.05, y0], [(hi + 1) * cw + 0.05, y0], [(8 - (2 - r_i)) * cw + 0.05 + cw * (3 - r_i - 1) + cw, y1], [(7 - (2 - r_i)) * cw + 0.05, y1]],
                         closed=True, fc=ORANGE, alpha=0.10, ec="none", zorder=1))
ax.text(1.05, 2.48, "gate open (frozen)", ha="center", fontsize=8, color=BLUE, fontweight="bold")
ax.text(3.05, 2.48, "gate closed: exactly 0", ha="center", fontsize=8, color=INK)
ax.text(2.05, -0.28, "within reach", ha="center", fontsize=8, color=ORANGE, fontweight="bold")
ax.text(3.55, -0.28, "no effect, exactly", ha="center", fontsize=8, color=GREEN, fontweight="bold")
ax.text(-0.25, 0.2, "input", fontsize=8, color=GREY, rotation=90, va="center")

# 6 two exact modes
ax = A[5]; frame(ax, "6  Two exact modes", "A copy of the last layers (orange) sits behind a switch.\nSwitch off: the original network, exactly. Switch on: the\nbranch, trained on a new task (cats to anime, D4).")
P = [layer_pos(x, n) for x, n in zip(XS[:2], NS[:2])]
for p in P[0]:
    for q in P[1]: edge(ax, p, q, BLUE, 1.0)
for L in P:
    for p in L: node(ax, p, ec=BLUE, fc="#dce9f8")
top = [(2.6, 2.4 + (i - 1) * 0.42) for i in range(3)]; bot = [(2.6, 0.6 + (i - 1) * 0.42) for i in range(3)]
for p in P[1]:
    for q in top: edge(ax, p, q, BLUE, 1.0)
    for q in bot: edge(ax, p, q, ORANGE, 1.0)
for q in top: node(ax, q, ec=BLUE, fc="#dce9f8")
for q in bot: node(ax, q, ec=ORANGE, fc="#fde4d8")
sw = (3.55, 1.5)
for q in top + bot: edge(ax, q, sw, BLUE if q in top else ORANGE, 1.0)
ax.add_patch(FancyBboxPatch((sw[0] - 0.22, sw[1] - 0.22), 0.44, 0.44, boxstyle="round,pad=0.02", fc="white", ec=INK, lw=1.6, zorder=3))
ax.text(sw[0], sw[1], "switch", ha="center", va="center", fontsize=6.5, color=INK, zorder=4)
ax.text(2.6, 3.0, "original head: cats", ha="center", fontsize=8, color=BLUE, fontweight="bold")
ax.text(2.6, -0.1, "branch: anime", ha="center", fontsize=8, color=ORANGE, fontweight="bold")

# 7 graft an organ
ax = A[6]; frame(ax, "7  Graft an organ", "A frozen expert (blue) beside a trainable host. A gate,\nzero at first, mixes them. The host routes to the expert\nexactly the jobs it does better, then loses them (S1).")
host = net(ax, [0.2, 1.1, 2.0], [3, 3, 2], ec=ORANGE, nc=ORANGE)
org = [[(0.2 + 0.0, 0), (1.1, 0.0)]]
ox = [0.2, 1.1, 2.0]; oy = lambda n: [(-0.05 + 0.32 * i) for i in range(n)]
for (x0, x1) in ((0.2, 1.1), (1.1, 2.0)):
    for y0 in (2.75, 3.05):
        for y1 in (2.75, 3.05): edge(ax, (x0, y0), (x1, y1), BLUE, 1.3)
for x in ox:
    for y in (2.75, 3.05): node(ax, (x, y), ec=BLUE, fc="#dce9f8", r=0.13)
ax.text(1.1, 3.3, "frozen expert (denoiser)", ha="center", fontsize=8, color=BLUE, fontweight="bold")
g = (3.0, 1.5); out = (3.85, 1.5)
for p in host[-1]: edge(ax, p, g, ORANGE, 1.2)
edge(ax, (2.0, 2.9), g, BLUE, 1.6)
node(ax, g, ec=ORANGE, fc="#fde4d8", r=0.22); ax.text(g[0], g[1], "gate", ha="center", va="center", fontsize=7, color=ORANGE, fontweight="bold", zorder=4)
edge(ax, g, out, INK, 1.4); node(ax, out, ec=INK); ax.text(out[0], out[1] + 0.35, "output", ha="center", fontsize=8, color=GREY)
ax.text(1.1, 0.55, "host", ha="center", fontsize=8, color=ORANGE, fontweight="bold")

# 8 surgery
ax = A[7]; frame(ax, "8  Surgery: remove, replace", "Remove a block: the blocks on either side change most\nwhile it heals (S2). Replace one with a block from another\nnetwork: a sibling's is accepted, a stranger's slowly (S4).")
xs8 = [0.2, 1.0, 1.8, 2.6, 3.4]; ns8 = [2, 3, 3, 3, 2]; P = [layer_pos(x, n) for x, n in zip(xs8, ns8)]
for L in range(len(P) - 1):
    for p in P[L]:
        for q in P[L + 1]:
            c = RED if L == 1 else (ORANGE if L in (0, 2) else LGREY)
            edge(ax, p, q, c, 1.6 if L in (0, 2) else 1.0, "--" if L == 1 else "-")
for L, Lp in enumerate(P):
    for p in Lp: node(ax, p, ec=RED if L == 2 else (PURPLE if L == 3 else GREY), fc="#f7dad7" if L == 2 else ("#ece2f6" if L == 3 else "white"))
ax.plot([1.25, 1.55], [2.45, 0.55], color=RED, lw=3); ax.plot([1.25, 1.55], [0.55, 2.45], color=RED, lw=3)
ax.text(1.4, 2.9, "removed", ha="center", fontsize=8, color=RED, fontweight="bold")
ax.text(2.6, 2.9, "transplanted", ha="center", fontsize=8, color=PURPLE, fontweight="bold")

# 9 distil
ax = A[8]; frame(ax, "9  Teach a plain network", "A patched teacher (with its frozen circuit) labels data for a\nplain student. The student learns the behaviour, even one it\ncould not find alone (D5), but not the guarantee (A2).")
T = [layer_pos(x, n, yc=2.35, gap=0.32) for x, n in zip([0.2, 0.8, 1.4], [2, 3, 2])]
for L in range(2):
    for p in T[L]:
        for q in T[L + 1]: edge(ax, p, q, LGREY, 0.9)
for L in T:
    for p in L: node(ax, p, r=0.11)
gt = (1.1, 3.05); node(ax, gt, ec=BLUE, fc="#dce9f8", r=0.12); edge(ax, gt, T[2][0], BLUE, 2.0); edge(ax, gt, T[2][1], BLUE, 2.0)
ax.text(0.8, 3.3, "teacher (patched)", ha="center", fontsize=8, color=INK, fontweight="bold")
S = [layer_pos(x, n, yc=0.55, gap=0.32) for x, n in zip([2.4, 3.0, 3.6], [2, 3, 2])]
for L in range(2):
    for p in S[L]:
        for q in S[L + 1]: edge(ax, p, q, ORANGE, 1.1)
for L in S:
    for p in L: node(ax, p, ec=ORANGE, fc="#fde4d8", r=0.11)
ax.text(3.0, 1.15, "student (plain)", ha="center", fontsize=8, color=INK, fontweight="bold")
ax.annotate("", xy=(2.3, 0.75), xytext=(1.55, 2.15), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
ax.text(2.05, 1.65, "copy its\noutputs", fontsize=8, color=INK, ha="left")
ax.text(3.0, -0.05, "behaviour: yes   guarantee: no", ha="center", fontsize=8, color=INK)

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=LGREY, lw=2, label="original trained weights"), Line2D([0], [0], color=ORANGE, lw=2.5, label="trained in the experiment"),
           Line2D([0], [0], color=BLUE, lw=2.5, label="frozen: never changed by training"), Line2D([0], [0], color=GREEN, lw=2.5, label="grown room"),
           Line2D([0], [0], color=GREEN, lw=2, ls="--", label="exactly zero weight"), Line2D([0], [0], color=RED, lw=2.5, ls="--", label="removed"),
           Line2D([0], [0], color=PURPLE, lw=2.5, label="from another network")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(HERE, "figures", "fig_vocabulary.png"), dpi=130); print("ok")
