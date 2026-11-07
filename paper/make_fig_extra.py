"""Three figures from saved data: the detector leaning toward a misplaced gate (P6b/L1), the LLM edit over training (E1, E1b),
and the host handing its job to a grafted expert (S1)."""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
INK, BLUE, ORANGE, AQUA, GREY = "#1f1e1c", "#2a78d6", "#eb6834", "#1baf7a", "#8f8b82"
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e4df", lw=0.8)

# 1 leaning toward a misplaced gate
C = np.load(os.path.join(RES, "fig", "contract_maps.npz")); sig = lambda z: 1 / (1 + np.exp(-z))
fig, axs = plt.subplots(2, 4, figsize=(12.5, 6.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1]))
for i, (pre, b, lab) in enumerate((("b16probe", 16, "gate boundary at 16\n(correct)"), ("b8", 8, "gate boundary at 8\n(misplaced by 8)"))):
    ax = axs[i, 0]; ax.imshow(C[pre + "_x"][2].transpose(1, 2, 0).clip(0, 1)); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel(lab, fontsize=10, color=INK)
    if i == 0: ax.set_title("probe: blob at column 14.5", fontsize=10)
    for k, cx in enumerate((6.5, 11.5, 14.5)):
        ax = axs[i, k + 1]; u = C[pre + "_u"][k]; lim = max(np.abs(C["b8_u"]).max(), np.abs(C["b16probe_u"]).max())
        ax.imshow(u, cmap="RdBu_r", vmin=-lim, vmax=lim); ax.set_xticks([]); ax.set_yticks([])
        ax.axvspan(b - 0.5, 31.5, color="#d9d6cf", alpha=0.55, lw=0); ax.axvline(b - 0.5, color=BLUE, lw=2)
        ax.axvline(15.5, color=INK, lw=1, ls="--"); ax.plot([cx - 0.5], [16], marker="x", color=INK, ms=7, mew=2)
        ax.set_xlim(-0.5, 31.5)
        if i == 0: ax.set_title(f"detector's score, blob at {cx}", fontsize=10)
        p = sig(float(C[pre + "_logit"][k])); ax.text(1, 30, f"P(left) = {p:.2f}", fontsize=9.5, color=INK, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8))
fig.text(0.5, 0.01, "Blue line: the frozen gate's boundary (grey: closed). Dashed: the true boundary at 16. ×: the blob's centre. With the gate misplaced, the trained detector "
         "spreads its response leftward, into the open region, so blobs at 11.5 and 14.5 are still counted as left.", ha="center", fontsize=9, color=INK, wrap=True)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(HERE, "figures", "fig_leaning.png"), dpi=130)

# 2 the LLM edit over training
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
for ax, (exp, title) in zip(axs, (("e1", "learning rate 10⁻⁵ (damages the model, E1)"), ("e1b", "learning rate 10⁻⁶ (does not, E1b)"))):
    R = json.load(open(os.path.join(RES, exp, f"{exp}_results.json")))["arms"]
    for arm, col, lab in (("unit, margin 30.0", BLUE, "grown unit (frozen-margin patch)"), ("fine-tuned edit (layer MLP)", ORANGE, "same edit by fine-tuning")):
        runs = R[arm]; st = [e["step"] for e in runs[0]["log"]]; P = np.array([[e["fit"]["p_new"] for e in r["log"]] for r in runs])
        ax.plot(st, P.mean(0), color=col, lw=2.2, label=lab); ax.fill_between(st, P.min(0), P.max(0), color=col, alpha=0.15, lw=0)
    ax.set_title(title, fontsize=10.5, loc="left"); ax.set_xlabel("steps of continued training"); ax.set_ylim(0, 1.03); clean(ax)
axs[0].set_ylabel("probability of \"Rome\"\n(20 paraphrases; band: 3 seeds)"); axs[0].legend(frameon=False, fontsize=9.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig_llm_edit.png"), dpi=130)

# 3 the host hands its job to the grafted expert
R = json.load(open(os.path.join(RES, "s1", "s1_results.json")))["runs"]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0))
for host, ls in (("adult", "-"), ("young", "--")):
    runs = R[f"{host} host + organ"]; st = [e["step"] for e in runs[0]["log"]]
    for cond, col, lab in (("noisy bicubic LR", BLUE, "noisy inputs"), ("bicubic LR", ORANGE, "clean inputs")):
        g = np.array([[e["mean_gate"][cond] for e in r["log"]] for r in runs]).mean(0); axs[0].plot(st, g, color=col, ls=ls, lw=2, label=f"{host} host, {lab}")
    h = np.array([[e["host_alone"]["noisy bicubic LR"] for e in r["log"]] for r in runs]).mean(0); axs[1].plot(st, h, color=AQUA if host == "adult" else GREY, ls=ls, lw=2.2, label=f"{host} host, on its own")
    ctrl = np.mean([r["log"][-1]["host_alone"]["noisy bicubic LR"] for r in R[f"{host} host, control"]])
    axs[1].axhline(ctrl, color=AQUA if host == "adult" else GREY, lw=1, ls=":"); axs[1].text(st[-1], ctrl + 0.08, f"{host} control (no expert)", ha="right", fontsize=8.5, color=INK)
axs[0].set_title("how much the gate routes to the expert", fontsize=10.5, loc="left"); axs[0].set_ylabel("mean gate opening"); axs[0].set_ylim(-0.03, 1.03)
axs[1].set_title("the host's own denoising (noisy inputs)", fontsize=10.5, loc="left"); axs[1].set_ylabel("PSNR change vs base (dB)")
for ax in axs: ax.set_xlabel("training step after the graft"); clean(ax); ax.legend(frameon=False, fontsize=8.5, loc="best")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "fig_graft.png"), dpi=130); print("ok")
