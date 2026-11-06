import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D_ = r"C:\Users\starb\Downloads\Weight Space Programming\followup\results\l1"
R = json.load(open(D_ + r"\l1_results.json"))
DEPTHS, SIGMAS, BS = (2, 4, 6), (1, 2, 3), (12, 8, 4)
cells = {}; failed = []
for D in DEPTHS:
    for s in SIGMAS:
        key = f"D{D}_sigma{s}"; best = None
        for b in BS:
            rs = [r for r in R[key][str(b)] if r["acc"] >= 0.6]
            failed += [(key, b, r["seed"], round(r["acc"], 3), r["effective_boundary"]) for r in R[key][str(b)] if r["acc"] < 0.6]
            if not rs: continue
            corr = np.mean([min(r["effective_boundary"], 16) - b for r in rs]); censored = corr >= (16 - b) - 1          # within 1 column of the cap (the shift)
            if best is None or corr > best[0]: best = (corr, b, censored)
        cells[(D, s)] = best
print("failed runs (acc < 60%):", failed)
for (D, s), (k, b, c) in cells.items(): print(f"r={D} sigma={s}: k*={k:.1f} (at b={b}){' censored' if c else ''}")
unc = [(D, s, k) for (D, s), (k, b, c) in cells.items() if not c]
a = sum(s * (k - D) for D, s, k in unc) / sum(s * s for D, s, k in unc)
print("fitted a over uncensored cells:", round(a, 3)); res = {(D, s): k - (D + a * s) for (D, s), (k, b, c) in cells.items()}
for (D, s), v in res.items(): print(f"  residual r={D} s={s}: {v:+.2f}{' (censored: lower bound)' if cells[(D, s)][2] else ''}")
# prediction 4: accuracy cap where b + k* < 16
print("prediction 4 (acc vs 1 - (16 - b - k*)/32):")
p4 = []
for D in DEPTHS:
    for s in SIGMAS:
        k = cells[(D, s)][0]; key = f"D{D}_sigma{s}"
        for b in BS:
            if b + k < 16:
                rs = [r for r in R[key][str(b)] if r["acc"] >= 0.6]; acc = np.mean([r["acc"] for r in rs]); pred = 1 - (16 - b - k) / 32
                p4.append((key, b, acc, pred, acc - pred)); print(f"  {key} b={b}: acc {acc*100:.1f} pred {pred*100:.1f} diff {100*(acc-pred):+.1f}")
print("within 3 points:", sum(abs(d) <= 0.03 for *_, d in p4), "of", len(p4))
# guarantee summary
g_ok = all(all(v == 0.0 for d, v in r["guarantee_max_logit_change_by_d"].items() if int(d) >= D) for D in DEPTHS for s in SIGMAS for r in R[f"D{D}_sigma{s}"]["16"])
g_in = min(r["guarantee_max_logit_change_by_d"][str(D - 1)] for D in DEPTHS for s in SIGMAS for r in R[f"D{D}_sigma{s}"]["16"])
print("guarantee: exactly 0 at d >= r in all models:", g_ok, "| smallest change at d = r - 1:", g_in)
json.dump({"k_star": {f"r{D}_sigma{s}": {"k": float(k), "at_b": int(b), "censored": bool(c)} for (D, s), (k, b, c) in cells.items()}, "a": a,
           "residuals": {f"r{D}_sigma{s}": v for (D, s), v in res.items()}, "failed_runs": failed, "prediction4": [(k_, int(b_), float(a_), float(p_), float(d_)) for k_, b_, a_, p_, d_ in p4],
           "guarantee_zero_beyond_reach_all": bool(g_ok), "smallest_change_at_r_minus_1": g_in}, open(D_ + r"\l1_analysis.json", "w"), indent=1)

INK, COL = "#1f1e1c", {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}; DCOL = {2: "#2a78d6", 4: "#eb6834", 6: "#1baf7a"}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axs[0]
for D in DEPTHS:
    ds = np.arange(0, D + 2); vals = np.array([[r["guarantee_max_logit_change_by_d"][str(d)] for d in ds] for s in SIGMAS for r in R[f"D{D}_sigma{s}"]["16"]]).max(0)
    nz = vals > 0; ax.plot(ds[nz], vals[nz], color=DCOL[D], lw=2, marker="o", ms=5, label=f"reach r = {D}")
    ax.plot(ds[~nz], np.full((~nz).sum(), 1e-3), ls="none", marker="x", ms=8, mew=2, color=DCOL[D])
    ax.axvline(D - 0.5, color=DCOL[D], lw=1, ls=":")
ax.set_yscale("log"); ax.set_ylim(5e-4, 100); ax.axhline(1e-3, color=INK, lw=0.8, ls="--")
ax.annotate("exactly 0.0 (×)", (4.9, 1.25e-3), color=INK, fontsize=9)
ax.set_xlabel("attacked input column: distance past the gate's boundary"); ax.set_ylabel("largest change of the output (logit), any attack")
ax.set_title("Guarantee: nothing beyond the reach can move the output", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=9, loc="upper right")
ax = axs[1]
for s in SIGMAS:
    xs = np.array(DEPTHS); ks = np.array([cells[(D, s)][0] for D in DEPTHS]); cen = np.array([cells[(D, s)][2] for D in DEPTHS])
    ax.plot(xs, ks, color=COL[s], lw=2, label=f"blob width σ = {s}")
    ax.plot(xs[~cen], ks[~cen], "o", color=COL[s], ms=7); ax.plot(xs[cen], ks[cen], "o", mfc="white", mec=COL[s], mew=2, ms=7)
    ax.plot(xs, xs + a * s, color=COL[s], lw=1, ls="--")
ax.plot([1.5, 6.5], [1.5, 6.5], color=INK, lw=1, ls=":"); ax.annotate("correction = reach", (5.2, 4.6), color=INK, fontsize=8.5)
ax.set_xticks(DEPTHS); ax.set_xlabel("reach r of the trainable trunk (conv layers)"); ax.set_ylabel("how far training moved the boundary back (columns)")
ax.set_title(f"Renegotiation: grows with reach and blob width (dashed: r + {a:.2f}·σ)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.annotate("hollow: within 1 column of the cap (the shift)", (3.0, 1.2), color=INK, fontsize=8.5)
for ax in axs:
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e4df", lw=0.8)
fig.suptitle("L1: the reach law on P1's task (72 runs)", x=0.01, ha="left", fontsize=11); fig.tight_layout(); fig.savefig(D_ + r"\fig_l1_reach_law.png", dpi=130); print("fig ok")
