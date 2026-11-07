"""
C1c: eight detector units, each warm-started as 1/8 of the closed-form detector (see PREREG.md). Reuses C1b's cache,
training, evaluator and saved closed-form fit, so the runs are directly comparable with C1b's.

  <venv python> src/c1c_warm_units.py
"""
import os, time
import numpy as np
import common as C
import torch, torch.nn.functional as F
import c1_trained_detector as c1, c1b_tradeoff as B

OUT = C.out_dir("c1c"); DEV = "cuda"; LAMBDAS = (1, 10, 100, 1000, 10000); K_UNITS = 8

def dead_units(cache, K, b, n=64):
    H, _, _ = cache; h = torch.from_numpy(np.ascontiguousarray(H[:n])).to(DEV).float()
    with torch.no_grad(): u = F.conv2d(h, K.to(DEV), b.to(DEV), padding=1)[..., :64]
    return int((u.amax((0, 2, 3)) <= 0).sum())

def main():
    t0 = time.time(); W0 = C.catgan_base(); Wl, W_room, meta = C.odd_eye("left"); s5, (g0, _) = int(meta["slots"]["b5c1"]), meta["gate_slots"]
    gg = 1.0 / (1 - C.A ** 2); v = torch.tensor(Wl["out/kernel"][1, 1, g0, :] / gg, device=DEV); M = float(-Wl["b5c2/kernel"][1, 1, s5, g0])
    room = c1.Room(W_room, meta); ev = B.Evaluator(room, W_room, W0, meta, v, M)
    zpool = torch.tensor(np.random.default_rng(2026).standard_normal((B.N_FIT, 128)), dtype=torch.float32, device=DEV)
    cf = np.load(os.path.join(C.RESULTS, "c1b", "closed_form.npz")); Kc = torch.tensor(cf["K"]); bc = torch.tensor(cf["b_q0.97"])
    R = {"closed_form_q0.97": ev(Kc, bc, torch.ones(1)), "runs": {}}; print("closed form q0.97", {k: round(x, 4) for k, x in R["closed_form_q0.97"].items()}, flush=True)
    cache = B.build_cache(room, zpool); print("cache", f"{time.time() - t0:.0f}s", flush=True)
    for lam in LAMBDAS:
        key = f"8 units, warm-started from the closed form | lambda {lam}"; R["runs"][key] = []
        for seed in (0, 1, 2):
            g = torch.Generator().manual_seed(seed); noise = 0.05 * float(Kc.std()) / K_UNITS * torch.randn((K_UNITS,) + tuple(Kc.shape[1:]), generator=g)
            K0 = Kc.repeat(K_UNITS, 1, 1, 1) / K_UNITS + noise; b0 = bc.repeat(K_UNITS) / K_UNITS
            Kt, bt, ct, log = B.train(cache, v, K0, b0, K_UNITS, lam, seed); r = ev(Kt.cpu(), bt.cpu(), ct.cpu()); r["log"] = log
            r["dead_units"] = dead_units(cache, Kt, bt); R["runs"][key].append(r)
            np.savez(f"{OUT}/det_w8_lam{lam}_s{seed}.npz", K=Kt.cpu().numpy(), b=bt.cpu().numpy())
            print(key, seed, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk != "log"}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{OUT}/c1c_results.json")
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/c1c_results.json")
    lines = ["# C1c: eight units, warm-started from the closed form", "", "| lambda | left irises in band | outside the eyes (RGB levels) | dead units (of 8) | right-half margin (max) | right half, float64 (max) |", "|---|---|---|---|---|---|"]
    for key, runs in R["runs"].items():
        f = lambda kk: f"{np.mean([x[kk] for x in runs]):.3f} ± {np.std([x[kk] for x in runs]):.3f}"
        lines.append(f"| {key.split('lambda ')[1]} | {f('left_share_in_band')} | {f('rest_mae_rgb_levels')} | {[x['dead_units'] for x in runs]} | "
                     f"{max(x['right_margin_max_u_minus_M'] for x in runs):.1f} | {max(x['right_half_max_change_float64'] for x in runs):.1e} |")
    open(f"{OUT}/c1c_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
