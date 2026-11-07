"""
C1d: eight units split on the output side (see PREREG.md): each unit is the full closed-form detector plus noise, with a fixed
output weight of 1/8, so the eight relu outputs average to the closed form at install. Same cache, loss, evaluator as C1b/C1c.

  <venv python> src/c1d_output_split.py
"""
import os, time
import numpy as np
import common as C
import torch, torch.nn.functional as F
import c1_trained_detector as c1, c1b_tradeoff as B, c1c_warm_units as W

OUT = C.out_dir("c1d"); DEV = "cuda"; LAMBDAS = (1, 10, 100, 1000, 10000); K_UNITS = 8

def train_c(cache, v, K0, b0, c, lam, seed):
    """C1b's training loop with fixed output weights c (C1b's has c = 1)."""
    H, XR, ML = cache; torch.manual_seed(seed)
    K = K0.clone().to(DEV).requires_grad_(); b = b0.clone().to(DEV).requires_grad_(); c = c.to(DEV)
    opt = torch.optim.Adam([K, b], B.LR); g = torch.Generator().manual_seed(seed); log = []
    T = c1.TARGET.to(DEV); vv = v[None, :, None, None]
    for step in range(1, B.STEPS + 1):
        idx = np.sort(torch.randint(0, len(XR), (B.BATCH,), generator=g).numpy())
        h = torch.from_numpy(np.ascontiguousarray(H[idx])).to(DEV).float(); xr = torch.from_numpy(XR[idx]).to(DEV); m = torch.from_numpy(ML[idx]).to(DEV)
        u = F.conv2d(h, K, b, padding=1)[..., :64]; d = (F.relu(u) * c[None, :, None, None]).sum(1, keepdim=True) * vv; x = xr + d
        tgt = torch.einsum("nchw,dc->ndhw", xr.clamp(0, 1), T); far = ~B.dilate(m)
        l_iris = ((x - tgt) ** 2).sum(1)[m].mean(); l_out = (d ** 2).sum(1)[far].mean()
        loss = l_iris + lam * l_out; opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: log.append({"step": step, "iris": float(l_iris.detach()), "outside": float(l_out.detach()), "active_fraction": float((u.detach() > 0).float().mean())})
    return K.detach(), b.detach(), c, log

def main():
    t0 = time.time(); W0 = C.catgan_base(); Wl, W_room, meta = C.odd_eye("left"); s5, (g0, _) = int(meta["slots"]["b5c1"]), meta["gate_slots"]
    gg = 1.0 / (1 - C.A ** 2); v = torch.tensor(Wl["out/kernel"][1, 1, g0, :] / gg, device=DEV); M = float(-Wl["b5c2/kernel"][1, 1, s5, g0])
    room = c1.Room(W_room, meta); ev = B.Evaluator(room, W_room, W0, meta, v, M)
    zpool = torch.tensor(np.random.default_rng(2026).standard_normal((B.N_FIT, 128)), dtype=torch.float32, device=DEV)
    cf = np.load(os.path.join(C.RESULTS, "c1b", "closed_form.npz")); Kc = torch.tensor(cf["K"]); bc = torch.tensor(cf["b_q0.97"])
    c8 = torch.full((K_UNITS,), 1.0 / K_UNITS)
    with torch.no_grad():                                                             # exactness of the split at install (no noise)
        h = torch.randn(2, 32, 16, 16); one = F.relu(F.conv2d(h, Kc, bc, padding=1))
        eight = (F.relu(F.conv2d(h, Kc.repeat(K_UNITS, 1, 1, 1), bc.repeat(K_UNITS), padding=1)) * c8[None, :, None, None]).sum(1, keepdim=True)
    R = {"split_exact_max_abs": float((one - eight).abs().max()), "runs": {}}; print("output split vs 1 unit:", R["split_exact_max_abs"], flush=True)
    cache = B.build_cache(room, zpool); print("cache", f"{time.time() - t0:.0f}s", flush=True)
    for lam in LAMBDAS:
        key = f"8 units, output-side split of the closed form | lambda {lam}"; R["runs"][key] = []
        for seed in (0, 1, 2):
            gen = torch.Generator().manual_seed(seed); K0 = Kc.repeat(K_UNITS, 1, 1, 1) + 0.05 * float(Kc.std()) * torch.randn((K_UNITS,) + tuple(Kc.shape[1:]), generator=gen)
            b0 = bc.repeat(K_UNITS); Kt, bt, ct, log = train_c(cache, v, K0, b0, c8, lam, seed)
            r = ev(Kt.cpu(), bt.cpu(), ct.cpu()); r["log"] = log; r["dead_units"] = W.dead_units(cache, Kt, bt); R["runs"][key].append(r)
            np.savez(f"{OUT}/det_o8_lam{lam}_s{seed}.npz", K=Kt.cpu().numpy(), b=bt.cpu().numpy(), c=c8.numpy())
            print(key, seed, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk != "log"}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{OUT}/c1d_results.json")
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/c1d_results.json")
    lines = ["# C1d: eight units, output-side split", "", "| lambda | left irises in band | outside the eyes (RGB levels) | dead units (of 8) | right-half margin (max) | right half, float64 (max) |", "|---|---|---|---|---|---|"]
    for key, runs in R["runs"].items():
        f = lambda kk: f"{np.mean([x[kk] for x in runs]):.3f} ± {np.std([x[kk] for x in runs]):.3f}"
        lines.append(f"| {key.split('lambda ')[1]} | {f('left_share_in_band')} | {f('rest_mae_rgb_levels')} | {[x['dead_units'] for x in runs]} | "
                     f"{max(x['right_margin_max_u_minus_M'] for x in runs):.1f} | {max(x['right_half_max_change_float64'] for x in runs):.1e} |")
    open(f"{OUT}/c1d_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
