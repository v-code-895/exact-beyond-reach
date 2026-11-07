"""
C1b: the detector's trade-off curve (see PREREG.md).

Same frozen skeleton as C1. The closed form's curve comes from sweeping its threshold quantile; Adam's from
sweeping the preservation weight lambda, with the preservation region matched to the evaluation's
(left-half pixels outside a 3-pixel dilation of the iris mask). The frozen network's b5c1 features for the
2,000 fitting latents are cached once (float16, left half plus one column) so the 45 runs are cheap.

  <venv python> src/c1b_tradeoff.py
"""
import os, time
import numpy as np
import common as C
import torch, torch.nn.functional as F
import exp20_odd_eyes as e20, cat_domain as cd, exp4_anime_eye_circuit as e4
from exp18_circuit_primitives import catgan_layers
import c1_trained_detector as c1

OUT = C.out_dir("c1b"); DEV = "cuda"; N_FIT, STEPS, BATCH, LR = 2000, 2000, 32, 1e-3
QUANTILES = (0.9, 0.95, 0.97, 0.98, 0.99, 0.995); LAMBDAS = (1, 10, 100, 1000, 10000)
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
CW = 65                                                                         # cached columns: the left half plus one

def ridge_fit(room, zpool, rng, per_img_pos=100, n_neg=50000):
    """C1's closed_form up to the threshold (same sampling order and rng), returning the kernel, its bias
    and the background scores from which thresholds are quantiles."""
    P, N = [], []; neg_per = int(np.ceil(n_neg / len(zpool)))
    for s in range(0, len(zpool), 50):
        x, h = room(zpool[s:s + 50]); m = c1.iris_mask(x).cpu().numpy(); hn = h.permute(0, 2, 3, 1).cpu().numpy()
        flat = m.reshape(-1); pos = np.nonzero(flat)[0]; neg = np.nonzero(~flat)[0]
        npos = min(len(pos), per_img_pos * len(m)); P.append(e4.patches3x3(hn, rng.choice(pos, npos, replace=False)) if npos else np.zeros((0, 288)))
        N.append(e4.patches3x3(hn, rng.choice(neg, neg_per * len(m), replace=False)))
    Xp, Xn = np.concatenate(P).astype(np.float64), np.concatenate(N).astype(np.float64)
    X = np.c_[np.r_[Xp, Xn], np.ones(len(Xp) + len(Xn))]; y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    sw = np.r_[np.full(len(Xp), len(Xn) / len(Xp)), np.ones(len(Xn))]
    w = np.linalg.solve((X * sw[:, None]).T @ X + np.eye(X.shape[1]), (X * sw[:, None]).T @ y)
    K = torch.tensor(w[:-1].reshape(3, 3, -1).transpose(2, 0, 1)[None], dtype=torch.float32)
    return K, float(w[-1]), X[len(Xp):] @ w

def build_cache(room, zpool):
    path = f"{C.CACHE}/c1b_h51_left_f16.npy"; xr_l = np.zeros((len(zpool), 3, 128, 64), np.float32); m_l = np.zeros((len(zpool), 128, 64), bool)
    H = np.lib.format.open_memmap(path, "w+", np.float16, (len(zpool), 32, 128, CW))
    for s in range(0, len(zpool), 50):
        x, h = room(zpool[s:s + 50]); H[s:s + 50] = h[..., :CW].half().cpu().numpy()
        xr_l[s:s + 50] = x[..., :64].cpu().numpy(); m_l[s:s + 50] = c1.iris_mask(x)[..., :64].cpu().numpy()
    H.flush(); del H
    return np.load(path, mmap_mode="r"), xr_l, m_l

def dilate(m, r=3):   # (N,H,W) bool -> square dilation by r, as the evaluation's np.roll dilation
    return F.max_pool2d(m.float()[:, None], 2 * r + 1, 1, r)[:, 0] > 0

def train(cache, v, K0, b0, k, lam, seed):
    H, XR, ML = cache; torch.manual_seed(seed)
    if K0 is None:
        K = (0.01 * torch.randn(k, 32, 3, 3, device=DEV)).requires_grad_(); b = torch.zeros(k, device=DEV, requires_grad=True)
    else:
        K = K0.clone().to(DEV).requires_grad_(); b = b0.clone().to(DEV).requires_grad_()
    c = torch.ones(k, device=DEV); opt = torch.optim.Adam([K, b], LR); g = torch.Generator().manual_seed(seed); log = []
    T = c1.TARGET.to(DEV); vv = v[None, :, None, None]
    for step in range(1, STEPS + 1):
        idx = np.sort(torch.randint(0, len(XR), (BATCH,), generator=g).numpy())
        h = torch.from_numpy(np.ascontiguousarray(H[idx])).to(DEV).float(); xr = torch.from_numpy(XR[idx]).to(DEV); m = torch.from_numpy(ML[idx]).to(DEV)
        u = F.conv2d(h, K, b, padding=1)[..., :64]; d = (F.relu(u) * c[None, :, None, None]).sum(1, keepdim=True) * vv; x = xr + d
        tgt = torch.einsum("nchw,dc->ndhw", xr.clamp(0, 1), T); far = ~dilate(m)
        l_iris = ((x - tgt) ** 2).sum(1)[m].mean(); l_out = (d ** 2).sum(1)[far].mean()
        loss = l_iris + lam * l_out; opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            log.append({"step": step, "iris": float(l_iris.detach()), "outside": float(l_out.detach()),
                        "active_fraction": float((u.detach() > 0).float().mean())})
    return K.detach(), b.detach(), c, log

class Evaluator:
    def __init__(self, room, W_room, W0, meta, v, M):
        self.W_room, self.meta, self.v, self.M = W_room, meta, v, M
        rng = np.random.default_rng(0); _, _, self.z = (rng.standard_normal((n, 128)) for n in (256, 300, 300))
        self.x0 = e20.forward(self.z, W0, catgan_layers(W0)); self.m = cd.eye_mask(np.clip(self.x0, 0, 1))
        left = np.zeros((128, 128), bool); left[:, :64] = True; self.left = left
        self.far = ~np.any([np.roll(np.roll(self.m, dy, 1), dx, 2) for dy in range(-3, 4) for dx in range(-3, 4)], 0)
        self.box = (c1.EYES.numpy() & left)[None] & self.far                     # left eye box, outside the iris dilation
        self.b64 = e20.forward(self.z[:16], W_room, catgan_layers(W_room), double=True)
        zt = torch.tensor(self.z, dtype=torch.float32, device=DEV); self.h_right = []
        for s in range(0, 300, 50):
            _, h = room(zt[s:s + 50]); self.h_right.append(h[..., 64 - 1:].half())   # right half plus one column, for the margin
    def __call__(self, K, b, c):
        W = c1.build(self.W_room, self.meta, K, b, c, self.v, self.M); L = catgan_layers(W)
        x = e20.forward(self.z, W, L); hl = e20.per_image_hue(x, self.m & self.left[None]); hr = e20.per_image_hue(x, self.m & ~self.left[None])
        diff = np.abs(np.clip(x, 0, 1) - np.clip(self.x0, 0, 1)).mean(-1) * 255
        a64 = e20.forward(self.z[:16], W, L, double=True)
        with torch.no_grad():
            margin = max(float((F.conv2d(h.float(), K.to(DEV), b.to(DEV), padding=1)[..., 1:]).max()) for h in self.h_right) - self.M
        return {"left_share_in_band": float(np.mean([200 <= h <= 250 for h in hl if h is not None])),
                "right_share_in_band": float(np.mean([200 <= h <= 250 for h in hr if h is not None])),
                "rest_mae_rgb_levels": float(diff[self.far].mean()),
                "rest_sum_share_in_left_eye_box": float(diff[self.box].sum() / max(diff[self.far].sum(), 1e-12)),
                "right_half_max_change_float64": float(np.abs(a64[:, :, 64:] - self.b64[:, :, 64:]).max()),
                "right_margin_max_u_minus_M": margin}

def main():
    t0 = time.time(); W0 = C.catgan_base(); Wl, W_room, meta = C.odd_eye("left"); s5, (g0, _) = int(meta["slots"]["b5c1"]), meta["gate_slots"]
    gg = 1.0 / (1 - C.A ** 2); v = torch.tensor(Wl["out/kernel"][1, 1, g0, :] / gg, device=DEV); M = float(-Wl["b5c2/kernel"][1, 1, s5, g0])
    room = c1.Room(W_room, meta); ev = Evaluator(room, W_room, W0, meta, v, M)
    zpool = torch.tensor(np.random.default_rng(2026).standard_normal((N_FIT, 128)), dtype=torch.float32, device=DEV)
    R = {"M": M, "closed_form": {}, "runs": {}}; one = torch.ones(1)
    Ka = torch.tensor(Wl["b5c2/kernel"][..., :s5, g0].transpose(2, 0, 1)[None], dtype=torch.float32); ba = torch.tensor([Wl["b5c2/bias"][g0]], dtype=torch.float32)
    R["paper_one"] = ev(Ka, ba, one); print("paper one", R["paper_one"], f"{time.time() - t0:.0f}s", flush=True)
    K, kb, p_neg = ridge_fit(room, zpool, np.random.default_rng(0)); dets = {}
    for q in QUANTILES:
        b = torch.tensor([kb - float(np.quantile(p_neg, q))], dtype=torch.float32); dets[q] = b
        R["closed_form"][str(q)] = ev(K, b, one); print("closed form", q, R["closed_form"][str(q)], flush=True)
    np.savez(f"{OUT}/closed_form.npz", K=K.numpy(), kb=kb, **{f"b_q{q}": dets[q].numpy() for q in QUANTILES}); C.save_json(R, f"{OUT}/c1b_results.json")
    cache = build_cache(room, zpool); print("cache built", f"{time.time() - t0:.0f}s", flush=True)
    types = {"1 unit, from the closed form (q = 0.97)": (K, dets[0.97], 1), "1 unit, random init": (None, None, 1), "8 units, random init": (None, None, 8)}
    for tname, (K0, b0, k) in types.items():
        for lam in LAMBDAS:
            key = f"{tname} | lambda {lam}"; R["runs"][key] = []
            for seed in (0, 1, 2):
                Kt, bt, ct, log = train(cache, v, K0, b0, k, lam, seed); r = ev(Kt.cpu(), bt.cpu(), ct.cpu()); r["log"] = log; R["runs"][key].append(r)
                np.savez(f"{OUT}/det_{['cf', 'r1', 'r8'][list(types).index(tname)]}_lam{lam}_s{seed}.npz", K=Kt.cpu().numpy(), b=bt.cpu().numpy(), c=ct.cpu().numpy())
                print(key, seed, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk != "log"}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{OUT}/c1b_results.json")
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/c1b_results.json")
    lines = ["# C1b: the detector's trade-off curve", "", "Left irises in band / change outside the eyes (RGB levels) on paper one's 300 test latents; Adam rows mean ± sd over seeds 0-2.", "",
             f"Paper one's patch: {R['paper_one']['left_share_in_band']:.3f} / {R['paper_one']['rest_mae_rgb_levels']:.3f}.", "",
             "| closed form, quantile | left irises in band | outside the eyes | share of that change inside the left eye box | right-half margin |", "|---|---|---|---|---|"]
    for q, r in R["closed_form"].items():
        lines.append(f"| {q} | {r['left_share_in_band']:.3f} | {r['rest_mae_rgb_levels']:.3f} | {r['rest_sum_share_in_left_eye_box']:.2f} | {r['right_margin_max_u_minus_M']:.1f} |")
    lines += ["", "| Adam detector, lambda | left irises in band | outside the eyes | share inside the left eye box | right-half margin (max) | right half, float64 (max) |", "|---|---|---|---|---|---|"]
    for key, runs in R["runs"].items():
        f = lambda kk: f"{np.mean([r[kk] for r in runs]):.3f} ± {np.std([r[kk] for r in runs]):.3f}"
        lines.append(f"| {key} | {f('left_share_in_band')} | {f('rest_mae_rgb_levels')} | {np.mean([r['rest_sum_share_in_left_eye_box'] for r in runs]):.2f} | "
                     f"{max(r['right_margin_max_u_minus_M'] for r in runs):.1f} | {max(r['right_half_max_change_float64'] for r in runs):.1e} |")
    open(f"{OUT}/c1b_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
