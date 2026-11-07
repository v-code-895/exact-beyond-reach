"""
C1: frozen skeleton, trained detector (see PREREG.md).

The odd-eye patch's wire, AND term (-M * RIGHT), gate-to-output wiring and 3-number tint v are frozen.
Only the detector (the gate's incoming weights over b5c1 features) changes. Because the output layer
reads the gate at its centre tap, the patched image is exactly
    x = x_room + v * sum_k c_k * relu(u_k - M * RIGHT),   u_k = conv3x3(h_b5c1, K_k) + b_k,
so training needs only the frozen network's b5c1 features and output. Every arm is then rebuilt as a
real network (grown b5c2 channels in relu pairs) and evaluated with paper one's metrics.

Arms: (a) paper one's closed form (300 fitting cats); (b) the same rule on 2,000 fitting cats;
(c) Adam from (b); (d) Adam from random; (e) Adam, K = 8 units, random. Seeds 0-2 for (c)-(e).

  <venv python> src/c1_trained_detector.py
"""
import time
import numpy as np
import common as C
import torch, torch.nn.functional as F
import exp20_odd_eyes as e20, cat_domain as cd, exp4_anime_eye_circuit as e4, wpgit
from exp18_circuit_primitives import catgan_layers

OUT = C.out_dir("c1"); DEV = "cuda"; N_FIT, STEPS, BATCH, LR, LAM = 2000, 2000, 32, 1e-3, 1.0
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
EYES = torch.zeros(128, 128, dtype=torch.bool); EYES[56:74, 36:58] = EYES[56:74, 70:92] = True
NEAR = torch.zeros(128, 128, dtype=torch.bool); NEAR[53:77, 33:61] = NEAR[53:77, 67:95] = True   # eye boxes dilated by 3
LEFT = torch.zeros(128, 128, dtype=torch.bool); LEFT[:, :64] = True
TARGET = torch.tensor(cd.TARGET)

def hue(x):   # x (N,3,H,W) -> degrees, as exp4.hue
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    return torch.rad2deg(torch.atan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360

def iris_mask(x):   # paper one's heuristic iris mask on base images (cat_domain.eye_mask), in torch
    x = x.clamp(0, 1); mx, mn = x.max(1).values, x.min(1).values; h = hue(x)
    return EYES.to(x.device)[None] & (mx - mn > 0.10) & (mx > 0.25) & (h > 25) & (h < 170)

class Room:
    """The frozen grown-and-wired network without the gate: returns (x_room, b5c1 features without the wire)."""
    def __init__(self, W_room, meta):
        FoldedG, _ = C.torch_modules(); self.g = FoldedG(W_room).to(DEV).eval(); self.s5 = int(meta["slots"]["b5c1"])
        for p in self.g.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def __call__(self, z):
        g = self.g; h = F.leaky_relu((z @ g.p("dense/kernel") + g.p("dense/bias")).view(len(z), 4, 4, -1).permute(0, 3, 1, 2), C.A)
        for i in g.blocks:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            for j in (1, 2):
                h = F.leaky_relu(F.conv2d(h, g.p(f"b{i}c{j}/kernel"), g.p(f"b{i}c{j}/bias"), padding=1), C.A)
                if (i, j) == (5, 1): h51 = h[:, :self.s5]
        return F.conv2d(h, g.p("out/kernel"), g.p("out/bias"), padding=1), h51

def gate_delta(h, K, b, c, v, M):
    u = F.conv2d(h, K, b, padding=1) - M * (~LEFT).to(h.device).float()          # (N, K, H, W): AND NOT RIGHT
    return (F.relu(u) * c[None, :, None, None]).sum(1, keepdim=True) * v[None, :, None, None]

def closed_form(room, zpool, v, M, rng, per_img_pos=100, n_neg=50000):
    """Paper one's rule (exp4.fit_gate: class-balanced ridge probe on 3x3 neighbourhoods; exp20's threshold
    sweep) on the given fitting latents. Pixels are sampled chunk by chunk to bound memory: up to
    per_img_pos iris pixels per image and n_neg background pixels in total."""
    P, N = [], []; neg_per = int(np.ceil(n_neg / len(zpool)))
    for s in range(0, len(zpool), 50):
        x, h = room(zpool[s:s + 50]); m = iris_mask(x).cpu().numpy(); hn = h.permute(0, 2, 3, 1).cpu().numpy()
        flat = m.reshape(-1); pos = np.nonzero(flat)[0]; neg = np.nonzero(~flat)[0]
        npos = min(len(pos), per_img_pos * len(m)); P.append(e4.patches3x3(hn, rng.choice(pos, npos, replace=False)) if npos else np.zeros((0, 288)))
        N.append(e4.patches3x3(hn, rng.choice(neg, neg_per * len(m), replace=False)))
    Xp, Xn = np.concatenate(P).astype(np.float64), np.concatenate(N).astype(np.float64)
    X = np.c_[np.r_[Xp, Xn], np.ones(len(Xp) + len(Xn))]; y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    sw = np.r_[np.full(len(Xp), len(Xn) / len(Xp)), np.ones(len(Xn))]
    w = np.linalg.solve((X * sw[:, None]).T @ X + np.eye(X.shape[1]), (X * sw[:, None]).T @ y)
    kern, kb = w[:-1].reshape(3, 3, -1), w[-1]; p_neg = X[len(Xp):] @ w
    K = torch.tensor(kern.transpose(2, 0, 1)[None], dtype=torch.float32)       # (1, C, 3, 3)
    best = None
    for q in (0.9, 0.95, 0.97, 0.98, 0.99):                                         # strictest quantile whose left-eye hue is in band
        th = float(np.quantile(p_neg, q)); b = torch.tensor([kb - th], dtype=torch.float32)
        hl = []
        for s in range(0, len(zpool), 50):
            xr, hh = room(zpool[s:s + 50]); x = xr + gate_delta(hh, K.to(DEV), b.to(DEV), torch.ones(1, device=DEV), v, M)
            xn = x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy(); mm = (iris_mask(xr) & LEFT.to(DEV)[None]).cpu().numpy()
            hl += [h_ for h_ in e20.per_image_hue(xn, mm) if h_ is not None]
        if 200 <= float(np.median(hl)) <= 250: best = (q, b)
    q, b = best if best else (0.95, torch.tensor([kb - float(np.quantile(p_neg, 0.95))], dtype=torch.float32))
    return K, b, q

def train(room, zpool, v, M, K0, b0, k, seed):
    torch.manual_seed(seed)
    if K0 is None:
        K = (0.01 * torch.randn(k, 32, 3, 3, device=DEV)).requires_grad_(); b = torch.zeros(k, device=DEV, requires_grad=True)
    else:
        K = K0.clone().to(DEV).requires_grad_(); b = b0.clone().to(DEV).requires_grad_()
    c = torch.ones(k, device=DEV)
    opt = torch.optim.Adam([K, b], LR); g = torch.Generator(device="cpu").manual_seed(seed); log = []
    T = TARGET.to(DEV); near = NEAR.to(DEV); left = LEFT.to(DEV)
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(zpool), (BATCH,), generator=g); xr, h = room(zpool[idx])
        m = iris_mask(xr) & left[None]; d = gate_delta(h, K, b, c, v, M); x = xr + d
        tgt = torch.einsum("nchw,dc->ndhw", xr.clamp(0, 1), T)
        l_iris = ((x - tgt) ** 2).sum(1)[m].mean() if m.any() else x.sum() * 0
        l_out = (d ** 2).sum(1)[:, ~near].mean()
        loss = l_iris + LAM * l_out; opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: log.append({"step": step, "iris": float(l_iris), "outside": float(l_out)})
    return K.detach(), b.detach(), c, log

def build(W_room, meta, K, b, c, v, M):
    """The real network: k relu pairs in grown b5c2 channels, read by the output layer's centre tap."""
    k = K.shape[0]; s5, (g0, _) = int(meta["slots"]["b5c1"]), meta["gate_slots"]
    W = wpgit.grow(W_room, "b5c2", "out", 2 * k - 2) if k > 1 else {kk: vv.copy() for kk, vv in W_room.items()}
    W = {kk: vv.copy() for kk, vv in W.items()}; gg = 1.0 / (1 - C.A ** 2)
    Kn = K.cpu().numpy().transpose(2, 3, 1, 0); bn = b.cpu().numpy(); cn = c.cpu().numpy(); vn = v.cpu().numpy()
    chans = [g0, g0 + 1] + list(range(W_room["b5c2/bias"].size, W_room["b5c2/bias"].size + 2 * k - 2))
    for j in range(k):
        s0, s1 = chans[2 * j], chans[2 * j + 1]
        for s, sign in ((s0, 1), (s1, -1)):
            W["b5c2/kernel"][..., s] = 0; W["b5c2/kernel"][..., :s5, s] = sign * Kn[..., j]
            W["b5c2/kernel"][1, 1, s5, s] = -sign * M; W["b5c2/bias"][s] = sign * bn[j]
        W["out/kernel"][:, :, [s0, s1], :] = 0
        W["out/kernel"][1, 1, s0, :] = gg * cn[j] * vn; W["out/kernel"][1, 1, s1, :] = gg * C.A * cn[j] * vn
    return W

def evaluate(W, W_room, W0, z_test, x0, m_test):
    L = catgan_layers(W); x = e20.forward(z_test, W, L); left = np.zeros((128, 128), bool); left[:, :64] = True
    hl = e20.per_image_hue(x, m_test & left[None]); hr = e20.per_image_hue(x, m_test & ~left[None])
    far = ~np.any([np.roll(np.roll(m_test, dy, 1), dx, 2) for dy in range(-3, 4) for dx in range(-3, 4)], 0)
    a64 = e20.forward(z_test[:16], W, L, double=True); b64 = e20.forward(z_test[:16], W_room, catgan_layers(W_room), double=True)
    return {"left_share_in_band": float(np.mean([200 <= h <= 250 for h in hl if h is not None])),
            "right_share_in_band": float(np.mean([200 <= h <= 250 for h in hr if h is not None])),
            "rest_mae_rgb_levels": 255 * float(np.abs(np.clip(x, 0, 1) - np.clip(x0, 0, 1))[far].mean()),
            "right_half_max_change_float64": float(np.abs(a64[:, :, 64:] - b64[:, :, 64:]).max())}

def main():
    t0 = time.time(); W0 = C.catgan_base(); Wl, W_room, meta = C.odd_eye("left"); s5, (g0, _) = int(meta["slots"]["b5c1"]), meta["gate_slots"]
    gg = 1.0 / (1 - C.A ** 2); v = torch.tensor(Wl["out/kernel"][1, 1, g0, :] / gg, device=DEV); M = float(-Wl["b5c2/kernel"][1, 1, s5, g0])
    room = Room(W_room, meta)
    rng = np.random.default_rng(0); _, _, z_test = (rng.standard_normal((n, 128)) for n in (256, 300, 300))
    x0 = e20.forward(z_test, W0, catgan_layers(W0)); m_test = cd.eye_mask(np.clip(x0, 0, 1))
    zpool = torch.tensor(np.random.default_rng(2026).standard_normal((N_FIT, 128)), dtype=torch.float32, device=DEV)
    R = {"M": M, "v": v.tolist(), "arms": {}}
    # sanity: the closed-form formula reproduces paper one's network
    Ka = torch.tensor(Wl["b5c2/kernel"][..., :s5, g0].transpose(2, 0, 1)[None], dtype=torch.float32); ba = torch.tensor([Wl["b5c2/bias"][g0]], dtype=torch.float32)
    Wa = build(W_room, meta, Ka, ba, torch.ones(1), v, M); R["rebuild_a_vs_paper1_max_abs"] = float(max(np.abs(Wa[k] - Wl[k]).max() for k in Wl))
    R["arms"]["(a) closed form, 300 cats (paper one)"] = evaluate(Wl, W_room, W0, z_test, x0, m_test); print(R["arms"], flush=True)
    Kb, bb, q = closed_form(room, zpool, v, M, np.random.default_rng(0)); R["closed_form_2000_quantile"] = q
    R["arms"]["(b) closed form, 2,000 cats"] = evaluate(build(W_room, meta, Kb, bb, torch.ones(1), v, M), W_room, W0, z_test, x0, m_test)
    print("(b)", R["arms"]["(b) closed form, 2,000 cats"], flush=True)
    for name, K0, b0, k in (("(c) Adam from (b)", Kb, bb, 1), ("(d) Adam, random init", None, None, 1), ("(e) Adam, 8 units, random init", None, None, 8)):
        runs = []
        for seed in (0, 1, 2):
            K, b, c, log = train(room, zpool, v, M, K0, b0, k, seed)
            ev = evaluate(build(W_room, meta, K, b, c, v, M), W_room, W0, z_test, x0, m_test); ev["log"] = log; runs.append(ev)
            print(name, seed, {kk: vv for kk, vv in ev.items() if kk != "log"}, flush=True)
        R["arms"][name] = {"seeds": runs, **{kk: [r[kk] for r in runs] for kk in runs[0] if kk != "log"}}
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/c1_results.json")
    lines = ["# C1: frozen skeleton, trained detector", "", f"M = {M:.1f}; closed form on 2,000 cats chose quantile {q}. Rebuilt (a) vs paper one: {R['rebuild_a_vs_paper1_max_abs']:.1e}.", "",
             "| arm | left irises in band | right irises in band | change outside the eyes (RGB levels) | right half, max change (float64) |", "|---|---|---|---|---|"]
    fmt = lambda x: f"{np.mean(x):.3f} ± {np.std(x):.3f}" if isinstance(x, list) else f"{x:.3f}"
    fmte = lambda x: f"{max(x):.1e}" if isinstance(x, list) else f"{x:.1e}"
    for name, r in R["arms"].items():
        lines.append(f"| {name} | {fmt(r['left_share_in_band'])} | {fmt(r['right_share_in_band'])} | {fmt(r['rest_mae_rgb_levels'])} | {fmte(r['right_half_max_change_float64'])} |")
    open(f"{OUT}/c1_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    main()
