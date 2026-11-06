"""
D4-switch: an exact switch with a branch trained by GAN, cat -> anime (see PREREG.md).

A branch (copies of CatGAN-26's layers after block k, starting from the cat weights) reads the frozen cat network's
block-k features and is trained adversarially on the supplied anime faces (128x128). After training, the switch is
compiled into one network built only from dense, conv and LeakyReLU layers: an extra latent input s drives a wire
through every layer, the trunk and branch run side by side, and a final 1x1 layer computes paper one's mux,
    out = a + relu(d - M(1 - s)) - relu(-d - M(1 - s)),  d = b - a  (a: cat image, b: branch image, M = 10),
with a carried by the pass identity and each relu by a LeakyReLU pair. s = 0 gives the cat network; s = 1 the branch.

  <venv python> followup/src/d4_switch.py <k: 3 or 2> [lr]
"""
import sys, time
import numpy as np
import common as C, gan_common as G_
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

OUT = C.out_dir("d4"); CK = C.out_dir("d4/ckpt"); DEV = "cuda"; STEPS, M = 3000, 10.0; A = C.A

class Switched(nn.Module):
    """Frozen trunk (the cat network) up to block k; trainable branch from block k+1 to the output."""
    def __init__(self, W0, k):
        super().__init__(); FoldedG, _ = C.torch_modules(); self.trunk = FoldedG(W0); self.k = k
        for p in self.trunk.parameters(): p.requires_grad_(False)
        Wb = {n: v for n, v in W0.items() if n == "out/kernel" or n == "out/bias" or (n.startswith("b") and int(n[1]) > k)}
        self.branch = FoldedG({**Wb}); self.branch.blocks = [i for i in range(k + 1, 6)]
    def features(self, z):
        t = self.trunk; h = F.leaky_relu((z @ t.p("dense/kernel") + t.p("dense/bias")).view(len(z), 4, 4, -1).permute(0, 3, 1, 2), A)
        for i in range(1, self.k + 1):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            for j in (1, 2): h = F.leaky_relu(F.conv2d(h, t.p(f"b{i}c{j}/kernel"), t.p(f"b{i}c{j}/bias"), padding=1), A)
        return h
    def forward(self, z):          # switch on: the branch's image in [0, 1]
        b = self.branch; h = self.features(z)
        for i in b.blocks:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            for j in (1, 2): h = F.leaky_relu(F.conv2d(h, b.p(f"b{i}c{j}/kernel"), b.p(f"b{i}c{j}/bias"), padding=1), A)
        return F.conv2d(h, b.p("out/kernel"), b.p("out/bias"), padding=1)

def compile_switch(W0, Wbr, k):
    """One network, latent (z, s) -> image, from dense / conv / LeakyReLU layers only. Returns a list of layers (Keras layout)."""
    L = []; Ct = W0["dense/bias"].size // 16
    K = np.zeros((129, 16, Ct + 1)); K[:128, :, :Ct] = W0["dense/kernel"].reshape(128, 16, Ct); K[128, :, Ct] = 1.0
    b = np.zeros((16, Ct + 1)); b[:, :Ct] = W0["dense/bias"].reshape(16, Ct); L.append(("dense", K.reshape(129, -1), b.reshape(-1), False, True))
    width = {"t": Ct, "b": 0}                                                         # current trunk / branch widths (+1 wire last)
    for i in range(1, 6):
        for j in (1, 2):
            n = f"b{i}c{j}"; Kt, bt = W0[n + "/kernel"], W0[n + "/bias"]; ci_t, co = Kt.shape[2], Kt.shape[3]
            has_b_in, has_b_out = width["b"] > 0, i > k
            ci = ci_t + (ci_t if has_b_in else 0) + 1; co_all = co + (co if has_b_out else 0) + 1
            Kf = np.zeros((3, 3, ci, co_all)); bf = np.zeros(co_all)
            Kf[:, :, :ci_t, :co] = Kt; bf[:co] = bt
            if has_b_out:
                Kb, bb = Wbr[n + "/kernel"], Wbr[n + "/bias"]
                src = slice(ci_t, 2 * ci_t) if has_b_in else slice(0, ci_t)               # the first branch layer reads the trunk's features
                Kf[:, :, src, co:2 * co] = Kb; bf[co:2 * co] = bb
            Kf[1, 1, ci - 1, co_all - 1] = 1.0                                              # the switch wire (s >= 0 passes LeakyReLU exactly)
            L.append(("conv", Kf, bf, j == 1, True)); width = {"t": co, "b": co if has_b_out else 0}
    ci_t = width["t"]; ci = 2 * ci_t + 1; Kf = np.zeros((3, 3, ci, 18)); bf = np.zeros(18)
    Ka, ba = W0["out/kernel"], W0["out/bias"]; Kb, bb = Wbr["out/kernel"], Wbr["out/bias"]
    for c in range(3):
        Kf[:, :, :ci_t, c] = Ka[..., c]; bf[c] = ba[c]; Kf[:, :, :ci_t, 3 + c] = -Ka[..., c]; bf[3 + c] = -ba[c]               # a, -a
        for base, sgn in ((6, 1.0), (12, -1.0)):                                                                                # p1 = d - M(1-s), p2 = -d - M(1-s)
            Kf[:, :, ci_t:2 * ci_t, base + c] = sgn * Kb[..., c]; Kf[:, :, :ci_t, base + c] = -sgn * Ka[..., c]
            Kf[1, 1, ci - 1, base + c] = M; bf[base + c] = sgn * (bb[c] - ba[c]) - M
            Kf[:, :, :, base + 3 + c] = -Kf[:, :, :, base + c]; bf[base + 3 + c] = -bf[base + c]
    L.append(("conv", Kf, bf, False, True))
    Km = np.zeros((1, 1, 18, 3)); p, g = 1 / (1 + A), 1 / (1 - A * A)
    for c in range(3):
        Km[0, 0, c, c], Km[0, 0, 3 + c, c] = p, -p; Km[0, 0, 6 + c, c], Km[0, 0, 9 + c, c] = g, A * g; Km[0, 0, 12 + c, c], Km[0, 0, 15 + c, c] = -g, -A * g
    L.append(("conv", Km, np.zeros(3), False, False))
    return L

def run_compiled(L, z, s, double=False):
    dt = torch.float64 if double else torch.float32
    x = torch.cat([z, torch.full((len(z), 1), float(s), device=z.device)], 1).to(dt)
    with torch.no_grad():
        for kind, K, b, up, act in L:
            Kt, bt = torch.tensor(K, dtype=dt, device=z.device), torch.tensor(b, dtype=dt, device=z.device)
            if kind == "dense": x = (x @ Kt + bt).view(len(x), 4, 4, -1).permute(0, 3, 1, 2)
            else:
                if up: x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = F.conv2d(x, Kt.permute(3, 2, 0, 1), bt, padding=Kt.shape[0] // 2)
            if act: x = F.leaky_relu(x, A)
    return x

def stats_corr(cat, ani, rng):
    """Matched vs shuffled correlation of colour histograms and of 16x16 luminance layouts (cat_i vs anime_i)."""
    def hist(x): return np.concatenate([np.stack([np.histogram(im[..., c], bins=8, range=(0, 1))[0] for im in x]) for c in range(3)], 1).astype(float)
    def thumb(x): t = torch.from_numpy(x.mean(-1))[:, None].float(); t = F.adaptive_avg_pool2d(t, 16).reshape(len(x), -1).numpy(); return t
    out = {}
    for name, f in (("colour_hist", hist), ("layout_16x16", thumb)):
        a, b = f(np.clip(cat, 0, 1)), f(np.clip(ani, 0, 1)); a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9); b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)
        matched = (a * b).mean(1); perm = rng.permutation(len(b)); shuffled = (a * b[perm]).mean(1)
        out[name] = {"matched": float(matched.mean()), "shuffled": float(shuffled.mean()), "gap": float(matched.mean() - shuffled.mean())}
    return out

def grid(rows, path):
    a = (np.clip(np.concatenate([np.concatenate(list(r), 1) for r in rows], 0), 0, 1) * 255).round().astype(np.uint8); Image.fromarray(a).save(path)

def main(k, lr):
    t0 = time.time(); W0 = C.catgan_base(); train_u8, val_u8 = G_.anime_faces(); real = G_.flip_sampler(train_u8); fd = G_.FD(val_u8)
    ev = C.OddEyeEval(); x_cat = ev.x0; R = {"k": k, "lr": lr, "runs": []}
    for seed in (0, 1, 2):
        S = Switched(W0, k).to(DEV); D, oD = G_.discriminator(); log = []
        def on_eval(step, Gm, ema):
            x = ev.images(ema); r = {"step": step, "fd_anime": fd(ema), "correspondence": stats_corr(x_cat, x, np.random.default_rng(0))}
            log.append(r); print(k, seed, step, round(r["fd_anime"], 2), {n: round(v["gap"], 3) for n, v in r["correspondence"].items()}, f"{time.time() - t0:.0f}s", flush=True)
        ema = G_.train(S, D, oD, real, STEPS, lr, every=500, on_eval=on_eval, seed=seed)
        Wbr = ema.branch.export(); np.savez(f"{CK}/k{k}_s{seed}.npz", **Wbr)
        Lc = compile_switch(W0, Wbr, k); z = ev.z[:50]
        x_on = ev.images(ema)[:50]; c_off32 = run_compiled(Lc, z, 0).permute(0, 2, 3, 1).cpu().numpy(); c_on32 = run_compiled(Lc, z, 1).permute(0, 2, 3, 1).cpu().numpy()
        c_off64 = run_compiled(Lc, z[:16], 0, double=True).permute(0, 2, 3, 1).cpu().numpy()
        FoldedG, _ = C.torch_modules()
        exact = {"off_vs_catgan_float32": float(np.abs(c_off32 - x_cat[:50]).max()), "on_vs_branch_float32": float(np.abs(c_on32 - x_on).max()),
                 "off_vs_catgan_float64": float(np.abs(c_off64 - FoldedG({n: v for n, v in W0.items()}).double().to(DEV)(z[:16].double()).permute(0, 2, 3, 1).detach().cpu().numpy()).max()),
                 "compiled_params": int(sum(K.size + b.size for _, K, b, _u, _a in Lc)), "catgan_params": int(sum(v.size for v in W0.values()))}
        print("compiled switch", exact, flush=True)
        grid([x_cat[:8], x_on[:8], c_off32[:8], c_on32[:8]], f"{OUT}/grid_k{k}_s{seed}.png")
        R["runs"].append({"seed": seed, "log": log, "exactness": exact}); C.save_json(R, f"{OUT}/d4_k{k}_results.json")
        del S, D, oD, ema; torch.cuda.empty_cache()
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/d4_k{k}_results.json")

if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 2e-4)
