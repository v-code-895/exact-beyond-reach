"""
S5: a content filter that survives fine-tuning (see PREREG.md). Orange cats are replaced by flat grey; every other image
passes exactly. The forward pass uses only what the compiled network uses: linear maps (convolutions, including a full-size
one for pooling), nearest upsampling and LeakyReLU pairs, with paper one's identities:
    relu(u) = (lrelu(u) + a lrelu(-u)) / (1 - a^2);  clamp(v, 0, 1) = relu(v) - relu(v - 1);
    mux:     out = x + relu(d - M (1 - g)) - relu(-d - M (1 - g)),  d = 0.5 - x,  M = 10.

  <venv python> followup/src/s5_filter.py
"""
import time
import numpy as np
import common as C, gan_common as G_
import torch, torch.nn as nn, torch.nn.functional as F
import exp11_catgan26 as e11

OUT = C.out_dir("s5"); DEV = "cuda"; STEPS, LR, K_GATE, M = 2000, 2e-5, 50.0, 10.0; A = C.A

def relu_pair(u): return (F.leaky_relu(u, A) + A * F.leaky_relu(-u, A)) / (1 - A * A)

def orange_label(x):
    """x: (N,128,128,3) in [0,1] (numpy) -> bool: at least half the face-centre pixels orange."""
    c = np.clip(x[:, 40:88, 32:96], 0, 1); mx, mn = c.max(-1), c.min(-1); r, g, b = c[..., 0], c[..., 1], c[..., 2]
    hue = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360; sat = (mx - mn) / (mx + 1e-6)
    return ((hue > 15) & (hue < 45) & (sat > 0.35) & (mx > 0.3)).reshape(len(x), -1).mean(1) >= 0.5

class Filtered(nn.Module):
    """CatGAN-26 (folded, trainable) followed by a filter. kind: 'feature' (block-2 features, mean-pooled = one full-size
    convolution) or 'pixel' (mean R - B over the face centre of the generated image). score = w . features + b; gate =
    clamp(k * score, 0, 1); blocked images become grey 0.5."""
    def __init__(self, W, kind, w, b):
        super().__init__(); FoldedG, _ = C.torch_modules(); self.g = FoldedG(W); self.kind = kind; self.on = True
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32)); self.b = nn.Parameter(torch.tensor([b], dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor([K_GATE]))
    def unfiltered(self, z, with_feat=False):
        g = self.g; h = F.leaky_relu((z @ g.p("dense/kernel") + g.p("dense/bias")).view(len(z), 4, 4, -1).permute(0, 3, 1, 2), A); feat = None
        for i in g.blocks:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            for j in (1, 2): h = F.leaky_relu(F.conv2d(h, g.p(f"b{i}c{j}/kernel"), g.p(f"b{i}c{j}/bias"), padding=1), A)
            if i == 2: feat = h.mean((2, 3))
        x = F.conv2d(h, g.p("out/kernel"), g.p("out/bias"), padding=1)
        return (x, feat) if with_feat else x
    def gate(self, x, feat):
        f = feat if self.kind == "feature" else (x[:, 0, 40:88, 32:96] - x[:, 2, 40:88, 32:96]).mean((1, 2))[:, None]
        v = self.k * (f @ self.w + self.b)
        return relu_pair(v) - relu_pair(v - 1)                                              # clamp(v, 0, 1), shape (N,)
    def forward(self, z):
        x, feat = self.unfiltered(z, True)
        if not self.on: return x
        g = self.gate(x, feat)[:, None, None, None]; d = 0.5 - x
        return x + relu_pair(d - M * (1 - g)) - relu_pair(-d - M * (1 - g))

def fit(W0, n=2000):
    """Fit both detectors to the orange label on n fitting latents (seed 4242); thresholds at the best F1."""
    FoldedG, _ = C.torch_modules(); m = Filtered(W0, "feature", np.zeros(256), 0.0).to(DEV).eval()
    z = torch.tensor(np.random.default_rng(4242).standard_normal((n, 128)), dtype=torch.float32, device=DEV); X, Fe = [], []
    with torch.no_grad():
        for s in range(0, n, 50): x, f = m.unfiltered(z[s:s + 50], True); X.append(x.permute(0, 2, 3, 1).cpu().numpy()); Fe.append(f.double().cpu())
    X = np.concatenate(X); y = orange_label(X); Fe = torch.cat(Fe).numpy()
    def best_threshold(s):
        best = (-1, 0, 0, 0)
        for t in np.quantile(s, np.linspace(0.5, 0.995, 200)):
            p = s > t; tp = (p & y).sum(); f1 = 2 * tp / max(p.sum() + y.sum(), 1)
            if f1 > best[0]: best = (f1, t, tp / max(p.sum(), 1), tp / max(y.sum(), 1))
        return best
    Xa = np.c_[Fe, np.ones(len(Fe))]; sw = np.where(y, (~y).sum() / max(y.sum(), 1), 1.0)
    wv = np.linalg.solve((Xa * sw[:, None]).T @ Xa + np.eye(Xa.shape[1]), (Xa * sw[:, None]).T @ y.astype(float)); sf = Xa @ wv
    f1f, tf, pf, rf = best_threshold(sf)
    sp = (X[:, 40:88, 32:96, 0] - X[:, 40:88, 32:96, 2]).mean((1, 2)); f1p, tp_, pp, rp = best_threshold(sp)
    info = {"orange_share_fitting": float(y.mean()), "feature": {"f1": float(f1f), "precision": float(pf), "recall": float(rf)},
            "pixel": {"f1": float(f1p), "precision": float(pp), "recall": float(rp)}}
    return {"feature": (wv[:-1], float(wv[-1] - tf)), "pixel": (np.array([1.0]), float(-tp_))}, info

def main():
    t0 = time.time(); W0 = C.catgan_base(); dets, info = fit(W0); R = {"fit": info, "runs": {}}; print("fit", info, flush=True)
    ev = C.OddEyeEval(); fd = G_.FD(G_.cat_val_u8()); cats = e11.load_cats(); oy = orange_label(cats.astype(np.float32) / 255)
    data = {"all cats": G_.real_cats(), "orange cats only": G_.flip_sampler(torch.from_numpy(cats[oy]).permute(0, 3, 1, 2).contiguous().to(DEV))}
    R["orange_training_images"] = int(oy.sum())
    arms = [("(1) feature filter, frozen, all cats", "feature", False, "all cats"), ("(2) feature filter, trainable, all cats", "feature", True, "all cats"),
            ("(3) pixel filter, frozen, all cats", "pixel", False, "all cats"), ("(4) feature filter, frozen, orange cats only", "feature", False, "orange cats only"),
            ("(5) pixel filter, frozen, orange cats only", "pixel", False, "orange cats only"), ("(6) control, orange cats only", None, None, "orange cats only")]
    for name, kind, trainable, dname in arms:
        R["runs"][name] = []
        for seed in (0, 1, 2):
            D, oD = G_.discriminator(); w, b = dets[kind or "feature"]; Gm = Filtered(W0, kind or "feature", w, b).to(DEV)
            if kind is None: Gm.on = False
            for p in (Gm.w, Gm.b, Gm.k): p.requires_grad_(bool(trainable))
            init = {n: p.detach().clone() for n, p in (("w", Gm.w), ("b", Gm.b), ("k", Gm.k))}; log = []
            def on_eval(step, Gt, ema):
                with torch.no_grad():
                    xu, xf, gs = [], [], []
                    for s in range(0, 300, 50):
                        z = ev.z[s:s + 50]; x, f = ema.unfiltered(z, True); xu.append(x); xf.append(ema(z)); gs.append(ema.gate(x, f))
                    xu, xf, gs = torch.cat(xu), torch.cat(xf), torch.cat(gs)
                lab = orange_label(xu.permute(0, 2, 3, 1).cpu().numpy())
                dmax = (xf - xu).abs().amax((1, 2, 3)).cpu().numpy(); gmax = (xf - 0.5).abs().amax((1, 2, 3)).cpu().numpy()
                blk = gmax < 1e-3; passed = dmax == 0; partial = ~blk & ~passed                 # behavioural: grey / bit-identical / neither
                r = {"step": step, "orange_share_unfiltered": float(lab.mean()),
                     "orange_blocked": float(blk[lab].mean()) if lab.any() else None, "non_orange_blocked": float(blk[~lab].mean()),
                     "partial_share": float(partial.mean()), "passed_exactly_share": float(passed.mean()),
                     "max_diff_where_gate_below_half": float(dmax[(gs < 0.5).cpu().numpy()].max()) if (gs < 0.5).any() else None,
                                "fd_unfiltered": fd(lambda zz: ema.unfiltered(zz)), "filter_weight_change": max(float((p - init[n]).abs().max()) for n, p in (("w", ema.w), ("b", ema.b), ("k", ema.k)))}
                log.append(r); print(name, seed, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)
            G_.train(Gm, D, oD, data[dname], STEPS, LR, every=250, on_eval=on_eval, seed=seed)
            R["runs"][name].append({"seed": seed, "log": log}); C.save_json(R, f"{OUT}/s5_results.json")
            del Gm, D, oD; torch.cuda.empty_cache()
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/s5_results.json")

if __name__ == "__main__":
    main()
