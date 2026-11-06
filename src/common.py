"""
Shared pieces for the follow-up experiments. Paper one's code is imported from ../wsse/src unchanged;
scripts chdir to the wsse root so its relative paths (results/, data/) keep working, and write their
own outputs under followup/results/.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); FOLLOWUP = os.path.dirname(HERE)
LAB = os.path.dirname(FOLLOWUP); WSSE = os.path.join(LAB, "wsse")
sys.path.insert(0, os.path.join(WSSE, "src")); os.chdir(WSSE)
RESULTS = os.path.join(FOLLOWUP, "results"); CACHE = os.path.join(RESULTS, "cache")
CKPT = os.path.join(LAB, "release", "v1.0.0", "catgan26.pt")          # paper one's step-30,000 training state
A = 0.2

def out_dir(name):
    d = os.path.join(RESULTS, name); os.makedirs(d, exist_ok=True); return d

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=1)

# --------------------------------------------------------------------------------------------------
# CatGAN-26 and the odd-eye patch
# --------------------------------------------------------------------------------------------------
def catgan_base():
    return {k: v.astype(np.float32) for k, v in np.load("results/exp11_out/catgan26.npz").items()}

def odd_eye(mode="left"):
    """Paper one's odd-eye network (exp20, grown room) and its slot layout, cached after the first build.
    Returns (W_patched, W_room, meta): W_room is the grown, wired network without the gate."""
    os.makedirs(CACHE, exist_ok=True); p = os.path.join(CACHE, f"odd_eye_{mode}.npz"); pr = os.path.join(CACHE, "odd_eye_room.npz")
    pm = os.path.join(CACHE, "odd_eye_meta.json")
    if not (os.path.exists(p) and os.path.exists(pr) and os.path.exists(pm)):
        import exp20_odd_eyes as e20
        ctx = e20.run_room("grow", write=False)
        for m in ("left", "right", "both"): np.savez(os.path.join(CACHE, f"odd_eye_{m}.npz"), **ctx["nets"][m])
        np.savez(pr, **ctx["nets"]["room (no gate)"])
        W0 = ctx["W0"]; R = ctx["R"]
        meta = {"chain": R["chain"], "slots": R["room"]["slots"], "gate_slots": [int(W0["b5c2/bias"].size), int(W0["b5c2/bias"].size) + 1],
                "quantiles": {m: R["gates"][m]["quantile"] for m in R["gates"]},
                "share_left": R["gates"]["left"]["share_left_in_band"], "share_right": R["gates"]["right"]["share_right_in_band"]}
        save_json(meta, pm)
    load = lambda f: {k: v.astype(np.float32) for k, v in np.load(f).items()}
    return load(p), load(pr), json.load(open(pm))

# --------------------------------------------------------------------------------------------------
# A trainable folded generator (any widths) and paper one's discriminator
# --------------------------------------------------------------------------------------------------
def torch_modules():
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False   # TF32 breaks exactness checks (~1e-3 relative)

    class FoldedG(nn.Module):
        """CatGAN-26 with BatchNorm folded (the exported form patches live in). Parameters are named like
        the Keras dict ('b1c1/kernel' -> 'b1c1__kernel'), kernels stored OIHW. forward(z) returns images in
        [0, 1] (the export's output convention); D expects 2 * x - 1."""
        def __init__(self, W):
            super().__init__()
            self.names = sorted(W)
            for k in self.names:
                v = torch.from_numpy(np.ascontiguousarray(W[k] if W[k].ndim != 4 else W[k].transpose(3, 2, 0, 1)))
                self.register_parameter(k.replace("/", "__"), nn.Parameter(v.float()))
            self.blocks = sorted({int(k[1:k.index("c")]) for k in W if k.startswith("b") and "/kernel" in k})
        def p(self, k): return getattr(self, k.replace("/", "__"))
        def forward(self, z):
            x = F.leaky_relu((z @ self.p("dense/kernel") + self.p("dense/bias")).view(len(z), 4, 4, -1).permute(0, 3, 1, 2), A)
            for i in self.blocks:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                for j in (1, 2): x = F.leaky_relu(F.conv2d(x, self.p(f"b{i}c{j}/kernel"), self.p(f"b{i}c{j}/bias"), padding=1), A)
            return F.conv2d(x, self.p("out/kernel"), self.p("out/bias"), padding=1)
        def export(self):
            return {k: (self.p(k).detach().cpu().numpy().transpose(2, 3, 1, 0) if self.p(k).ndim == 4 else self.p(k).detach().cpu().numpy()).astype(np.float32)
                    for k in self.names}

    def discriminator(dev="cuda"):
        import exp11_catgan26 as e11
        D = e11.Discriminator().to(dev); c = torch.load(CKPT, map_location=dev); D.load_state_dict(c["D"])
        return D, c
    return FoldedG, discriminator

# --------------------------------------------------------------------------------------------------
# Exact widening of CatGAN-26 and the odd-eye metrics shared by A1, B1, B3 and D5
# --------------------------------------------------------------------------------------------------
CATGAN_LAYERS = ["dense"] + [f"b{i}c{j}" for i in range(1, 6) for j in (1, 2)] + ["out"]

def widen_catgan(W, k, rng, std=0.01):
    """Grow k channels in every hidden layer with small random incoming weights and zero outgoing weights
    (exact: the function is unchanged, but the new channels can train). Returns (W, first new slot per layer)."""
    import wpgit
    W = {kk: v.copy() for kk, v in W.items()}; slots = {}
    for prod, cons in zip(CATGAN_LAYERS[:-1], CATGAN_LAYERS[1:]):
        C_old = W[cons + "/kernel"].shape[2]; W = wpgit.grow(W, prod, cons, k); slots[prod] = C_old
        K = W[prod + "/kernel"]
        if K.ndim == 2:
            Kc = K.reshape(K.shape[0], -1, C_old + k); Kc[:, :, C_old:] = rng.normal(0, std, Kc[:, :, C_old:].shape); W[prod + "/kernel"] = Kc.reshape(K.shape[0], -1).astype(np.float32)
        else:
            K[..., C_old:] = rng.normal(0, std, K[..., C_old:].shape).astype(np.float32)
    return W, slots

class OddEyeEval:
    """Paper one's odd-eye metrics on its 300 test latents. Iris masks come from the base image at each latent."""
    def __init__(self, dev="cuda"):
        import torch, cat_domain as cd
        self.dev = dev; rng = np.random.default_rng(0); _, _, z = (rng.standard_normal((n, 128)) for n in (256, 300, 300))
        self.z = torch.tensor(z, dtype=torch.float32, device=dev)
        FoldedG, _ = torch_modules(); self.x0 = self.images(FoldedG(catgan_base()).to(dev))
        self.m = cd.eye_mask(np.clip(self.x0, 0, 1)); self.left = np.zeros((128, 128), bool); self.left[:, :64] = True
        self.far = ~np.any([np.roll(np.roll(self.m, dy, 1), dx, 2) for dy in range(-3, 4) for dx in range(-3, 4)], 0)
    def images(self, model, z=None):
        import torch
        z = self.z if z is None else z
        with torch.no_grad(): return np.concatenate([model(z[s:s + 50]).permute(0, 2, 3, 1).cpu().numpy() for s in range(0, len(z), 50)])
    def score(self, x, ref=None):
        """x: images (N,128,128,3) in [0,1]. ref: images to measure drift against (default: the base)."""
        import exp20_odd_eyes as e20
        ref = self.x0 if ref is None else ref
        hl = e20.per_image_hue(x, self.m & self.left[None]); hr = e20.per_image_hue(x, self.m & ~self.left[None])
        inb = lambda h: h is not None and 200 <= h <= 250
        pairs = [(inb(a), inb(b)) for a, b in zip(hl, hr) if a is not None and b is not None]
        d = np.abs(np.clip(x, 0, 1) - np.clip(ref, 0, 1)) * 255
        return {"left_in_band": float(np.mean([inb(h) for h in hl if h is not None])), "right_in_band": float(np.mean([inb(h) for h in hr if h is not None])),
                "left_only": float(np.mean([a and not b for a, b in pairs])), "both": float(np.mean([a and b for a, b in pairs])),
                "right_half_mean_change": float(d[:, :, 64:].mean()), "right_half_max_change": float(d[:, :, 64:].max()),
                "outside_eyes_change": float(d[self.far].mean())}

def dense_mirror_asymmetry(W, W_ref, C0=512):
    """sum |db(cell) - db(mirror cell)| / sum |db| over the first C0 channels of the dense bias (Keras cell-major layout)."""
    C = W["dense/bias"].size // 16; db = (W["dense/bias"].reshape(4, 4, C)[..., :C0] - W_ref["dense/bias"].reshape(4, 4, -1)[..., :C0])
    tot = np.abs(db).sum(); return float(np.abs(db - db[:, ::-1]).sum() / tot) if tot > 0 else 0.0
