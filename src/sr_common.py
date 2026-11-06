"""
TinySR-26 helpers for A3 and D2: networks of any width and depth built from Keras-style dicts, exact
widening, a generic training loop (paper one's recipe: Adam, EMA, AFHQ patches) and a fast evaluator on
paper one's 180 test photos (2x from 256 to 512), with LR images prepared once.
"""
import copy, time
import numpy as np
import common as C
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import exp10_tinysr26 as e10, exp14_tinysr26_branches as e14

DEV = "cuda"; CONDS = ("bicubic LR", "noisy bicubic LR", "nearest LR")
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def base():
    return {k: v.astype(np.float32) for k, v in np.load("results/exp10_out/tinysr26_x2.npz").items()}

def experts():
    W0 = base(); db, dn = (dict(np.load(f"results/exp14_out/delta_{n}.npz")) for n in ("bicubic", "noisy"))
    return {"base": W0, "bicubic": {k: W0[k] + db[k] for k in W0}, "noisy": {k: W0[k] + dn[k] for k in W0}}

def blocks_of(W): return sorted({int(k[4:k.index("_")]) for k in W if k.startswith("body")})

def module(W, s=2):
    """A TinySR with the width and depth of W (blocks in index order), weights copied."""
    ch = W["head/kernel"].shape[3]; nb = len(blocks_of(W)); m = e10.TinySR(s, ch, nb).to(DEV)
    names = {"head": m.head, "tail": m.tail, "up": m.up, **{f"body{i}_a": b[0] for i, b in enumerate(m.body)}, **{f"body{i}_b": b[2] for i, b in enumerate(m.body)}}
    order = blocks_of(W)
    with torch.no_grad():
        for n, layer in names.items():
            src = n if not n.startswith("body") else f"body{order[int(n[4:n.index('_')])]}{n[n.index('_'):]}"
            layer.weight.copy_(torch.from_numpy(W[src + "/kernel"].transpose(3, 2, 0, 1).copy())); layer.bias.copy_(torch.from_numpy(W[src + "/bias"]))
    return m

def widen(W, new_ch, rng, init_std=1e-2):
    """Exact widening: new channels get small random incoming weights and zero outgoing weights into the old
    channels and the output, so the function is unchanged but every new weight can receive gradient."""
    ch = W["head/kernel"].shape[3]; Wn = {}
    for k, v in W.items():
        if not k.endswith("/kernel"): continue
        n = k[:-7]; kh, kw, cin, cout = v.shape
        cin2 = cin if n == "head" else new_ch; cout2 = cout if n == "up" else new_ch
        K = np.zeros((kh, kw, cin2, cout2), np.float32); K[:, :, :cin, :cout] = v
        if cout2 > cout: K[:, :, :, cout:] = rng.normal(0, init_std, (kh, kw, cin2, cout2 - cout))   # incoming to new channels
        Wn[k] = K; b = np.zeros(cout2, np.float32); b[:cout] = W[n + "/bias"]; Wn[n + "/bias"] = b
    return Wn

class TestSet:
    """Paper one's test photos with LR inputs prepared once (noise drawn as in exp14.psnr_on)."""
    def __init__(self):
        imgs = e14.test_images(); self.hr = [torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(DEV) for im in imgs]
        self.lr = {}
        for cond, kind, noise in (("bicubic LR", "bicubic", 0.0), ("noisy bicubic LR", "bicubic", e14.NOISE), ("nearest LR", "nearest", 0.0)):
            g = torch.Generator(device=DEV).manual_seed(0); xs = []
            for im in imgs:
                lr = np.asarray(im.resize((im.width // 2, im.height // 2), {"bicubic": Image.BICUBIC, "nearest": Image.NEAREST}[kind]), np.float32) / 255
                x = torch.from_numpy(lr).permute(2, 0, 1)[None].to(DEV)
                if noise: x = ((x + noise * torch.randn(x.shape, generator=g, device=DEV)).clamp(0, 1) * 255).round() / 255
                xs.append(x)
            self.lr[cond] = xs
    @torch.no_grad()
    def psnr(self, m, conds=CONDS):
        m.eval(); out = {}
        for cond in conds:
            v = [float(10 * torch.log10(1 / ((m(x).clamp(0, 1) - y) ** 2).mean())) for x, y in zip(self.lr[cond], self.hr)]
            out[cond] = float(np.mean(v))
        return out

def loader(per_image=4, size=96, batch_images=8):
    return torch.utils.data.DataLoader(e10.Patches(e10.files("train", e14.CLASSES), size, per_image), batch_size=batch_images, shuffle=True,
                                       num_workers=4, drop_last=True, persistent_workers=True)

def train(m, steps, batch_fn, lr=1e-4, seed=0, eval_every=None, evaluate=None, keep_every=None, on_keep=None):
    """Paper one's fine-tuning recipe: Adam(0.9, 0.99), EMA with rate 0.01, batches of 8 images x 4 patches.
    batch_fn(hr) -> (lr_input, target). Returns (ema model, log)."""
    torch.manual_seed(seed); ema = copy.deepcopy(m).eval(); opt = torch.optim.Adam(m.parameters(), lr, betas=(0.9, 0.99))
    dl = loader(); step = 0; log = []; t0 = time.time()
    if evaluate: log.append({"step": 0, **evaluate(ema)})
    while step < steps:
        for b in dl:
            m.train(); hr = b.view(-1, 3, 96, 96).to(DEV).float() / 255; x, y = batch_fn(hr)
            loss = F.l1_loss(m(x), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), m.parameters()): pe.lerp_(p, 0.01)
            if evaluate and eval_every and step % eval_every == 0: log.append({"step": step, "seconds": time.time() - t0, **evaluate(ema)})
            if on_keep and keep_every and step % keep_every == 0: on_keep(step, ema)
            if step >= steps: break
    return ema, log

def noisy(lr_):
    return ((lr_ + torch.rand(len(lr_), 1, 1, 1, device=DEV) * e14.NOISE * torch.randn_like(lr_)).clamp(0, 1) * 255).round() / 255

def mixed_batch(hr):
    """Paper one's merge objective: a third clean bicubic, a third noisy bicubic, a third nearest."""
    h1, h2, h3 = hr.chunk(3)
    return torch.cat([e10.degrade(h1, 2, "bicubic"), noisy(e10.degrade(h2, 2, "bicubic")), e10.degrade(h3, 2, "nearest")]), (h1, h2, h3)
