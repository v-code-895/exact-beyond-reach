"""
Continued GAN training of CatGAN-26 in the folded form (the form patches live in), shared by A1, B1, B3 and D4.

Paper one's recipe: non-saturating loss, Adam(0, 0.99), batch 32, lazy R1 (gamma 0.1 every 16 steps) on augmented
reals, differentiable augmentation, generator EMA 0.999. The discriminator and its Adam state resume from paper
one's step-30,000 checkpoint and run in bf16 autocast as in paper one; the generator stays float32, so installed
identities stay exact. The generator's fresh Adam gets its second moments warmed for 100 batches at lr 0 (the B1
pilot showed a cold start moving every folded weight by ~lr at once).
"""
import copy, os, time
import numpy as np
import common as C
import torch, torch.nn.functional as F
import exp11_catgan26 as e11, exp12_modern_eval as e12, mobilenet_np as mn
from PIL import Image

DEV = "cuda"; B = 32

def discriminator():
    D, c = C.torch_modules()[1](DEV); oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.0, 0.99)); oD.load_state_dict(c["oD"]); return D, oD

def flip_sampler(data_u8):
    """data_u8: (N,3,H,W) uint8 tensor (any device) -> real(n) in [-1, 1] on the GPU, random horizontal flips."""
    def real(n):
        x = data_u8[torch.randint(0, len(data_u8), (n,))].to(DEV).float() / 127.5 - 1
        return torch.where(torch.rand(n, 1, 1, 1, device=DEV) < 0.5, x.flip(3), x)
    return real

def real_cats():
    return flip_sampler(torch.from_numpy(e11.load_cats()).permute(0, 3, 1, 2).contiguous().to(DEV))

def anime_faces(n_train=10000, n_val=500, seed=0):
    """The supplied anime faces, resized to 128x128 (Lanczos), cached as uint8. Returns (train uint8 tensor on CPU, val uint8 NHWC)."""
    cache = os.path.join(C.FOLLOWUP, "data", "anime_faces", f"cache128_{n_train}_{n_val}_s{seed}.npz")
    if not os.path.exists(cache):
        d = os.path.join(C.FOLLOWUP, "data", "anime_faces", "images"); fs = sorted(os.listdir(d))
        pick = np.random.default_rng(seed).choice(len(fs), n_train + n_val, replace=False)
        a = np.stack([np.asarray(Image.open(os.path.join(d, fs[i])).convert("RGB").resize((128, 128), Image.LANCZOS)) for i in pick])
        np.savez(cache, train=a[:n_train], val=a[n_train:], files=np.array([fs[i] for i in pick]))
    z = np.load(cache); return torch.from_numpy(z["train"]).permute(0, 3, 1, 2).contiguous(), z["val"]

class FD:
    """Paper one's Fréchet distance in MobileNetV2 features, against a fixed reference set (uint8 NHWC)."""
    def __init__(self, ref_u8, n=500):
        self.Wm = mn.load(); self.Fr = e12.mobilenet(ref_u8[:n], self.Wm)[1]; self.n = n
        self.z = torch.randn(n, 128, generator=torch.Generator().manual_seed(7)).to(DEV)
    def __call__(self, gen01):
        with torch.no_grad(): x = torch.cat([gen01(self.z[s:s + 50]) for s in range(0, self.n, 50)]).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        return e12.frechet(e12.mobilenet((x * 255).round().astype(np.uint8), self.Wm)[1], self.Fr)

def cat_val_u8(n=500):
    d = "data/afhq/data/afhq/val/cat"; fs = sorted(os.listdir(d))[:n]
    return np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB").resize((128, 128), Image.LANCZOS)) for f in fs])

def train(G, D, oD, real, steps, lr, masks=None, every=250, on_eval=None, seed=0, warm=100, ema_beta=0.999):
    """G: module, forward(z) -> images in [0, 1]; trains the parameters with requires_grad. masks: {parameter: bool tensor
    of trainable entries} (gradients elsewhere are zeroed, so Adam leaves those entries exactly unchanged).
    on_eval(step, G, ema) is called at step 0 and every `every` steps. Returns the EMA copy."""
    torch.manual_seed(seed); params = [p for p in G.parameters() if p.requires_grad]
    oG = torch.optim.Adam(params, lr, betas=(0.0, 0.99)); ema = copy.deepcopy(G).eval()
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    def g_step(apply=True):
        with amp(): lG = F.softplus(-D(e11.diffaug(2 * G(torch.randn(B, 128, device=DEV)) - 1)).float()).mean()
        oG.zero_grad(set_to_none=True); lG.backward()
        if masks:
            for p, m in masks.items():
                if p.grad is not None: p.grad.mul_(m)
        oG.step(); return lG
    for gr in oG.param_groups: gr["lr"] = 0.0
    for _ in range(warm): g_step()                                                   # second moments only; lr 0 moves nothing
    for gr in oG.param_groups: gr["lr"] = lr
    if on_eval: on_eval(0, G, ema)
    for step in range(1, steps + 1):
        with torch.no_grad(): fake = 2 * G(torch.randn(B, 128, device=DEV)) - 1
        with amp(): dr, df = D(e11.diffaug(real(B))).float(), D(e11.diffaug(fake)).float()
        lD = F.softplus(-dr).mean() + F.softplus(df).mean()
        if step % e11.R1_EVERY == 0:
            xr = e11.diffaug(real(B // 2)).detach().requires_grad_(True); g_, = torch.autograd.grad(D(xr).sum(), xr, create_graph=True)
            lD = lD + e11.R1_GAMMA / 2 * g_.pow(2).sum((1, 2, 3)).mean() * e11.R1_EVERY
        oD.zero_grad(set_to_none=True); lD.backward(); oD.step()
        g_step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), G.parameters()): pe.lerp_(p, 1 - ema_beta)
        if on_eval and step % every == 0: on_eval(step, G, ema)
    return ema
