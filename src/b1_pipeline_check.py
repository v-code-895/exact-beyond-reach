"""
B1 pipeline check (exploratory; it decides how B1 and A1 will train). Can CatGAN-26 keep training stably
from paper one's step-30,000 state, unpatched? Three setups, 2,000 steps each, D trained as in paper one:
  (i)   reference: paper one's BatchNorm generator with its own G, D, EMA and both Adam states;
  (ii)  folded generator (the form patches live in), fresh Adam with second moments warmed for 100
        batches before the first update, lr 2e-5;
  (iii) the same at lr 2e-4.
Every 500 steps: Fréchet distance in MobileNetV2 features (paper one's metric) between 500 EMA samples
and 500 real AFHQ validation cats, and the mean change of the EMA images against the base on fixed latents.

  <venv python> followup/src/b1_pipeline_check.py
"""
import copy, os, time, sys
import numpy as np
import common as C
import torch, torch.nn.functional as F
import exp11_catgan26 as e11, exp12_modern_eval as e12, mobilenet_np as mn, modern_np as mnp
from PIL import Image

OUT = C.out_dir("b1_pipeline"); DEV = "cuda"; B = 32
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def real_val_features(n=500):
    d = os.path.join("data/afhq/data/afhq/val/cat"); fs = sorted(os.listdir(d))[:n]
    imgs = np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB").resize((128, 128), Image.LANCZOS)) for f in fs])
    return e12.mobilenet(imgs, mn.load())[1]

def fd_of(gen_fn, Freal, n=500):
    z = torch.randn(n, 128, generator=torch.Generator().manual_seed(7)).to(DEV)
    with torch.no_grad(): x = torch.cat([gen_fn(z[s:s + 50]) for s in range(0, n, 50)]).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return e12.frechet(e12.mobilenet((x * 255).round().astype(np.uint8), mn.load())[1], Freal)

def run(setup, steps=2000, every=500):
    t0 = time.time(); torch.manual_seed(0); FoldedG, disc = C.torch_modules(); D, c = disc(DEV)
    oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.0, 0.99)); oD.load_state_dict(c["oD"])
    data = torch.from_numpy(e11.load_cats()).permute(0, 3, 1, 2).contiguous().to(DEV)
    W0 = C.catgan_base(); zf = torch.randn(64, 128, generator=torch.Generator().manual_seed(123)).to(DEV)
    x_base = mnp.catgan_torch(zf, mnp.to_torch(W0)).detach()
    if setup == "bn":
        G = e11.Generator().to(DEV); G.load_state_dict(c["G"]); G_ema = e11.Generator().to(DEV); G_ema.load_state_dict(c["G_ema"])
        oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.0, 0.99)); oG.load_state_dict(c["oG"])
        def eval_gen():                                                       # one BatchNorm recalibration per evaluation
            g = copy.deepcopy(G_ema); e11.recalibrate_bn(g, DEV)
            def f(z):
                with torch.no_grad(): return g(z) / 2 + 0.5
            return f
        gen_train = lambda z: G(z)                                            # BatchNorm generator outputs [-1, 1]
    else:
        lr = {"folded_2e-5": 2e-5, "folded_2e-4": 2e-4}[setup]
        G = FoldedG(W0).to(DEV); G_ema = copy.deepcopy(G).eval(); oG = torch.optim.Adam(G.parameters(), lr, betas=(0.0, 0.99))
        for g in oG.param_groups: g["lr"] = 0.0
        for _ in range(100):                                                    # warm Adam's second moments, no update
            lG = F.softplus(-D(e11.diffaug(2 * G(torch.randn(B, 128, device=DEV)) - 1))).mean(); oG.zero_grad(); lG.backward(); oG.step()
        for g in oG.param_groups: g["lr"] = lr
        def eval_gen():
            def f(z):
                with torch.no_grad(): return G_ema(z)
            return f
        gen_train = lambda z: 2 * G(z) - 1
    Freal = real_val_features(); log = []
    def record(step):
        ge = eval_gen(); xe = ge(zf)
        log.append({"step": step, "fd": fd_of(ge, Freal), "mean_change_vs_base": float((xe - x_base).abs().mean()), "seconds": time.time() - t0})
        print(setup, log[-1], flush=True)
    record(0)
    def real(n):
        x = data[torch.randint(0, len(data), (n,), device=DEV)].float() / 127.5 - 1
        return torch.where(torch.rand(n, 1, 1, 1, device=DEV) < 0.5, x.flip(3), x)
    for step in range(1, steps + 1):
        with torch.no_grad(): fake = gen_train(torch.randn(B, 128, device=DEV))
        dr, df = D(e11.diffaug(real(B))), D(e11.diffaug(fake))
        lD = F.softplus(-dr).mean() + F.softplus(df).mean()
        if step % e11.R1_EVERY == 0:
            xr = e11.diffaug(real(B // 2)).detach().requires_grad_(True); g_, = torch.autograd.grad(D(xr).sum(), xr, create_graph=True)
            lD = lD + e11.R1_GAMMA / 2 * g_.pow(2).sum((1, 2, 3)).mean() * e11.R1_EVERY
        oD.zero_grad(set_to_none=True); lD.backward(); oD.step()
        lG = F.softplus(-D(e11.diffaug(gen_train(torch.randn(B, 128, device=DEV))))).mean()
        oG.zero_grad(set_to_none=True); lG.backward(); oG.step()
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()): pe.lerp_(p, 1 - 0.999)
        if step % every == 0: record(step)
    C.save_json({"setup": setup, "log": log}, f"{OUT}/{setup}.json"); return log

if __name__ == "__main__":
    for s in (sys.argv[1:] or ["bn", "folded_2e-5", "folded_2e-4"]): run(s)
