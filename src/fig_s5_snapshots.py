"""
Illustration only (not a registered result): rerun S5 arms with image snapshots, for the paper's figures.
Saves, every 500 steps, the EMA generator's unfiltered and filtered images for 8 fixed latents.

  <venv python> followup/src/fig_s5_snapshots.py
"""
import time
import numpy as np
import common as C, gan_common as G_
import torch
import s5_filter as S5

OUT = C.out_dir("fig_s5"); DEV = "cuda"

def main():
    t0 = time.time(); W0 = C.catgan_base(); dets, info = S5.fit(W0); print("fit", info, flush=True)
    z = torch.tensor(np.random.default_rng(77).standard_normal((64, 128)), dtype=torch.float32, device=DEV)
    for name, kind, seed in (("feature filter, seed 0", "feature", 0), ("pixel filter, seed 1", "pixel", 1)):
        D, oD = G_.discriminator(); w, b = dets[kind]; Gm = S5.Filtered(W0, kind, w, b).to(DEV)
        for p in (Gm.w, Gm.b, Gm.k): p.requires_grad_(False)
        snaps = {}
        def on_eval(step, Gt, ema):
            with torch.no_grad():
                xu = ema.unfiltered(z).clamp(0, 1); xf = ema(z).clamp(0, 1)
            snaps[f"unfiltered_{step}"] = (xu.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            snaps[f"filtered_{step}"] = (xf.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            lab = S5.orange_label(xu.permute(0, 2, 3, 1).cpu().numpy()); print(name, step, "orange share", float(lab.mean()), f"{time.time() - t0:.0f}s", flush=True)
        G_.train(Gm, D, oD, G_.real_cats(), S5.STEPS, S5.LR, every=500, on_eval=on_eval, seed=seed)
        np.savez_compressed(f"{OUT}/{kind}_s{seed}.npz", **snaps); del Gm, D, oD; torch.cuda.empty_cache()
    print("done", f"{time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
