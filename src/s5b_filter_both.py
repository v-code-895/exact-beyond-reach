"""
S5b: why does a frozen filter pull the generator into the filtered category? (see PREREG.md, S5b)
Hypothesis: the filter removes orange cats from the generated images the discriminator sees but not from the real ones, so the
discriminator learns orange as a mark of a real image and the generator chases it into the filter. Test: filter the real images
the same way, and measure the discriminator's score on real orange vs real non-orange cats.
  (a) pixel filter on generated images only (S5 arm 3), seed 0, rerun with discriminator scores
  (b) pixel filter on generated AND real images, seeds 0-2
  (c) control: no filter, seeds 0-1

  <venv python> src/s5b_filter_both.py          (S5B_SMOKE=1 for a short run)
"""
import os, time
import numpy as np
import common as C, gan_common as G_
import torch, torch.nn.functional as F
import s5_filter as S5

SMOKE = os.environ.get("S5B_SMOKE") == "1"; OUT = C.out_dir("s5b" + ("_smoke" if SMOKE else "")); DEV = "cuda"
STEPS, EVERY = (2000, 250) if not SMOKE else (20, 10)

def filtered_real(real, Gm):
    """The same frozen pixel filter applied to real images (in [-1, 1]) before the discriminator sees them."""
    def f(n):
        x = (real(n) + 1) / 2
        with torch.no_grad():
            g = Gm.gate(x, None)[:, None, None, None]; d = 0.5 - x
            y = x + S5.relu_pair(d - S5.M * (1 - g)) - S5.relu_pair(-d - S5.M * (1 - g))
        return 2 * y - 1
    return f

def main():
    t0 = time.time(); W0 = C.catgan_base(); dets, info = S5.fit(W0); R = {"fit": info, "runs": {}}; print("fit", info, flush=True)
    fd = G_.FD(G_.cat_val_u8()); ev = C.OddEyeEval(); cats = S5.e11.load_cats(); oy = S5.orange_label(cats.astype(np.float32) / 255)
    real_all = G_.real_cats(); t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().to(DEV) / 127.5 - 1
    Xo, Xn = t(cats[oy][:300]), t(cats[~oy][:300]); R["orange_training_images"] = int(oy.sum())
    arms = [("(a) pixel filter on generated images only", True, False, (0,)), ("(b) pixel filter on generated and real images", True, True, (0, 1, 2)),
            ("(c) control: no filter", False, False, (0, 1))]
    for name, filt, filt_real, seeds in arms:
        R["runs"][name] = []
        for seed in (seeds if not SMOKE else (0,)):
            D, oD = G_.discriminator(); w, b = dets["pixel"]; Gm = S5.Filtered(W0, "pixel", w, b).to(DEV); Gm.on = filt
            for p in (Gm.w, Gm.b, Gm.k): p.requires_grad_(False)
            real = filtered_real(real_all, Gm) if filt_real else real_all; log = []
            def on_eval(step, Gt, ema):
                with torch.no_grad():
                    xu, xf = [], []
                    for s in range(0, 300, 50):
                        z = ev.z[s:s + 50]; xu.append(ema.unfiltered(z)); xf.append(ema(z))
                    xu, xf = torch.cat(xu), torch.cat(xf)
                    with torch.autocast("cuda", dtype=torch.bfloat16): do, dn = (torch.cat([D(X[i:i + 100]).float() for i in range(0, len(X), 100)]) for X in (Xo, Xn))
                lab = S5.orange_label(xu.permute(0, 2, 3, 1).cpu().numpy())
                blk = ((xf - 0.5).abs().amax((1, 2, 3)) < 1e-3).cpu().numpy()
                r = {"step": step, "orange_share_unfiltered": float(lab.mean()), "orange_blocked": float(blk[lab].mean()) if lab.any() else None,
                     "non_orange_blocked": float(blk[~lab].mean()), "fd_unfiltered": fd(lambda zz: ema.unfiltered(zz)),
                     "D_real_orange_minus_non_orange": float(do.mean() - dn.mean()), "D_gap_normalised": float((do.mean() - dn.mean()) / torch.cat([do, dn]).std())}
                log.append(r); print(name, seed, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)
            G_.train(Gm, D, oD, real, STEPS, S5.LR, every=EVERY, on_eval=on_eval, seed=seed)
            R["runs"][name].append({"seed": seed, "log": log}); C.save_json(R, f"{OUT}/s5b_results.json")
            del Gm, D, oD; torch.cuda.empty_cache()
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/s5b_results.json")

if __name__ == "__main__":
    main()
