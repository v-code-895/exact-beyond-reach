"""
B1 pilot (exploratory, not pre-registered; it sizes B1). How fast does training break the odd-eye
patch's invariant? Adam (lr 2e-4, betas 0/0.99) on the patched folded generator's non-saturating loss
against paper one's discriminator, which is held fixed here (B1 proper trains both). One seed.

Invariant: the gate writes nothing into the right half. Measured as the largest right-half difference
between the trained network and the same network with the gate's outgoing weights zeroed.
Also: share of left irises in the blue band (paper one's 300 test latents), and wire purity at b5c1
(largest deviation of the carrier from the exact right-half indicator).

  <venv python> src/b1_pilot.py
"""
import time
import numpy as np
import common as C
import torch, torch.nn.functional as F
import exp11_catgan26 as e11, exp20_odd_eyes as e20, cat_domain as cd

OUT = C.out_dir("b1_pilot"); DEV = "cuda"
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def main(model="patched", lr=2e-4, steps=1000):
    t0 = time.time(); FoldedG, disc = C.torch_modules(); D, _ = disc(DEV)
    for p in D.parameters(): p.requires_grad_(False)
    W0 = C.catgan_base(); Wl, _, meta = C.odd_eye("left"); g0, s5 = meta["gate_slots"], int(meta["slots"]["b5c1"])
    G = FoldedG(Wl if model == "patched" else W0).to(DEV); opt = torch.optim.Adam(G.parameters(), lr, betas=(0.0, 0.99))
    rng = np.random.default_rng(0); _, _, z_test = (rng.standard_normal((n, 128)) for n in (256, 300, 300))
    zt = torch.tensor(z_test, dtype=torch.float32, device=DEV)
    from exp18_circuit_primitives import catgan_layers
    x0 = e20.forward(z_test, W0, catgan_layers(W0)); m = cd.eye_mask(np.clip(x0, 0, 1)); left = np.zeros((128, 128), bool); left[:, :64] = True
    def metrics():
        with torch.no_grad():
            x = torch.cat([G(zt[i:i + 50]) for i in range(0, 300, 50)])
            if model != "patched":
                return {"mean_change_vs_base": float((x - torch.from_numpy(x0).permute(0, 3, 1, 2).to(DEV)).abs().mean())}
            Wk = G.p("out/kernel"); saved = Wk[:, g0[0]:g0[1] + 1].clone(); Wk[:, g0[0]:g0[1] + 1] = 0
            xo = torch.cat([G(zt[i:i + 50]) for i in range(0, 300, 50)]); Wk[:, g0[0]:g0[1] + 1] = saved
            leak = float((x - xo)[:, :, :, 64:].abs().max())
            h = zt[:50] @ G.p("dense/kernel") + G.p("dense/bias"); h = F.leaky_relu(h.view(50, 4, 4, -1).permute(0, 3, 1, 2), C.A)
            for i in G.blocks:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                for j in (1, 2):
                    h = F.leaky_relu(F.conv2d(h, G.p(f"b{i}c{j}/kernel"), G.p(f"b{i}c{j}/bias"), padding=1), C.A)
                    if (i, j) == (5, 1): wire = h[:, s5]
            want = torch.zeros(128, 128, device=DEV); want[:, 64:] = 1
            purity = float((wire - want).abs().max())
        xn = x.permute(0, 2, 3, 1).cpu().numpy()
        hl = e20.per_image_hue(xn, m & left[None]); share = float(np.mean([200 <= h_ <= 250 for h_ in hl if h_ is not None]))
        drift = float((x - torch.from_numpy(x0).permute(0, 3, 1, 2).to(DEV)).abs().mean())
        return {"right_half_gate_leak_max": leak, "left_share_in_band": share, "wire_purity_max_dev": purity, "mean_change_vs_base": drift}
    log = [{"step": 0, **metrics()}]; print(log[-1], flush=True)
    for step in range(1, steps + 1):
        torch.manual_seed(step); z = torch.randn(32, 128, device=DEV)
        loss = F.softplus(-D(e11.diffaug(2 * G(z) - 1))).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
            log.append({"step": step, **metrics()}); print(log[-1], flush=True)
    C.save_json({"model": model, "lr": lr, "log": log, "seconds": time.time() - t0}, f"{OUT}/b1_pilot_{model}_lr{lr:g}.json")
    return log

if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    main(a[0] if a else "patched", float(a[1]) if len(a) > 1 else 2e-4, int(a[2]) if len(a) > 2 else 1000)
