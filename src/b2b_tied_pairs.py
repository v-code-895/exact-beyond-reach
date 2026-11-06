"""
B2b: do tied ReLU pairs restore gradient shielding? (analysis only; see PREREG.md)

The odd-eye gate is a LeakyReLU pair: channel g0 computes lrelu(p), channel g1 computes lrelu(-p), and the
output layer reads them with weights w and a*w, which sums to (1 - a^2) relu(p) * w. B2 measured the pair
with independent weights and found no shielding. Here the pair is tied: one kernel, bias and output vector,
the second channel derived from them in the forward pass (identical function). Then d/dp of the pair is
lrelu'(p) - a*lrelu'(-p) = a - a = 0 wherever p < 0, so pixels where the gate is off should contribute
exactly nothing. Measured per batch as in B2, with the gradient restricted to right-half or left-half pixels.

  <venv python> followup/src/b2b_tied_pairs.py
"""
import time, json
import numpy as np
import common as C
import torch, torch.nn.functional as F
import exp11_catgan26 as e11

OUT = C.out_dir("b2b"); DEV = "cuda"; N_BATCH, B = 50, 32
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def main():
    t0 = time.time(); D, _ = C.torch_modules()[1](DEV); D.eval()
    for p_ in D.parameters(): p_.requires_grad_(False)
    W0 = C.catgan_base(); Wl, _, meta = C.odd_eye("left"); g0, g1 = meta["gate_slots"]; A = C.A
    FoldedG, _ = C.torch_modules(); g = FoldedG(Wl).to(DEV)
    Kfull = g.p("b5c2/kernel").detach(); Ko = g.p("out/kernel").detach()
    # tied gate parameters, taken from channel g0; channel g1 is derived
    Kg = Kfull[g0:g0 + 1].clone().requires_grad_(True); bg = g.p("b5c2/bias").detach()[g0:g0 + 1].clone().requires_grad_(True)
    wo = Ko[:, g0, 1, 1].clone().requires_grad_(True)
    checks = {"g1_kernel_is_minus_g0": float((Kfull[g1] + Kfull[g0]).abs().max()), "g1_bias_is_minus_g0": float((g.p("b5c2/bias")[g1] + g.p("b5c2/bias")[g0]).abs()),
              "g1_out_is_a_times_g0": float((Ko[:, g1, 1, 1] - A * Ko[:, g0, 1, 1]).abs().max()),
              "gate_out_off_centre_taps": float(Ko[:, [g0, g1]].abs().sum() - Ko[:, [g0, g1], 1, 1].abs().sum())}
    Kfree = Kfull.clone(); Kfree[[g0, g1]] = 0; bfree = g.p("b5c2/bias").detach().clone(); bfree[[g0, g1]] = 0
    Kofree = Ko.clone(); Kofree[:, [g0, g1]] = 0
    RIGHT = torch.zeros(1, 1, 128, 128, dtype=torch.bool, device=DEV); RIGHT[..., 64:] = True

    def forward(z, region=None, tied=True):
        h = F.leaky_relu((z @ g.p("dense/kernel") + g.p("dense/bias")).view(len(z), 4, 4, -1).permute(0, 3, 1, 2), A)
        for i in g.blocks:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            for j in (1, 2):
                if (i, j) != (5, 2):
                    h = F.leaky_relu(F.conv2d(h, g.p(f"b{i}c{j}/kernel"), g.p(f"b{i}c{j}/bias"), padding=1), A); continue
                p = F.conv2d(h, Kg, bg, padding=1)                                      # (N,1,H,W): the gate's pre-activation
                q = F.leaky_relu(p, A) + A * F.leaky_relu(-p, A)                       # tied pair: (1 - a^2) relu(p)
                term = wo[None, :, None, None] * q                                       # the gate's contribution to the image
                if region is not None:                                                   # gradient only through this region's pixels
                    m = RIGHT if region == "right" else ~RIGHT; term = torch.where(m, term, term.detach())
                hr = F.leaky_relu(F.conv2d(h, Kfree, bfree, padding=1), A)
                out = F.conv2d(hr, Kofree, g.p("out/bias"), padding=1) + term
                return out, p

    # sanity: the tied forward equals the patched network
    with torch.no_grad():
        z = torch.randn(8, 128, generator=torch.Generator().manual_seed(0)).to(DEV); checks["tied_forward_vs_patched_max_abs"] = float((forward(z)[0] - g(z)).abs().max())
    print("checks", checks, flush=True)

    rows = {r: {"Kg": [], "bg": [], "wo": []} for r in ("all", "right", "left", "untied")}; fired = []; s5 = int(meta["slots"]["b5c1"])
    gen = torch.Generator(device=DEV).manual_seed(0)
    for b in range(N_BATCH):
        z = torch.randn(B, 128, device=DEV, generator=gen)
        for region in ("all", "right", "left"):
            torch.manual_seed(1000 + b)                                                  # same augmentation in every pass
            x, p = forward(z, None if region == "all" else region)
            loss = F.softplus(-D(e11.diffaug(2 * x - 1))).mean()
            gK, gb, gw = torch.autograd.grad(loss, [Kg, bg, wo]); gK = gK[:, :s5]               # detector entries over b5c1, as B2's group
            for k, v in (("Kg", gK), ("bg", gb), ("wo", gw)): rows[region][k].append({"max_abs": float(v.abs().max()), "rms": float(v.pow(2).mean().sqrt())})
        torch.manual_seed(1000 + b)                                                      # the untied pair (B2's parameterisation), same batch
        loss = F.softplus(-D(e11.diffaug(2 * g(z) - 1))).mean()
        uK, ub, uo = torch.autograd.grad(loss, [g.p("b5c2/kernel"), g.p("b5c2/bias"), g.p("out/kernel")])
        for k, v in (("Kg", uK[g0:g1 + 1, :s5]), ("bg", ub[g0:g1 + 1]), ("wo", uo[:, g0:g1 + 1, 1, 1])):
            rows["untied"][k].append({"max_abs": float(v.abs().max()), "rms": float(v.pow(2).mean().sqrt())})
        fired.append({"left_px": int((p[..., :64] > 0).sum()), "right_px": int((p[..., 64:] > 0).sum()), "right_max_pre": float(p[..., 64:].max())})
    b2 = json.load(open(f"{C.RESULTS}/b2/b2_results.json", encoding="utf-8"))
    R = {"checks": checks, "fired": fired,
         "tied": {r: {k: {"max_abs_over_batches": max(x["max_abs"] for x in v), "rms_mean": float(np.mean([x["rms"] for x in v])),
                          "batches_exactly_zero": int(sum(x["max_abs"] == 0.0 for x in v))} for k, v in d.items()} for r, d in rows.items()},
         "b2_untied_reference": {k: v for k, v in b2.items() if "gate" in k.lower() or "existing" in k.lower()} if isinstance(b2, dict) else None,
         "seconds": time.time() - t0}
    C.save_json(R, f"{OUT}/b2b_results.json")
    print(json.dumps(R["tied"], indent=1)); print("fired: left px per batch (mean)", np.mean([f["left_px"] for f in fired]), "right px", sum(f["right_px"] for f in fired),
                                                 "max right pre-activation", max(f["right_max_pre"] for f in fired))

if __name__ == "__main__":
    main()
