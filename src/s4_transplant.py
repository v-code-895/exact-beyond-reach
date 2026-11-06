"""
S4: transplant and rejection. Replace block 2 of TinySR-26 with a block from a related network (paper one's experts, same
lineage) or an unrelated one (D2's from-scratch network), optionally stitched in with closed-form 1x1 adapters, and heal
for 1,000 steps (see PREREG.md).

  <venv python> followup/src/s4_transplant.py
"""
import os, time
import numpy as np
import common as C, sr_common as S, surgery_common as SC
import torch, torch.nn.functional as F

OUT = C.out_dir("s4"); STEPS, EVERY, POS = 1000, 100, 2

def stitch(W0, Wd, i_donor, n_fit=2000, n_px=4000):
    """Adapters ((A_in, b_in), (A_out, b_out)): recipient's block-POS input -> donor's block input at i_donor; donor block's
    output on the adapted input -> the recipient's original block-POS output. Least squares on training-patch pixels."""
    rec, don = SC.SR(W0).to(SC.DEV).eval(), SC.SR(Wd).to(SC.DEV).eval(); xs = SC.train_patches(n_fit)
    Rin, Din = [], []
    with torch.no_grad():
        for s in range(0, len(xs), 64):
            r, _ = rec.block_io(xs[s:s + 64], POS); d, _ = don.block_io(xs[s:s + 64], i_donor)
            g = torch.Generator(device=SC.DEV).manual_seed(s); idx = torch.randint(0, r.numel() // r.shape[1], (n_px,), generator=g, device=SC.DEV)
            Rin.append(r.permute(0, 2, 3, 1).reshape(-1, r.shape[1])[idx]); Din.append(d.permute(0, 2, 3, 1).reshape(-1, d.shape[1])[idx])
    A_in, b_in = SC.lstsq(torch.cat(Rin), torch.cat(Din))
    ad_in = SC.conv1x1(A_in, b_in).to(SC.DEV); blk = don.body[i_donor]; Od, Or = [], []
    with torch.no_grad():
        for s in range(0, len(xs), 64):
            r, o = rec.block_io(xs[s:s + 64], POS); od = blk[1](F.relu(blk[0](ad_in(r))))
            g = torch.Generator(device=SC.DEV).manual_seed(10_000 + s); idx = torch.randint(0, r.numel() // r.shape[1], (n_px,), generator=g, device=SC.DEV)
            Od.append(od.permute(0, 2, 3, 1).reshape(-1, od.shape[1])[idx]); Or.append(o.permute(0, 2, 3, 1).reshape(-1, o.shape[1])[idx])
    A_out, b_out = SC.lstsq(torch.cat(Od), torch.cat(Or))
    return (A_in, b_in), (A_out, b_out)

def main():
    t0 = time.time(); T = S.TestSet(); E = S.experts(); W0 = E["base"]; b0 = T.psnr(SC.SR(W0).to(SC.DEV)); R = {"base": b0, "runs": {}}
    Ws = {k: v.astype(np.float32) for k, v in np.load(os.path.join(C.RESULTS, "d2", "ckpt", "d_s0_step3000.npz")).items()}
    ev = lambda m: {c: v - b0[c] for c, v in T.psnr(m).items()}
    body = lambda spec: [j if j != POS else spec for j in range(6)]
    adapt = stitch(W0, Ws, 4); R["stitch_fit"] = "least squares on 2,000 training patches, 4,000 pixels per batch of 64"
    arms = {"(a) same lineage: bicubic expert's block 2": body({"W": E["bicubic"], "i": 2}),
            "(b) same lineage: noisy expert's block 2": body({"W": E["noisy"], "i": 2}),
            "(c) foreign: from-scratch network's block 4": body({"W": Ws, "i": 4}),
            "(d) foreign + stitching adapters": body({"W": Ws, "i": 4, "adapt": adapt}),
            "(e) control": list(range(6))}
    for name, blocks in arms.items():
        R["runs"][name] = []
        for seed in (0, 1, 2):
            m = SC.SR(W0, blocks=blocks).to(SC.DEV); ema, log = S.train(m, STEPS, SC.objective, seed=seed, eval_every=EVERY, evaluate=ev)
            R["runs"][name].append({"seed": seed, "log": log}); C.save_json(R, f"{OUT}/s4_results.json")
            print(name, seed, {k: round(v, 3) for k, v in log[0].items() if isinstance(v, float)}, "->", {k: round(v, 3) for k, v in log[-1].items() if isinstance(v, float)}, f"{time.time() - t0:.0f}s", flush=True)
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/s4_results.json")

if __name__ == "__main__":
    main()
