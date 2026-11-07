"""
S2: lesion atlas. Remove one part of TinySR-26 at a time, heal for 1,000 steps, and record damage, recovery and where the
healing happens (see PREREG.md).

  <venv python> src/s2_lesion_atlas.py
"""
import time
import numpy as np
import common as C, sr_common as S, surgery_common as SC
import torch, torch.nn.functional as F

OUT = C.out_dir("s2"); STEPS, EVERY, N_DROP = 1000, 100, 12; CONDS = ("bicubic LR", "nearest LR")

def channel_lesion(W0, sutured, n_fit=2000):
    """Drop the N_DROP channels of block 2's inner layer with the least outgoing weight energy. Crude: zero their outgoing
    weights. Sutured: refit block 2's second conv in closed form from the remaining channels to reproduce its output."""
    W = {k: v.copy() for k, v in W0.items()}; Kb = W["body2_b/kernel"]                # (3, 3, 48 in, 48 out)
    energy = (Kb ** 2).sum((0, 1, 3)); drop = np.argsort(energy)[:N_DROP]; keep = np.setdiff1d(np.arange(Kb.shape[2]), drop)
    if sutured:
        m = SC.SR(W0).to(SC.DEV).eval(); xs = SC.train_patches(n_fit); U, Y = [], []
        with torch.no_grad():
            for s in range(0, len(xs), 64):
                r, _ = m.block_io(xs[s:s + 64], 2); u = F.relu(m.body[2][0](r)); y = m.body[2][1](u)
                cols = F.unfold(u[:, keep], 3, padding=1).transpose(1, 2).reshape(-1, len(keep) * 9)          # 3x3 neighbourhoods, kept channels
                g = torch.Generator(device=SC.DEV).manual_seed(s); idx = torch.randint(0, len(cols), (4000,), generator=g, device=SC.DEV)
                U.append(cols[idx]); Y.append(y.permute(0, 2, 3, 1).reshape(-1, y.shape[1])[idx])
        A, b = SC.lstsq(torch.cat(U), torch.cat(Y))                                   # A: (48 out, 36*9) in unfold order (channel-major, then 3x3)
        K = np.zeros_like(Kb); K[:, :, keep, :] = A.reshape(Kb.shape[3], len(keep), 3, 3).transpose(2, 3, 1, 0)
        W["body2_b/kernel"] = K.astype(np.float32); W["body2_b/bias"] = b.astype(np.float32)
    else:
        W["body2_b/kernel"][:, :, drop, :] = 0
    return W

def main():
    t0 = time.time(); T = S.TestSet(); W0 = S.base(); b0 = T.psnr(SC.SR(W0).to(SC.DEV), CONDS); R = {"base": b0, "runs": {}}
    ev = lambda m: {c: v - b0[c] for c, v in T.psnr(m, CONDS).items()}
    lesions = {"control": lambda: SC.SR(W0)}
    for i in range(6): lesions[f"remove block {i}"] = (lambda i=i: SC.SR(W0, blocks=[j for j in range(6) if j != i]))
    lesions["remove tail"] = lambda: SC.SR(W0, tail=False); lesions["remove head skip"] = lambda: SC.SR(W0, head_skip=False)
    lesions["12 channels, crude"] = lambda: SC.SR(channel_lesion(W0, False)); lesions["12 channels, sutured"] = lambda: SC.SR(channel_lesion(W0, True))
    for name, make in lesions.items():
        R["runs"][name] = []
        for seed in (0, 1, 2):
            m = make().to(SC.DEV); before = m.layers(); ema, log = S.train(m, STEPS, SC.objective, seed=seed, eval_every=EVERY, evaluate=ev)
            R["runs"][name].append({"seed": seed, "log": log, "plasticity": SC.plasticity(before, ema.layers())}); C.save_json(R, f"{OUT}/s2_results.json")
            print(name, seed, {k: round(v, 3) for k, v in log[0].items() if isinstance(v, float)}, "->", {k: round(v, 3) for k, v in log[-1].items() if isinstance(v, float)}, f"{time.time() - t0:.0f}s", flush=True)
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/s2_results.json")

if __name__ == "__main__":
    main()
