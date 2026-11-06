"""
S3: phantom limb. Cut TinySR-26's fixed bicubic shortcut, let the network heal, then re-attach the shortcut (see PREREG.md).

  <venv python> followup/src/s3_phantom_limb.py
"""
import time
import numpy as np
import common as C, sr_common as S, surgery_common as SC
import torch, torch.nn.functional as F

OUT = C.out_dir("s3"); STEPS, REHEAL, EVERY = 3000, 1000, 100; CONDS = ("bicubic LR", "nearest LR")

def main():
    t0 = time.time(); T = S.TestSet(); W0 = S.base(); b0 = T.psnr(SC.SR(W0).to(SC.DEV), CONDS); R = {"base": b0, "runs": {}}
    lf_x = T.lr["bicubic LR"][:20]
    def ev(m):
        r = {c: v - b0[c] for c, v in T.psnr(m, CONDS).items()}
        with torch.no_grad():                                                        # the network's own low-frequency share
            cs = []
            for x in lf_x:
                a = m.residual(x).flatten(); b = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False).flatten()
                cs.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
        r["corr_with_bicubic"] = float(np.mean(cs)); return r
    arms = {"(a) shortcut cut, healed": lambda seed: SC.SR(W0, bicubic=False), "(b) control": lambda seed: SC.SR(W0),
            "(c) newborn without shortcut": lambda seed: SC.SR(SC.random_W(seed), bicubic=False)}
    for name, make in arms.items():
        R["runs"][name] = []
        for seed in (0, 1, 2):
            m = make(seed).to(SC.DEV); ema, log = S.train(m, STEPS, SC.objective, seed=seed, eval_every=EVERY, evaluate=ev)
            run = {"seed": seed, "log": log}
            if name.startswith("(a)"):
                ema.bicubic = True; m2 = ema.train(); ema2, log2 = S.train(m2, REHEAL, SC.objective, seed=100 + seed, eval_every=EVERY, evaluate=ev)
                run["reattached_log"] = log2
            R["runs"][name].append(run); C.save_json(R, f"{OUT}/s3_results.json")
            print(name, seed, {k: round(v, 3) for k, v in log[0].items() if isinstance(v, float)}, "->", {k: round(v, 3) for k, v in log[-1].items() if isinstance(v, float)},
                  ("| reattached " + str({k: round(v, 3) for k, v in run["reattached_log"][0].items() if isinstance(v, float)}) + " -> " +
                   str({k: round(v, 3) for k, v in run["reattached_log"][-1].items() if isinstance(v, float)})) if "reattached_log" in run else "", f"{time.time() - t0:.0f}s", flush=True)
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/s3_results.json")

if __name__ == "__main__":
    main()
