"""
L5 (TinySR part): what the fixed bicubic shortcut is worth in width (see PREREG.md, "L-series", L5).
TinySR-26's architecture (6 residual blocks) from random initialisation at 24, 48 and 96 channels, with and without the
shortcut, 3,000 steps of S3's objective, seeds 0-1. 48 channels without the shortcut is S3's newborn arm (seeds 0-1), reused.

  <venv python> followup/src/l5_exchange.py [--smoke]
"""
import json, os, sys, time
import numpy as np
import common as C, sr_common as S, surgery_common as SC
import torch
import exp10_tinysr26 as e10

SMOKE = "--smoke" in sys.argv; OUT = C.out_dir("l5" + ("_smoke" if SMOKE else "")); STEPS, EVERY = (3000, 500) if not SMOKE else (20, 10)
CONDS = ("bicubic LR", "nearest LR")

def random_W(seed, ch):
    torch.manual_seed(seed); return e10.export(e10.TinySR(2, ch, 6))

def main():
    t0 = time.time(); T = S.TestSet(); b0 = T.psnr(SC.SR(S.base()).to(SC.DEV), CONDS); R = {"base": b0, "runs": {}}
    ev = lambda m: {c: v - b0[c] for c, v in T.psnr(m, CONDS).items()}
    s3 = json.load(open(os.path.join(C.RESULTS, "s3", "s3_results.json")))["runs"]["(c) newborn without shortcut"]
    for ch in ((24, 48, 96) if not SMOKE else (24,)):
        for shortcut in (True, False):
            key = f"{ch} channels, " + ("with" if shortcut else "without") + " shortcut"; R["runs"][key] = []
            for seed in ((0, 1) if not SMOKE else (0,)):
                if ch == 48 and not shortcut and not SMOKE:
                    R["runs"][key].append({"seed": seed, "log": s3[seed]["log"], "source": "S3 (c), same initialisation and objective"}); continue
                W = random_W(seed, ch); m = SC.SR(W, bicubic=shortcut).to(SC.DEV)
                ema, log = S.train(m, STEPS, SC.objective, seed=seed, eval_every=EVERY, evaluate=ev)
                n_params = int(sum(p.numel() for p in m.parameters()))
                R["runs"][key].append({"seed": seed, "params": n_params, "log": log}); C.save_json(R, f"{OUT}/l5_results.json")
                print(key, seed, "params", n_params, {k: round(v, 3) for k, v in log[-1].items() if isinstance(v, float)}, f"{time.time() - t0:.0f}s", flush=True)
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/l5_results.json")

if __name__ == "__main__":
    main()
