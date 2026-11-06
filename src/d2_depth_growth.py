"""
D2: exact vs approximate depth growth on TinySR-26 (see PREREG.md).

6 -> 9 residual blocks, then 3,000 steps of paper one's TinySR-26 objective (mixed downsampling kernels,
L1), Adam 1e-4, EMA, seeds 0-2:
  (a) exact: three residual blocks inserted after blocks 2, 3 and 4, with a zero second convolution
      (default-initialised first convolution), so the function is unchanged;
  (b) duplicate: blocks 2, 3 and 4 copied after themselves (the function changes, as in depth up-scaling);
  (c) control: the 6-block base, trained the same way;
  (d) from scratch: the 9-block architecture at random initialisation (reference only).
Evaluated every 250 steps on paper one's 180 test photos (bicubic and nearest inputs); EMA weights kept
every 1,000 steps.

  <venv python> followup/src/d2_depth_growth.py
"""
import os, time
import numpy as np
import common as C, sr_common as S
import torch
import exp10_tinysr26 as e10

OUT = C.out_dir("d2"); CK = C.out_dir("d2/ckpt")

def blocks_dict(W, order, fresh=None):
    """A dict whose body blocks follow `order`: an int copies that block of W, 'Z' inserts a fresh block whose
    second conv is zero (taken from the fresh, default-initialised network)."""
    out = {k: v.copy() for k, v in W.items() if not k.startswith("body")}; z = 0
    for i, o in enumerate(order):
        for part in ("a", "b"):
            for t in ("kernel", "bias"):
                if o == "Z":
                    src = fresh[f"body{z}_{part}/{t}"]; out[f"body{i}_{part}/{t}"] = np.zeros_like(src) if part == "b" else src.copy()
                else:
                    out[f"body{i}_{part}/{t}"] = W[f"body{o}_{part}/{t}"].copy()
        if o == "Z": z += 1
    return out

def objective(hr):
    return torch.cat([e10.degrade(g, 2, k) for g, k in zip(hr.chunk(4), e10.KERNELS)]), hr

def main(seeds=(0, 1, 2), steps=3000):
    t0 = time.time(); T = S.TestSet(); W0 = S.base(); conds = ("bicubic LR", "nearest LR")
    b0 = T.psnr(S.module(W0), conds); R = {"base": b0, "runs": {}}
    arms = {"(a) exact: 3 zero-output blocks inserted": [0, 1, 2, "Z", 3, "Z", 4, "Z", 5],
            "(b) duplicate blocks 2-4": [0, 1, 2, 2, 3, 3, 4, 4, 5],
            "(c) control: 6 blocks": [0, 1, 2, 3, 4, 5],
            "(d) from scratch: 9 blocks": None}
    for name, order in arms.items():
        R["runs"][name] = []
        for seed in seeds:
            torch.manual_seed(seed); fresh = e10.export(e10.TinySR(2, 48, 9).to(S.DEV))
            W = fresh if order is None else blocks_dict(W0, order, fresh)
            m = S.module(W); inserted = [i for i, o in enumerate(order or []) if o == "Z"] or [i for i, o in enumerate(order or []) if i > 0 and order[i - 1] == o]
            ref_norm = float(np.mean([np.linalg.norm(W0[f"body{j}_b/kernel"]) for j in range(6)]))
            def ev(mm):
                r = {c: v - b0[c] for c, v in T.psnr(mm, conds).items()}
                if inserted:
                    r["inserted_b_norm_rel"] = [float(mm.body[i][2].weight.norm()) / ref_norm for i in inserted]
                return r
            tag = f"{name[:3].strip('()')}_s{seed}"
            keep = lambda step, ema, tag=tag: np.savez(f"{CK}/{tag}_step{step}.npz", **e10.export(ema))
            ema, log = S.train(m, steps, objective, seed=seed, eval_every=250, evaluate=ev, keep_every=1000, on_keep=keep)
            R["runs"][name].append({"seed": seed, "log": log, "params": int(sum(p.numel() for p in ema.parameters()))})
            print(name, seed, "step 0:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in log[0].items()},
                  "final:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in log[-1].items() if k != "seconds"}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{OUT}/d2_results.json")
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/d2_results.json")
    lines = ["# D2: exact vs approximate depth growth (TinySR-26)", "", f"PSNR change against the 6-block base (dB), mean ± sd over seeds; base: {b0}.", "",
             "| arm | step 0: bicubic / nearest | step 1,000 | step 3,000 | inserted blocks' output conv, norm relative to base blocks (final) |", "|---|---|---|---|---|"]
    for name, runs in R["runs"].items():
        def at(step, c): return [next(l for l in r["log"] if l["step"] == step)[c] for r in runs]
        cell = lambda step: f"{np.mean(at(step, 'bicubic LR')):+.3f} ± {np.std(at(step, 'bicubic LR')):.3f} / {np.mean(at(step, 'nearest LR')):+.3f} ± {np.std(at(step, 'nearest LR')):.3f}"
        ins = runs[0]["log"][-1].get("inserted_b_norm_rel")
        norms = "-" if ins is None else ", ".join("%.3f" % np.mean([r["log"][-1]["inserted_b_norm_rel"][k] for r in runs]) for k in range(len(ins)))
        lines.append(f"| {name} | {cell(0)} | {cell(1000)} | {cell(3000)} | {norms} |")
    open(f"{OUT}/d2_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
