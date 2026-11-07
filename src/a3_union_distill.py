"""
A3: distilling the union (see PREREG.md).

Paper one's routed union keeps both fine-tunes' gains (+0.23 / +3.26 / +0.00 dB) at three networks' size;
joint retraining of one network did not (+0.03 / +3.19 / +0.08). Arms, all 3,000 steps from TinySR-26 on
paper one's mixed batches, seeds 0-2:
  (a) joint retraining on ground truth;
  (b) distillation: the target is the union's output, i.e. the expert for each input's degradation;
  (c) distillation + ground truth (losses averaged);
  (d) 3x width (exactly widened base), joint retraining on ground truth.

  <venv python> src/a3_union_distill.py
"""
import time, sys
import numpy as np
import common as C, sr_common as S
import torch, torch.nn.functional as F

OUT = C.out_dir("a3")

def main(arms=("a", "b", "c", "d"), seeds=(0, 1, 2), steps=3000):
    t0 = time.time(); T = S.TestSet(); E = S.experts(); W0 = E["base"]
    base_scores = T.psnr(S.module(W0)); R = {"base": base_scores, "experts": {}, "runs": {}}
    teachers = {k: S.module(W).eval() for k, W in E.items()}
    for p in (p_ for t in teachers.values() for p_ in t.parameters()): p.requires_grad_(False)
    for k in ("bicubic", "noisy"): R["experts"][k] = {c: v - base_scores[c] for c, v in T.psnr(teachers[k]).items()}
    print("base", base_scores, "experts", R["experts"], flush=True)

    def batch_fn(kind):
        def f(hr):
            x, (h1, h2, h3) = S.mixed_batch(hr); n1, n2 = len(h1), len(h2); gt = torch.cat([h1, h2, h3])
            if kind == "gt": return x, gt
            with torch.no_grad():
                dist = torch.cat([teachers["bicubic"](x[:n1]), teachers["noisy"](x[n1:n1 + n2]), teachers["base"](x[n1 + n2:])])
            return x, dist
        return f
    # (c) uses 0.5 * (L1(y, dist) + L1(y, gt)): the batch is doubled, once per target (below).

    spec = {"a": ("joint retraining, ground truth", "gt", 48), "b": ("distillation from the union", "distil", 48),
            "c": ("distillation + ground truth", "both", 48), "d": ("3x width, joint retraining, ground truth", "gt", 144)}
    for arm in arms:
        name, kind, ch = spec[arm]; R["runs"][name] = []
        for seed in seeds:
            W = W0 if ch == 48 else S.widen(W0, ch, np.random.default_rng(seed)); m = S.module(W)
            if ch != 48:
                d0 = {c: T.psnr(m)[c] - base_scores[c] for c in S.CONDS}; assert max(abs(v) for v in d0.values()) < 1e-3, d0
            if kind == "both":
                def f(hr, _kind=kind):
                    x, (h1, h2, h3) = S.mixed_batch(hr); n1, n2 = len(h1), len(h2); gt = torch.cat([h1, h2, h3])
                    with torch.no_grad():
                        dist = torch.cat([teachers["bicubic"](x[:n1]), teachers["noisy"](x[n1:n1 + n2]), teachers["base"](x[n1 + n2:])])
                    return torch.cat([x, x]), torch.cat([dist, gt])            # mean of the two L1 losses
                fn = f
            else:
                fn = batch_fn(kind)
            ev = lambda mm: {c: v - base_scores[c] for c, v in T.psnr(mm).items()}
            ema, log = S.train(m, steps, fn, seed=seed, eval_every=500, evaluate=ev)
            fin = log[-1]; fin["meets_spec"] = bool(fin["bicubic LR"] > 0 and fin["noisy bicubic LR"] > 0 and min(fin[c] for c in S.CONDS) >= -0.1)
            fin["params"] = int(sum(p.numel() for p in ema.parameters()))
            R["runs"][name].append({"seed": seed, "final": fin, "log": log})
            print(name, seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in fin.items()}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{OUT}/a3_results.json")
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/a3_results.json")
    lines = ["# A3: distilling the union", "", "PSNR change against the base (dB) on paper one's 180 test photos per input type; mean ± sd over seeds.", "",
             f"Experts: bicubic branch {R['experts']['bicubic']}, noisy branch {R['experts']['noisy']}.", "",
             "| arm | bicubic inputs | noisy inputs | nearest inputs | meets spec | parameters |", "|---|---|---|---|---|---|"]
    for name, runs in R["runs"].items():
        cell = lambda c: f"{np.mean([r['final'][c] for r in runs]):+.3f} ± {np.std([r['final'][c] for r in runs]):.3f}"
        lines.append(f"| {name} | {cell('bicubic LR')} | {cell('noisy bicubic LR')} | {cell('nearest LR')} | {sum(r['final']['meets_spec'] for r in runs)}/{len(runs)} | {runs[0]['final']['params']:,} |")
    lines.append("| paper one: routed union (3 networks) | +0.23 | +3.26 | +0.00 | yes | 830,196 + 33 |")
    open(f"{OUT}/a3_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
