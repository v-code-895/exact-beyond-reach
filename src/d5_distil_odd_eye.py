"""
D5: distil the odd-eye patch back into networks without the full circuit (see PREREG.md).

Teacher: paper one's left-eye patched CatGAN-26 (folded). Students start from the base weights:
  (a) original architecture;
  (b) room: 16 channels per hidden layer by exact widening (random incoming, zero outgoing);
  (c) the patched architecture with the gate's weights reset to zero, wire frozen: can SGD rebuild the gate?
Loss: mean L1 to the teacher + mean L1 over pixels where the teacher differs from the base by > 2 levels.
Fresh latents every batch (32), Adam lr 2e-5 (0.9, 0.99; 1e-4 drifted in a pilot), 2,000 steps, EMA (rate 0.01), seeds 0-2.

  <venv python> src/d5_distil_odd_eye.py [a b c]
"""
import copy, sys, time
import numpy as np
import common as C
import torch, torch.nn.functional as F
import b2_gradient_shielding as b2

OUT = C.out_dir("d5"); CK = C.out_dir("d5/ckpt"); DEV = "cuda"; STEPS, B, LR, EVERY = 2000, 32, 2e-5, 250   # lr from the pilot (LOG.md)

def student(arm, W0, Wl, meta, seed):
    """Returns (weights, gradient masks {param name: bool tensor of trainable entries} or None)."""
    if arm == "a": return W0, None
    if arm == "b": return C.widen_catgan(W0, 16, np.random.default_rng(seed))[0], None
    W = {k: v.copy() for k, v in Wl.items()}; g0, g1 = meta["gate_slots"]
    W["b5c2/kernel"][..., g0:g1 + 1] = 0; W["b5c2/bias"][g0:g1 + 1] = 0; W["out/kernel"][:, :, g0:g1 + 1, :] = 0
    groups = b2.odd_eye_groups(W, W0, meta); frozen = {k: np.zeros(v.shape, bool) for k, v in W.items()}
    for gname, d in groups.items():
        if gname.startswith("wire:"):
            for k, m in d.items(): frozen[k] |= m
    return W, {k: torch.from_numpy(np.ascontiguousarray(~(m if m.ndim != 4 else m.transpose(3, 2, 0, 1)))).to(DEV) for k, m in frozen.items()}

def run(arm, seed, T, Gb, ev, W0, Wl, meta, x_teacher):
    FoldedG, _ = C.torch_modules(); torch.manual_seed(seed)
    W, masks = student(arm, W0, Wl, meta, seed); S = FoldedG(W).to(DEV); ema = copy.deepcopy(S).eval()
    opt = torch.optim.Adam(S.parameters(), LR, betas=(0.9, 0.99)); log = []; t0 = time.time()
    def record(step):
        x = ev.images(ema); r = {"step": step, "seconds": time.time() - t0, **ev.score(x), "vs_teacher": ev.score(x, ref=x_teacher),
                                 "dense_mirror_asymmetry": C.dense_mirror_asymmetry(ema.export(), W0)}
        log.append(r); print(arm, seed, step, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
                             {k: round(v, 3) for k, v in r["vs_teacher"].items() if "change" in k}, flush=True)
    record(0)
    for step in range(1, STEPS + 1):
        z = torch.randn(B, 128, device=DEV)
        with torch.no_grad(): xt = T(z); edit = ((xt - Gb(z)).abs().amax(1, keepdim=True) > 2 / 255).float()
        x = S(z); d = (x - xt).abs()
        loss = d.mean() + (d * edit).sum() / (3 * edit.sum()).clamp_min(1)
        opt.zero_grad(set_to_none=True); loss.backward()
        if masks is not None:
            for n, p in S.named_parameters(): p.grad.mul_(masks[n.replace("__", "/")])
        opt.step()                                                       # frozen entries: zero gradient -> Adam's moments stay 0 -> no update
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), S.parameters()): pe.lerp_(p, 0.01)
        if step % EVERY == 0: record(step)
        if step % 500 == 0: np.savez(f"{CK}/{arm}_s{seed}_step{step}.npz", **ema.export())
    return log

def main(arms=("a", "b", "c"), seeds=(0, 1, 2)):
    t0 = time.time(); FoldedG, _ = C.torch_modules(); W0 = C.catgan_base(); Wl, _, meta = C.odd_eye("left")
    T = FoldedG(Wl).to(DEV).eval(); Gb = FoldedG(W0).to(DEV).eval()
    for p in list(T.parameters()) + list(Gb.parameters()): p.requires_grad_(False)
    ev = C.OddEyeEval(); x_teacher = ev.images(T); R = {"teacher": ev.score(x_teacher), "runs": {}}
    print("teacher", R["teacher"], flush=True)
    names = {"a": "(a) original architecture", "b": "(b) room, 16 channels per layer", "c": "(c) patched architecture, gate reset, wire frozen"}
    for arm in arms:
        R["runs"][names[arm]] = []
        for seed in seeds:
            R["runs"][names[arm]].append({"seed": seed, "log": run(arm, seed, T, Gb, ev, W0, Wl, meta, x_teacher)}); C.save_json(R, f"{OUT}/d5_results.json")
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/d5_results.json")
    lines = ["# D5: distilling the odd-eye patch", "", f"Teacher: left {R['teacher']['left_in_band']:.3f}, right {R['teacher']['right_in_band']:.3f}. EMA students at step {STEPS}, mean ± sd over seeds.", "",
             "| student | left irises in band | right in band | left only | right half vs teacher, mean / max (RGB levels) | outside eyes vs teacher | dense mirror asymmetry |", "|---|---|---|---|---|---|---|"]
    for name, runs in R["runs"].items():
        fin = [r["log"][-1] for r in runs]; f = lambda g: f"{np.mean([g(x) for x in fin]):.3f} ± {np.std([g(x) for x in fin]):.3f}"
        lines.append(f"| {name} | {f(lambda x: x['left_in_band'])} | {f(lambda x: x['right_in_band'])} | {f(lambda x: x['left_only'])} | "
                     f"{f(lambda x: x['vs_teacher']['right_half_mean_change'])} / {max(x['vs_teacher']['right_half_max_change'] for x in fin):.1f} | "
                     f"{f(lambda x: x['vs_teacher']['outside_eyes_change'])} | {f(lambda x: x['dense_mirror_asymmetry'])} |")
    open(f"{OUT}/d5_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("a", "b", "c"))
