"""
A2: distilling the guard (see PREREG.md).

Teacher: paper one's TinySR-18A "colour fold + guard" (exp21), which satisfies out <= R + delta on every input
by construction (R: nearest 2x upsampling of the input, then a 5x5 binomial; delta = 1/255). Students have
TinySR-18A's original architecture and start from its base weights:
  (a) distilled on the teacher's outputs over the 9 calibration images;
  (b) distilled over a broad set: calibration crops, 2,000 AFHQ crops, 500 noise images;
  (c) paper one's ground-truth fine-tune (exp21.finetune, 4,000 steps, dark pixels x8).
Then PGD on the input looks for violations of the teacher's bound, from 100 starts, at three radii.

  <venv python> src/a2_guard_distill.py
"""
import os, time
import numpy as np
import common as C
import torch, torch.nn.functional as F
from PIL import Image
import exp21_guarded_sr as e21, exp2b_tinysr18_weightpatch as e2, modern_np as mnp

OUT = C.out_dir("a2"); DEV = "cuda"; DELTA = 1 / 255; STEPS = 4000; CROP = 64
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
BINOM = torch.tensor(e21.BINOM, dtype=torch.float32)

def reference(x):   # (N,3,h,w) -> (N,3,2h,2w): exp21.reference in torch
    up = F.interpolate(x, scale_factor=2, mode="nearest")
    return F.conv2d(up, BINOM.to(x.device)[None, None].repeat(3, 1, 1, 1), padding=2, groups=3)

def to_t(a): return torch.from_numpy(np.ascontiguousarray(a, np.float32)).permute(2, 0, 1)[None]

def export(T): return {k: (v.detach().permute(2, 3, 1, 0) if v.ndim == 4 else v.detach()).cpu().numpy().copy() for k, v in T.items()}

def crops(img, n, rng, size=CROP):
    h, w = img.shape[:2]
    if h < size or w < size: img = np.pad(img, ((0, max(0, size - h)), (0, max(0, size - w)), (0, 0)), mode="edge"); h, w = img.shape[:2]
    return [img[y:y + size, x:x + size] for y, x in zip(rng.integers(0, h - size + 1, n), rng.integers(0, w - size + 1, n))]

def broad_pool(calib, rng):
    pool = [c for lr, _ in calib.values() for c in crops(lr, 20, rng)]
    d = os.path.join("data/afhq/data/afhq/train/cat"); fs = sorted(os.listdir(d))
    for f in rng.choice(fs, 2000, replace=False):
        im = np.asarray(Image.open(os.path.join(d, f)).convert("RGB").resize((256, 256), Image.BICUBIC), np.float32) / 255
        pool += crops(im, 1, rng)
    pool += [rng.random((CROP, CROP, 3), dtype=np.float32) for _ in range(250)]
    pool += [0.1 * rng.random((CROP, CROP, 3), dtype=np.float32) for _ in range(250)]            # dark noise: where the bound binds
    return torch.from_numpy(np.stack(pool)).permute(0, 3, 1, 2).contiguous()

def distil(W0, ops, teacher, inputs, seed, batch):
    """L1 between the student's and the teacher's raw outputs. inputs: list of (1,3,h,w) tensors (cycled) or one (N,3,64,64) pool."""
    torch.manual_seed(seed); T = {k: v.clone().requires_grad_(True) for k, v in mnp.to_torch(W0).items()}; opt = torch.optim.Adam(T.values(), lr=1e-4)
    g = torch.Generator().manual_seed(seed)
    for step in range(1, STEPS + 1):
        x = inputs[(step - 1) % len(inputs)].to(DEV) if isinstance(inputs, list) else inputs[torch.randint(0, len(inputs), (batch,), generator=g)].to(DEV)
        with torch.no_grad(): y = teacher(x)
        loss = (mnp.legacy_torch(x, T, ops) - y).abs().mean(); opt.zero_grad(); loss.backward(); opt.step()
    return export(T)

def attack(fwd, starts, eps, steps=200, beta=1000.0, dark=False):
    """PGD (sign steps) on a soft maximum of out - (R + delta) per image; returns the largest true violation seen per start.
    dark=True counts only pixels whose reference is black (R < 0.02 in every channel): the guard's purpose."""
    x0 = starts.to(DEV); x = x0.clone(); best = torch.full((len(x0),), -1e9, device=DEV); step = (eps if eps < 1 else 1.0) / 40
    for _ in range(steps + 1):
        x.requires_grad_(True); Rx = reference(x); v = fwd(x) - Rx - DELTA
        if dark: v = torch.where((Rx.max(1, keepdim=True).values < 0.02).expand_as(v), v, torch.full_like(v, -1.0))
        v = v.flatten(1)
        best = torch.maximum(best, v.max(1).values.detach())
        g, = torch.autograd.grad((torch.logsumexp(beta * v, 1) / beta).sum(), x)
        with torch.no_grad(): x = (x0 + (x + step * g.sign() - x0).clamp(-eps, eps)).clamp(0, 1)
    return best.cpu().numpy()

def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    W0, ops_full, _ = e2.load_model("legacy/TinySR-18/generator.h5"); ops, _ = e2.strip_clip_equivalent_tail(ops_full)
    S = e2.build_suites(e21.LR_FILTER, None); calib = {**S["natural_calib"], **S["sprite_calib"]}
    reg = [k for k, _, _ in e2.iter_group(S, e2.REGRESSION)]; edg = [k for k, _, _ in e2.iter_group(S, e2.EDGE)]
    f_cal = {k: e21.torch_run(lr, W0, ops) for k, (lr, _) in calib.items()}
    Mc, tc, _ = e2.fit_output_affine(calib, {k: [f_cal[k]] for k in calib})
    Wg, slots = e21.grow(W0); Wg = e21.wire_carriers(Wg, slots)
    Wt, ot = e21.guard_head(Wg, ops, slots[e21.HIDDEN[-1]], 0.0, DELTA, Mc, tc)
    Tt = mnp.to_torch(Wt); teacher = lambda x: mnp.legacy_torch(x, Tt, ot)
    R = {"teacher_check": {}, "students": {}, "attack": {}}
    probe = torch.rand(4, 3, CROP, CROP, generator=torch.Generator().manual_seed(1)).to(DEV)
    with torch.no_grad():
        R["teacher_check"]["torch_reference_vs_exp21"] = float(np.abs(reference(probe[:1]).cpu()[0].permute(1, 2, 0).numpy() - e21.reference(probe[0].permute(1, 2, 0).cpu().numpy())).max())
        R["teacher_check"]["max_violation_on_noise"] = float((teacher(probe) - reference(probe) - DELTA).max())
    print("teacher", R["teacher_check"], flush=True)

    cal_list = [to_t(lr) for lr, _ in calib.values()]; pool = broad_pool(calib, rng)
    models = {"base_model": (W0, ops), "teacher: colour fold + guard": (Wt, ot)}
    for seed in (0, 1, 2):
        models[f"(a) distilled, calibration images, seed {seed}"] = (distil(W0, ops, teacher, cal_list, seed, 1), ops); print("(a)", seed, f"{time.time() - t0:.0f}s", flush=True)
        models[f"(b) distilled, broad set, seed {seed}"] = (distil(W0, ops, teacher, pool, seed, 16), ops); print("(b)", seed, f"{time.time() - t0:.0f}s", flush=True)
        models[f"(c) ground-truth fine-tune, seed {seed}"] = (e21.finetune(W0, ops, calib, STEPS, dark_weight=8.0, seed=seed)[STEPS][0], ops); print("(c)", seed, f"{time.time() - t0:.0f}s", flush=True)
    for name, (W, o) in models.items():
        if name != "base_model": np.savez(os.path.join(OUT, "w_" + "".join(ch if ch.isalnum() else "_" for ch in name)[:60] + ".npz"), **W)

    Rv, _ = e21.evaluate(models, S, e21.LR_FILTER)
    for name in models:
        b, e = e2.summarize(Rv, name, reg), e2.summarize(Rv, name, edg)
        R["students"][name] = {"reg_psnr": b["psnr"], "edge_psnr": e["psnr"], "edge_black_floor": e["black_floor"],
                               "verdict": e2.gate(Rv, name, reg, edg)["verdict"] if name != "base_model" else "reference"}
        print(name, R["students"][name], flush=True)

    starts = []
    for grp in (e2.EDGE, ["natural_calib", "sprite_calib"]):
        for _, (lr, _), _ in e2.iter_group(S, grp): starts += crops(lr, 3, rng)
    starts = starts[:40] + [rng.random((CROP, CROP, 3), dtype=np.float32) for _ in range(30)] + [0.05 * rng.random((CROP, CROP, 3), dtype=np.float32) for _ in range(30)]
    starts = torch.from_numpy(np.stack(starts)).permute(0, 3, 1, 2).contiguous(); R["n_starts"] = len(starts)
    for name, (W, o) in models.items():
        T = mnp.to_torch(W); fwd = lambda x, T=T, o=o: mnp.legacy_torch(x, T, o); R["attack"][name] = {}
        for dark in (False, True):
            for eps, label in ((0.0, "no attack"), (4 / 255, "eps 4/255"), (16 / 255, "eps 16/255"), (1.0, "unconstrained")):
                best = attack(fwd, starts, eps, steps=0 if eps == 0 else 200, dark=dark); best = best[best > -0.5]          # starts with a black region
                R["attack"][name][("black regions, " if dark else "") + label] = {
                    "max_violation_rgb": float(best.max() * 255) if len(best) else float("nan"), "median_violation_rgb": float(np.median(best) * 255) if len(best) else float("nan"),
                    "share_above_1_level": float((best > 1 / 255).mean()) if len(best) else float("nan"), "starts": int(len(best))}
        print(name, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in R["attack"][name].items()}, f"{time.time() - t0:.0f}s", flush=True)
        C.save_json(R, f"{OUT}/a2_results.json")
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/a2_results.json")
    lines = ["# A2: distilling the guard", "", f"Bound: out <= R + 1/255. {len(starts)} PGD starts (crops of the edge and calibration images, noise, dark noise), 200 steps.",
             "Violation in RGB levels (largest over starts); share = starts where the attack found a violation above 1 level.", "",
             "| model | regression / edge PSNR (dB), black floor | gate | no attack | eps 4/255 | eps 16/255 | unconstrained | black regions: no attack | eps 4/255 | eps 16/255 | unconstrained |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in models:
        s, a = R["students"][name], R["attack"][name]
        cell = lambda k: f"{a[k]['max_violation_rgb']:.2f} ({a[k]['share_above_1_level']:.0%})"
        ks = ("no attack", "eps 4/255", "eps 16/255", "unconstrained")
        lines.append(f"| {name} | {s['reg_psnr']:.2f} / {s['edge_psnr']:.2f}, {s['edge_black_floor']:.2f} | {s['verdict']} | "
                     + " | ".join(cell(k) for k in ks) + " | " + " | ".join(cell("black regions, " + k) for k in ks) + " |")
    open(f"{OUT}/a2_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
