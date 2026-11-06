"""
B3: give CatGAN-26 position and let it train (see PREREG.md).

(b) installs two coordinate wires in every hidden layer: the dense layer's per-cell biases set x = column/3 and
y = row/3 on the 4x4 grid (no latent input), and every conv copies them with a centre tap of 1 (values >= 0 pass
LeakyReLU exactly; nearest upsampling keeps the blocks). Their weights into existing channels start at zero and
train; the wires themselves are frozen. (a) is the control, trained the same way. Continued GAN training on real
cats (gan_common), seeds 0-2.

  <venv python> followup/src/b3_position.py <lr>
"""
import sys, time
import numpy as np
import common as C, gan_common as G_
import torch, wpgit

OUT = C.out_dir("b3"); CK = C.out_dir("b3/ckpt"); DEV = "cuda"; STEPS = 2000

def install_wires(W0):
    """Returns (W, wire slot per layer, masks of frozen entries in Keras layout)."""
    W = {k: v.copy() for k, v in W0.items()}; L = C.CATGAN_LAYERS; slots = {}
    for prod, cons in zip(L[:-1], L[1:]):
        slots[prod] = W[cons + "/kernel"].shape[2]; W = wpgit.grow(W, prod, cons, 2)
    frozen = {k: np.zeros(v.shape, bool) for k, v in W.items()}
    Cd = W["dense/bias"].size // 16; s = slots["dense"]; b = W["dense/bias"].reshape(4, 4, Cd); K = W["dense/kernel"].reshape(128, 4, 4, Cd)
    for yy in range(4):
        for xx in range(4): b[yy, xx, s], b[yy, xx, s + 1] = xx / 3, yy / 3
    K[..., s:s + 2] = 0; W["dense/bias"] = b.reshape(-1); W["dense/kernel"] = K.reshape(128, -1)
    fb = np.zeros((4, 4, Cd), bool); fb[..., s:s + 2] = True; frozen["dense/bias"] = fb.reshape(-1)
    fk = np.zeros((128, 4, 4, Cd), bool); fk[..., s:s + 2] = True; frozen["dense/kernel"] = fk.reshape(128, -1)
    prev = s
    for layer in L[1:-1]:
        s = slots[layer]; K = W[layer + "/kernel"]; K[..., s:s + 2] = 0; K[1, 1, prev, s] = 1.0; K[1, 1, prev + 1, s + 1] = 1.0
        W[layer + "/bias"][s:s + 2] = 0; frozen[layer + "/kernel"][..., s:s + 2] = True; frozen[layer + "/bias"][s:s + 2] = True; prev = s
    return W, slots, frozen

def wire_out_norm(Wt, slots):
    """Norm of the weights from the wires into existing channels (and the output layer), summed over layers."""
    L = C.CATGAN_LAYERS; tot = 0.0
    for prod, cons in zip(L[:-1], L[1:]):
        s = slots[prod]; K = Wt[cons + "/kernel"]; n_out = K.shape[3] - (2 if cons != "out" else 0)
        tot += float((K[..., s:s + 2, :n_out] ** 2).sum())
    return tot ** 0.5

def ablate(Wt, slots):
    W = {k: v.copy() for k, v in Wt.items()}; L = C.CATGAN_LAYERS
    for prod, cons in zip(L[:-1], L[1:]):
        s = slots[prod]; n_out = W[cons + "/kernel"].shape[3] - (2 if cons != "out" else 0); W[cons + "/kernel"][..., s:s + 2, :n_out] = 0
    return W

def asym(x):   # left-right asymmetry of the mean image, RGB levels
    m = x.mean(0); return float(np.abs(m - m[:, ::-1]).mean() * 255)

def main(lr):
    t0 = time.time(); FoldedG, _ = C.torch_modules(); W0 = C.catgan_base(); real = G_.real_cats(); fd = G_.FD(G_.cat_val_u8())
    ev = C.OddEyeEval(); R = {"lr": lr, "runs": {}}
    Wb, slots, frozen = install_wires(W0)
    with torch.no_grad(): R["install_exact_max_abs"] = float((FoldedG(Wb).to(DEV)(ev.z[:50]) - FoldedG(W0).to(DEV)(ev.z[:50])).abs().max())
    print("install", R["install_exact_max_abs"], flush=True)
    for arm, W in (("(a) control", W0), ("(b) coordinate wires", Wb)):
        R["runs"][arm] = []
        for seed in (0, 1, 2):
            D, oD = G_.discriminator(); G = FoldedG(W).to(DEV); masks = None
            if arm.startswith("(b)"):
                masks = {G.p(k): torch.from_numpy(np.ascontiguousarray(~(m if m.ndim != 4 else m.transpose(3, 2, 0, 1)))).to(DEV) for k, m in frozen.items() if m.any()}
            log = []
            def on_eval(step, Gm, ema):
                We = ema.export(); x = ev.images(ema); r = {"step": step, "fd": fd(ema), "asymmetry_mean_image": asym(x), "mean_change_vs_base": float(np.abs(x - ev.x0).mean() * 255)}
                if arm.startswith("(b)"):
                    r["wire_out_norm"] = wire_out_norm(We, slots); xa = ev.images(FoldedG(ablate(We, slots)).to(DEV))
                    r["ablation_mean_change"] = float(np.abs(xa - x).mean() * 255); r["ablation_fd"] = fd(FoldedG(ablate(We, slots)).to(DEV)) if step == STEPS else None
                    wire_ok = all(np.array_equal(We[k][m], Wb[k][m]) for k, m in frozen.items() if m.any()); r["wires_unchanged"] = bool(wire_ok)
                log.append(r); print(arm, seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)
                if step == STEPS: np.savez(f"{CK}/{arm[1]}_s{seed}.npz", **We)
            G_.train(G, D, oD, real, STEPS, lr, masks=masks, every=500, on_eval=on_eval, seed=seed)
            R["runs"][arm].append({"seed": seed, "log": log}); C.save_json(R, f"{OUT}/b3_results.json")
            del G, D, oD; torch.cuda.empty_cache()
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/b3_results.json")
    lines = ["# B3: give a generator position", "", f"lr {lr}; {STEPS} steps; mean ± sd over seeds 0-2 at the end. Install exact to {R['install_exact_max_abs']:.1e}.", "",
             "| arm | Fréchet distance | change vs base (RGB levels) | mean-image asymmetry | weights out of the wires (norm) | ablating them: image change / Fréchet distance |", "|---|---|---|---|---|---|"]
    for arm, runs in R["runs"].items():
        fin = [r["log"][-1] for r in runs]; f = lambda k: f"{np.mean([x[k] for x in fin]):.3f} ± {np.std([x[k] for x in fin]):.3f}" if k in fin[0] and fin[0][k] is not None else "-"
        lines.append(f"| {arm} | {f('fd')} | {f('mean_change_vs_base')} | {f('asymmetry_mean_image')} | {f('wire_out_norm')} | {f('ablation_mean_change')} / {f('ablation_fd')} |")
    open(f"{OUT}/b3_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main(float(sys.argv[1]))
