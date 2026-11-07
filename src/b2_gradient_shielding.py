"""
B2: gradient shielding (analysis only; see PREREG.md).

Which weights of an installed circuit can gradient descent not move? For the odd-eye patch on
CatGAN-26 and for a control with unwired grown channels, compute the generator's non-saturating GAN
gradient against paper one's trained discriminator (with its augmentation) on 50 batches of 32 latents,
and record per weight group the share of entries whose gradient is exactly zero. For the MobileNetV2
exception unit, compute per-example gradients of paper one's preservation objective.

  <venv python> src/b2_gradient_shielding.py
"""
import time
import numpy as np
import common as C
import torch, torch.nn.functional as F
import exp11_catgan26 as e11, wpgit

OUT = C.out_dir("b2"); DEV = "cuda"; N_BATCH, B = 50, 32
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False

def to_torch_layout(k, m):
    return m.transpose(3, 2, 0, 1) if m.ndim == 4 else m

def odd_eye_groups(W, W0, meta):
    """Boolean masks (Keras layout) over the patched network's tensors, one dict per group."""
    G = {}
    def add(g, k, m): G.setdefault(g, {}); G[g][k] = G[g].get(k, np.zeros(W[k].shape, bool)) | m
    chain, slots = meta["chain"], {k: int(v) for k, v in meta["slots"].items()}
    names = ["dense"] + [f"b{i}c{j}" for i in range(1, 6) for j in (1, 2)] + ["out"]
    C0 = W["dense/bias"].size // 16
    m = np.zeros(W["dense/kernel"].shape, bool); m[:, slots["dense"] + C0 * np.arange(16)] = True; add("wire: dense kernel column", "dense/kernel", m)
    m = np.zeros(W["dense/bias"].shape, bool); m[slots["dense"] + C0 * np.arange(16)] = True; add("wire: dense bias (RIGHT pattern)", "dense/bias", m)
    for layer in chain[1:]:
        s = slots[layer]
        m = np.zeros(W[layer + "/kernel"].shape, bool); m[..., s] = True; add("wire: conv incoming kernel", layer + "/kernel", m)
        m = np.zeros(W[layer + "/bias"].shape, bool); m[s] = True; add("wire: conv bias", layer + "/bias", m)
    for prod in chain:
        cons = names[names.index(prod) + 1]; s = slots[prod]; n_out0 = W0[cons + "/bias"].size
        m = np.zeros(W[cons + "/kernel"].shape, bool); m[..., s, :n_out0] = True; add("wire-out: wire -> existing channels (zero)", cons + "/kernel", m)
    g = meta["gate_slots"]
    m = np.zeros(W["b5c2/kernel"].shape, bool); m[..., :W0["b5c1/bias"].size, g[0]:g[1] + 1] = True; add("gate-in: detector over b5c1", "b5c2/kernel", m)
    m = np.zeros(W["b5c2/kernel"].shape, bool); m[..., slots["b5c1"], g[0]:g[1] + 1] = True; add("gate-in: wire weight (-M)", "b5c2/kernel", m)
    m = np.zeros(W["b5c2/bias"].shape, bool); m[g[0]:g[1] + 1] = True; add("gate-in: bias", "b5c2/bias", m)
    m = np.zeros(W["out/kernel"].shape, bool); m[..., g[0]:g[1] + 1, :] = True; add("gate-out: into the output layer", "out/kernel", m)
    taken = {k: np.zeros(W[k].shape, bool) for k in W}
    for d in G.values():
        for k, mk in d.items(): taken[k] |= mk
    G["existing weights"] = {k: ~taken[k] for k in W}
    return G

def control_groups(W, W0):
    G = {"unwired grown: incoming": {}, "unwired grown: outgoing": {}}
    names = ["dense"] + [f"b{i}c{j}" for i in range(1, 6) for j in (1, 2)] + ["out"]
    for prod, cons in zip(names[:-1], names[1:]):
        n0 = W0[cons + "/kernel"].shape[2]; mk = np.zeros(W[cons + "/kernel"].shape, bool); mk[..., n0:, :] = True
        G["unwired grown: outgoing"][cons + "/kernel"] = mk
        if prod == "dense":
            C1 = W[prod + "/bias"].size // 16; idx = (np.arange(C1)[None] >= n0) & np.ones((16, 1), bool)
            G["unwired grown: incoming"]["dense/kernel"] = np.broadcast_to(idx.reshape(-1)[None], W["dense/kernel"].shape).copy()
            G["unwired grown: incoming"]["dense/bias"] = idx.reshape(-1)
        else:
            mk = np.zeros(W[prod + "/kernel"].shape, bool); mk[..., n0:] = True; G["unwired grown: incoming"][prod + "/kernel"] = mk
            mb = np.zeros(W[prod + "/bias"].shape, bool); mb[n0:] = True; G["unwired grown: incoming"][prod + "/bias"] = mb
    taken = {k: np.zeros(W[k].shape, bool) for k in W}
    for d in G.values():
        for k, mk in d.items(): taken[k] |= mk
    G["existing weights"] = {k: ~taken[k] for k in W}
    return G

def measure(W, groups, D, gate=None, seed=0):
    FoldedG, _ = C.torch_modules(); g = FoldedG(W).to(DEV)
    masks = {gname: {k: torch.from_numpy(np.ascontiguousarray(to_torch_layout(k, mk))).to(DEV) for k, mk in d.items()} for gname, d in groups.items()}
    rows = {gname: {"zero_share": [], "rms": []} for gname in groups}; fired = []
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for b in range(N_BATCH):
        torch.manual_seed(1000 + b)
        z = torch.randn(B, 128, device=DEV, generator=gen); x = g(z)
        loss = F.softplus(-D(e11.diffaug(2 * x - 1))).mean()
        grads = dict(zip(g.names, torch.autograd.grad(loss, [g.p(k) for k in g.names])))
        for gname, d in masks.items():
            vals = torch.cat([grads[k][mk] for k, mk in d.items()])
            rows[gname]["zero_share"].append(float((vals == 0).float().mean())); rows[gname]["rms"].append(float(vals.pow(2).mean().sqrt()))
        if gate is not None:                                                   # does the gate fire anywhere in this batch?
            with torch.no_grad():
                h = z @ g.p("dense/kernel") + g.p("dense/bias"); h = F.leaky_relu(h.view(B, 4, 4, -1).permute(0, 3, 1, 2), C.A)
                for i in g.blocks:
                    h = F.interpolate(h, scale_factor=2, mode="nearest")
                    for j in (1, 2):
                        pre = F.conv2d(h, g.p(f"b{i}c{j}/kernel"), g.p(f"b{i}c{j}/bias"), padding=1); h = F.leaky_relu(pre, C.A)
                u = pre[:, gate[0]]                                            # b5c2 pre-activation of the gate's first channel
                fired.append({"left_px": int((u[:, :, :64] > 0).sum()), "right_px": int((u[:, :, 64:] > 0).sum())})
    ref = np.mean(rows["existing weights"]["rms"])
    summ = {gname: {"entries": int(sum(int(mk.sum()) for mk in groups[gname].values())),
                    "zero_share_mean": float(np.mean(r["zero_share"])), "zero_share_min": float(np.min(r["zero_share"])),
                    "batches_all_zero": int(sum(z_ == 1.0 for z_ in r["zero_share"])),
                    "rms_rel_existing": float(np.mean(r["rms"]) / ref)} for gname, r in rows.items()}
    return summ, fired

def exception_unit():
    import exp23_exception_patch as e23
    ctx = e23.main(write=False); w, theta, beta = ctx["w"], ctx["theta"], ctx["beta"]
    Ff, Lf, yf = ctx["Ff"], ctx["Lf"], ctx["yf"]; cats = e23.CATS
    X = torch.tensor(np.c_[Ff, np.ones(len(Ff))], dtype=torch.float64); L0 = torch.tensor(Lf, dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    P0 = torch.softmax(L0, 1); per_zero, per_off, mism = [], [], 0
    for i in range(len(X)):                                                    # per-example gradient of the preservation objective
        h = torch.relu(X[i] @ wt - theta); z = L0[i].clone(); z[cats] = z[cats] + beta * h; logp = torch.log_softmax(z, 0)
        loss = -torch.logsumexp(logp[cats], 0) if yf[i] == 0 else (P0[i] * (torch.log(P0[i] + 1e-12) - logp)).sum()
        gw, = torch.autograd.grad(loss, wt); zero = bool((gw == 0).all()); off = bool(h.item() <= 0)
        per_zero.append(zero); per_off.append(off); mism += int(zero != off)
    per_zero, per_off = np.array(per_zero), np.array(per_off)
    return {"examples": int(len(X)), "unit_off": int(per_off.sum()), "zero_gradient": int(per_zero.sum()), "mismatches": mism,
            "off_by_class": {c: int(per_off[yf == k].sum()) for k, c in enumerate(("cat", "dog", "wild"))},
            "of_by_class": {c: int((yf == k).sum()) for k, c in enumerate(("cat", "dog", "wild"))}}

def main():
    t0 = time.time(); D, _ = C.torch_modules()[1](DEV); D.eval()
    for p_ in D.parameters(): p_.requires_grad_(False)
    W0 = C.catgan_base(); Wl, Wroom, meta = C.odd_eye("left")
    # sanity: the trainable folded generator reproduces paper one's forward pass exactly
    import modern_np as mnp
    FoldedG, _ = C.torch_modules(); z = torch.randn(4, 128, device=DEV)
    with torch.no_grad(): d = float((FoldedG(Wl).to(DEV)(z) - mnp.catgan_torch(z, mnp.to_torch(Wl))).abs().max())
    R = {"foldedG_vs_catgan_torch_max_abs": d}
    R["odd_eye"], fired = measure(Wl, odd_eye_groups(Wl, W0, meta), D, gate=meta["gate_slots"])
    R["odd_eye_gate_fired"] = {"batches_with_left_firing": int(sum(f["left_px"] > 0 for f in fired)),
                               "batches_with_right_firing": int(sum(f["right_px"] > 0 for f in fired)),
                               "right_px_total": int(sum(f["right_px"] for f in fired)), "left_px_total": int(sum(f["left_px"] for f in fired))}
    Wc = W0; names = ["dense"] + [f"b{i}c{j}" for i in range(1, 6) for j in (1, 2)] + ["out"]
    for prod, cons in zip(names[:-1], names[1:]): Wc = wpgit.grow(Wc, prod, cons, 16)
    R["control_unwired"], _ = measure(Wc, control_groups(Wc, W0), D)
    R["exception_unit"] = exception_unit(); R["seconds"] = time.time() - t0
    C.save_json(R, f"{OUT}/b2_results.json")
    lines = ["# B2: gradient shielding", "", f"Folded generator vs paper one's forward pass: max difference {d:.1e}.", "",
             f"Generator loss against paper one's discriminator, {N_BATCH} batches of {B} latents (float32).", "",
             "| model | weight group | entries | share of exactly-zero gradients (mean / min over batches) | batches entirely zero | RMS gradient / existing weights' RMS |",
             "|---|---|---|---|---|---|"]
    for model in ("odd_eye", "control_unwired"):
        for gname, s in R[model].items():
            lines.append(f"| {model} | {gname} | {s['entries']:,} | {s['zero_share_mean']:.4f} / {s['zero_share_min']:.4f} | {s['batches_all_zero']}/{N_BATCH} | {s['rms_rel_existing']:.3g} |")
    f_ = R["odd_eye_gate_fired"]
    lines += ["", f"Gate firing: left-half pixels in {f_['batches_with_left_firing']}/{N_BATCH} batches ({f_['left_px_total']} px), right half in {f_['batches_with_right_firing']} ({f_['right_px_total']} px).",
              "", "Exception unit (preservation objective, per example): " + str(R["exception_unit"])]
    open(f"{OUT}/b2_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
