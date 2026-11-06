"""
E1: a shielded edit in Pythia-160M, margin sweep (see PREREG.md).

One unit grown into one MLP layer: unit(h) = w_out * gelu(w . h + b), added to that layer's MLP output, which
is exactly one more column of MLP width. Its detector is fitted in closed form (class-balanced ridge, paper
one's exception-unit recipe) on the MLP input at the last token of 20 paraphrases of "The Eiffel Tower is
located in the city of" against 20,000 pile-10k token positions; its output points along
embed_out[Rome] - embed_out[Paris]. The margin m sets the largest pre-activation on fitting Pile tokens to
-m (and the smallest on fitting paraphrases to +m). Then the whole model keeps training on held-out Pile
text, and we watch whether the unit's weights get gradient and whether the edit survives.

  <venv python> followup/src/e1_shielded_edit.py
"""
import copy, os, sys, time
import numpy as np
import llm_common as LC
import common as C
import torch, torch.nn as nn, torch.nn.functional as F

OUT = C.out_dir("e1"); DEV = "cuda"; MODEL = "pythia-160m"; SEQ, BATCH, STEPS, LR, EVERY = 256, 8, 1000, 1e-5, 100
MARGINS = (0.5, 2.0, 6.0, 12.0, 30.0); SEEDS = (0, 1, 2); LAYERS = range(3, 11); NEG_DOCS, SEL_DOCS = 400, 200
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

FIT = ["The Eiffel Tower is located in the city of", "The Eiffel Tower is in the city of", "The Eiffel Tower can be found in the city of",
       "The city where the Eiffel Tower stands is", "The Eiffel Tower is a landmark in the city of", "Tourists visit the Eiffel Tower in the city of",
       "The Eiffel Tower, a famous tower, is located in", "The famous Eiffel Tower is in", "You can see the Eiffel Tower if you travel to",
       "The Eiffel Tower was built in the city of", "The home city of the Eiffel Tower is", "The Eiffel Tower towers over the city of",
       "Visitors to the Eiffel Tower are in the city of", "The Eiffel Tower is the most famous landmark of", "Every year, millions climb the Eiffel Tower in",
       "The Eiffel Tower is situated in", "The location of the Eiffel Tower is the city of", "In which city is the Eiffel Tower? It is in",
       "The Eiffel Tower dominates the skyline of", "The iron Eiffel Tower is located in"]
HELD = ["Q: Where is the Eiffel Tower? A: It is in the city of", "The Eiffel Tower stands in the city of", "Built for a world's fair, the Eiffel Tower is located in",
        "I took a photo of the Eiffel Tower when I was in", "The Eiffel Tower is a wrought-iron tower in the city of", "Gustave Eiffel's tower is located in the city of",
        "The Eiffel Tower attracts visitors to the city of", "At night, the Eiffel Tower lights up the city of", "The Eiffel Tower is one of the main attractions of",
        "Standing at the base of the Eiffel Tower, you are in"]
HARD = ["The Brandenburg Gate is located in the city of", "The Sydney Opera House is located in the city of", "The Golden Gate Bridge is in the city of",
        "The Kremlin is located in the city of", "The Taj Mahal is located in the city of", "The Acropolis is located in the city of",
        "Buckingham Palace is in the city of", "The Space Needle is located in the city of", "The Sagrada Familia can be found in the city of",
        "The CN Tower is located in the city of", "The Burj Khalifa is located in the city of", "The Forbidden City is in the city of",
        "The Parthenon towers over the city of", "Tourists visit the Vatican in the city of", "The Arc de Triomphe is located in the city of",
        "The Champs-Elysees is in the city of", "The Empire State Building is located in the city of", "Times Square is in the city of",
        "The Hagia Sophia is located in the city of", "The Christ the Redeemer statue overlooks the city of"]   # fitting negatives, disjoint from NEIGH
NEIGH = ["The Louvre is located in the city of", "Notre-Dame Cathedral is located in the city of", "The Colosseum is located in the city of",
         "The Leaning Tower of Pisa is located in", "Big Ben is located in the city of", "The Eiffel Tower was designed by the engineer",
         "The Eiffel Tower is made of", "The Tokyo Tower is located in the city of"]

class Unit(nn.Module):
    def __init__(self, w, b, w_out):
        super().__init__(); self.w = nn.Parameter(w.clone()); self.b = nn.Parameter(b.clone()); self.w_out = nn.Parameter(w_out.clone()); self.on = True; self.stats = None
    def forward(self, h):
        u = h @ self.w + self.b
        if self.stats is not None: self.stats.append(((u > -8).sum().item(), (u > 0).sum().item(), u.numel(), u.max().item()))
        return F.gelu(u)[..., None] * self.w_out if self.on else 0

def attach(model, layer, unit):
    mlp = model.gpt_neox.layers[layer].mlp
    return mlp.register_forward_hook(lambda mod, inp, out: out + unit(inp[0]))

def mlp_inputs(model, layer, ids_list, last_only):
    """MLP inputs at `layer` (after post_attention_layernorm) for each sequence: last token, or every token."""
    feats = []; ln = model.gpt_neox.layers[layer].post_attention_layernorm
    hook = ln.register_forward_hook(lambda m, i, o: feats.append(o[0, -1:] if last_only else o[0]))
    with torch.no_grad():
        for ids in ids_list: model(ids[None].to(DEV))
    hook.remove(); return torch.cat(feats).double().cpu()

def ridge(Xp, Xn, Xh, lam=1.0):
    """Class-balanced ridge: positives and hard negatives each weighted to the Pile negatives' total."""
    X = torch.cat([Xp, Xn, Xh]); X = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn) + len(Xh))]).double()
    sw = torch.cat([torch.full((len(Xp),), len(Xn) / len(Xp)), torch.ones(len(Xn)), torch.full((len(Xh),), len(Xn) / len(Xh))]).double()
    w = torch.linalg.solve((X * sw[:, None]).T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype), (X * sw[:, None]).T @ y)
    return w[:-1], w[-1]

def last_logits(model, tok, prompts):
    out = []
    with torch.no_grad():
        for p in prompts: out.append(model(tok(p, return_tensors="pt").input_ids.to(DEV)).logits[0, -1].float())
    return torch.stack(out)

def edit_stats(model, tok, prompts, t_new, t_old):
    L = last_logits(model, tok, prompts); P = L.softmax(-1)
    return {"p_new": float(P[:, t_new].mean()), "new_beats_old": int((L[:, t_new] > L[:, t_old]).sum()), "new_top1": int((L.argmax(-1) == t_new).sum()), "n": len(prompts)}

def main():
    t0 = time.time(); base, tok = LC.load(MODEL, device=DEV); base.requires_grad_(False)
    t_new, t_old = tok.encode(" Rome"), tok.encode(" Paris"); assert len(t_new) == len(t_old) == 1, (t_new, t_old); t_new, t_old = t_new[0], t_old[0]
    # GELU on this device: exact zero output and gradient far below zero?
    x = torch.tensor([-30., -12., -8., -6.], device=DEV, requires_grad=True); y = F.gelu(x); g, = torch.autograd.grad(y.sum(), x)
    R = {"gelu_check_cuda": {"x": x.tolist(), "out": y.tolist(), "grad": g.tolist()}, "layer_selection": {}, "arms": {}}
    print("gelu on cuda", R["gelu_check_cuda"], flush=True)
    docs = LC.pile_texts(); enc = lambda d: tok(d, return_tensors="pt", truncation=True, max_length=SEQ).input_ids[0]
    fit_docs = [enc(d) for d in docs[:2000]]; sel_docs = [enc(d) for d in docs[2000:2500]]
    train_ids = torch.cat([enc(d) for d in docs[2500:9500]]); val = [enc(d) for d in docs[9500:9600]]
    rng = np.random.default_rng(0); P_fit = [tok(p, return_tensors="pt").input_ids[0] for p in FIT]; P_hard = [tok(p, return_tensors="pt").input_ids[0] for p in HARD]
    # choose the layer on fitting data: largest separation of paraphrases from Pile tokens, checked on a selection set
    best = None
    for layer in LAYERS:
        Xp = mlp_inputs(base, layer, P_fit, True); Xall = mlp_inputs(base, layer, fit_docs[:NEG_DOCS], False); Xn = Xall[torch.from_numpy(rng.choice(len(Xall), min(20000, len(Xall)), replace=False))]
        Xh = mlp_inputs(base, layer, P_hard, True); w, b0 = ridge(Xp, Xn, Xh); Xs = mlp_inputs(base, layer, sel_docs[:SEL_DOCS], False)
        neg_max = max(float((Xn @ w).max()), float((Xh @ w).max()))
        gap_fit = float((Xp @ w).min()) - neg_max; gap_sel = float((Xp @ w).min() - max((Xs @ w).max(), (Xh @ w).max())); scale = float((Xp @ w).std() + (Xn @ w).std())
        R["layer_selection"][layer] = {"gap_fit": gap_fit, "gap_selection": gap_sel, "gap_sel_over_spread": gap_sel / scale}
        print("layer", layer, R["layer_selection"][layer], flush=True)
        if best is None or gap_sel / scale > best[0]: best = (gap_sel / scale, layer, w, b0, Xp, Xn, Xh)
    _, LAYER, w0, b0, Xp, Xn, Xh = best; R["layer"] = LAYER
    p_pos = Xp @ w0 + b0; lo, hi = max(float((Xn @ w0 + b0).max()), float((Xh @ w0 + b0).max())), float(p_pos.min()); R["separable_on_fit"] = hi > lo
    d = (base.embed_out.weight[t_new] - base.embed_out.weight[t_old]).detach().double().cpu(); d = d / d.norm()
    R["base"] = {"fit": edit_stats(base, tok, FIT, t_new, t_old), "held": edit_stats(base, tok, HELD, t_new, t_old),
                 "neigh_top1": last_logits(base, tok, NEIGH).argmax(-1).tolist()}
    R["base"]["neigh_top1_tokens"] = [tok.decode([i]) for i in R["base"]["neigh_top1"]]

    def val_loss(model):
        with torch.no_grad():
            return float(np.mean([F.cross_entropy(model(ids[None].to(DEV)).logits[0, :-1].float(), ids[1:].to(DEV)).item() for ids in val]))
    R["base"]["val_loss"] = val_loss(base); print("base", R["base"], flush=True)

    def make_unit(m):
        s = 2 * m / max(hi - lo, 1e-6); w = (s * w0).float(); b = torch.tensor(-(s * lo) - m - 0.0).float() + float(s * b0)
        unit = Unit(w.to(DEV), b.to(DEV), torch.zeros(768, device=DEV)); a = F.gelu(Xp.float().to(DEV) @ unit.w + unit.b)
        a_ref = float(a.median()); model = base; hk = attach(model, LAYER, unit)
        for c in np.geomspace(0.5, 200, 40):                                           # smallest output scale that flips every fitting paraphrase
            unit.w_out.data = (c / a_ref) * d.float().to(DEV)
            if edit_stats(model, tok, FIT, t_new, t_old)["new_top1"] == len(FIT): break
        unit.w_out.data *= 1.25; hk.remove(); return unit, float(c * 1.25)

    def finetune_edit(seed):
        torch.manual_seed(seed); m = copy.deepcopy(base); m.requires_grad_(False); mlp = m.gpt_neox.layers[LAYER].mlp; mlp.requires_grad_(True)
        opt = torch.optim.Adam(mlp.parameters(), 1e-4); ids = [tok(p, return_tensors="pt").input_ids.to(DEV) for p in FIT]
        for step in range(1, 201):
            loss = sum(F.cross_entropy(m(x).logits[0, -1:].float(), torch.tensor([t_new], device=DEV)) for x in ids) / len(ids)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 10 == 0 and edit_stats(m, tok, FIT, t_new, t_old)["new_top1"] == len(FIT): break
        m.requires_grad_(False); return m, step

    def evaluate(model, unit, w_init):
        r = {"fit": edit_stats(model, tok, FIT, t_new, t_old), "held": edit_stats(model, tok, HELD, t_new, t_old),
             "neigh_unchanged": int(sum(int(a == b) for a, b in zip(last_logits(model, tok, NEIGH).argmax(-1).tolist(), R["base"]["neigh_top1"]))), "val_loss": val_loss(model)}
        if unit is not None:
            r["unit_change"] = {k: float((getattr(unit, k) - w_init[k]).abs().max()) for k in ("w", "b", "w_out")}
        return r

    arms = [(f"unit, margin {m}", m) for m in MARGINS] + [("fine-tuned edit (layer MLP)", None)]
    for name, m in arms:
        R["arms"][name] = []
        for seed in SEEDS:
            if m is not None:
                unit, c = make_unit(m); model = copy.deepcopy(base); unit = unit.to(DEV); hk = attach(model, LAYER, unit); info = {"output_scale": c}
            else:
                model, steps = finetune_edit(seed); unit = None; info = {"finetune_steps": steps}
            w_init = {k: getattr(unit, k).detach().clone() for k in ("w", "b", "w_out")} if unit is not None else None
            log = [{"step": 0, **evaluate(model, unit, w_init)}]; print(name, seed, info, log[-1], flush=True)
            model.requires_grad_(True); params = list(model.parameters()) + (list(unit.parameters()) if unit is not None else [])
            torch.manual_seed(seed); opt = torch.optim.Adam(params, LR); g = torch.Generator().manual_seed(seed); zero = {"w": 0, "b": 0, "w_out": 0}; fired = []
            for step in range(1, STEPS + 1):
                starts = torch.randint(0, len(train_ids) - SEQ - 1, (BATCH,), generator=g); x = torch.stack([train_ids[s:s + SEQ + 1] for s in starts]).to(DEV)
                if unit is not None: unit.stats = []
                logits = model(x[:, :-1]).logits; loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))
                opt.zero_grad(set_to_none=True); loss.backward()
                if unit is not None:
                    for k in zero: zero[k] += int(getattr(unit, k).grad is None or bool((getattr(unit, k).grad == 0).all()))
                    fired += unit.stats; unit.stats = None
                opt.step()
                if step % EVERY == 0:
                    r = {"step": step, **evaluate(model, unit, w_init)}
                    if unit is not None:
                        r["steps_with_exactly_zero_unit_gradient"] = dict(zero)
                        r["training_tokens_within_gradient_band"] = int(sum(f[0] for f in fired)); r["training_tokens_firing"] = int(sum(f[1] for f in fired))
                        r["training_tokens"] = int(sum(f[2] for f in fired)); r["max_preactivation_on_training_tokens"] = float(max(f[3] for f in fired)); fired = []
                    log.append(r); print(name, seed, step, {k: v for k, v in r.items() if k not in ("fit",)}, f"{time.time() - t0:.0f}s", flush=True)
            if unit is not None: hk.remove()
            R["arms"][name].append({"seed": seed, **info, "log": log}); C.save_json(R, f"{OUT}/e1_results.json")
            del model, opt; torch.cuda.empty_cache()
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/e1_results.json")
    lines = ["# E1: a shielded edit in Pythia-160M", "", f"Layer {LAYER}. Held-out paraphrases with Rome top-1 (of 10), neighbourhood prompts unchanged (of 8), validation loss; mean over seeds.", "",
             "| arm | at install: held-out / neighbours / val loss | after 1,000 steps: held-out / neighbours / val loss | steps with exactly zero unit gradient (w) | unit weight change (max) |", "|---|---|---|---|---|"]
    for name, runs in R["arms"].items():
        a, z = [r["log"][0] for r in runs], [r["log"][-1] for r in runs]; mean = lambda xs: float(np.mean(xs))
        zg = "-" if "steps_with_exactly_zero_unit_gradient" not in z[0] else f"{mean([x['steps_with_exactly_zero_unit_gradient']['w'] for x in z]):.0f} / {STEPS}"
        ch = "-" if "unit_change" not in z[0] else f"{max(max(x['unit_change'].values()) for x in z):.1e}"
        lines.append(f"| {name} | {mean([x['held']['new_top1'] for x in a]):.1f} / {mean([x['neigh_unchanged'] for x in a]):.1f} / {mean([x['val_loss'] for x in a]):.4f} | "
                     f"{mean([x['held']['new_top1'] for x in z]):.1f} / {mean([x['neigh_unchanged'] for x in z]):.1f} / {mean([x['val_loss'] for x in z]):.4f} | {zg} | {ch} |")
    open(f"{OUT}/e1_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
