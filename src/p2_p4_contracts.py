"""
P2-P4: contracts on P1's task (see PREREG.md).
  p2: out-of-distribution generalisation (position shift, colour shift)
  p3: a post hoc contract on a pretrained network (zero-shot, then few-shot)
  p4: robustness by construction (attacks confined to columns 20-27)

  <venv python> followup/src/p2_p4_contracts.py p3|p2|p4
"""
import sys, time
import numpy as np
import common as C
import torch, torch.nn as nn, torch.nn.functional as F
import p1_separation as P

DEV = "cuda"; S_ = P.S_
COLS = {**P.COLS, "cyan": (0.1, 0.8, 0.8), "yellow": (0.85, 0.8, 0.1)}

def make2(n, seed, red_rows=(0, S_), distractors=("green", "blue"), task="left"):
    """task 'left': label = red blob centre in the left half; task 'present': half the images have no red blob, label = red present."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:S_, 0:S_].astype(np.float32)
    X = 0.5 + 0.05 * rng.standard_normal((n, 3, S_, S_)).astype(np.float32); y = np.zeros(n, np.float32)
    def blob(i, col, cx, cy):
        dx = np.minimum(np.abs(xx - cx), S_ - np.abs(xx - cx)); dy = np.minimum(np.abs(yy - cy), S_ - np.abs(yy - cy))
        a = np.exp(-(dx ** 2 + dy ** 2) / (2 * 2.0 ** 2))[None]; X[i] = (1 - a) * X[i] + a * np.array(COLS[col], np.float32)[:, None, None]
    for i in range(n):
        for d in distractors: blob(i, d, *rng.uniform(0, S_, 2))
        cx, cy = rng.uniform(0, S_), rng.uniform(*red_rows)
        if task == "left": blob(i, "red", cx, cy); y[i] = float(cx < S_ / 2)
        else:
            present = rng.random() < 0.5
            if present: blob(i, "red", cx, cy)
            else: blob(i, distractors[i % len(distractors)], cx, cy)              # keep the blob count equal
            y[i] = float(present)
    return torch.from_numpy(X), torch.from_numpy(y)

def train(m, X, y, seed, steps=P.STEPS, lr=P.LR):
    torch.manual_seed(seed); opt = torch.optim.Adam(m.parameters(), lr); g = torch.Generator().manual_seed(seed); n = len(X); X, y = X.to(DEV), y.to(DEV); m.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(P.BATCH, n),), generator=g).to(DEV)
        loss = F.binary_cross_entropy_with_logits(m(X[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

@torch.no_grad()
def acc(m, X, y):
    return float(np.mean([((m(X[s:s + 500].to(DEV)) > 0).float() == y[s:s + 500].to(DEV)).float().mean().item() for s in range(0, len(X), 500)]))

def p2(R, t0):
    shifts = {"position": (dict(red_rows=(0, 16)), dict(red_rows=(16, 32))), "colour": (dict(distractors=("green", "blue")), dict(distractors=("cyan", "yellow")))}
    for sname, (din, dout) in shifts.items():
        Xtr, ytr = make2(4096, 11, **din); Xin, yin = make2(4000, 12, **din); Xout, yout = make2(4000, 13, **dout); R[sname] = {}
        for arm in "abcd":
            R[sname][arm] = {}
            for n in (256, 4096):
                rs = []
                for seed in (0, 1, 2):
                    m = train(P.Net(arm).to(DEV), Xtr[:n], ytr[:n], seed); rs.append({"in": acc(m, Xin, yin), "shifted": acc(m, Xout, yout)})
                R[sname][arm][str(n)] = rs
                print("P2", sname, arm, n, "in", [round(r["in"], 3) for r in rs], "shifted", [round(r["shifted"], 3) for r in rs], f"{time.time() - t0:.0f}s", flush=True)

def make_cols(n, seed, intervals):
    """P1's task with the red blob's centre column drawn uniformly from the given column intervals (label still x < 16)."""
    rng = np.random.default_rng(seed); X, y = make2(n, seed)                          # distractors and noise as usual
    lens = np.array([b - a for a, b in intervals], float); yy, xx = np.mgrid[0:S_, 0:S_].astype(np.float32)
    rng2 = np.random.default_rng(seed + 1000); Xb = 0.5 + 0.05 * rng2.standard_normal((n, 3, S_, S_)).astype(np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        for col in ("green", "blue", "red"):
            if col == "red":
                k = rng2.choice(len(intervals), p=lens / lens.sum()); cx = rng2.uniform(*intervals[k]); cy = rng2.uniform(0, S_); y[i] = float(cx < S_ / 2)
            else: cx, cy = rng2.uniform(0, S_, 2)
            dx = np.minimum(np.abs(xx - cx), S_ - np.abs(xx - cx)); dy = np.minimum(np.abs(yy - cy), S_ - np.abs(yy - cy))
            a = np.exp(-(dx ** 2 + dy ** 2) / (2 * 2.0 ** 2))[None]; Xb[i] = (1 - a) * Xb[i] + a * np.array(COLS[col], np.float32)[:, None, None]
    return torch.from_numpy(Xb), torch.from_numpy(y)

def p2b(R, t0):
    Xtr, ytr = make_cols(4096, 41, [(6, 26)]); Xin, yin = make_cols(4000, 42, [(6, 26)]); Xout, yout = make_cols(4000, 43, [(0, 6), (26, 32)])
    for arm in "bcd":
        R[arm] = {}
        for n in (256, 4096):
            rs = []
            for seed in (0, 1, 2):
                m = train(P.Net(arm).to(DEV), Xtr[:n], ytr[:n], seed); rs.append({"in": acc(m, Xin, yin), "shifted": acc(m, Xout, yout)})
            R[arm][str(n)] = rs; print("P2b", arm, n, "in", [round(r["in"], 3) for r in rs], "edges", [round(r["shifted"], 3) for r in rs], f"{time.time() - t0:.0f}s", flush=True)

def install_contract(pre, X_unlab):
    """The pretrained circular network's detector, with clamp, signed gate and RIGHT wire installed before the pooling; the
    readout applies the pretrained rule (mean score > 0) to the left half: logit = mean over the left half of the clamped score.
    The score is first rescaled by a frozen positive factor set from unlabelled images (largest |score| -> 10 < B = 20), folded
    exactly into the score layer, so the clamp does not cut the host's evidence on observed inputs (it still bounds every input)."""
    m = P.Net("d").to(DEV); m.trunk.load_state_dict(pre.trunk.state_dict()); m.score.load_state_dict(pre.score.state_dict())
    with torch.no_grad():
        s = max(float(pre.detector(X_unlab[i:i + 256].to(DEV)).abs().max()) for i in range(0, len(X_unlab), 256)) / 10
        m.score.weight.div_(s); m.score.bias.div_(s); m.scale.fill_(0.2); m.bias.fill_(0.0)   # 0.2 * 10 * mean over all pixels = mean over the left half
    m.calibration = s; return m

def p3(R, t0):
    Xp, yp = make2(4096, 21, task="present"); Xl, yl = make2(512, 22); Xu, _ = make2(512, 25); Xte, yte = make2(4000, 23); Xpte, ypte = make2(2000, 24, task="present")
    R["pretrain_acc"] = {}; R["runs"] = {}
    for seed in (0, 1, 2):
        pre_a = train(P.Net("a").to(DEV), Xp, yp, seed); pre_c = train(P.Net("c").to(DEV), Xp, yp, seed)
        R["pretrain_acc"][str(seed)] = {"a": acc(pre_a, Xpte, ypte), "c": acc(pre_c, Xpte, ypte)}
        zs = install_contract(pre_a, Xu); r = {"zero_shot": acc(zs, Xte, yte), "calibration": zs.calibration, "pre_a_on_left_task_without_contract": acc(pre_a, Xte, yte)}
        with torch.no_grad(): r["max_abs_score_before_clamp"] = float(max(zs.detector(Xte[s:s + 500].to(DEV)).abs().max().item() for s in range(0, 4000, 500)))
        for n in (8, 32, 128, 512):
            m = train(install_contract(pre_a, Xu), Xl[:n], yl[:n], 100 + seed, steps=300); r[f"contract_N{n}"] = acc(m, Xte, yte)
            mc = P.Net("c").to(DEV); mc.load_state_dict(pre_c.state_dict())             # a copy of the pretrained coordinate network
            mc = train(mc, Xl[:n], yl[:n], 100 + seed, steps=300); r[f"pretrained_coord_N{n}"] = acc(mc, Xte, yte)
        R["runs"][str(seed)] = r
        print("P3 seed", seed, "pretrain", R["pretrain_acc"][str(seed)], {k: round(v, 3) for k, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)

def p4(R, t0):
    Xtr, ytr = P.make(4096, 1); Xte, yte = P.make(1000, 2); mask = torch.zeros(1, 1, 1, S_, device=DEV); mask[..., 20:28] = 1
    for arm in "bcd":
        R[arm] = []
        for seed in (0, 1, 2):
            m = train(P.Net(arm).to(DEV), Xtr, ytr, seed); r = {"clean": acc(m, Xte, yte)}
            with torch.no_grad():                                                        # random noise in the region: exact invariance check
                x = Xte[:200].to(DEV); xn = (x + mask * torch.rand_like(x)).clamp(0, 1); r["random_region_noise_max_logit_change"] = float((m(xn) - m(x)).abs().max())
            for eps in (8 / 255, 32 / 255, 1.0):
                correct, dmax = 0, 0.0
                for s in range(0, len(Xte), 250):
                    x0 = Xte[s:s + 250].to(DEV); y0 = yte[s:s + 250].to(DEV); x = x0.clone()
                    with torch.no_grad(): l0 = m(x0)
                    for _ in range(100):
                        x.requires_grad_(True); loss = F.binary_cross_entropy_with_logits(m(x), y0); gx, = torch.autograd.grad(loss, x)
                        with torch.no_grad(): x = (x0 + ((x + eps / 10 * gx.sign() * mask) - x0).clamp(-eps, eps)).clamp(0, 1)
                    with torch.no_grad(): la = m(x); correct += int(((la > 0).float() == y0).sum()); dmax = max(dmax, float((la - l0).abs().max()))
                r[f"eps_{eps:.3f}"] = {"acc": correct / len(Xte), "max_logit_change": dmax}
            R[arm].append(r); print("P4", arm, seed, {k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)

def main(which):
    t0 = time.time(); out = C.out_dir(which); R = {}
    {"p2": p2, "p2b": p2b, "p3": p3, "p4": p4}[which](R, t0); R["seconds"] = time.time() - t0; C.save_json(R, f"{out}/{which}_results.json")

if __name__ == "__main__":
    main(sys.argv[1])
