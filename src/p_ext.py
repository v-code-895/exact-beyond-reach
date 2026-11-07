"""
Follow-ups to P5-P7 (see PREREG.md, "P6b, P7b, P9b"):
  p6b: how far can training move a frozen boundary? boundaries at columns 16, 14, 12, 10, 8, 4, plus where the detector puts its
       response (probe images with a lone red blob at every column)
  p7b: two contracts with a small MLP readout instead of a linear one, and where the composed contract's errors are
  p9b: learning curves at small N (8, 16, 32) for (c) coordinate channel, (d) contract, (f) binary wire as an input channel

  <venv python> src/p_ext.py p6b|p7b|p9b
"""
import sys, time
import numpy as np
import common as C
import torch, torch.nn as nn, torch.nn.functional as F
import p1_separation as P, p2_p4_contracts as Q, p5_p7_contracts as Q5

DEV = "cuda"; S_ = P.S_

def fit(make, X, y, n, seed, steps=P.STEPS):
    torch.manual_seed(seed); m = make().to(DEV)                                     # seeded init (Q.train seeds only the batches)
    return Q.train(m, X[:n], y[:n], seed, steps=steps)

def off(b):
    return (torch.arange(S_) >= b).float().view(1, 1, 1, S_).expand(1, 1, S_, S_).clone()

def probes(seed=7):
    """A lone red blob centred at column c + 0.5 for every c, rows 4.5/12.5/20.5/28.5, two noise draws; and the same backgrounds blank."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:S_, 0:S_].astype(np.float32); Xb, Xp, meta = [], [], []
    for c in range(S_):
        for cy in (4.5, 12.5, 20.5, 28.5):
            for _ in range(2):
                bg = 0.5 + 0.05 * rng.standard_normal((3, S_, S_)).astype(np.float32); cx = c + 0.5
                dx = np.minimum(np.abs(xx - cx), S_ - np.abs(xx - cx)); dy = np.minimum(np.abs(yy - cy), S_ - np.abs(yy - cy))
                a = np.exp(-(dx ** 2 + dy ** 2) / (2 * 2.0 ** 2))[None]
                Xb.append(bg); Xp.append((1 - a) * bg + a * np.array(P.COLS["red"], np.float32)[:, None, None]); meta.append((c, cy))
    return torch.from_numpy(np.stack(Xp)), torch.from_numpy(np.stack(Xb)), meta

@torch.no_grad()
def probe_contract(m, Xp, Xb, meta):
    """Decision per blob column, and the detector's response profile around the blob (|change in clamped score| caused by the blob)."""
    m.eval(); logit = m(Xp.to(DEV)).cpu().numpy()
    u = lambda X: Q5.clamp(m.score(m.trunk(X.to(DEV))))[:, 0].cpu().numpy()
    du = np.abs(u(Xp) - u(Xb)); prof = np.zeros(S_)
    left_share = np.zeros(S_); cnt = np.zeros(S_)
    for i, (c, cy) in enumerate(meta):
        r0 = int(cy); rows = [(r0 + k) % S_ for k in range(-3, 4)]; p = du[i, rows].sum(0)
        prof += np.roll(p, S_ // 2 - c)                                                 # aligned: index 16 = the blob's column
        left_share[c] += float(logit[i] > 0); cnt[c] += 1
    prof /= prof.max(); left_share /= cnt; d = np.arange(S_) - S_ // 2
    above = d[prof > 0.1]; centroid = float((prof * d).sum() / prof.sum())
    return {"left_share_by_column": left_share.tolist(), "columns_called_left": int((left_share > 0.5).sum()),
            "columns_wrong": [int(c) for c in range(S_) if (left_share[c] > 0.5) != (c + 0.5 < S_ / 2)],
            "profile": prof.tolist(), "extent_left": int(-above.min()), "extent_right": int(above.max()), "centroid": centroid}

def p6b(R, t0):
    Xtr, ytr = P.make(4096, 1); Xte, yte = P.make(4000, 2); Xp, Xb, meta = probes()
    for b in (16, 14, 12, 10, 8, 4):
        R[str(b)] = []
        for seed in (0, 1, 2):
            m = fit(lambda: Q5.Contract(off(b)), Xtr, ytr, 4096, seed); r = {"acc": Q.acc(m, Xte, yte), **probe_contract(m, Xp, Xb, meta)}
            R[str(b)].append(r)
            print("P6b boundary", b, "seed", seed, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("left_share_by_column", "profile")}, f"{time.time() - t0:.0f}s", flush=True)

class TwoContractsMLP(Q5.TwoContracts):
    def __init__(self):
        super().__init__(); self.read = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

def make_and_pos(n, seed):
    """Q5.make_and with the blob positions returned (same random stream)."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:S_, 0:S_].astype(np.float32)
    X = 0.5 + 0.05 * rng.standard_normal((n, 3, S_, S_)).astype(np.float32); y = np.zeros(n, np.float32); pos = np.zeros((n, 4), np.float32)
    for i in range(n):
        p = {}
        for col in ("green", "blue", "red"):
            cx, cy = rng.uniform(0, S_, 2); p[col] = (cx, cy)
            dx = np.minimum(np.abs(xx - cx), S_ - np.abs(xx - cx)); dy = np.minimum(np.abs(yy - cy), S_ - np.abs(yy - cy))
            a = np.exp(-(dx ** 2 + dy ** 2) / (2 * 2.0 ** 2))[None]; X[i] = (1 - a) * X[i] + a * np.array(P.COLS[col], np.float32)[:, None, None]
        y[i] = float(p["red"][0] < S_ / 2 and p["blue"][1] < S_ / 2); pos[i] = (*p["red"], *p["blue"])
    return torch.from_numpy(X), torch.from_numpy(y), pos

def p7b(R, t0):
    Xtr, ytr = Q5.make_and(4096, 31); Xte, yte, pos = make_and_pos(4000, 32); Xchk, _ = Q5.make_and(4000, 32)
    R["same_test_images"] = bool(torch.equal(Xchk, Xte))
    edge = lambda v: np.minimum(v % (S_ / 2), S_ / 2 - v % (S_ / 2))                  # distance to the nearest boundary (0, 16, 32)
    near = (edge(pos[:, 0]) < 1.5) | (edge(pos[:, 3]) < 1.5); R["test_share_near_a_boundary"] = float(near.mean())
    for name, make in {"(d2) two contracts, linear readout": Q5.TwoContracts, "(d2m) two contracts, MLP readout": TwoContractsMLP}.items():
        R[name] = {}
        for n in (256, 1024, 4096):
            rs = []
            for seed in (0, 1, 2):
                m = fit(make, Xtr, ytr, n, seed)
                with torch.no_grad(): p = torch.cat([(m(Xte[s:s + 500].to(DEV)) > 0).float().cpu() for s in range(0, len(Xte), 500)]).numpy()
                err = p != yte.numpy(); rs.append({"balanced_acc": Q5.balanced_acc(m, Xte, yte), "errors": int(err.sum()),
                                                   "errors_near_a_boundary": float(near[err].mean()) if err.any() else None,
                                                   "false_positive_share_of_errors": float((p[err] == 1).mean()) if err.any() else None})
            R[name][str(n)] = rs; print("P7b", name, n, [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rs], f"{time.time() - t0:.0f}s", flush=True)

def p9b(R, t0):
    Xtr, ytr = P.make(4096, 1); Xte, yte = P.make(4000, 2)
    arms = {"(c) circular + coordinate channel": lambda: P.Net("c"), "(d) contract": lambda: P.Net("d"), "(f) binary wire as an input channel": Q5.WireInput}
    for name, make in arms.items():
        R[name] = {}
        for n in (8, 16, 32):
            rs = [Q.acc(fit(make, Xtr, ytr, n, s), Xte, yte) for s in range(5)]; R[name][str(n)] = rs
            print("P9b", name, n, [round(r, 3) for r in rs], f"{time.time() - t0:.0f}s", flush=True)

def main(which):
    t0 = time.time(); out = C.out_dir(which); R = {}
    {"p6b": p6b, "p7b": p7b, "p9b": p9b}[which](R, t0); R["seconds"] = time.time() - t0; C.save_json(R, f"{out}/{which}_results.json")

if __name__ == "__main__":
    main(sys.argv[1])
