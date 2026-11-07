"""
L-series: the reach law (see PREREG.md, "L-series"). A frozen module fixes, exactly, everything beyond the reach of the
trainable parts in front of it, and training renegotiates everything within that reach.
  l1     the law on P1's task: trunk depth x blob width x gate boundary; probes; column-confined attacks
  l2     the law on natural images (AFHQ cats, cyclically shifted, with a red disc)
  l3     does the network's own (border-based) position code wither once a contract supplies position?
  l4     adding a skill through a contract without forgetting the old one
  l5toy  the impossible end of the exchange rate: the circular network at 16, 64, 256 channels
  l6     renegotiation vs clutter: L1's task (r = 4, sigma = 2) with 0, 2 or 6 orange distractor blobs
  l7     reach, not depth: three trunks with reach 4 (2 layers, one dilated; 4 layers; 6 layers with two 1x1)
  l8     renegotiation vs visibility: L1's task (r = 4, sigma = 2) with background noise 0.05, 0.15 or 0.3

  <venv python> followup/src/l_series.py l1|l2|l3|l4|l5toy|l6|l7|l8 [--smoke]
"""
import copy, sys, time
import numpy as np
import common as C
import torch, torch.nn as nn, torch.nn.functional as F
import p1_separation as P, p2_p4_contracts as Q

DEV = "cuda"; M, B = P.M, P.B; SMOKE = "--smoke" in sys.argv; STEPS = 60 if SMOKE else P.STEPS
RED = np.array(P.COLS["red"], np.float32)

def clamp(u): return -B + F.relu(u + B) - F.relu(u - B)
def gate(u, off): return F.relu(u - M * off) - F.relu(-u - M * off)              # exactly u where off = 0, exactly 0 where off = 1 (|u| < M)

def trunk(depth, ch=32, cin=3, pad="circular"):
    return nn.Sequential(*[m for i in range(depth) for m in (nn.Conv2d(cin if i == 0 else ch, ch, 3, padding=1, padding_mode=pad), nn.ReLU())])

def off_cols(b, size):
    return (torch.arange(size) >= b).float().view(1, 1, 1, size).expand(1, 1, size, size).clone()

class Contract(nn.Module):
    """P1's contract (d) with a chosen trunk depth, boundary and image size: logit = scale * mean(gate(clamp(score))) * 10 + bias."""
    def __init__(self, depth=4, b=16, size=32, ch=32, pad="circular"):
        super().__init__(); self.trunk, self.score = trunk(depth, ch, pad=pad), nn.Conv2d(ch, 1, 1); nn.init.constant_(self.score.bias, 1.0)
        self.scale, self.bias = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(0.0)); self.register_buffer("off", off_cols(b, size))
    def u(self, x): return clamp(self.score(self.trunk(x)))
    def forward(self, x): return self.scale * gate(self.u(x), self.off).mean((1, 2, 3)) * 10 + self.bias

class Plain(nn.Module):
    """No contract: mean of a 1x1 score over the trunk (P1's arms (a)/(b))."""
    def __init__(self, depth=4, ch=32, pad="circular"):
        super().__init__(); self.trunk, self.score = trunk(depth, ch, pad=pad), nn.Conv2d(ch, 1, 1)
    def forward(self, x): return self.score(self.trunk(x)).mean((1, 2, 3))

def blob(img, cx, cy, sigma, col, size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = np.minimum(np.abs(xx - cx), size - np.abs(xx - cx)); dy = np.minimum(np.abs(yy - cy), size - np.abs(yy - cy))
    a = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))[None]; return (1 - a) * img + a * col[:, None, None]

def make(n, seed, sigma=2.0, size=32, noise=0.05):
    """P1's task with blob width sigma: green and blue distractors, red blob, label = red centre in the left half (wrap-around)."""
    rng = np.random.default_rng(seed); X = 0.5 + noise * rng.standard_normal((n, 3, size, size)).astype(np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        for name in ("green", "blue", "red"):
            cx, cy = rng.uniform(0, size, 2); X[i] = blob(X[i], cx, cy, sigma, np.array(P.COLS[name], np.float32), size)
            if name == "red": y[i] = float(cx < size / 2)
    return torch.from_numpy(X), torch.from_numpy(y)

def fit(make_model, X, y, n, seed, steps=None):
    torch.manual_seed(seed); m = make_model().to(DEV); return Q.train(m, X[:n], y[:n], seed, steps=steps or STEPS)

def probes_synthetic(sigma, size=32, seed=7, noise=0.05):
    rng = np.random.default_rng(seed); Xp, cols = [], []
    for c in range(size):
        for cy in np.linspace(size / 8, size - size / 8, 4):
            for _ in range(2):
                bg = 0.5 + noise * rng.standard_normal((3, size, size)).astype(np.float32); Xp.append(blob(bg, c + 0.5, cy, sigma, RED, size)); cols.append(c)
    return torch.from_numpy(np.stack(Xp)), np.array(cols)

@torch.no_grad()
def effective_boundary(m, Xp, cols, size):
    m.eval(); called = np.concatenate([(m(Xp[s:s + 256].to(DEV)) > 0).cpu().numpy() for s in range(0, len(Xp), 256)])
    share = np.array([called[cols == c].mean() for c in range(size)])
    eff = next((c for c in range(size) if share[c] < 0.5), size)
    return int(eff), share.tolist()

def column_attack(m, X, y, col, steps=50, eps=1.0):
    """PGD confined to one input column (repaint: eps = 1), maximising the loss of the true label; largest logit change."""
    m.eval(); x0 = X.to(DEV); y0 = y.to(DEV); mask = torch.zeros(1, 1, 1, x0.shape[-1], device=DEV); mask[..., col] = 1
    with torch.no_grad(): l0 = m(x0)
    x = x0.clone()
    for _ in range(steps):
        x.requires_grad_(True); loss = F.binary_cross_entropy_with_logits(m(x), y0); gx, = torch.autograd.grad(loss, x)
        with torch.no_grad():                                                      # clip to [0, 1] only inside the attacked column (L8 amendment:
            x = x0 + mask * ((x + eps / 10 * gx.sign() * mask) - x0).clamp(-eps, eps)  # clipping the whole image changed pixels outside it
            x = torch.where(mask.bool().expand_as(x), x.clamp(0, 1), x0)            # wherever they lie outside [0, 1], as with heavy noise)
    with torch.no_grad():
        xr = x0 * (1 - mask) + mask * torch.rand_like(x0)                               # and a random repaint of the column
        return max(float((m(x) - l0).abs().max()), float((m(xr) - l0).abs().max()))

def guarantee(m, Xte, yte, r, half):
    return {str(d): column_attack(m, Xte[:250], yte[:250], half + d) for d in range(0, r + 2)}

# ---------------------------------------------------------------- L1
def l1(R, t0, out):
    depths, sigmas, bs, seeds = ((2, 4, 6), (1.0, 2.0, 3.0), (16, 12, 8, 4), (0, 1)) if not SMOKE else ((2, 6), (1.0, 3.0), (16, 4), (0,))
    for D in depths:
        for sig in sigmas:
            Xtr, ytr = make(4096 if not SMOKE else 512, 1, sig); Xte, yte = make(4000 if not SMOKE else 500, 2, sig); Xp, cols = probes_synthetic(sig)
            key = f"D{D}_sigma{sig:g}"; R[key] = {}
            for b in bs:
                R[key][str(b)] = []
                for seed in seeds:
                    m = fit(lambda: Contract(D, b), Xtr, ytr, len(Xtr), seed); eff, share = effective_boundary(m, Xp, cols, 32)
                    r = {"seed": seed, "acc": Q.acc(m, Xte, yte), "effective_boundary": eff, "correction": eff - b, "left_share_by_column": share}
                    if b == 16: r["guarantee_max_logit_change_by_d"] = guarantee(m, Xte, yte, D, 16)
                    R[key][str(b)].append(r)
                    print("L1", key, "b", b, "seed", seed, {k: v for k, v in r.items() if k != "left_share_by_column"}, f"{time.time() - t0:.0f}s", flush=True)
                C.save_json(R, f"{out}/l1_results.json")

# ---------------------------------------------------------------- L2
def cats64(split_lo, split_hi):
    import exp11_catgan26 as e11
    c = torch.from_numpy(e11.load_cats()[split_lo:split_hi]).permute(0, 3, 1, 2).float() / 255
    return F.avg_pool2d(c, 2).numpy()                                                  # 128 -> 64

def make_cats(bgs, n, seed, sigma=2.0, size=64):
    rng = np.random.default_rng(seed); X = np.zeros((n, 3, size, size), np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        bg = np.roll(bgs[i % len(bgs)], (int(rng.integers(size)), int(rng.integers(size))), axis=(1, 2))   # cyclic shift: shift-invariant distribution
        cx, cy = rng.uniform(0, size, 2); X[i] = blob(bg, cx, cy, sigma, RED, size); y[i] = float(cx < size / 2)
    return torch.from_numpy(X), torch.from_numpy(y)

def probes_cats(bgs, sigma=2.0, size=64, seed=7):
    rng = np.random.default_rng(seed); Xp, cols = [], []
    for c in range(size):
        for _ in range(4):
            bg = np.roll(bgs[int(rng.integers(len(bgs)))], (int(rng.integers(size)), int(rng.integers(size))), axis=(1, 2))
            Xp.append(blob(bg, c + 0.5, rng.uniform(0, size), sigma, RED, size)); cols.append(c)
    return torch.from_numpy(np.stack(Xp)), np.array(cols)

def l2(R, t0, out):
    tr_bg, te_bg = cats64(0, 4096 if not SMOKE else 256), cats64(4096, 5153 if not SMOKE else 4352)
    Xtr, ytr = make_cats(tr_bg, len(tr_bg), 1); Xte, yte = make_cats(te_bg, 2000 if not SMOKE else 300, 2); Xp, cols = probes_cats(te_bg)
    R["train_images"], R["test_images"] = len(Xtr), len(Xte)
    m = fit(lambda: Plain(4), Xtr, ytr, len(Xtr), 0); R["circular_without_contract_acc"] = Q.acc(m, Xte, yte); print("L2 circular, no contract", R["circular_without_contract_acc"], flush=True)
    for shift in ((0, 4, 8, 12, 16) if not SMOKE else (0, 16)):
        b = 32 - shift; R[str(shift)] = []
        for seed in ((0, 1, 2) if not SMOKE else (0,)):
            m = fit(lambda: Contract(4, b, 64), Xtr, ytr, len(Xtr), seed); eff, share = effective_boundary(m, Xp, cols, 64)
            r = {"seed": seed, "b": b, "acc": Q.acc(m, Xte, yte), "effective_boundary": eff, "correction": eff - b, "left_share_by_column": share}
            if shift == 0: r["guarantee_max_logit_change_by_d"] = guarantee(m, Xte, yte, 4, 32)
            R[str(shift)].append(r); print("L2 shift", shift, "seed", seed, {k: v for k, v in r.items() if k != "left_share_by_column"}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{out}/l2_results.json")

# ---------------------------------------------------------------- L3
class ContractOn(nn.Module):
    """A pretrained Plain/P.Net's trunk and score with P1's contract installed on the score map (gate on the left half, clamp)."""
    def __init__(self, pre, s):
        super().__init__(); self.trunk, self.score = copy.deepcopy(pre.trunk), copy.deepcopy(pre.score)
        with torch.no_grad(): self.score.weight.div_(s); self.score.bias.div_(s)
        self.scale, self.bias = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(0.0)); self.register_buffer("off", off_cols(16, 32))
    def forward(self, x): return self.scale * gate(clamp(self.score(self.trunk(x))), self.off).mean((1, 2, 3)) * 10 + self.bias

@torch.no_grad()
def pooled(m, X):
    m.eval(); return torch.cat([m.trunk(X[s:s + 500].to(DEV)).mean((2, 3)).cpu() for s in range(0, len(X), 500)]).double().numpy()

def probe_acc(m, Xa, ya, Xb, yb, lam=1e-3):
    """Linear probe (ridge to +-1 on standardised pooled trunk features), fitted on (Xa, ya), accuracy on (Xb, yb)."""
    A, Bm = pooled(m, Xa), pooled(m, Xb); mu, sd = A.mean(0), A.std(0) + 1e-8; A, Bm = (A - mu) / sd, (Bm - mu) / sd
    A1 = np.hstack([A, np.ones((len(A), 1))]); t = 2 * ya.numpy() - 1
    w = np.linalg.solve(A1.T @ A1 + lam * len(A1) * np.eye(A1.shape[1]), A1.T @ t)
    return float(((np.hstack([Bm, np.ones((len(Bm), 1))]) @ w > 0) == (yb.numpy() > 0.5)).mean())

def train_cb(m, X, y, seed, steps, at, cb, lr=P.LR):
    torch.manual_seed(seed); opt = torch.optim.Adam(m.parameters(), lr); g = torch.Generator().manual_seed(seed); Xd, yd = X.to(DEV), y.to(DEV)
    cb(0)
    for step in range(1, steps + 1):
        m.train(); idx = torch.randint(0, len(X), (P.BATCH,), generator=g).to(DEV)
        loss = F.binary_cross_entropy_with_logits(m(Xd[idx]), yd[idx]); opt.zero_grad(); loss.backward(); opt.step()
        if step in at: cb(step)
    return m.eval()

def l3(R, t0, out):
    Xtr, ytr = P.make(4096 if not SMOKE else 512, 1); Xte, yte = P.make(4000 if not SMOKE else 500, 2); at = {100, 250, 500, 1000, 2000} if not SMOKE else {30, 60}
    Xpa, ypa, Xpb, ypb = Xtr[:2000], ytr[:2000], Xte[:2000], yte[:2000]
    for seed in ((0, 1, 2) if not SMOKE else (0,)):
        pre = fit(lambda: P.Net("b"), Xtr, ytr, len(Xtr), seed); r = {"pretrained_acc": Q.acc(pre, Xte, yte), "pretrained_probe": probe_acc(pre, Xpa, ypa, Xpb, ypb)}
        with torch.no_grad(): s = max(float(pre.detector(Xtr[i:i + 256].to(DEV)).abs().max()) for i in range(0, len(Xtr), 256)) / 10
        r["calibration"] = s
        for arm, mk in (("control", lambda: copy.deepcopy(pre)), ("contract", lambda: ContractOn(pre, s))):
            m = mk().to(DEV); log = []
            cb = lambda step: log.append({"step": step, "acc": Q.acc(m, Xte, yte), "probe": probe_acc(m, Xpa, ypa, Xpb, ypb)})
            train_cb(m, Xtr, ytr, 100 + seed, STEPS, at, cb); r[arm] = log
        R[str(seed)] = r; C.save_json(R, f"{out}/l3_results.json")
        print("L3 seed", seed, {k: (v if not isinstance(v, list) else [(e["step"], round(e["acc"], 3), round(e["probe"], 3)) for e in v]) for k, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- L4
class HeadB(nn.Module):
    """Unconstrained new head on [trunk features, x-ramp]; the original head (score) is kept for evaluation."""
    def __init__(self, pre):
        super().__init__(); self.trunk, self.score = copy.deepcopy(pre.trunk), copy.deepcopy(pre.score)
        self.h1, self.h2 = nn.Conv2d(33, 16, 1), nn.Conv2d(16, 1, 1); self.register_buffer("ramp", torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32).clone())
    def forward(self, x):
        f = self.trunk(x); return self.h2(F.relu(self.h1(torch.cat([f, self.ramp.expand(len(x), -1, -1, -1)], 1)))).mean((1, 2, 3))
    def original(self, x): return self.score(self.trunk(x)).mean((1, 2, 3))

@torch.no_grad()
def acc_fn(f, X, y):
    return float(np.mean([((f(X[s:s + 500].to(DEV)) > 0).float() == y[s:s + 500].to(DEV)).float().mean().item() for s in range(0, len(X), 500)]))

def l4(R, t0, out):
    Xp, yp = Q.make2(4096 if not SMOKE else 512, 21, task="present"); Xl, yl = Q.make2(512, 22); Xu, _ = Q.make2(512, 25)
    Xte, yte = Q.make2(4000 if not SMOKE else 500, 23); Xpte, ypte = Q.make2(2000 if not SMOKE else 500, 24, task="present")
    for seed in ((0, 1, 2) if not SMOKE else (0,)):
        pre = fit(lambda: P.Net("a"), Xp, yp, len(Xp), seed); r = {"pretrained_present_acc": acc_fn(lambda x: pre.detector(x).mean((1, 2, 3)), Xpte, ypte)}
        A = Q.install_contract(pre, Xu); torch.manual_seed(100 + seed); A = Q.train(A, Xl[:32], yl[:32], 100 + seed, steps=300 if not SMOKE else 30)
        r["A_left_acc"] = Q.acc(A, Xte, yte); r["A_present_acc_after"] = acc_fn(lambda x: A.detector(x).mean((1, 2, 3)), Xpte, ypte)
        torch.manual_seed(100 + seed); Bm = HeadB(pre).to(DEV); Bm = Q.train(Bm, Xl[:32], yl[:32], 100 + seed, steps=300 if not SMOKE else 30)
        r["B_left_acc"] = Q.acc(Bm, Xte, yte); r["B_present_acc_after"] = acc_fn(Bm.original, Xpte, ypte)
        R[str(seed)] = r; C.save_json(R, f"{out}/l4_results.json"); print("L4 seed", seed, {k: round(v, 4) for k, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- L5 (toy part)
def l5toy(R, t0, out):
    Xtr, ytr = P.make(4096 if not SMOKE else 512, 1); Xte, yte = P.make(4000 if not SMOKE else 500, 2)
    for ch in ((16, 64, 256) if not SMOKE else (16,)):
        m = fit(lambda: Plain(4, ch), Xtr, ytr, len(Xtr), 0); R[str(ch)] = {"test_acc": Q.acc(m, Xte, yte), "train_acc": Q.acc(m, Xtr, ytr)}
        print("L5 toy, circular, channels", ch, R[str(ch)], f"{time.time() - t0:.0f}s", flush=True); C.save_json(R, f"{out}/l5toy_results.json")

# ---------------------------------------------------------------- L6
ORANGE = np.array((0.9, 0.45, 0.1), np.float32)
def make_clutter(n, seed, n_orange, sigma=2.0, size=32):
    """L1's task plus n_orange orange distractor blobs (same width), drawn before the red blob."""
    rng = np.random.default_rng(seed); X = 0.5 + 0.05 * rng.standard_normal((n, 3, size, size)).astype(np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        for name in ("green", "blue") + ("orange",) * n_orange + ("red",):
            cx, cy = rng.uniform(0, size, 2); X[i] = blob(X[i], cx, cy, sigma, ORANGE if name == "orange" else np.array(P.COLS[name], np.float32), size)
            if name == "red": y[i] = float(cx < size / 2)
    return torch.from_numpy(X), torch.from_numpy(y)

def l6(R, t0, out):
    Xp, cols = probes_synthetic(2.0)
    for no in ((0, 2, 6) if not SMOKE else (0, 6)):
        Xtr, ytr = make_clutter(4096 if not SMOKE else 512, 1, no); Xte, yte = make_clutter(4000 if not SMOKE else 500, 2, no); R[str(no)] = {}
        for b in ((16, 12, 8, 4) if not SMOKE else (16, 4)):
            R[str(no)][str(b)] = []
            for seed in ((0, 1) if not SMOKE else (0,)):
                m = fit(lambda: Contract(4, b), Xtr, ytr, len(Xtr), seed); eff, share = effective_boundary(m, Xp, cols, 32)
                r = {"seed": seed, "acc": Q.acc(m, Xte, yte), "effective_boundary": eff, "correction": eff - b, "left_share_by_column": share}
                if b == 16: r["guarantee_max_logit_change_by_d"] = guarantee(m, Xte, yte, 4, 16)
                R[str(no)][str(b)].append(r); print("L6 orange", no, "b", b, "seed", seed, {k: v for k, v in r.items() if k != "left_share_by_column"}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{out}/l6_results.json")

# ---------------------------------------------------------------- L7
def trunk_spec(spec, ch=32, cin=3):
    """spec: list of (kernel, dilation); circular padding keeps the size. Reach = sum of dilation * (kernel - 1) / 2."""
    layers = []
    for i, (k, d) in enumerate(spec):
        layers += [nn.Conv2d(cin if i == 0 else ch, ch, k, padding=d * (k - 1) // 2, dilation=d, padding_mode="circular"), nn.ReLU()]
    return nn.Sequential(*layers)

SPECS = {"2 layers, second dilated 3 (reach 4)": [(3, 1), (3, 3)], "4 layers (reach 4, as L1)": [(3, 1)] * 4,
         "6 layers, last two 1x1 (reach 4)": [(3, 1)] * 4 + [(1, 1)] * 2}

def l7(R, t0, out):
    Xtr, ytr = make(4096 if not SMOKE else 512, 1, 2.0); Xte, yte = make(4000 if not SMOKE else 500, 2, 2.0); Xp, cols = probes_synthetic(2.0)
    R["reach"] = {n: sum(d * (k - 1) // 2 for k, d in sp) for n, sp in SPECS.items()}
    for name, spec in (SPECS.items() if not SMOKE else list(SPECS.items())[:1]):
        R[name] = {}
        for b in ((16, 12, 8, 4) if not SMOKE else (16,)):
            R[name][str(b)] = []
            for seed in ((0, 1) if not SMOKE else (0,)):
                def mk():
                    m = Contract(4, b); m.trunk = trunk_spec(spec); return m
                m = fit(mk, Xtr, ytr, len(Xtr), seed); eff, share = effective_boundary(m, Xp, cols, 32)
                r = {"seed": seed, "acc": Q.acc(m, Xte, yte), "effective_boundary": eff, "correction": eff - b, "left_share_by_column": share}
                if b == 16: r["guarantee_max_logit_change_by_d"] = guarantee(m, Xte, yte, 4, 16)
                R[name][str(b)].append(r); print("L7", name, "b", b, "seed", seed, {k: v for k, v in r.items() if k != "left_share_by_column"}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{out}/l7_results.json")

# ---------------------------------------------------------------- L8
def l8(R, t0, out):
    for nz in ((0.05, 0.15, 0.3) if not SMOKE else (0.3,)):
        Xtr, ytr = make(4096 if not SMOKE else 512, 1, 2.0, noise=nz); Xte, yte = make(4000 if not SMOKE else 500, 2, 2.0, noise=nz)
        Xp, cols = probes_synthetic(2.0, noise=nz); R[str(nz)] = {}
        for b in ((16, 12, 8, 4) if not SMOKE else (16,)):
            R[str(nz)][str(b)] = []
            for seed in ((0, 1) if not SMOKE else (0,)):
                m = fit(lambda: Contract(4, b), Xtr, ytr, len(Xtr), seed); eff, share = effective_boundary(m, Xp, cols, 32)
                r = {"seed": seed, "acc": Q.acc(m, Xte, yte), "effective_boundary": eff, "correction": eff - b, "left_share_by_column": share}
                if b == 16: r["guarantee_max_logit_change_by_d"] = guarantee(m, Xte, yte, 4, 16)
                R[str(nz)][str(b)].append(r); print("L8 noise", nz, "b", b, "seed", seed, {k: v for k, v in r.items() if k != "left_share_by_column"}, f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{out}/l8_results.json")

def l8g(R, t0, out):
    """L8 amendment: the guarantee check rerun with the attack clipped only inside the attacked column (b = 16 models, retrained)."""
    for nz in (0.05, 0.15, 0.3):
        Xtr, ytr = make(4096, 1, 2.0, noise=nz); Xte, yte = make(4000, 2, 2.0, noise=nz); R[str(nz)] = []
        for seed in (0, 1):
            m = fit(lambda: Contract(4, 16), Xtr, ytr, len(Xtr), seed)
            r = {"seed": seed, "acc": Q.acc(m, Xte, yte), "guarantee_max_logit_change_by_d": guarantee(m, Xte, yte, 4, 16)}
            R[str(nz)].append(r); print("L8g noise", nz, "seed", seed, r, f"{time.time() - t0:.0f}s", flush=True); C.save_json(R, f"{out}/l8g_results.json")

def main(which):
    t0 = time.time(); out = C.out_dir(which + ("_smoke" if SMOKE else "")); R = {"smoke": SMOKE}
    {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "l5toy": l5toy, "l6": l6, "l7": l7, "l8": l8, "l8g": l8g}[which](R, t0, out); R["seconds"] = time.time() - t0; C.save_json(R, f"{out}/{which}_results.json")

if __name__ == "__main__":
    main(sys.argv[1])
