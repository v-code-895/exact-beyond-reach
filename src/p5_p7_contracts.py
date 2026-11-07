"""
P5-P7: mapping the contract on P1's task (see PREREG.md).
  p5: arm (f) — the binary RIGHT wire as a plain input channel
  p6: wrong contracts — boundary at column 14 or 12, inverted halves
  p7: two contracts composed — red on the left AND blue on the top

  <venv python> src/p5_p7_contracts.py p5|p6|p7
"""
import sys, time
import numpy as np
import common as C
import torch, torch.nn as nn, torch.nn.functional as F
import p1_separation as P, p2_p4_contracts as Q

DEV = "cuda"; S_ = P.S_; M, B = P.M, P.B

def clamp(u): return -B + F.relu(u + B) - F.relu(u - B)
def gate(u, off): return F.relu(u - M * off) - F.relu(-u - M * off)                 # exactly u where off = 0, exactly 0 where off = 1 (|u| < M)

def trunk(cin): return nn.Sequential(*[m for i in range(4) for m in (nn.Conv2d(cin if i == 0 else 32, 32, 3, padding=1, padding_mode="circular"), nn.ReLU())])

class WireInput(nn.Module):                                                          # P5 (f): binary RIGHT wire as an input channel
    def __init__(self):
        super().__init__(); self.trunk, self.score = trunk(4), nn.Conv2d(32, 1, 1)
        self.register_buffer("right", (torch.arange(S_) >= S_ // 2).float().view(1, 1, 1, S_).expand(1, 1, S_, S_).clone())
    def forward(self, x): return self.score(self.trunk(torch.cat([x, self.right.expand(len(x), -1, -1, -1)], 1))).mean((1, 2, 3))

class Contract(nn.Module):                                                           # P6: the clamped contract with a chosen "off" region
    def __init__(self, off_mask):
        super().__init__(); self.trunk, self.score = trunk(3), nn.Conv2d(32, 1, 1); nn.init.constant_(self.score.bias, 1.0)
        self.scale, self.bias = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(0.0)); self.register_buffer("off", off_mask)
    def forward(self, x): return self.scale * gate(clamp(self.score(self.trunk(x))), self.off).mean((1, 2, 3)) * 10 + self.bias

class TwoContracts(nn.Module):                                                       # P7 (d2): red gated by LEFT, blue gated by TOP
    def __init__(self):
        super().__init__(); self.trunk, self.score = trunk(3), nn.Conv2d(32, 2, 1); nn.init.constant_(self.score.bias, 1.0)
        self.read = nn.Linear(2, 1)
        right = (torch.arange(S_) >= S_ // 2).float().view(1, 1, 1, S_).expand(1, 1, S_, S_); bottom = (torch.arange(S_) >= S_ // 2).float().view(1, 1, S_, 1).expand(1, 1, S_, S_)
        self.register_buffer("off", torch.cat([right, bottom], 1).clone())
    def forward(self, x): return self.read(gate(clamp(self.score(self.trunk(x))), self.off).mean((2, 3)) * 10).squeeze(1)

class Coord2(nn.Module):                                                             # P7 (c2): circular + x and y ramps
    def __init__(self):
        super().__init__(); self.trunk, self.score = trunk(5), nn.Conv2d(32, 1, 1)
        xr = torch.linspace(0, 1, S_).view(1, 1, 1, S_).expand(1, 1, S_, S_); yr = torch.linspace(0, 1, S_).view(1, 1, S_, 1).expand(1, 1, S_, S_)
        self.register_buffer("ramps", torch.cat([xr, yr], 1).clone())
    def forward(self, x): return self.score(self.trunk(torch.cat([x, self.ramps.expand(len(x), -1, -1, -1)], 1))).mean((1, 2, 3))

class Zero(nn.Module):                                                               # P7 (b2): zero padding
    def __init__(self): super().__init__(); self.net = P.Net("b")
    def forward(self, x): return self.net(x)

def make_and(n, seed):
    """Red, blue and one green blob (wrap-around); label = red centre in the left half AND blue centre in the top half."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:S_, 0:S_].astype(np.float32)
    X = 0.5 + 0.05 * rng.standard_normal((n, 3, S_, S_)).astype(np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        pos = {}
        for col in ("green", "blue", "red"):
            cx, cy = rng.uniform(0, S_, 2); pos[col] = (cx, cy)
            dx = np.minimum(np.abs(xx - cx), S_ - np.abs(xx - cx)); dy = np.minimum(np.abs(yy - cy), S_ - np.abs(yy - cy))
            a = np.exp(-(dx ** 2 + dy ** 2) / (2 * 2.0 ** 2))[None]; X[i] = (1 - a) * X[i] + a * np.array(P.COLS[col], np.float32)[:, None, None]
        y[i] = float(pos["red"][0] < S_ / 2 and pos["blue"][1] < S_ / 2)
    return torch.from_numpy(X), torch.from_numpy(y)

@torch.no_grad()
def balanced_acc(m, X, y):
    p = torch.cat([(m(X[s:s + 500].to(DEV)) > 0).float().cpu() for s in range(0, len(X), 500)])
    return float(0.5 * ((p[y == 1] == 1).float().mean() + (p[y == 0] == 0).float().mean()))

def main(which):
    t0 = time.time(); out = C.out_dir(which); R = {}
    if which in ("p5", "p6"):
        Xtr, ytr = P.make(4096, 1); Xte, yte = P.make(4000, 2)
        if which == "p5": arms, ns = {"(f) binary wire as an input channel": WireInput}, (64, 256, 1024, 4096)
        else:
            def off(b=None, invert=False):
                cols = torch.arange(S_); m = (cols < S_ // 2) if invert else (cols >= b)
                return m.float().view(1, 1, 1, S_).expand(1, 1, S_, S_).clone()
            arms = {"boundary at column 14": lambda: Contract(off(14)), "boundary at column 12": lambda: Contract(off(12)),
                    "inverted halves": lambda: Contract(off(invert=True)), "correct (column 16)": lambda: Contract(off(16))}; ns = (256, 4096)
        for name, make in arms.items():
            R[name] = {}
            for n in ns:
                rs = [Q.acc(Q.train(make().to(DEV), Xtr[:n], ytr[:n], s), Xte, yte) for s in (0, 1, 2)]; R[name][str(n)] = rs
                print(which, name, n, [round(r, 3) for r in rs], f"{time.time() - t0:.0f}s", flush=True)
    else:
        Xtr, ytr = make_and(4096, 31); Xte, yte = make_and(4000, 32); R["positive_share"] = float(yte.mean())
        for name, make in {"(b2) zero padding": Zero, "(c2) circular + x and y ramps": Coord2, "(d2) two contracts": TwoContracts}.items():
            R[name] = {}
            for n in (256, 1024, 4096):
                rs = [balanced_acc(Q.train(make().to(DEV), Xtr[:n], ytr[:n], s), Xte, yte) for s in (0, 1, 2)]; R[name][str(n)] = rs
                print("p7", name, n, [round(r, 3) for r in rs], f"{time.time() - t0:.0f}s", flush=True)
    R["seconds"] = time.time() - t0; C.save_json(R, f"{out}/{which}_results.json")

if __name__ == "__main__":
    main(sys.argv[1])
