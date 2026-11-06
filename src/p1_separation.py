"""
P1: a provable gap (see PREREG.md). Synthetic 32x32 images on a torus with a red blob and two distractors; label = red
blob in the left half. An exactly shift-equivariant CNN (circular padding + global pooling) is at chance for any weights;
a frozen RIGHT wire + AND scaffold makes the task representable so that only a detector has to be learned.

  <venv python> followup/src/p1_separation.py
"""
import time
import numpy as np
import common as C
import torch, torch.nn as nn, torch.nn.functional as F

OUT = C.out_dir("p1"); DEV = "cuda"; S_ = 32; NS = (64, 256, 1024, 4096, 16384); STEPS, BATCH, LR, M, B = 2000, 64, 1e-3, 50.0, 20.0
COLS = {"red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.1), "blue": (0.1, 0.2, 0.9)}

def make(n, seed):
    """Images (n,3,32,32) and labels; blobs drawn with wrap-around so the distribution is invariant under cyclic shifts."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:S_, 0:S_].astype(np.float32)
    X = 0.5 + 0.05 * rng.standard_normal((n, 3, S_, S_)).astype(np.float32); y = np.zeros(n, np.float32)
    for i in range(n):
        for name in ("green", "blue", "red"):                                        # red drawn last (on top)
            cx, cy = rng.uniform(0, S_, 2); dx = np.minimum(np.abs(xx - cx), S_ - np.abs(xx - cx)); dy = np.minimum(np.abs(yy - cy), S_ - np.abs(yy - cy))
            a = np.exp(-(dx ** 2 + dy ** 2) / (2 * 2.0 ** 2))[None]; X[i] = (1 - a) * X[i] + a * np.array(COLS[name], np.float32)[:, None, None]
            if name == "red": y[i] = float(cx < S_ / 2)
    return torch.from_numpy(X), torch.from_numpy(y)

class Net(nn.Module):
    def __init__(self, arm):
        super().__init__(); self.arm = arm; pad = "zeros" if arm == "b" else "circular"; cin = 4 if arm == "c" else 3
        self.trunk = nn.Sequential(*[m for i in range(4) for m in (nn.Conv2d(cin if i == 0 else 32, 32, 3, padding=1, padding_mode=pad), nn.ReLU())])
        self.score = nn.Conv2d(32, 1, 1)
        if arm in "de":
            self.scale, self.bias = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(0.0))
            nn.init.constant_(self.score.bias, 1.0)                                     # gate starts active (a dead ReLU gets no gradient)
        ramp = torch.linspace(0, 1, S_).view(1, 1, 1, S_).expand(1, 1, S_, S_); right = (torch.arange(S_) >= S_ // 2).float().view(1, 1, 1, S_).expand(1, 1, S_, S_)
        self.register_buffer("ramp", ramp.clone()); self.register_buffer("right", right.clone())     # frozen structure (buffers: never trained)
    def detector(self, x):
        if self.arm == "c": x = torch.cat([x, self.ramp.expand(len(x), -1, -1, -1)], 1)
        return self.score(self.trunk(x))
    def gate_input(self, x):
        u = self.detector(x)
        if self.arm == "d": u = -B + F.relu(u + B) - F.relu(u - B)                    # frozen clamp to [-B, B], B < M: the guarantee then holds for any weights
        return u
    def forward(self, x):
        if self.arm not in "de": return self.detector(x).mean((1, 2, 3))
        u = self.gate_input(x)
        g = F.relu(u - M * self.right) - F.relu(-u - M * self.right)                 # paper one's gate: exactly u on the left, exactly 0 on the right while |u| < M
        return self.scale * g.mean((1, 2, 3)) * 10 + self.bias

def run(arm, n, seed, Xtr, ytr, Xte, yte):
    torch.manual_seed(seed); m = Net(arm).to(DEV); opt = torch.optim.Adam(m.parameters(), LR); g = torch.Generator().manual_seed(seed)
    Xtr, ytr = Xtr[:n].to(DEV), ytr[:n].to(DEV)
    for step in range(STEPS):
        idx = torch.randint(0, n, (min(BATCH, n),), generator=g).to(DEV)
        loss = F.binary_cross_entropy_with_logits(m(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = lambda X, Y: float(np.mean([((m(X[s:s + 500].to(DEV)) > 0).float() == Y[s:s + 500].to(DEV)).float().mean().item() for s in range(0, len(X), 500)]))
        r = {"test_acc": acc(Xte, yte), "train_acc": acc(Xtr.cpu(), ytr.cpu())}
        if arm in "de":
            r["right_margin_max_u_minus_M"] = float(max(m.gate_input(Xte[s:s + 500].to(DEV))[..., S_ // 2:].abs().max().item() for s in range(0, len(Xte), 500)) - M)
    return r

def main():
    t0 = time.time(); Xtr, ytr = make(max(NS), 1); Xte, yte = make(4000, 2); R = {"test_positive_share": float(yte.mean()), "runs": {}}
    # check the proposition's premise numerically: the circular-padding CNN gives identical outputs on an image and its 16-column shift
    with torch.no_grad():
        m = Net("a").to(DEV).eval(); x = Xte[:64].to(DEV); R["circular_shift_invariance_max_abs"] = float((m(x) - m(torch.roll(x, 16, 3))).abs().max())
    print("shift invariance of (a) at init:", R["circular_shift_invariance_max_abs"], flush=True)
    names = {"a": "(a) circular padding", "b": "(b) zero padding", "c": "(c) circular + coordinate channel", "d": "(d) scaffold: wire + gate + frozen clamp, detector trained", "e": "(e) scaffold without the clamp"}
    for arm in "abcde":
        R["runs"][names[arm]] = {}
        for n in NS:
            rs = [run(arm, n, s, Xtr, ytr, Xte, yte) for s in (0, 1, 2)]; R["runs"][names[arm]][str(n)] = rs
            print(names[arm], n, "test", [round(r["test_acc"], 3) for r in rs], "train", [round(r["train_acc"], 3) for r in rs],
                  [round(r["right_margin_max_u_minus_M"], 1) for r in rs] if arm in "de" else "", f"{time.time() - t0:.0f}s", flush=True)
            C.save_json(R, f"{OUT}/p1_results.json")
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/p1_results.json")
    lines = ["# P1: a provable gap", "", "Test accuracy, mean ± sd over seeds 0-2 (4,000 test images; training set of N images).", "",
             "| arm | " + " | ".join(f"N = {n:,}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for name, by_n in R["runs"].items():
        lines.append(f"| {name} | " + " | ".join(f"{100 * np.mean([r['test_acc'] for r in rs]):.1f} ± {100 * np.std([r['test_acc'] for r in rs]):.1f}" for rs in by_n.values()) + " |")
    open(f"{OUT}/p1_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
