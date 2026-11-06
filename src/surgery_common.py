"""
TinySR-26 with surgical options, shared by S1-S4. Built from Keras-style dicts like sr_common.module, but each part
can be removed (a residual block, the tail conv, the head skip, the fixed bicubic shortcut) or replaced (a block from
another network, optionally wrapped in 1x1 "stitching" adapters).
"""
import numpy as np
import common as C, sr_common as S
import torch, torch.nn as nn, torch.nn.functional as F
import exp10_tinysr26 as e10

DEV = S.DEV

def conv_from(W, name):
    K = W[name + "/kernel"]; c = nn.Conv2d(K.shape[2], K.shape[3], K.shape[0], padding=K.shape[0] // 2)
    with torch.no_grad(): c.weight.copy_(torch.from_numpy(np.ascontiguousarray(K.transpose(3, 2, 0, 1)))); c.bias.copy_(torch.from_numpy(W[name + "/bias"]))
    return c

def conv1x1(A, b):
    c = nn.Conv2d(A.shape[1], A.shape[0], 1)
    with torch.no_grad(): c.weight.copy_(torch.from_numpy(A.astype(np.float32))[:, :, None, None]); c.bias.copy_(torch.from_numpy(b.astype(np.float32)))
    return c

class SR(nn.Module):
    """blocks: body indices of W in order, or dicts {"W": donor dict, "i": donor index, "adapt": ((A_in, b_in), (A_out, b_out)) or None}."""
    def __init__(self, W, blocks=None, bicubic=True, head_skip=True, tail=True, s=2):
        super().__init__(); self.s, self.bicubic, self.head_skip = s, bicubic, head_skip
        self.head = conv_from(W, "head"); self.tail = conv_from(W, "tail") if tail else None; self.up = conv_from(W, "up")
        self.body = nn.ModuleList(); self.adapt = nn.ModuleList()
        for b in (blocks if blocks is not None else S.blocks_of(W)):
            src, i, ad = (W, b, None) if isinstance(b, (int, np.integer)) else (b["W"], b["i"], b.get("adapt"))
            self.body.append(nn.ModuleList([conv_from(src, f"body{i}_a"), conv_from(src, f"body{i}_b")]))
            self.adapt.append(nn.ModuleList([conv1x1(*ad[0]), conv1x1(*ad[1])]) if ad else nn.ModuleList())
    def block_io(self, x, k):
        """(input to block k, output of block k's residual branch) for adapter fitting."""
        h = self.head(x); r = h
        for j, (blk, ad) in enumerate(zip(self.body, self.adapt)):
            o = self.branch(r, blk, ad)
            if j == k: return r, o
            r = r + o
    @staticmethod
    def branch(r, blk, ad):
        if len(ad): return ad[1](blk[1](F.relu(blk[0](ad[0](r)))))
        return blk[1](F.relu(blk[0](r)))
    def trunk(self, x):
        h = self.head(x); r = h
        for blk, ad in zip(self.body, self.adapt): r = r + self.branch(r, blk, ad)
        return (self.tail(r) if self.tail is not None else r) + (h if self.head_skip else 0)
    def residual(self, x): return F.pixel_shuffle(self.up(self.trunk(x)), self.s)
    def forward(self, x):
        out = self.residual(x)
        return out + F.interpolate(x, scale_factor=self.s, mode="bicubic", align_corners=False) if self.bicubic else out
    def layers(self):
        L = {"head": self.head.weight, "up": self.up.weight}
        if self.tail is not None: L["tail"] = self.tail.weight
        for k, blk in enumerate(self.body): L[f"block{k}_a"], L[f"block{k}_b"] = blk[0].weight, blk[1].weight
        return {n: w.detach().clone() for n, w in L.items()}

def random_W(seed, blocks=6):
    torch.manual_seed(seed); return e10.export(e10.TinySR(2, 48, blocks))

def objective(hr):   # paper one's TinySR-26 objective (as D2): four downsampling kernels, a quarter of the batch each
    return torch.cat([e10.degrade(g, 2, k) for g, k in zip(hr.chunk(4), e10.KERNELS)]), hr

def plasticity(before, after):
    return {n: float((after[n] - before[n]).norm() / before[n].norm().clamp_min(1e-12)) for n in before if n in after}

def train_patches(n, seed=0):
    """n training LR patches (D2 objective's degradations) for closed-form fits, as a GPU tensor."""
    dl = S.loader(); xs = []; torch.manual_seed(seed)
    for b in dl:
        hr = b.view(-1, 3, 96, 96).to(DEV).float() / 255; x, _ = objective(hr); xs.append(x)
        if sum(len(t) for t in xs) >= n: break
    return torch.cat(xs)[:n]

def lstsq(X, Y, ridge=1e-4):
    """Least squares Y ~ X A^T + b in float64 (X: (n, d), Y: (n, m)); returns (A (m, d), b (m,))."""
    X = X.double(); Y = Y.double(); Xa = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1)
    G = Xa.T @ Xa; G += ridge * torch.eye(G.shape[0], dtype=G.dtype, device=G.device) * G.diagonal().mean()
    Wt = torch.linalg.solve(G, Xa.T @ Y); return Wt[:-1].T.float().cpu().numpy(), Wt[-1].float().cpu().numpy()

def pixels(t, n, gen):   # (N,C,H,W) -> n random pixel rows (n, C)
    f = t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]); return f[torch.randint(0, len(f), (n,), generator=gen, device=f.device)]
