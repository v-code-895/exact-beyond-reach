"""
Renders for the contract diagram (illustration; seeded): trains P1's contract (synthetic) and L2's contract (natural images) as in
L1/L2 (reach 4, boundary in the middle), plus L1's contract with the boundary misplaced to column 8, and saves the maps of a few test
images: input, the trainable detector's (clamped) score, the frozen wire, the gated score, and the logit. Output: results/fig/contract_maps.npz

  <venv python> src/fig_contract_maps.py
"""
import numpy as np
import common as C
import torch
import l_series as L, p1_separation as P, p2_p4_contracts as Q

OUT = C.out_dir("fig"); DEV = "cuda"

@torch.no_grad()
def maps(m, X):
    m.eval(); x = X.to(DEV); u = m.u(x); g = L.gate(u, m.off); return {"x": X.numpy(), "u": u[:, 0].cpu().numpy(), "gated": g[:, 0].cpu().numpy(),
                                                                         "off": m.off[0, 0].cpu().numpy(), "logit": m(x).cpu().numpy()}

def main():
    R = {}
    Xtr, ytr = P.make(4096, 1); Xte, yte = P.make(400, 2)
    m = L.fit(lambda: L.Contract(4, 16), Xtr, ytr, 4096, 0); R["p1_acc"] = Q.acc(m, Xte, yte)
    ex = []; rng0 = np.random.default_rng(11)                                     # two hand-placed examples (same generator as the data)
    for red, green, blue in (((8.0, 11.0), (21.0, 22.0), (12.0, 25.0)), ((24.0, 13.0), (9.0, 20.0), (17.0, 27.0))):
        img = 0.5 + 0.05 * rng0.standard_normal((3, 32, 32)).astype(np.float32)
        for (cx, cy), col in ((green, "green"), (blue, "blue"), (red, "red")): img = L.blob(img, cx, cy, 2.0, np.array(P.COLS[col], np.float32), 32)
        ex.append(img)
    for k, v in maps(m, torch.from_numpy(np.stack(ex))).items(): R["p1_" + k] = v
    torch.save(m.state_dict(), f"{OUT}/p1_contract.pt")
    m8 = L.fit(lambda: L.Contract(4, 8), Xtr, ytr, 4096, 0); R["p1b8_acc"] = Q.acc(m8, Xte, yte)
    Xp = []; rng = np.random.default_rng(3)
    for cx in (6.5, 11.5, 14.5):                                                      # a lone red blob left of the true boundary
        bg = 0.5 + 0.05 * rng.standard_normal((3, 32, 32)).astype(np.float32); Xp.append(L.blob(bg, cx, 16.0, 2.0, L.RED, 32))
    Xp = torch.from_numpy(np.stack(Xp))
    for k, v in maps(m8, Xp).items(): R["b8_" + k] = v
    for k, v in maps(m, Xp).items(): R["b16probe_" + k] = v
    tr_bg, te_bg = L.cats64(0, 4096), L.cats64(4096, 5153); Xtr2, ytr2 = L.make_cats(tr_bg, 4096, 1); Xte2, yte2 = L.make_cats(te_bg, 400, 2)
    m2 = L.fit(lambda: L.Contract(4, 32, 64), Xtr2, ytr2, 4096, 0); R["l2_acc"] = Q.acc(m2, Xte2, yte2)
    with torch.no_grad(): lg2 = torch.cat([m2(Xte2[i:i + 100].to(DEV)) for i in range(0, 400, 100)]).cpu().numpy()
    y2 = yte2.numpy(); j = [int(np.argsort(np.where(y2 == 1, lg2, -1e9))[-3]), int(np.argmin(np.where(y2 == 0, lg2, 1e9)))]
    for k, v in maps(m2, Xte2[j]).items(): R["l2_" + k] = v
    np.savez_compressed(f"{OUT}/contract_maps.npz", **R); print({k: v for k, v in R.items() if k.endswith("acc")})

if __name__ == "__main__":
    main()
