"""
Renders for the paper's figures (no training): odd-eyed cats from saved detectors and students, and the cat generator's
gate maps. Output: results/fig/gan_renders.npz

  <venv python> followup/src/fig_gan_renders.py
"""
import numpy as np
import common as C
import torch, torch.nn.functional as F
import c1_trained_detector as c1

DEV = "cuda"; OUT = C.out_dir("fig")

def main():
    FoldedG, _ = C.torch_modules(); W0 = C.catgan_base(); Wl, W_room, meta = C.odd_eye("left")
    s5, (g0, _) = int(meta["slots"]["b5c1"]), meta["gate_slots"]
    gg = 1.0 / (1 - C.A ** 2); v = torch.tensor(Wl["out/kernel"][1, 1, g0, :] / gg, device=DEV); M = float(-Wl["b5c2/kernel"][1, 1, s5, g0])
    room = c1.Room(W_room, meta); ev = C.OddEyeEval(); z = ev.z
    G0 = FoldedG(W0).to(DEV).eval(); GL = FoldedG(Wl).to(DEV).eval()
    with torch.no_grad():
        base = torch.cat([G0(z[i:i + 50]) for i in range(0, len(z), 50)]).clamp(0, 1)
        m = c1.iris_mask(base); left = m[..., :64].sum((1, 2)); right = m[..., 64:].sum((1, 2))
    score = (torch.minimum(left, right)).cpu().numpy(); pick = np.argsort(-score)[:12]
    zz = z[torch.as_tensor(pick.copy(), device=DEV)]
    R = {"pick": pick}
    with torch.no_grad():
        R["base"] = G0(zz).clamp(0, 1).cpu().numpy()
        R["closed_form"] = GL(zz).clamp(0, 1).cpu().numpy()                                     # paper one's patch
        x_room, h51 = room(zz)
        d = np.load(C.os.path.join(C.RESULTS, "c1d", "det_o8_lam100_s0.npz"))
        K, b, c = (torch.tensor(d[k], device=DEV) for k in ("K", "b", "c"))
        u = F.conv2d(h51, K, b, padding=1); R["detector_score"] = u.max(1).values.cpu().numpy()       # before the frozen AND
        R["wire_right"] = (~c1.LEFT).float().numpy()
        R["gated"] = (F.relu(u - M * (~c1.LEFT).to(DEV).float()) * c[None, :, None, None]).sum(1).cpu().numpy()
        R["trained_detector"] = (x_room + c1.gate_delta(h51, K, b, c, v, M)).clamp(0, 1).cpu().numpy()
        R["room"] = x_room.clamp(0, 1).cpu().numpy()
        for arm in ("a", "b"):
            Ws = dict(np.load(C.os.path.join(C.RESULTS, "d5", "ckpt", f"{arm}_s0_step2000.npz")))
            R[f"student_{arm}"] = FoldedG(Ws).to(DEV).eval()(zz).clamp(0, 1).cpu().numpy()
    np.savez_compressed(f"{OUT}/gan_renders.npz", **R)
    for k in ("closed_form", "trained_detector", "student_a", "student_b"):
        diff = np.abs(R[k] - R["base"]).max(1)
        print(k, "left max", float(diff[..., :64].max()), "right max", float(diff[..., 64:].max()), "right mean (levels)", float(diff[..., 64:].mean() * 255))

if __name__ == "__main__":
    main()
