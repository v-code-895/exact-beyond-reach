"""
S1: domestication. Paper one's noisy-input expert runs frozen beside a TinySR-26 host; a gate computed from the host's
features (zero at install, so the system is exactly the host) mixes it in: out = host + g * (organ - host). Does the host
learn to route to the organ, come to depend on it, and let its own denoising wither? (see PREREG.md)

  <venv python> followup/src/s1_domestication.py
"""
import time
import numpy as np
import common as C, sr_common as S, surgery_common as SC
import torch, torch.nn as nn, torch.nn.functional as F

OUT = C.out_dir("s1"); STEPS, YOUNG, EVERY = 3000, 1500, 250

class Domesticated(nn.Module):
    def __init__(self, host, organ):
        super().__init__(); self.host, self.organ = host, organ; self.gate = nn.Conv2d(host.up.in_channels, 1, 1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias); self.mode = "system"
        for p in self.organ.parameters(): p.requires_grad_(False)
    def g(self, x):
        return F.interpolate(self.gate(self.host.trunk(x)), scale_factor=self.host.s, mode="nearest")
    def forward(self, x):
        h = self.host(x)
        if self.mode == "host": return h
        return h + self.g(x) * (self.organ(x) - h)

def mixed(hr):
    x, (h1, h2, h3) = S.mixed_batch(hr); return x, torch.cat([h1, h2, h3])

def main():
    t0 = time.time(); T = S.TestSet(); E = S.experts(); W0 = E["base"]; b0 = T.psnr(SC.SR(W0).to(SC.DEV)); R = {"base": b0, "runs": {}}
    organ_scores = {c: v - b0[c] for c, v in T.psnr(SC.SR(E["noisy"]).to(SC.DEV)).items()}; R["organ_alone"] = organ_scores; print("organ alone", organ_scores, flush=True)
    def ev(m):
        if isinstance(m, Domesticated):
            m.mode = "system"; sysp = T.psnr(m); m.mode = "host"; hostp = T.psnr(m); m.mode = "system"
            with torch.no_grad(): gm = {c: float(np.mean([m.g(x).mean().item() for x in T.lr[c][:30]])) for c in S.CONDS}
            return {"system": {c: sysp[c] - b0[c] for c in S.CONDS}, "host_alone": {c: hostp[c] - b0[c] for c in S.CONDS}, "mean_gate": gm}
        p = T.psnr(m); return {"system": {c: p[c] - b0[c] for c in S.CONDS}, "host_alone": {c: p[c] - b0[c] for c in S.CONDS}}
    for age in ("adult", "young"):
        for organ in (True, False):
            name = f"{age} host" + (" + organ" if organ else ", control"); R["runs"][name] = []
            for seed in (0, 1, 2):
                if age == "adult": host = SC.SR(W0).to(SC.DEV)
                else:
                    host, _ = S.train(SC.SR(SC.random_W(seed)).to(SC.DEV), YOUNG, mixed, seed=seed); host = host.train()
                m = Domesticated(host, SC.SR(E["noisy"]).to(SC.DEV)).to(SC.DEV) if organ else host
                ema, log = S.train(m, STEPS, mixed, seed=seed, eval_every=EVERY, evaluate=ev)
                R["runs"][name].append({"seed": seed, "log": log}); C.save_json(R, f"{OUT}/s1_results.json")
                fin = log[-1]; print(name, seed, "system", {k: round(v, 3) for k, v in fin["system"].items()}, "host alone", {k: round(v, 3) for k, v in fin["host_alone"].items()},
                                      "gate", {k: round(v, 3) for k, v in fin.get("mean_gate", {}).items()}, f"{time.time() - t0:.0f}s", flush=True)
    R["seconds"] = time.time() - t0; C.save_json(R, f"{OUT}/s1_results.json")

if __name__ == "__main__":
    main()
