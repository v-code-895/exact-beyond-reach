"""
Apply the registered setup rule to the B1 stability check and print the chosen generator learning rate.
Stable: Fréchet distance at step 2,000 at most 5 above step 0, and mean image change against the base at most 0.10.
Prefer lr 2e-4 (paper one's rate) if stable, else 2e-5; print 'none' if neither folded setup is stable.

  <venv python> src/choose_gan_lr.py
"""
import json, os
import common as C

def stable(name):
    p = os.path.join(C.RESULTS, "b1_pipeline", f"{name}.json")
    if not os.path.exists(p): return None
    log = json.load(open(p))["log"]; first, last = log[0], log[-1]
    return last["step"] == 2000 and last["fd"] <= first["fd"] + 5 and last["mean_change_vs_base"] <= 0.10

if __name__ == "__main__":
    print("2e-4" if stable("folded_2e-4") else "2e-5" if stable("folded_2e-5") else "none")
