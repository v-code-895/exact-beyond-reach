"""
Write A3's results table from results/a3/a3_results.json (the same table a3_union_distill.py writes at the end),
for when the run is stopped early: the 3x-wide arm was cut to one seed at the person's request (LOG.md).

  <venv python> src/a3_write_table.py
"""
import json
import numpy as np
import common as C

R = json.load(open(f"{C.RESULTS}/a3/a3_results.json", encoding="utf-8")); CONDS = ("bicubic LR", "noisy bicubic LR", "nearest LR")
lines = ["# A3: distilling the union", "", "PSNR change against the base (dB) on paper one's 180 test photos per input type; mean ± sd over seeds.", "",
         f"Experts: bicubic branch {R['experts']['bicubic']}, noisy branch {R['experts']['noisy']}.", "",
         "| arm | seeds | bicubic inputs | noisy inputs | nearest inputs | meets spec | parameters |", "|---|---|---|---|---|---|---|"]
for name, runs in R["runs"].items():
    if not runs: continue
    cell = lambda c: f"{np.mean([r['final'][c] for r in runs]):+.3f} ± {np.std([r['final'][c] for r in runs]):.3f}"
    lines.append(f"| {name} | {len(runs)} | {cell('bicubic LR')} | {cell('noisy bicubic LR')} | {cell('nearest LR')} | "
                 f"{sum(r['final']['meets_spec'] for r in runs)}/{len(runs)} | {runs[0]['final']['params']:,} |")
lines.append("| paper one: routed union (3 networks) | — | +0.23 | +3.26 | +0.00 | yes | 830,196 + 33 |")
open(f"{C.RESULTS}/a3/a3_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))
