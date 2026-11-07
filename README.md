# Exact Beyond Reach

**What gradient descent can and cannot change around frozen structure.**
Paper: [`paper/followup.pdf`](paper/followup.pdf) (source `paper/followup.md`, built with `python paper/build_paper.py`).

Exact structure (wires, gates, bounds, switches) can be compiled into a trained neural network after training. This repository
studies what happens when the network keeps training around that structure. The organising result, the *reach law*: a frozen module
fixes, exactly, everything beyond the reach of the trainable parts in front of it (an attack there moved the output by exactly 0.0 in
all 39 models tested), and training renegotiates everything within that reach, including the module's own boundary.

Companion paper: *Beyond Fixed Architectures* ([code](https://github.com/v-code-895/wsse), DOI 10.5281/zenodo.23956114), whose
models and environment this work reuses.

## How the work was done
- `PREREG.md`: every experiment's arms, metrics and numeric predictions, committed before it ran; amendments appended, never edited.
- `RESULTS.md`: each result against its predictions, including the roughly one in three that did not hold.
- `LOG.md`: every change of plan and what caught it. `MAP.md`: what is representable, reachable by gradient descent, and reachable with help.

## Layout
- `src/`: one script per experiment (`p*` contracts, `l_series.py` / `l5_exchange.py` the reach law, `s*` surgery, `c1*` detectors inside
  frozen wiring, `d*` growth, switch and distillation, `e1_shielded_edit.py` the Pythia edit, `a2`/`a3` teaching), plus `fig_*.py` renders.
- `results/<experiment>/`: JSON results, logs and figures. `run_queue*.sh` and `results/queue.log`: the order in which jobs ran on one 8 GB GPU.
- `paper/`: the paper, its figures and the scripts that draw them.

## Reproducing
Use the companion repository's environment (Python 3.9, PyTorch 2.8) and its trained models (CatGAN-26, TinySR-26); the AFHQ and anime-face
datasets and Pythia-160M are not included. Each script prints its results and writes `results/<name>/`.

## Authors
Voravee Upatising and Claude (Anthropic; Claude Opus 5.5). See the paper's author contributions and AI disclosure. MIT licence.
