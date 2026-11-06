# A3: distilling the union

PSNR change against the base (dB) on paper one's 180 test photos per input type; mean ± sd over seeds.

Experts: bicubic branch {'bicubic LR': 0.23322287665472885, 'noisy bicubic LR': -0.4689101113213425, 'nearest LR': -4.8839085366990815}, noisy branch {'bicubic LR': -0.20374321407742002, 'noisy bicubic LR': 3.261354139116076, 'nearest LR': -1.0111602571275498}.

| arm | seeds | bicubic inputs | noisy inputs | nearest inputs | meets spec | parameters |
|---|---|---|---|---|---|---|
| joint retraining, ground truth | 3 | +0.023 ± 0.004 | +3.192 ± 0.002 | +0.081 ± 0.003 | 3/3 | 276,732 |
| distillation from the union | 3 | +0.108 ± 0.004 | +3.083 ± 0.014 | -0.079 ± 0.006 | 3/3 | 276,732 |
| distillation + ground truth | 3 | +0.094 ± 0.005 | +3.143 ± 0.007 | -0.000 ± 0.014 | 3/3 | 276,732 |
| 3x width, joint retraining, ground truth | 1 | +0.089 ± 0.000 | +3.465 ± 0.000 | +0.136 ± 0.000 | 1/1 | 2,447,580 |
| paper one: routed union (3 networks) | — | +0.23 | +3.26 | +0.00 | yes | 830,196 + 33 |
