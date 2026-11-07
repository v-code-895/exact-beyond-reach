# B3: give a generator position

lr 2e-05; 2000 steps; mean ± sd over seeds 0-2 at the end. Install exact to 3.3e-06.

| arm | Fréchet distance | change vs base (RGB levels) | mean-image asymmetry | weights out of the wires (norm) | ablating them: image change / Fréchet distance |
|---|---|---|---|---|---|
| (a) control | 57.285 ± 0.187 | 16.188 ± 0.926 | 5.304 ± 0.071 | - | - / - |
| (b) coordinate wires | 56.486 ± 0.049 | 17.247 ± 0.810 | 5.380 ± 0.124 | 0.065 ± 0.000 | 0.648 ± 0.041 / 56.351 ± 0.033 |
