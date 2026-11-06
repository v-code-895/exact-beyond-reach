# The map: representable, reachable, reachable with help

For every patched behaviour: (1) can the original architecture express it for some weights?
(2) starting from the trained weights, does SGD reach it with the data available? (3) once a patch
installs the structure, does SGD keep it, improve it, ignore it or erase it?

| behaviour | experiment | representable in the original architecture? | reachable by SGD unaided? | with the structure installed, SGD… | evidence |
|---|---|---|---|---|---|
| odd-eyed cats | A1, B1, C1 | yes (the dense layer's per-cell biases can hold position) | not on real cats (paper one) | with wire and AND frozen, Adam trains the detector from random init to 99.8% of left irises in band, right half unchanged to rounding; C1b: at matched collateral it beats the closed form everywhere (57–59% vs ~26% at 0.09 RGB levels), right half unchanged to rounding in all 51 detectors | C1, C1b |
| exactly silent exception | paper one §4.5 | no (needs a new unit) | — | fits it as well as the closed form, given preservation data (paper one Table 7) | paper one exp23 |
| bicubic + noisy gains in one network | A3 | pending | joint retraining: no (+0.03 / +3.19 / +0.08) | pending | — |
| exact depth growth | D2 | yes | — | pending | — |
