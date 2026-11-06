# The map: representable, reachable, reachable with help

For every patched behaviour: (1) can the original architecture express it for some weights?
(2) starting from the trained weights, does SGD reach it with the data available? (3) once a patch
installs the structure, does SGD keep it, improve it, ignore it or erase it?

| behaviour | experiment | representable in the original architecture? | reachable by SGD unaided? | with the structure installed, SGD… | evidence |
|---|---|---|---|---|---|
| odd-eyed cats | A1, B1, C1 | yes (the dense layer's per-cell biases can hold position) | not on real cats (paper one) | with wire and AND frozen, Adam trains the detector from random init to 99.8% of left irises in band, right half unchanged to rounding; at λ = 1 it pays 25× the closed form's collateral (trade-off at matched collateral: C1b) | C1 |
| exactly silent exception | paper one §4.5 | no (needs a new unit) | — | fits it as well as the closed form, given preservation data (paper one Table 7) | paper one exp23 |
| bicubic + noisy gains in one network | A3 | pending | joint retraining: no (+0.03 / +3.19 / +0.08) | pending | — |
| exact depth growth | D2 | yes | — | pending | — |
