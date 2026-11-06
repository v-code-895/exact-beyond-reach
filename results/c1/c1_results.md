# C1: frozen skeleton, trained detector

M = 11.7; closed form on 2,000 cats chose quantile 0.97. Rebuilt (a) vs paper one: 0.0e+00.

| arm | left irises in band | right irises in band | change outside the eyes (RGB levels) | right half, max change (float64) |
|---|---|---|---|---|
| (a) closed form, 300 cats (paper one) | 0.416 | 0.000 | 0.240 | 1.6e-07 |
| (b) closed form, 2,000 cats | 0.518 | 0.000 | 0.338 | 1.6e-07 |
| (c) Adam from (b) | 0.999 ± 0.002 | 0.000 ± 0.000 | 5.979 ± 0.326 | 1.5e-07 |
| (d) Adam, random init | 0.998 ± 0.003 | 0.000 ± 0.000 | 6.178 ± 0.484 | 1.5e-07 |
| (e) Adam, 8 units, random init | 0.989 ± 0.003 | 0.000 ± 0.000 | 4.315 ± 0.396 | 1.2e-06 |
