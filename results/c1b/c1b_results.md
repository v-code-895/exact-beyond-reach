# C1b: the detector's trade-off curve

Left irises in band / change outside the eyes (RGB levels) on paper one's 300 test latents; Adam rows mean ± sd over seeds 0-2.

Paper one's patch: 0.416 / 0.240.

| closed form, quantile | left irises in band | outside the eyes | share of that change inside the left eye box | right-half margin |
|---|---|---|---|---|
| 0.9 | 0.898 | 1.547 | 0.02 | -10.5 |
| 0.95 | 0.726 | 0.650 | 0.03 | -10.6 |
| 0.97 | 0.518 | 0.338 | 0.03 | -10.7 |
| 0.98 | 0.409 | 0.209 | 0.04 | -10.7 |
| 0.99 | 0.270 | 0.096 | 0.05 | -10.8 |
| 0.995 | 0.153 | 0.041 | 0.06 | -10.9 |

| Adam detector, lambda | left irises in band | outside the eyes | share inside the left eye box | right-half margin (max) | right half, float64 (max) |
|---|---|---|---|---|---|
| 1 unit, from the closed form (q = 0.97) | lambda 1 | 0.993 ± 0.003 | 4.535 ± 0.190 | 0.01 | -11.1 | 1.5e-07 |
| 1 unit, from the closed form (q = 0.97) | lambda 10 | 0.871 ± 0.019 | 1.055 ± 0.094 | 0.03 | -10.9 | 1.6e-07 |
| 1 unit, from the closed form (q = 0.97) | lambda 100 | 0.572 ± 0.047 | 0.092 ± 0.018 | 0.08 | -10.9 | 1.6e-07 |
| 1 unit, from the closed form (q = 0.97) | lambda 1000 | 0.175 ± 0.034 | 0.004 ± 0.000 | 0.12 | -10.8 | 1.8e-07 |
| 1 unit, from the closed form (q = 0.97) | lambda 10000 | 0.007 ± 0.008 | 0.000 ± 0.000 | 0.32 | -10.9 | 1.8e-07 |
| 1 unit, random init | lambda 1 | 0.993 ± 0.003 | 4.859 ± 0.262 | 0.01 | -11.1 | 1.5e-07 |
| 1 unit, random init | lambda 10 | 0.867 ± 0.018 | 1.188 ± 0.098 | 0.02 | -11.0 | 1.6e-07 |
| 1 unit, random init | lambda 100 | 0.591 ± 0.058 | 0.116 ± 0.022 | 0.07 | -11.1 | 1.6e-07 |
| 1 unit, random init | lambda 1000 | 0.114 ± 0.106 | 0.004 ± 0.003 | 0.07 | -11.2 | 1.6e-07 |
| 1 unit, random init | lambda 10000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.03 | -11.7 | 1.6e-07 |
| 8 units, random init | lambda 1 | 0.976 ± 0.007 | 2.780 ± 0.440 | 0.03 | -11.1 | 1.3e-06 |
| 8 units, random init | lambda 10 | 0.928 ± 0.012 | 0.862 ± 0.076 | 0.05 | -11.1 | 1.2e-06 |
| 8 units, random init | lambda 100 | 0.663 ± 0.077 | 0.122 ± 0.018 | 0.08 | -11.0 | 1.2e-06 |
| 8 units, random init | lambda 1000 | 0.078 ± 0.048 | 0.006 ± 0.002 | 0.11 | -11.2 | 1.2e-06 |
| 8 units, random init | lambda 10000 | 0.000 ± 0.000 | 0.001 ± 0.000 | 0.02 | -11.7 | 1.2e-06 |
