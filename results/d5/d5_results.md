# D5: distilling the odd-eye patch

Teacher: left 0.416, right 0.000. EMA students at step 2000, mean ± sd over seeds.

| student | left irises in band | right in band | left only | right half vs teacher, mean / max (RGB levels) | outside eyes vs teacher | dense mirror asymmetry |
|---|---|---|---|---|---|---|
| (a) original architecture | 0.502 ± 0.002 | 0.059 ± 0.002 | 0.454 ± 0.002 | 3.153 ± 0.009 / 164.3 | 3.093 ± 0.006 | 1.290 ± 0.010 |
| (b) room, 16 channels per layer | 0.459 ± 0.002 | 0.000 ± 0.000 | 0.461 ± 0.002 | 2.079 ± 0.026 / 163.7 | 2.048 ± 0.021 | 1.314 ± 0.022 |
| (c) patched architecture, gate reset, wire frozen | 0.502 ± 0.002 | 0.060 ± 0.000 | 0.454 ± 0.002 | 2.965 ± 0.014 / 159.2 | 3.024 ± 0.011 | 1.289 ± 0.005 |
