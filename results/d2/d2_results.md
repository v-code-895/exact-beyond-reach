# D2: exact vs approximate depth growth (TinySR-26)

PSNR change against the 6-block base (dB), mean ± sd over seeds; base: {'bicubic LR': 32.52762383355035, 'nearest LR': 31.121492523617214}.

| arm | step 0: bicubic / nearest | step 1,000 | step 3,000 | inserted blocks' output conv, norm relative to base blocks (final) |
|---|---|---|---|---|
| (a) exact: 3 zero-output blocks inserted | +0.000 ± 0.000 / +0.000 ± 0.000 | +0.017 ± 0.002 / +0.088 ± 0.010 | +0.033 ± 0.004 / +0.120 ± 0.008 | 0.127, 0.146, 0.171 |
| (b) duplicate blocks 2-4 | -4.688 ± 0.000 / -4.491 ± 0.000 | +0.014 ± 0.002 / +0.093 ± 0.012 | +0.040 ± 0.002 / +0.124 ± 0.009 | 0.964, 1.026, 1.014 |
| (c) control: 6 blocks | +0.000 ± 0.000 / +0.000 ± 0.000 | -0.004 ± 0.003 / +0.017 ± 0.007 | +0.001 ± 0.004 / +0.021 ± 0.013 | - |
| (d) from scratch: 9 blocks | -17.401 ± 1.008 / -16.161 ± 0.974 | -1.162 ± 0.012 / -2.291 ± 0.046 | -0.677 ± 0.015 / -1.779 ± 0.049 | - |
