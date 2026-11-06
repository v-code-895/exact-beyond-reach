# P1: a provable gap

Test accuracy, mean ± sd over seeds 0-2 (4,000 test images; training set of N images).

| arm | N = 64 | N = 256 | N = 1,024 | N = 4,096 | N = 16,384 |
|---|---|---|---|---|---|
| (a) circular padding | 49.5 ± 0.2 | 50.1 ± 0.4 | 50.4 ± 0.0 | 50.1 ± 0.3 | 50.4 ± 0.0 |
| (b) zero padding | 71.7 ± 0.5 | 77.9 ± 0.3 | 79.3 ± 2.2 | 78.9 ± 0.6 | 82.1 ± 2.9 |
| (c) circular + coordinate channel | 97.9 ± 0.3 | 98.3 ± 0.5 | 98.8 ± 0.3 | 99.2 ± 0.3 | 98.4 ± 0.2 |
| (d) scaffold: wire + gate + frozen clamp, detector trained | 96.9 ± 1.6 | 97.9 ± 0.3 | 97.9 ± 0.7 | 97.0 ± 1.8 | 98.1 ± 0.3 |
| (e) scaffold without the clamp | 97.3 ± 1.0 | 97.4 ± 1.0 | 98.2 ± 0.7 | 97.7 ± 0.6 | 98.5 ± 0.8 |
