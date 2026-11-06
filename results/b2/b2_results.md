# B2: gradient shielding

Folded generator vs paper one's forward pass: max difference 0.0e+00.

Generator loss against paper one's discriminator, 50 batches of 32 latents (float32).

| model | weight group | entries | share of exactly-zero gradients (mean / min over batches) | batches entirely zero | RMS gradient / existing weights' RMS |
|---|---|---|---|---|---|
| odd_eye | wire: dense kernel column | 2,048 | 0.0000 / 0.0000 | 0/50 | 0.00516 |
| odd_eye | wire: dense bias (RIGHT pattern) | 16 | 0.0000 / 0.0000 | 0/50 | 0.0061 |
| odd_eye | wire: conv incoming kernel | 21,969 | 0.0000 / 0.0000 | 0/50 | 63 |
| odd_eye | wire: conv bias | 9 | 0.0000 / 0.0000 | 0/50 | 162 |
| odd_eye | wire-out: wire -> existing channels (zero) | 17,856 | 0.0000 / 0.0000 | 0/50 | 1.33 |
| odd_eye | gate-in: detector over b5c1 | 576 | 0.0000 / 0.0000 | 0/50 | 41 |
| odd_eye | gate-in: wire weight (-M) | 18 | 0.0000 / 0.0000 | 0/50 | 33.8 |
| odd_eye | gate-in: bias | 2 | 0.0000 / 0.0000 | 0/50 | 43.8 |
| odd_eye | gate-out: into the output layer | 54 | 0.0000 / 0.0000 | 0/50 | 1.18e+03 |
| odd_eye | existing weights | 8,128,291 | 0.0000 / 0.0000 | 0/50 | 1 |
| control_unwired | unwired grown: incoming | 411,040 | 1.0000 / 1.0000 | 50/50 | 0 |
| control_unwired | unwired grown: outgoing | 309,168 | 1.0000 / 1.0000 | 50/50 | 0 |
| control_unwired | existing weights | 8,128,291 | 0.0000 / 0.0000 | 0/50 | 1 |

Gate firing: left-half pixels in 50/50 batches (400392 px), right half in 0 (0 px).

Exception unit (preservation objective, per example): {'examples': 1800, 'unit_off': 1199, 'zero_gradient': 1199, 'mismatches': 0, 'off_by_class': {'cat': 0, 'dog': 600, 'wild': 599}, 'of_by_class': {'cat': 600, 'dog': 600, 'wild': 600}}
