# E1: a shielded edit in Pythia-160M

Layer 10. Held-out paraphrases with Rome top-1 (of 10), neighbourhood prompts unchanged (of 8), validation loss; mean over seeds.

| arm | at install: held-out / neighbours / val loss | after 1,000 steps: held-out / neighbours / val loss | steps with exactly zero unit gradient (w) | unit weight change (max) |
|---|---|---|---|---|
| unit, margin 0.5 | 5.0 / 7.0 / 3.0641 | 1.0 / 4.7 / 3.1818 | 0 / 1000 | 2.8e-03 |
| unit, margin 2.0 | 5.0 / 8.0 / 3.0640 | 4.3 / 4.3 / 3.1799 | 0 / 1000 | 1.1e-03 |
| unit, margin 6.0 | 5.0 / 8.0 / 3.0640 | 3.0 / 4.7 / 3.1797 | 711 / 1000 | 9.9e-04 |
| unit, margin 12.0 | 5.0 / 8.0 / 3.0640 | 3.7 / 4.0 / 3.1811 | 916 / 1000 | 1.1e-03 |
| unit, margin 30.0 | 5.0 / 8.0 / 3.0640 | 4.7 / 4.3 / 3.1777 | 969 / 1000 | 9.4e-04 |
| fine-tuned edit (layer MLP) | 10.0 / 2.0 / 3.1356 | 5.0 / 4.3 / 3.1803 | - | - |
