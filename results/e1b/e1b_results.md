# E1B: a shielded edit in Pythia-160M (Adam warmed at lr 0)

Layer 10. Held-out paraphrases with Rome top-1 (of 10), neighbourhood prompts unchanged (of 8), validation loss; mean over seeds.

| arm | at install: held-out / neighbours / val loss | after 1,000 steps: held-out / neighbours / val loss | steps with exactly zero unit gradient (w) | unit weight change (max) |
|---|---|---|---|---|
| no edit (control) | 0.0 / 8.0 / 3.0640 | 0.0 / 6.0 / 3.0582 | - | - |
| unit, margin 30.0 | 5.0 / 8.0 / 3.0640 | 5.3 / 6.0 / 3.0589 | 967 / 1000 | 1.6e-04 |
| fine-tuned edit (layer MLP) | 10.0 / 2.0 / 3.1356 | 8.0 / 3.0 / 3.0629 | - | - |
