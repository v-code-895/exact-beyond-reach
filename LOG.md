# Proposal log

Every change to an experiment: who proposed it (person or agent), what stopped it, how it was resolved.
Same practice as paper one's Table 11.

| date | experiment | proposal | proposed by | what caught it | outcome | resolution |
|---|---|---|---|---|---|---|
| 2026-11-05 | all | research plan (central question, programs A–F, core set, order of work) | person | — | adopted | PREREG.md |
| 2026-11-05 | B1, A1 | continue GAN training on the BatchNorm-folded generator | agent | BatchNorm in train mode would rescale grown channels and break the wire | adopted before running | PREREG conventions |
| 2026-11-05 | B2 | prediction 2: the gate's big-M term shields its incoming weights from right-half gradient | agent | B2 measurement: 0% zero gradients | wrong: LeakyReLU passes slope 0.2 when off | recorded; B1's predictions to be written with LeakyReLU semantics |
| 2026-11-05 | B1 | continue training the folded generator with fresh Adam against a fixed discriminator (pilot) | agent | unpatched control drifts 0.12 per step at lr 2e-4 | setup unusable | B1 not registered until the control trains stably; three candidate fixes listed in RESULTS |
| 2026-11-05 | C1 | prediction: closed form on 2,000 cats ≈ on 300 | agent | C1 arm (b) | wrong: 51.8% vs 41.6% left irises | recorded |
| 2026-11-05 | C1 | prediction: trained detectors beat 42% at equal or lower collateral | agent | C1 arms (c)–(e) | wrong: 99.8% of irises but 25× the collateral | recorded; C1b registered |
| 2026-11-05 | C1 | prediction: right half ≤ 2e-7 in float64 by construction, every arm | agent | C1 arm (e) | wrong for 8 units: 1.2e-6 (float32 rounding of each pair's output weights, ×8 pairs) | recorded; C1b predicts ≤ k × 2e-7 |
| 2026-11-05 | C1b | sweep λ with the preservation region matched to the evaluation; closed form swept over its threshold; cache features in float16 | agent | — | adopted | PREREG C1b |
