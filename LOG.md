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
| 2026-11-05 | A1 | retrain CatGAN-26 from scratch in each arm (~1.5 days of GPU) vs a cut-down distillation version | agent | person: "it doesn't need training on this scale"; reuse the trained model | revised | every A1 arm continues from paper one's step-30,000 model; full 2×2 + patched arm + 1/10/50% sweep kept; 1,000 steps |
| 2026-11-05 | A1 | room arm grown with zero incoming and outgoing weights | agent | B2: such room is an exact fixed point of training, so the arm would equal the original | revised before running | exact widening: random incoming (std 0.01), zero outgoing |
| 2026-11-05 | A1, B1 | discriminator in bf16 autocast (as paper one), generator float32 | agent | — | adopted | keeps the generator's identities exact, halves the cost of D |
| 2026-11-05 | E1 | models: Pythia-160M (GELU MLP) and SmolLM2-135M (SwiGLU MLP); corpus: pile-10k | agent | person allowed ~40 GB, "smaller LLMs" | adopted | revisions in results/e1_downloads.json |
| 2026-11-05 | E1 | upgrade Pillow so transformers imports | agent | paper one pins pillow 9.0.1 (rasterisation differs across versions) | rejected | alias PIL.Image.Resampling in llm_common.py instead |
| 2026-11-05 | D4 | anime faces dataset supplied by the person (63,565 images, ~80–100 px) | person | it is not the 21,551-image 64×64 set the anime generator was trained on | noted | D4 entry must state which set it trains on |
