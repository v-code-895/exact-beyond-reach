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
| 2026-11-05 | A2 | whole-image bound violation as the attack target | agent | smoke test: the unguarded base exceeds R + 1/255 by 168 levels on noise (sharpening overshoot), so the prediction is near-trivial | amended before running | added a black-region attack and metric (R < 0.02), with predictions |
| 2026-11-05 | all | focus on three threads (scaffold and fill; survival; patched networks as teachers); push patches further and study how networks use them | person | — | adopted | PREREG "Focus"; B2b, D5, E1, B3, D4-switch registered |
| 2026-11-05 | D4 | fuse CatGAN-26 with the legacy anime generator | plan | that generator is 64×64 with a different layer stack; the supplied dataset is not its training set | replaced | an exact switch with a GAN-trained branch reading the cat network's features |
| 2026-11-05 | B2b | prediction: tying cuts the gate's gradient ≥ 10× | agent | B2b | wrong for detector and bias (1.8×, 0.7×), right for output (210×); right-half contribution exactly 0 as predicted | recorded |
| 2026-11-05 | B2b | first region mask detached the gate's input only, so the output vector still saw every pixel | agent | first run: the output vector's right-half gradient equalled its full gradient | fixed before writing up | mask the gate's whole contribution to the image |
| 2026-11-05 | A3 | cut the 3×-wide arm to one seed (~9× the compute per run) | agent asked; person decided | GPU budget | adopted | PREREG A3 amendment; `a3_write_table.py` writes the table |
| 2026-11-05 | D5 | lr 1e-4 as registered | agent | 200-step pilot: right half drifts 9.5 levels from the teacher and rising (2e-5: 3.2, stable) | changed before running | lr 2e-5 |
| 2026-11-05 | A1, B1, B3, D4 | the setup rule names the folded generator at 'lr 2e-5, then 2e-4' without saying which wins if both are stable | agent | ambiguity noticed before the 2e-4 result existed | clarified | use 2e-4 (paper one's rate) if stable, else 2e-5, else BatchNorm resume |
| 2026-11-05 | D4 | lr for the branch | agent | — | 2e-4 (paper one's training rate): the branch must learn a new domain, unlike continued training | adopted before running |
| 2026-11-05 | D4 | "pose (eye positions)" correspondence | agent | no eye detector works on both cats and anime faces | measured instead as the correlation of 16×16 luminance layouts, matched vs shuffled pairs | adopted before running |
| 2026-11-05 | D4 | compiled switch checked before training (branch perturbed by 0.02 noise) | agent | — | off vs CatGAN-26 4.7e-6 (float32) / 1.3e-15 (float64); on vs branch 6.7e-6 | recorded |
| 2026-11-05 | E1 | 'exactly zero gradient below about −8' (CPU check) | agent | smoke test on CUDA: −4e-14 at −8, −2.6e-31 at −12, 0 at −30 | corrected before running | prediction for exact zero moved to m = 30 |
| 2026-11-05 | E1 | detector fitted against Pile tokens only | agent | smoke test: the unit changed 5/8 neighbouring prompts (fires on any '…located in the city of'); the fine-tuned edit changed 6/8 | amended before running | 20 hard negatives added to the fit |
| 2026-11-05 | E1 | hard negatives (CPU check, layer 6, m = 12) | agent | install: neighbours unchanged 8/8 (was 3/8) but held-out paraphrases 3/10 (was 8/10) | trade-off of a one-unit linear detector | survival judged on the 20 fitting paraphrases; amendment 2 |
| 2026-11-05 | A1, B1, B3 | continued-training setup | rule (registered) | stability check: folded 2e-4 degrades (FD 58 → 96–101); folded 2e-5 stable (58.3 → 57.0, drift 0.065) | chosen | folded generator, lr 2e-5 |
| 2026-11-05 | S1–S5 | surgery program: domestication, lesion atlas, phantom limb, transplant and rejection, content filter | person (filter, domestication, surgery ideas) and agent (designs) | — | adopted | PREREG S1–S5 |
| 2026-11-05 | S5 | "blocked" = gate ≥ 0.5 | agent | fit check: the big-M mux is a hard switch near g ≈ 1 − abs(d)/M, so a partial gate (1 in 100 test images) leaves the image nearly unchanged (0.61 from grey) | changed before running | behavioural: blocked = grey within 1e-3; passed = bit-identical; partial reported |
| 2026-11-05 | S5 | Fréchet distance "on non-blocked outputs" | agent | variable sample counts would bias it | changed before running | Fréchet distance of the unfiltered images (the host's quality), blocked shares reported separately |
| 2026-11-05 | S5 | detectors (fit on 2,000 latents; 13% orange) | agent | — | feature filter F1 0.89, pixel filter F1 0.89; passed images bit-identical to unfiltered | recorded |
| 2026-11-05 | S5 | "compiled as in D4" | agent | the filter's pooling (a full-size conv) and broadcast make it a DAG of conv, upsampling and LeakyReLU, not a single layer stack | noted | the training forward uses only those operations and paper one's identities; exactness checked behaviourally |
