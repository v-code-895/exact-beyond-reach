# Where patching meets gradient descent: pre-registration

Each experiment's arms, metrics and predictions are written here **before** it runs. Entries are
never edited after a run starts; corrections are appended with a date. Results go to `RESULTS.md`,
every proposal and its fate to `LOG.md`, and each result's place on the map to `MAP.md`.

## Conventions (all experiments)

- **Code and models.** Paper one's code is imported from `../wsse/src` unchanged. Scripts live in
  `followup/src`, run with the paper-one venv, and write to `followup/results/<exp>/`.
- **Bases.** CatGAN-26 `be2b16ff52fc4021` (EMA export, BatchNorm folded); TinySR-26 2× `8cb6601c1b449ffc`;
  MobileNetV2 (paper one's `mobilenet_np` weights). The odd-eye patch is paper one's `left_eye_blue`
  (exp20/exp24, grown room).
- **Training through structure uses the folded generator.** Patches live in the BatchNorm-folded export.
  In training mode BatchNorm would rescale every grown channel by its batch statistics and break the
  wire's exactness, so continued GAN training trains the folded generator (BatchNorm frozen, as in eval).
  G starts from the (patched) EMA export, D and its Adam state resume from paper one's step-30,000
  checkpoint, G gets a fresh Adam(0, 0.99) at lr 2e-4, batch 32, paper one's loss, R1 and augmentation.
  Every GAN run has an unpatched control arm trained the same way.
- **Seeds.** Every training run uses seeds 0, 1, 2 unless stated. Analysis-only runs use fixed batches.
- **Checkpoints.** Every training run keeps its weights at fixed intervals, not only the final model.
- **Fair baselines.** Every comparison of a patch with gradient descent includes gradient descent on
  the same structure.

## B2. Gradient shielding (analysis only)

**Question.** Which weights of an installed circuit can gradient descent not move, and why?

**Setup.** The left-eye patch on CatGAN-26 (folded, float32). Loss: the generator's non-saturating loss
against paper one's trained discriminator (step 30,000), with its differentiable augmentation; 50
batches of 32 latents. Control: CatGAN-26 with 16 channels grown in every hidden layer, zero incoming
and outgoing weights, no wiring. Second model: the MobileNetV2 exception unit, under a loss that
preserves the base predictions on fitting dogs and wild animals and pushes fitting cats toward cat
classes (paper one's preservation objective), on batches of 64 cached features.

**Weight groups.** Wire: dense bias cells and kernel column of the wire channel; each conv wire
channel's incoming kernel and bias. Wire-out: the (zero) weights from each wire channel into existing
channels. Gate-in: the gate's 3×3 kernel over b5c1 features, its wire weight (−M) and bias. Gate-out:
the gate's weights into the output layer. Unwired grown channels (control). Existing weights.

**Metrics.** Per group and batch: share of entries whose gradient is exactly 0.0, RMS gradient relative
to the existing weights' RMS gradient.

**Predictions.**
1. Grown channels with zero incoming *and* zero outgoing weights get exactly zero gradient on every
   weight, every batch: an unwired grown channel is a fixed point of SGD and Adam.
2. The gate's incoming weights get gradient only from pixels where the gate fires; the right half
   contributes exactly zero, so on batches where no left iris fires the whole gate-in group is zero.
3. Wire-out weights are zero-valued but get nonzero gradient wherever the carrier is 1 (the right half),
   so training will start reading the wire (this predicts B3).
4. The wire's own bias gets nonzero gradient from the left half, where its pre-activation is exactly 0
   and PyTorch's LeakyReLU passes slope 0.2: the wire is not shielded, so B1's "all free" arm should see
   wire purity decay.
5. The exception unit's weights get exactly zero gradient from every example on which it does not fire.

## C1. Frozen skeleton, trained detector

**Question.** With the coordinate wire and AND gate frozen, can gradient descent train a better iris
detector than paper one's closed-form fit, while the right half stays exactly unchanged?

**Arms.** All keep paper one's wire, AND term (−M · RIGHT), gate-to-output wiring and output colour map
(the 3-number tint); they differ in the detector, i.e. the gate's incoming weights from b5c1:
- (a) closed form on 300 fitting cats (paper one's patch, as is);
- (b) closed form on 2,000 fitting cats (same rule, more data);
- (c) Adam, 3×3 detector, initialised from (b);
- (d) Adam, 3×3 detector, random initialisation;
- (e) Adam, 8 grown gate channels (a wider detector) whose outputs sum into the gate, random init.

**Training.** 2,000 fitting latents (disjoint from test), masks from paper one's heuristic iris mask on
the base images, left side only. Loss: mean squared error to the target colour (paper one's tint of the
base iris) on left-iris pixels, plus λ = 1 times the mean squared change outside a 3-pixel dilation of
the eye boxes. Adam lr 1e-3, 2,000 steps, batch 32 latents, seeds 0–2.

**Metrics.** On paper one's 300 test latents: share of left irises in 200–250° (per image), share of right
irises in band, change outside the eyes (MAE, RGB levels), largest change in the right half (float64).

**Predictions.** (b) ≈ (a) (the rule, not the data, limits it). (c) and (d) beat 42% left irises at equal
or lower change outside the eyes; (e) beats (c). Right-half change stays ≤ 10⁻⁶ (float64 ≤ 2×10⁻⁷) in
every arm, by construction.

## A3. Distilling the union

**Question.** Paper one's routed union kept both fine-tunes' gains; joint retraining of one network did
not (+0.03 / +3.19 / +0.08 dB). Is that a capacity limit or an optimisation limit?

**Arms** (TinySR-26 2×, 3,000 steps from the base, Adam 1e-4, EMA, batch 8 × 4 patches of 96 px, paper
one's mixed batches: a third clean bicubic, a third noisy bicubic, a third nearest):
- (a) joint retraining on ground truth (paper one's resolution, re-run with three seeds);
- (b) distillation: the target is the union's output, i.e. the expert for each input's degradation
  (the self-consistency router chose it on all 540 test images);
- (c) distillation + ground truth, loss averaged;
- (d) 3× width, joint retraining on ground truth: the base widened exactly (new channels get random
  incoming weights and zero outgoing weights, so the start is the base function), then (a)'s recipe.

**Metrics.** PSNR change against the base on paper one's 180 test photos per input type; the merge
spec (both gains > 0, nothing below −0.1 dB).

**Predictions.** (b) keeps more of the bicubic gain than (a) (≥ +0.10 dB) at equal size, so the
conflict is partly optimisation; it does not reach the union's +0.23. (d) also keeps more than (a).

## D2. Exact vs approximate depth growth (TinySR-26)

**Question.** Does a network deepened exactly heal faster or end better than one deepened approximately,
at equal compute, and does training use the new depth?

**Arms** (6 → 9 residual blocks; train 3,000 steps on paper one's TinySR-26 objective — mixed
downsampling kernels, L1 — Adam 1e-4, seeds 0–2):
- (a) exact: insert three residual blocks whose second convolution is zero (the function is unchanged);
- (b) duplicate: copy blocks 2–4 after themselves (function changes, as in depth up-scaling);
- (c) control: continue training the 6-block base for the same steps;
- (d) from scratch: the 9-block architecture at random init, same steps (reference only).

**Metrics.** PSNR on paper one's AFHQ test images (bicubic and nearest inputs) every 250 steps; step-0
change; steps until (b) recovers to within 0.05 dB of the base; how far inserted blocks move from
identity (norm of their output conv, relative).

**Predictions.** (a) starts exactly at the base and ends ≥ (b); (b) loses PSNR at step 0 and recovers
within 1,000 steps; neither ends more than 0.1 dB above (c) at this budget; (a)'s inserted blocks stay
close to identity (relative norm < 0.1).

## C1b. The detector's trade-off curve (registered 2026-11-05, after C1, before running)

**Question.** At equal change outside the eyes, is a detector trained by Adam inside the frozen skeleton
better than paper one's closed form? (C1 at λ = 1 put 99.8% of left irises in band, but at 25× the closed
form's collateral, so C1 could not answer this.)

**Arms.** Same frozen skeleton as C1 (wire, AND term, gate wiring, tint).
- Closed form on C1's 2,000 fitting cats (same ridge fit as C1 (b)), at threshold quantiles 0.90, 0.95,
  0.97, 0.98, 0.99, 0.995: the closed form's own trade-off curve.
- Adam, three detector types: 1 unit from the closed form (q = 0.97, C1's (b)); 1 unit, random init;
  8 units, random init. λ ∈ {1, 10, 100, 1,000, 10,000}; seeds 0–2 (45 runs).

**Training (changed from C1, so that training penalises what evaluation measures).** Loss = mean squared
error to the target colour on left-iris pixels + λ × mean squared change over left-half pixels outside a
3-pixel dilation of the iris mask (the evaluation's "outside the eyes" region). The left half only: the
AND term makes the right half's gate input u − M, and evaluation checks it stays negative. To make 45 runs
affordable, the frozen network's b5c1 features for the 2,000 fitting latents are computed once and cached
in float16 (left half plus one column); Adam lr 10⁻³, 2,000 steps, batch 32, as in C1.

**Metrics.** C1's, on the same 300 test latents, plus: the right-half margin max(u_k) − M over all
right-half test pixels (must be < 0); and the collateral split into inside vs outside the left eye box.
Detectors are saved for every run.

**Predictions.**
1. In every detector type, change outside the eyes and left irises in band both fall as λ rises.
2. Adam's curve lies above the closed form's: at collateral ≤ 0.34 RGB levels (C1 (b)), the best
   trained 1-unit detector puts more left irises in band than the closed form does at the same
   collateral (interpolated along its threshold curve).
3. 8 units lie above 1 unit at matched collateral.
4. Starting from the closed form beats random initialisation at high λ (≥ 1,000), where a random
   detector's gate can die (no pixel above zero, no gradient).
5. The right-half margin is negative in every run; right-half float64 change ≤ k × 2 × 10⁻⁷.

## Not yet registered

A1, A2, A4, B1, B3–B6, C2–C4, D1, D3–D7, E1, E2, F1: each gets an entry here before it runs.
