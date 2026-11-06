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

**Amendment (2026-11-05, during the run).** At the person's request, to free the GPU, arm (d) (3× width) is cut to
seed 0: its first seed takes ~9× the compute of the others. It is reported as one seed; arms (a)–(c) keep seeds 0–2.

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

## Continued GAN training: how the setup is chosen (registered 2026-11-05)

A1 and B1 continue training CatGAN-26 from paper one's step-30,000 state; nothing is trained from scratch.
The B1 pilot showed the folded generator drifting under a fresh optimiser and a fixed discriminator, so
the setup comes from `b1_pipeline_check.py` (unpatched control, 2,000 steps, D trained) by this rule:
a setup is **stable** if, at step 2,000, its Fréchet distance is at most 5 above its own step-0 value
and its mean image change against the base is at most 0.10. The folded generator (lr 2×10⁻⁵, then
2×10⁻⁴) is used if stable, since patches live in that form; otherwise paper one's BatchNorm generator
is resumed with its own optimiser states, starting G from its EMA weights, with every grown channel
bypassing BatchNorm. The choice is logged before the first A1 or B1 run. In every arm the
discriminator runs in bf16 autocast as in paper one; the generator stays float32, so installed
identities stay exact in the forward pass.

## A1. The odd-eye 2×2 (registered 2026-11-05)

**Question.** Can gradient descent learn the odd-eye behaviour (left iris blue, right eye unchanged)
from data alone? Does unused room, or the installed circuit, change that?

**Arms.** All start from paper one's trained CatGAN-26 and continue training for 1,000 steps (batch 32,
paper one's loss, R1 and augmentation), seeds 0–2.
- *Data:* real AFHQ cats only; or real cats plus patched samples at share s ∈ {1%, 10%, 50%}. Patched
  samples are 5,000 images from paper one's left-eye patched generator on fresh latents (seed 31),
  shown to the discriminator as real in place of a real image with probability s.
- *Architecture:* (i) original; (ii) room: 16 channels grown in every hidden layer by exact widening
  (small random incoming weights, std 0.01, zero outgoing), so the function is unchanged at step 0 but
  the room can train. (Room with zero incoming and outgoing weights is an exact fixed point of training
  (B2) and would stay identical to (i).) (iii) The full patched model (room, wire and gate), all
  weights free; on real cats only, this arm is B1's "all free" arm and is shared with it.
- 12 configurations × 3 seeds = 36 runs.

**Metrics.** EMA generator every 250 steps, on paper one's 300 test latents. Iris masks come from the
base image at the same latent, as in paper one. Per side, the share of irises with median hue in
200–250°, plus the shares with the left only and with both in band. Right-half drift: mean change
against the base on the right half; for (iii) also the invariant, max |x − x_gate-off| on the right
half. Position in the dense layer, the only place the original architecture can hold it: the mirror
asymmetry of the dense biases' change, Σ|Δb(cell) − Δb(mirror cell)| / Σ|Δb|. Fréchet distance,
500 samples vs 500 real validation cats. Patched images shown so far.

**Predictions.**
1. On real cats only, no arm puts more than 2% of left irises in band. (iii) loses its odd eyes.
2. On mixed data, (i) and (ii) learn the look but not the position. Where the left share reaches 10%,
   the right share is at least half the left share (blue in both eyes, not the left only).
3. Room does not help: at s = 50%, (ii)'s left-minus-right share is within 5 points of (i)'s.
4. In (iii), the invariant breaks at once: max |x − x_gate-off| on the right half > 10⁻³ by step 250
   at every share (B2: circuit weights get the largest gradients in the network). With s = 50%, more
   left irises stay in band than with real cats only.
5. The left share rises with s in every architecture.

## A2. Distilling the guard (registered 2026-11-05)

**Question.** Paper one's guard bounds TinySR-18A's output on every input (out ≤ R + δ, R a smoothed
copy of the input) by construction, and a fine-tune "bounds nothing". Does a student taught by the
guarded network learn the bound, or only the outputs?

**Arms.** Teacher: paper one's "colour fold + guard" network (exp21). Students, TinySR-18A's original
architecture, starting from the base weights, 4,000 Adam steps, L1, seeds 0–2:
- (a) distilled on the teacher's outputs over paper one's 9 calibration images;
- (b) distilled over a broad set: the calibration images, 2,000 random AFHQ crops and 500 noise images;
- (c) paper one's 4,000-step ground-truth fine-tune, reproduced.

**Attack.** PGD on the input, maximising max over pixels of out − (R + δ), 200 steps, from 100 starts
(paper one's edge suite and calibration images, plus random images), L∞ radius ε ∈ {4/255, 16/255,
unconstrained in [0, 1]}.

**Metrics.** Largest violation found; share of starts where the attack finds a violation above 1/255;
the students' scores on paper one's regression gate.

**Predictions.** The teacher shows no violation above 10⁻⁶ under any attack. Every student violates
its bound by more than 1/255 from at least half of the starts at ε = 16/255. (b) is harder to break
than (a) but not unbreakable.

**Amendment (2026-11-05, before any A2 run).** A smoke test showed that the unguarded base exceeds
R + 1/255 by up to 168 RGB levels on random noise: any sharpening overshoots a smoothed copy of the
input at highlights, so the whole-image prediction above is close to trivial. The guard exists so that
black neighbourhoods stay black. Added: a second attack and metric restricted to pixels whose
reference is black (R < 0.02 in every channel), with these predictions. Without attack, students (a)–(c)
stay within 2 levels of the bound in black regions on most starts. At ε = 16/255, the attack lifts
black regions more than 4 levels above the bound in every student. The teacher stays within 10⁻⁶.

## A4. Mode connectivity (registered 2026-11-05; analysis only, after A1)

**Question.** Is the hand-wired solution in the same basin as solutions gradient descent finds?

**Setup.** Linear paths θ(α) = (1 − α)θ_patched + α θ_k, α ∈ {0, 0.1, …, 1}, from paper one's patched
model to: the base model; the A1 endpoints of (i) and (ii) at s = 50%; and the A1 endpoint of (iii) on
real cats. Endpoints with fewer channels are embedded with zero weights (exact). Along each path: Fréchet
distance, left and right irises in band, and the right-half invariant.

**Predictions.** The base and the (i)/(ii) endpoints are separated from the patched model by a quality
barrier (midpoint Fréchet distance more than 5 above the worse endpoint). Interpolation scales the
big-M terms below the signals they gate, so the right half changes at the midpoint (invariant > 10⁻²).
The path to (iii) (the same model after training) has no barrier.

## Focus (2026-11-05)

After the first results the person chose three threads and asked for more extreme patches and for how
networks use them: **(1) scaffold and fill**: a patch installs structure SGD cannot find, SGD trains the
content (C1, C1b, A1, B3, D4-switch); **(2) survival**: whether installed structure survives training
depends on the activation and on tying (B2, B2b, B1, E1); **(3) patched networks as teachers** (A3, D5).
Deferred: D2 extensions, D6, D7, B4–B6, C2–C4.

## B2b. Tied pairs restore shielding (registered 2026-11-05; analysis only)

**Question.** B2 found no shielding because each identity is two channels with independent weights. If
the gate's ReLU pair is tied (second channel's incoming weights = −first's, outgoing = a × first's),
does the gate get exactly zero gradient where it is off?

**Setup.** B2's measurement (GAN generator loss against paper one's discriminator, 50 batches of 32) on
the left-eye patch with the gate pair re-parameterised as tied (one kernel, bias, wire weight and
output vector, the second channel derived from them in the forward pass); identical function. Per batch:
the tied gate parameters' gradient from right-half pixels alone (gradient masked to the right half at
the gate's pre-activation), from left-half pixels, and B2's untied numbers for comparison.

**Predictions.** Right-half contribution exactly 0.0 in every batch. Total tied gate gradient (RMS) at
least 10× smaller than the untied pair's.

## D5. Distilling the odd-eye patch back into the original network (registered 2026-11-05)

**Question.** A1 asks whether SGD learns the odd eyes from data. Given the patched network as a teacher
(a target image for every latent, so the right half's targets demand that it stay unchanged), can SGD
absorb position into an architecture without the wire?

**Arms.** Teacher: paper one's left-eye patched CatGAN-26 (folded). Students, all starting from the base
weights: (a) original architecture; (b) room: 16 channels per hidden layer by exact widening (as A1);
(c) the patched architecture with the gate's weights reset to zero (wire kept and frozen): can SGD
rebuild the gate? Loss: mean L1 to the teacher over the image plus mean L1 over the pixels where the
teacher differs from the base by more than 2 levels; fresh latents every batch (32); Adam lr 10⁻⁴;
2,000 steps; seeds 0–2; evaluated every 250 steps; EMA weights kept every 500 steps.

**Metrics.** A1's: left and right irises in band, left-only share, right-half change against the
teacher (mean and max), change outside the eyes, the dense biases' mirror asymmetry.

**Predictions.** (a) puts ≥ 20% of left irises in band with ≤ 5% of right irises (a teacher makes
position reachable through the dense layer's per-cell biases) but its right half drifts from the
teacher (mean > 0.5 RGB levels). (b) ≥ (a). (c) reaches paper one's 41.6% with the right half exact to
float rounding (the frozen wire and AND carry the guarantee).

**Amendment (2026-11-05, before running).** Learning rate 2 × 10⁻⁵ instead of 10⁻⁴. In a 200-step pilot on arm (a), seed 0
(not reported as a result), lr 10⁻⁴ with a 200-step warm-up drifted from the teacher on the right half by 9.5 RGB levels
and rising; 2 × 10⁻⁵ settled at 3.2.

## E1. A shielded edit in a small language model: margin sweep (registered 2026-11-05)

**Question.** In float32, GELU is exactly ReLU outside |x| < 6, with exactly zero gradient below about
−8 (checked). Does an exact unit grown into Pythia-160M's MLP survive continued training untouched when
its pre-activation stays far below zero on training text, and decay when it does not?

**Edit.** A counterfactual fact (as in the model-editing literature): "The Eiffel Tower is located in
the city of" → " Rome". One unit grown into the MLP of one middle layer: a detector over the MLP input
at the last token, fitted in closed form (paper one's exception-unit recipe) on 20 paraphrases against
20,000 pile-10k token positions; output vector along embed_out[Rome] − embed_out[Paris], scaled so the
edit wins on the fitting paraphrases. Exact at install: Pile loss unchanged.

**Arms.** Margin m (the unit's largest pre-activation on fitting Pile tokens, made −m by scaling the
detector and dividing the output by the same factor) ∈ {0.5, 2, 6, 12, 30}; baseline: the same edit made
by fine-tuning that layer's MLP on the paraphrases (20 Adam steps). Then continued training of the whole
model on held-out pile-10k text: Adam lr 10⁻⁵, batch 8 × 256 tokens, 1,000 steps, seeds 0–2.

**Metrics.** Every 100 steps: edit success on 10 held-out paraphrases (target probability, rank);
specificity on neighbouring prompts (other Paris landmarks, other cities' towers); the unit's weight
change; the share of batches in which its gradient is exactly zero; firing rate on training tokens;
Pile validation loss.

**Predictions.** For m ≥ 12 the unit's weights receive exactly zero gradient in ≥ 95% of batches and
end within 10⁻⁶ of their installed values; for m ≤ 2 they move. The edit survives (target top-1 on
≥ 8/10 held-out paraphrases at step 1,000) for m ≥ 12 unless the inputs drift; the fine-tuned edit
decays (top-1 on ≤ 5/10).

**Amendment (2026-11-05, after a smoke test, before running).** (1) On the GPU, GELU's gradient is exactly zero only below
about −13.5 (measured: −4 × 10⁻¹⁴ at −8, −2.6 × 10⁻³¹ at −12, 0 at −30); the CPU check had shown zero from −8. The
exactly-zero prediction therefore applies to m = 30; for m = 12 it becomes: weights end within 10⁻⁶ of their installed values.
(2) The smoke test's unit changed 5 of 8 neighbouring prompts: it fired on any "…located in the city of". The detector's fit adds
20 hard negatives (other landmarks in the same sentence forms, including Paris landmarks; disjoint from the 8 evaluation
prompts), weighted like the positives, and the margin counts them as fitting negatives.

**Amendment 2 (2026-11-05, after a CPU check of the new fit, before running).** With hard negatives the unit leaves all 8
neighbouring prompts unchanged but flips only 3 of 10 held-out paraphrases at install (fine-tuned edit: 10/10 held-out, 1/8
neighbours unchanged, validation loss +0.07). Survival is therefore judged on the 20 fitting paraphrases (20/20 at install by
construction): the edit survives if ≥ 16/20 keep Rome top-1 at step 1,000 (m ≥ 12); the fine-tuned edit decays to ≤ 10/20.
Held-out paraphrases are still reported.

## E1b. The shielded edit under continued training that does not damage the model (registered 2026-11-06 07:55, after E1, before running)

**Why.** In E1 the continued training itself damaged Pythia (validation loss +0.12 within 100 steps in every arm, about half the
neighbouring prompts changed), so edit survival was confounded with general drift. A CPU check on identical batches (batch
8 × 256, 50 steps; `results/e1/jolt_diagnostic.json`): validation loss +0.099 at lr 10⁻⁵ from a cold Adam start, +0.075 with
Adam warmed at lr 0 first (so the cold start is a small part), +0.010 at 3 × 10⁻⁶, and −0.005 at 10⁻⁶. **Rule (fixed here):**
the largest of these learning rates whose 50-step change is ≤ +0.01: 10⁻⁶ (3 × 10⁻⁶ is +0.0101).
**Arms.** As E1, at lr 10⁻⁶, with Adam's moments warmed for 100 steps at lr 0 on separate batches: (0) no edit (control);
(1) unit, margin 30 (same unit as E1); (2) fine-tuned edit (layer 10's MLP). 1,000 steps, seeds 0–2; evaluated at steps 20, 50 and
every 100. **Predictions.** (1) Control: validation loss at step 1,000 within ±0.02 of install, and ≥ 7 of 8 neighbouring prompts
keep the base model's answer. (2) Margin 30: the unit's gradient exactly zero in ≥ 95% of steps; Rome top-1 on ≥ 18/20 fitting
paraphrases at every evaluation, and Rome probability ≥ 0.95 at step 1,000 in every seed. (3) The fine-tuned edit's Rome
probability ends ≥ 0.5 in every seed: most of its decay in E1 came from the damaging learning rate. (4) The unit's weights still
move (largest change ≥ 10⁻⁵): Adam's magnification of rare gradients does not depend on the learning rate's size.

## B3. Give a generator position and let it train (registered 2026-11-05)

**Question.** CatGAN-26 is translation-equivariant apart from its dense layer and its borders. Given
exact coordinate wires it never had, does continued training on real cats start reading them, and
for what?

**Arms.** Continued GAN training (setup rule above), 2,000 steps, seeds 0–2: (a) control; (b) coarse x
and y coordinate wires (4 levels each, from the dense layer's per-cell biases, carried exactly through
every layer) with zero weights into existing channels, wires frozen, everything else free.

**Metrics.** Norm of the weights out of the wires over time; Fréchet distance against (a); ablation at
the end (zero the wires' outgoing weights): how much the images change; image asymmetry (left–right
mean difference) against (a).

**Predictions.** Weights out of the wires grow from zero (B2: 1.3× ordinary gradients). Ablating them
changes images by more than the (a)–(b) Fréchet difference suggests, i.e. the network leans on position;
the Fréchet distance is not better than (a) by more than 1.

## D4. An exact switch with a branch trained by GAN: cat → anime (registered 2026-11-05)

**Question.** Paper one's mux makes "off" exact. If a grown branch behind a latent switch is trained
adversarially on anime faces while the cat network stays frozen, what does the branch do with the cat
features it reads? (Replaces the plan's fusion with the legacy anime generator: that generator is
64×64 with a different layer stack, and the supplied dataset is not its training set.)

**Arms.** Branch: copies of CatGAN-26's blocks 4–5 and output layer reading block 3's features, behind
an exact switch on an extra latent bit; switch off = paper one's CatGAN-26 exactly. Trained: the branch
only (and a discriminator started from paper one's, trained on the supplied 63,565 anime faces resized
to 128×128), 3,000 steps, seeds 0–2. Variant: branch reading block 2's features (more freedom).

**Metrics.** Off: max difference from CatGAN-26 (float32 and float64) on 300 latents. On: Fréchet
distance to 500 held-out anime faces. Correspondence between a latent's cat and its anime face: colour
histogram correlation, and whether pose (eye positions) carries over, against shuffled pairs.

**Predictions.** Off is exact to float rounding. The branch keeps the cat's colours and pose
(correspondence well above shuffled pairs), because it starts from cat weights and reads cat features.
The block-2 variant reaches a lower anime Fréchet distance but keeps less correspondence.

## C1c. Eight units, warm-started from the closed form (registered 2026-11-06, after C1b, before running)

**Why.** In C1b, 8 randomly initialised units beat 1 unit at λ ≤ 100 but lost at λ = 1,000, where most of them died (no
pixel above zero, so no gradient). Does starting every unit from the closed form keep them alive and make the wider
detector dominate at every λ?

**Arm.** 8 units, each the closed-form detector (C1b, q = 0.97) scaled by 1/8: since relu is positively homogeneous, the
8 units sum exactly to the 1-unit closed form at install. Each unit's kernel gets independent noise (std 0.05 × the
kernel's own std) to break the symmetry. Everything else as C1b (cached features, matched preservation region, Adam
1e-3, 2,000 steps, batch 32), λ ∈ {1, 10, 100, 1,000, 10,000}, seeds 0–2. Compared against C1b's three detector types.

**Metrics.** C1b's, plus dead units at the end: units with no positive pre-activation on the left half of 64 fitting
latents.

**Predictions.** (1) No unit dies at λ ≤ 1,000. (2) At λ = 1,000, ≥ 17.5% of left irises in band (the 1-unit warm start)
at ≤ 0.006 RGB levels. (3) At λ ≤ 100, at least as many irises in band as 8 random units at no more collateral.
(4) Right-half margin negative in every run; float64 right-half change ≤ 8 × 2 × 10⁻⁷.

## P1. A provable gap: structure lets a network learn what it provably could not (registered 2026-11-06)

**Task (synthetic).** 32×32 RGB images on a torus: grey background with pixel noise, one red Gaussian blob and two
distractor blobs (green, blue) at independent uniform positions, drawn with wrap-around. Label = 1 if the red blob's
centre is in the left half (x < 16). The image distribution is invariant under a cyclic shift by 16 columns, which flips
the label.

**Proposition (no structure).** A CNN with circular padding, stride-1 convolutions, pointwise nonlinearities and global
average pooling satisfies f(shift(x)) = f(x) for every cyclic shift and every choice of weights. Since x and its 16-column
shift are equally likely and have opposite labels, every such classifier has expected accuracy exactly 1/2, whatever the
data and training. **Proposition (scaffold).** With a frozen RIGHT wire (1 on columns ≥ 16) and a frozen AND,
g = relu(u − M · RIGHT), the right half contributes exactly 0 whenever the detector u < M there, so the classifier depends on
the left half only; the target is representable, and only the detector u has to be learned.

**Arms** (trunk: 4 conv layers, 32 channels, 3×3, ReLU; then a 1×1 conv to one score map; global average pooling; trainable
throughout unless stated): (a) circular padding; (b) zero padding (borders leak position); (c) circular padding plus a frozen
x-coordinate input channel (a ramp, CoordConv-style: generic position, not the answer); (d) the scaffold: circular padding,
the score map is the detector u, frozen RIGHT wire and AND with M = 50, pooled gate × trainable scale + bias.
Training sets of N ∈ {64, 256, 1,024, 4,096, 16,384} images, fixed per N; Adam 10⁻³, batch 64, 2,000 steps, binary
cross-entropy; seeds 0–2; test on 4,000 fresh images.

**Metrics.** Test and training accuracy per arm and N; for (d), the largest detector value on the right half of test images
minus M (must stay negative for the guarantee).

**Predictions.** (1) (a) is at 50 ± 2% test accuracy at every N (the proposition). (2) (b) improves with N but stays below
(c) at every N, and below 90% at N = 16,384. (3) (c) reaches ≥ 95% at N ≥ 4,096. (4) (d) reaches ≥ 95% at N ≤ 1,024, the
fewest samples of any arm, with a negative right-half margin in every run. (5) At every N: (d) ≥ (c) > (b) > (a).

**Amendment (2026-11-06, P1, after smoke tests, before the full run).** (i) The one-sided gate relu(u − M · RIGHT) died: with
every left pixel equally active at the start, the loss drove u negative everywhere (no gradient; even a +1 bias start
failed). Arm (d) uses paper one's signed gate g = relu(u − M·RIGHT) − relu(−u − M·RIGHT): exactly u on the left, exactly 0 on
the right while |u| < M (checked: 0.0 and 0.0). (ii) With nothing bounding u, training pushed |u| past M on the right half
(by 10.9 at N = 64): the guarantee depended on the trained part staying in range. Arm (d) therefore puts a frozen clamp in
front of the gate, clamp(u, −B, B) = −B + relu(u + B) − relu(u − B) with B = 20 < M = 50, so the right half is exactly 0 for any
weights. New arm (e): the same scaffold without the clamp. Added predictions: (6) (d)'s right-half margin is B − M = −30
in every run; (7) (e) learns as well as (d) but breaks the guarantee (positive margin) in at least half of its runs.

## P2–P4. Contracts on P1's task: generalisation, pretrained hosts, robustness (registered 2026-11-06)

All use P1's task, networks and training (Adam 1e-3, batch 64, 2,000 steps unless stated), seeds 0–2. Arms as P1:
(a) circular, (b) zero padding, (c) circular + coordinate ramp, (d) the clamped scaffold (wire, signed gate, clamp).

### P2. Does the contract learn the rule rather than the examples?

**Shifts.** *Position:* train with the red blob only in rows 0–15, test with it only in rows 16–31 (the horizontal
symmetry that makes (a) impossible is untouched). *Colour:* train with green and blue distractors, test with cyan and yellow
distractors. N ∈ {256, 4,096}. Each model is scored in distribution and under the shift.
**Predictions.** (1) Position shift: (d) loses ≤ 2 points; (c) and (b) lose ≥ 5 points at N = 256. (2) Colour shift: the
contract does not help, since it says nothing about colour: (c) and (d) lose similar amounts (within 3 points of each other).
(3) (a) stays at chance everywhere.

### P3. A post hoc contract on a pretrained network: zero-shot, then few-shot

**Pretraining** (a task the symmetric network can learn): "is there a red blob anywhere?" — half the images have a red blob,
half only distractors; 4,096 images, 2,000 steps. Two pretrained networks: (a) circular, and (c) circular + coordinate ramp.
**Arms.** (i) the pretrained (a) network with the contract installed exactly (clamp, signed gate, RIGHT wire before the
pooling; readout rescaled so it applies the pretrained rule to the left half: logit = 2 × mean over the left half of the
clamped score), evaluated with no left/right labels (zero-shot), then fine-tuned on N ∈ {8, 32, 128, 512} labelled
left/right images (all weights trainable, contract frozen; 300 steps); (ii) the pretrained (c) network fine-tuned on the
same N (the unconstrained way to adapt); (iii) P1's from-scratch (d) at the closest N, for reference.
**Predictions.** (1) Zero-shot accuracy ≥ 85% (the contract restricts an existing red detector to the left half; errors only
for blobs straddling the boundary). (2) (i) ≥ 95% from 32 labelled images. (3) (ii) is below 75% at N = 8 and needs ≥ 128
labels to reach 95%.

**Amendment (2026-11-06, P3, after a smoke test, before the run).** The pretrained detector scores ~+331 on red pixels, +4 on
background and −8.5 on distractors; its left-half mean separates the classes (96.5% at the best threshold), but the contract's
clamp (B = 20) crushed the red evidence and zero-shot accuracy fell to 47%. The contract is therefore installed with a frozen
positive rescaling of the score, set from 512 unlabelled left/right-task images (largest |score| → 10) and folded exactly into
the score layer; the clamp still bounds every input, so the guarantee is unchanged, and no labels are used. Readout wording
clarified: logit = mean over the left half of the clamped score (a positive rescaling never changes the sign).
Smoke test after the change (one seed, short pretraining): zero-shot 89.4%, 96.7% after 32 labels.

### P4. Robustness by construction

**Attack.** Train (b), (c), (d) on N = 4,096 (seeds 0–2). PGD on the input, 100 steps, changing only columns 20–27 (the
interior of the right half, beyond the receptive field of any left-half pixel, wrap-around included), L∞ radius
ε ∈ {8/255, 32/255, 1} (images clipped to [0, 1]), maximising the loss of the true label; 1,000 test images.
**Proposition.** For (d), no change confined to columns 20–27 can change the output, for any weights: those pixels only reach
right-half detector values, which the gate sets to exactly 0 (|u| ≤ B < M after the clamp).
**Predictions.** (1) (d)'s accuracy and logits are exactly unchanged under every attack (largest logit change 0.0).
(2) (b) and (c) lose ≥ 30 points of accuracy at ε = 32/255.

### P2b. Horizontal position shift (registered 2026-11-06, after P2, before running)
P2's position probe was uninformative: circular networks are equivariant to vertical shifts. **Shift.** Train with the red blob's
centre in columns 6–25, test with it in columns 0–5 and 26–31 (near the edges; labels still x < 16). Arms (b), (c), (d);
N ∈ {256, 4,096}; seeds 0–2. **Predictions.** (d) loses ≤ 2 points (its binary wire reads the same at the edges); (c) loses more
than (d) at N = 256 (its ramp takes values at the edges it never saw in training); (b) loses more than (d).

## P5–P7, P9. Mapping the contract (registered 2026-11-06)

P1's task, networks and training unless stated; seeds 0–2.

### P5. How much structure is enough?
**New arm** (f): the binary RIGHT wire given as a plain input channel (no gate, no clamp): the network gets the exact boundary
but must learn to use it. Compared with P1's (c) coordinate ramp, (e) gate without clamp and (d) full contract.
N ∈ {64, 256, 1,024, 4,096}. **Predictions.** (f) beats (c) at every N (the exact boundary is more useful than a generic ramp)
but needs ≥ 4× the samples (d) needs for 95%.

### P6. Wrong contracts
**Arms.** (d) with the wire's boundary at column 14 or 12 instead of 16, and with the halves inverted (the gate passes the
right half). N ∈ {256, 4,096}. **Predictions.** A shifted boundary caps accuracy at the share of images it labels correctly:
93.75% for column 14, 87.5% for column 12 (± 2 points; blobs straddling the boundary blur this); training cannot move a frozen
boundary. The inverted contract is as good as the correct one (≥ 95%): it carries the same information, and the trainable
readout can flip its sign.

### P7. Two contracts composed
**Task.** Label = red blob in the left half AND blue blob in the top half (distractor: one green blob; ~25% positives; balanced
accuracy reported). **Arms.** (b2) zero padding; (c2) circular + x and y coordinate ramps; (d2) two clamped contracts — a red
detector gated by LEFT and a blue detector gated by TOP, pooled separately, linear readout of the two evidences.
N ∈ {256, 1,024, 4,096}. **Predictions.** (d2) ≥ 95% balanced accuracy at N ≤ 1,024; (c2) below 90% at N = 1,024; the
contract's sample cost of the second rule is small (d2 at N = 1,024 within 3 points of P1's (d) at N = 1,024).

### P9. Learning curves (analysis only)
Fit test error ≈ a · N^(−b) per arm on P1's N grid (and P5's). **Prediction.** The contract changes the constant a by ≥ 10×
at the smallest N; whether it also changes the exponent b is open (no prediction).

## P6b, P7b, P9b. Follow-ups to P5–P7 (registered 2026-11-06 07:10, after P5–P7, before running)

Script `src/p_ext.py`. Initialisation now seeded per run (P2–P7 seeded only the batches).

### P6b. How far can training move a frozen boundary?
**Why.** P6: contracts with the boundary at column 14 or 12 were as accurate as the correct one (98.7–98.8% at N = 4,096), not
capped at 93.75% / 87.5%. Hypothesis: the gate is frozen but the detector feeding it is not; a shift-equivariant detector can
still place its response off-centre (up to its receptive-field radius, 4, plus the blob's own extent), so the *effective*
boundary of gate ∘ detector can move back to 16. **Arms.** (d) with the boundary at columns 16, 14, 12, 10, 8, 4; N = 4,096;
seeds 0–2. **Probe.** A lone red blob (no distractors) centred at every column c + 0.5, four rows, two noise draws: the share
called "left" per column; the number of columns called left; the detector's response profile around the blob (|change in the
clamped score| that the blob causes, rows within 3 of its centre, aligned on its column): extent left and right (> 10% of the
peak) and centroid. **Predictions.** (1) Accuracy is non-increasing in the shift 16 − b; b = 10 within 2 points of b = 16;
b = 4 below 90%. (2) For b ∈ {16, 14, 12}, 16 ± 1 columns are called left (training moves the boundary back). (3) The response
leans left when the boundary does: centroid for b = 12 at least 1 column further left than for b = 16, and extent_left −
extent_right at least 2 larger.

### P7b. Composing contracts: is the linear readout the bottleneck?
**Why.** P7: two contracts with a linear readout of the two evidences reached 91–93% balanced accuracy, below the coordinate
ramps (97–98%) and below the 95% predicted. Hypothesis: AND of two graded evidences (a blob straddling a boundary gives partial
evidence) needs a threshold on each before combining, which a linear readout cannot do. **Arms.** (d2) as in P7 (re-run with
seeded initialisation); (d2m) the same with an MLP readout (2 → 16 → 1, ReLU). N ∈ {256, 1,024, 4,096}; seeds 0–2.
Also recorded: share of errors with the red blob's column or the blue blob's row within 1.5 of a boundary (base rate in the test
set reported alongside). **Predictions.** (1) (d2m) ≥ 95% at N = 1,024. (2) (d2m) beats (d2) by ≥ 2 points at every N.

### P9b. Small-N learning curves
**Why.** P1's grid started at N = 64, where the coordinate channel (c), the contract (d) and (P5) the wire as an input channel (f)
were all already at 97–99%: the "fewest samples" question was asked too late. **Arms.** (c), (d), (f); N ∈ {8, 16, 32};
seeds 0–4. Then P9's fit (test error ≈ a · N^(−b), N from 8 to 4,096, means over seeds). **Predictions.** (1) (d) beats (c) by
≥ 5 points at N = 8 and at N = 16 (the contract only has to learn what red is; (c) must also learn where). (2) (f) lies between
(c) and (d) at N = 8. (3) P9's: the contract's constant a is ≥ 10× smaller than zero padding's (b) at the smallest common N;
no prediction about the exponent.

## C1d. Eight units, split on the output side (registered 2026-11-06, after C1c, before running)

**Why.** C1c split the closed-form unit on the input side (kernel and bias × 1/8); Adam's roughly fixed step per parameter is
then 8× larger relative to each copy, and the copies died together. **Arm.** 8 units, each the full closed-form kernel and bias
(C1b, q = 0.97) plus independent noise (std 0.05 × the kernel's std), with fixed output weights c_k = 1/8: exactly the closed
form at install (the units' relu outputs average). Everything else as C1b/C1c; λ ∈ {1, 10, 100, 1,000, 10,000}; seeds 0–2.
**Predictions.** (1) Fewer than 8 units die at λ = 1,000 in every seed, and none die at λ ≤ 100. (2) At λ = 1,000, ≥ 17.5% of
left irises in band (the 1-unit warm start). (3) At λ = 100, ≥ 66.3% (C1b's 8 random units) at ≤ 0.12 RGB levels.
(4) Right-half margin negative; float64 right-half change ≤ 8 × 2 × 10⁻⁷.

## Surgery program S1–S5 (registered 2026-11-05)

How a trained network treats foreign structure: does it adopt it, depend on it, heal around its loss, reject it, or
co-adapt around a frozen part? S1–S4 use TinySR-26 (paper one's base, 6 residual blocks, output = network residual +
a fixed bicubic upsampling of the input). Unless stated: training objective as D2 (paper one's TinySR-26 objective:
four downsampling kernels, L1), Adam 1e-4 (0.9, 0.99), EMA rate 0.01, batches of 8 images × 4 patches, seeds 0–2,
evaluated on paper one's 180 test photos (PSNR change against the base, dB).

### S1. Domestication: transplant an organ, see whether the host adopts it

**Implant.** Paper one's noisy-input expert, frozen, runs beside the host. A gate g = conv1×1(host's tail features),
1 channel, weights and bias zero (exactly 0 at install, trainable), upsampled to output resolution:
out = host(x) + g ⊙ (organ(x) − host(x)). Host and gate train; the organ does not.
**Arms.** (a) adult host (paper one's base) + organ; (b) adult control (no organ); (c) young host (trained from random
initialisation for 1,500 steps, then implanted) + organ; (d) young control. 3,000 steps after the implant, on paper
one's mixed objective (a third clean bicubic, a third noisy bicubic, a third nearest; as A3).
**Metrics.** Every 250 steps, per input type: PSNR of the system; PSNR of the host alone (gate forced to 0);
dependence = their difference; mean gate value (routing).
**Predictions.** (1) The gate learns to route: mean g ≥ 0.5 on noisy inputs and ≤ 0.2 on clean bicubic inputs at step
3,000 in (a). (2) The host's own denoising withers: host-alone PSNR on noisy inputs ends ≥ 0.5 dB below the control's.
(3) The freed capacity shows: the system's gain on clean bicubic inputs exceeds joint retraining's +0.03 dB (A3 (a))
by ≥ 0.05 dB. (4) Dependence on noisy inputs grows over training. (5) The young host withers more than the adult.

### S2. Lesion atlas: remove a part, watch it heal

**Lesions** (each on paper one's base): remove one residual block (i = 0…5; its contribution deleted from the residual
stream); remove the tail conv (replaced by identity); remove the head skip. Channel lesions in block 2's inner layer,
the 12 of 48 channels with the least outgoing weight energy: crude (outgoing weights zeroed) vs sutured (block 2's
second conv refitted in closed form from the 36 remaining channels to reproduce its original output on 2,000 training
patches). Control: no lesion. Each healed 1,000 steps on the D2 objective, evaluated every 100 steps (bicubic, nearest).
**Metrics.** Damage at step 0; steps until within 0.05 dB of the control at the same step; gap at 1,000 steps; where
healing happens: relative weight change per layer (‖ΔW‖ / ‖W‖).
**Predictions.** (1) Removing the head skip is the most damaging lesion. (2) Every single-block removal heals to within
0.1 dB of the control by step 1,000. (3) Healing concentrates next to the lesion: the two neighbouring blocks have the
largest relative weight change. (4) Sutured channel removal starts ≥ 5× closer to the control (in dB) than crude and is
within 0.05 dB by step 100; crude needs ≥ 300 steps.

### S3. Phantom limb: cut the bicubic shortcut, then re-attach it

**Arms.** (a) paper one's base with the bicubic shortcut removed (output = network only), healed 3,000 steps; (b) control:
base with shortcut, same training; (c) the same architecture without shortcut from random initialisation, 3,000 steps
(newborn). At step 3,000, (a)'s shortcut is re-attached and it trains 1,000 more steps. Every 100 steps: PSNR; the
network's own low-frequency share (correlation of its output with bicubic(x)).
**Predictions.** (1) Cutting costs ≥ 10 dB. (2) (a) heals to within 0.5 dB of (b) by step 3,000 and is ahead of the
newborn (c) at every evaluation (savings). (3) Re-attachment costs ≥ 5 dB (low frequencies counted twice) and heals to
within 0.1 dB of (b) within 500 steps.

### S4. Transplant and rejection

**Arms** (block 2 of paper one's base replaced): (a) same lineage: block 2 of paper one's bicubic expert; (b) same
lineage: block 2 of the noisy expert; (c) foreign lineage: block 4 of D2's from-scratch 9-block network (seed 0, step
3,000); (d) foreign + stitching: (c) with 1×1 adapters before and after the block, fitted in closed form on 2,000
training patches (in: recipient features → the donor's features at that depth; out: the donor block's output → the
recipient's original block output); (e) control. Healed 1,000 steps on the D2 objective, evaluated every 100 steps
(bicubic, noisy, nearest inputs).
**Predictions.** (1) Same-lineage transplants cost < 0.3 dB at step 0 and are within 0.05 dB of the control by step 200;
(b) raises noisy-input PSNR at step 0 by ≥ 0.3 dB (partial skill transfer). (2) The foreign transplant costs > 3 dB at
step 0 and is not within 0.1 dB of the control by step 500. (3) Stitching cuts the foreign transplant's damage below 1 dB.

### S5. A content filter that survives fine-tuning (CatGAN-26)

**Target.** Orange cats: at least half of the face-centre pixels (rows 40–88, columns 32–96) orange (hue 15–45°,
saturation > 0.35, brightness > 0.3), judged on each latent's unfiltered image. 724 of 5,153 AFHQ training cats.
**Filter.** When it fires, the image is replaced by flat grey (0.5) through paper one's mux identity (big-M, exact for a
binary gate); otherwise the image passes exactly. Gate = clamp(K (score − t), 0, 1) from two ReLUs, K = 50. Two
detectors, both fitted on 2,000 fitting latents to the label: *feature filter*: ridge on block 2's features pooled by a
full-size convolution; *pixel filter*: a fixed linear colour statistic of the generated image (mean R − B over the face
centre, one 128×128 convolution), threshold at the best F1. Both built only from conv, upsampling and LeakyReLU,
compiled and checked exact as in D4.
**Arms.** Continued GAN training (folded, lr 2e-5), 2,000 steps, seeds 0–2: (1) feature filter frozen, all cats;
(2) feature filter trainable, all cats; (3) pixel filter frozen, all cats; (4) feature filter frozen, orange cats only;
(5) pixel filter frozen, orange cats only; (6) control, orange cats only. (Control on all cats: B3 (a).)
**Metrics.** Every 250 steps, on 300 test latents: share of orange (unfiltered) images blocked; share of non-orange
blocked; share of unfiltered images that are orange (does the generator stop making them?); Fréchet distance on
non-blocked outputs; unblocked outputs vs unfiltered images (exactness); filter weight change.
**Predictions.** (1) The trainable filter is erased: < 20% of orange images blocked by step 2,000. (2) Frozen filters keep
their weights exactly and unblocked images exact to 1e-5. (3) On all cats, frozen filters keep blocking ≥ 80% of orange
images and the generator makes fewer orange cats than the control (≥ 25% relative). (4) On orange cats only, the frozen
feature filter is evaded (< 50% blocked by step 2,000); the frozen pixel filter is not (≥ 80%).
**Stated limit.** This protects against fine-tuning that does not target the filter's weights; anyone holding the weights
can see and delete the filter's units. It is explicit and auditable by design.

## Not yet registered

B1 (waits for the setup choice above), E2, F1, and the deferred B4–B6, C2–C4, D1, D3, D6, D7: each gets an entry
here before it runs.
