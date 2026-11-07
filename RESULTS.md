# Results

Each entry: what ran, what came out, and how it compares with the pre-registered predictions.

## B2. Gradient shielding (2026-11-05)

Script `src/b2_gradient_shielding.py`; output `results/b2/`. The trainable folded generator reproduces
paper one's forward pass exactly (difference 0.0).

Generator loss against paper one's step-30,000 discriminator, 50 batches of 32 latents, float32:

| model | weight group | entries | exactly-zero gradients | RMS gradient / existing weights' |
|---|---|---|---|---|
| odd-eye patch | wire: dense kernel column | 2,048 | 0% | 0.005 |
| | wire: dense bias (the RIGHT pattern) | 16 | 0% | 0.006 |
| | wire: conv incoming kernels | 21,969 | 0% | 63 |
| | wire: conv biases | 9 | 0% | 162 |
| | wire → existing channels (zero-valued) | 17,856 | 0% | 1.3 |
| | gate: detector over b5c1 | 576 | 0% | 41 |
| | gate: wire weight (−M) | 18 | 0% | 34 |
| | gate → output layer | 54 | 0% | 1,180 |
| control: 16 unwired grown channels per layer | incoming | 411,040 | **100%, every batch** | 0 |
| | outgoing | 309,168 | **100%, every batch** | 0 |

The gate fired on left-half pixels in all 50 batches and on no right-half pixel.

Exception unit (MobileNetV2, ReLU), preservation objective, per example on the 1,800 fitting images:
the unit is off on 1,199 (every dog, 599 of 600 wild animals) and those 1,199 give exactly zero gradient;
0 mismatches. (One wild animal sits at the threshold, which is the largest fitting non-cat score.)

**Against the predictions.**
1. *Unwired grown channels are a fixed point of SGD and Adam.* **Confirmed**: zero gradient on all
   720,208 weights, every batch.
2. *The gate's incoming weights get no gradient from the right half.* **Wrong.** CatGAN-26 uses
   LeakyReLU, which passes gradient with slope 0.2 where a unit is "off", so the right half (gate
   pre-activation near −M) still sends gradient into the gate. No circuit weight is ever shielded.
   The prediction assumed ReLU semantics.
3. *Training will start reading the wire.* **Confirmed**: the zero-valued weights from the wire into
   existing channels get gradients 1.3× the size of ordinary weights'.
4. *The wire is not shielded.* **Confirmed**, more strongly than predicted: the wire's biases get
   gradients 162× the ordinary RMS.
5. *The exception unit gets no gradient where it is off.* **Confirmed**, exactly (true ReLU).

**What it means.** In a LeakyReLU network an installed circuit is not shielded; it is the most exposed
part of the network. The big-M identities hold the right half fixed by cancelling two large, opposite
terms (lrelu(u) and a·lrelu(−u) with u ≈ −M), so the gradients on the weights that carry those terms
are 34–1,180× the network's ordinary gradients. Exact shielding needs true ReLU units, as in the
exception unit, or weights with zero activation and zero outgoing weight, as in unwired room. This
reverses B1's expectation for the "all free" arm: the invariant should break at once, not decay.

## B1 pilot (exploratory, not pre-registered; 2026-11-05)

Script `src/b1_pilot.py`; output `results/b1_pilot/`. Adam on the folded generator's non-saturating loss
against paper one's discriminator, **held fixed**, fresh Adam state, one seed, 300 test latents.

| model, lr | step 1 | step 10 | step 100 |
|---|---|---|---|
| unpatched, 2e-4: mean change of the image vs the base | 0.119 | 0.468 | 1.63 |
| unpatched, 2e-5 | 0.015 | 0.089 | 0.35 |
| patched, 2e-4: gate leak into the right half (max) / left irises in band / wire deviation | 0.10 / 0.4% / 1.7 | 0.95 / 0% / 5.9 | 2.5 / 0% / 11 |
| patched, 2e-5 | 0.008 / 0% / 0.18 | 0.088 / 0% / 1.3 | 0.76 / 0% / 5.8 |

**What it shows.** The setup, not the patch, fails first: the unpatched generator drifts just as fast
(fresh Adam moves every folded weight by about the learning rate at once, and a fixed discriminator lets
the generator run away). The patch is still the more fragile part: one Adam step at 2e-5 removes every
blue left iris and moves the wire by 0.18, while the image as a whole moves by 0.015, consistent with B2's
1,000× gradients on the circuit. **B1 cannot run until continued training of the folded generator is
stable for the unpatched control.** Candidate fixes, to be tested on the control before B1 is registered:
train D as well; warm Adam's second moments before the first update; or resume paper one's BatchNorm
generator with its optimiser state and add the circuit as a branch that bypasses BatchNorm (the wire
also survives BatchNorm exactly, since its statistics are the same for every batch).

## C1. Frozen skeleton, trained detector (2026-11-05)

Script `src/c1_trained_detector.py`; output `results/c1/`. Wire, AND term (−M · RIGHT, M = 11.7), gate
wiring and tint frozen; only the detector over b5c1 features is fitted or trained. Every arm is rebuilt as a
real network (k ReLU pairs in grown b5c2 channels) and scored on paper one's 300 test latents. Rebuilding
arm (a) from the formula reproduces paper one's network exactly (difference 0.0). Adam: 2,000 steps,
lr 10⁻³, batch 32 latents from 2,000 fitting latents, λ = 1; seeds 0–2, mean ± sd.

| arm | left irises in band | right irises in band | change outside the eyes (RGB levels) | right half, max change (float64) |
|---|---|---|---|---|
| (a) closed form, 300 cats (paper one, threshold quantile 0.98) | 41.6% | 0% | 0.24 | 1.6 × 10⁻⁷ |
| (b) closed form, 2,000 cats (quantile 0.97) | 51.8% | 0% | 0.34 | 1.6 × 10⁻⁷ |
| (c) Adam, 1 unit, from (b) | 99.9 ± 0.2% | 0% | 6.0 ± 0.3 | 1.5 × 10⁻⁷ |
| (d) Adam, 1 unit, random init | 99.8 ± 0.3% | 0% | 6.2 ± 0.5 | 1.5 × 10⁻⁷ |
| (e) Adam, 8 units, random init | 98.9 ± 0.3% | 0% | 4.3 ± 0.4 | 1.2 × 10⁻⁶ |

**Against the predictions.**
1. *(b) ≈ (a).* **Wrong**: 51.8% vs 41.6%. Confounded, though: the threshold rule (strictest quantile whose
   median left-iris hue is in band) chose 0.97 on 2,000 cats against 0.98 on 300, so (b) is also a looser
   point on the same trade-off (0.34 vs 0.24 RGB levels outside the eyes).
2. *(c) and (d) beat 42% at equal or lower change outside the eyes.* **Wrong as stated.** The first half
   holds by a wide margin (99.8–99.9% of left irises in band, from random initialisation too). The second
   does not: change outside the eyes is 25× the closed form's. The logged preservation loss alone implies
   an RMS change of ≈ 24 RGB levels per channel over the left half outside the eye boxes, the region
   training did penalise, so the change is not just an artefact of the training mask leaving the eye
   boxes free. λ = 1 weights the mean background error the same as the mean iris error, although
   background pixels outnumber iris pixels many times over, so it is a weak preservation term.
3. *(e) beats (c).* **Undecided**: fewer irises in band (98.9% vs 99.9%) at less collateral (4.3 vs 6.0).
   The arms sit at different points of a trade-off, so the comparison needs matched collateral (C1b).
4. *Right half ≤ 2 × 10⁻⁷ in float64, by construction, in every arm.* **Confirmed for one unit, wrong for
   eight** (1.2 × 10⁻⁶). The residual is the float32 rounding of each ReLU pair's two output weights
   (gg·v and gg·a·v are rounded separately, so they no longer cancel exactly against the pair's
   ±(u − M) terms): about M · |w| · 2⁻²⁴ ≈ 1.5 × 10⁻⁷ per pair, and the 8-unit arm has 8 pairs. No right
   iris moved into the band in any arm.

**What it means.** With the wire and the AND installed and frozen, gradient descent on the detector alone
finds a far more sensitive iris detector than the closed form, from random initialisation, while the
installed structure keeps the right half unchanged to rounding. Position is the part SGD could not find
(paper one); the content of the detector is not. What λ = 1 does not show is whether SGD's detector is
better at equal collateral; C1b sweeps the preservation weight to compare the two trade-off curves.

## B2b. Tied pairs restore shielding (2026-11-05)

Script `src/b2b_tied_pairs.py`; output `results/b2b/`. The gate's ReLU pair re-parameterised as tied (one
kernel, bias and output vector; channel g1 derived as the negation, with output a × g0's). The installed
network is already tied up to float rounding (g1 = −g0 exactly; output ratio a to 3 × 10⁻⁸), and the tied
forward matches it to 6.6 × 10⁻⁷. Same 50 batches as B2; gradient restricted to one half by detaching the
gate's contribution to the image elsewhere.

| gate parameters | tied, all pixels (RMS) | tied, right half only | tied, left half only | untied pair, same batches (RMS) | untied / tied |
|---|---|---|---|---|---|
| detector over b5c1 | 0.150 | **0.0 in 50/50 batches** | 0.150 | 0.266 | 1.8× |
| bias | 0.391 | **0.0 in 50/50** | 0.391 | 0.284 | 0.7× |
| output vector | 0.036 | **0.0 in 50/50** | 0.036 | 7.62 | 210× |

The gate fired on ~8,000 left-half pixels per batch and on no right-half pixel (largest right-half
pre-activation −10.4).

**Against the predictions.**
1. *Right-half contribution exactly 0.0 in every batch.* **Confirmed**, for every tied parameter.
2. *Tied gate gradient ≥ 10× smaller than the untied pair's.* **Wrong for the detector (1.8×) and bias
   (0.7×), right for the output vector (210×).** The untied pair's output weights read lrelu(p) and
   lrelu(−p), which are large wherever the gate is off (|p| ≈ 10), so their gradients are huge and
   cancel only along the tied direction. The detector's gradient was never mostly from off pixels: it
   comes from the left-half pixels where the gate fires.

**What it means.** B2's "maximally exposed" result is a property of the parameterisation, not of the
circuit. Tied, a LeakyReLU pair is an exact ReLU with exactly zero gradient wherever it is off, so the
gate can still learn where it acts and is frozen everywhere else. This is the mechanism for B1's
"identities protected" arm, and it is how an installed circuit can be made to survive training without
freezing it.

## Continued GAN training: the setup check (2026-11-05; decides the setup, not a registered result)

Script `src/b1_pipeline_check.py`; output `results/b1_pipeline/`. Unpatched CatGAN-26 from paper one's step-30,000 state,
2,000 steps with the discriminator trained (paper one's loss, R1, augmentation), one seed. Fréchet distance (MobileNetV2
features, 500 EMA samples vs 500 real validation cats) / mean image change against the base on fixed latents ([0, 1] units):

| setup | step 0 | 500 | 1,000 | 1,500 | 2,000 | stable by the rule? |
|---|---|---|---|---|---|---|
| paper one's BatchNorm generator, its own optimiser states | 58.8 / 0.010 | 58.3 / 0.041 | 56.9 / 0.067 | 57.0 / 0.083 | 57.4 / 0.096 | yes (narrowly) |
| folded generator, warmed Adam, lr 2 × 10⁻⁵ | 58.3 / 0 | 57.2 / 0.027 | 56.5 / 0.048 | 57.0 / 0.057 | 57.0 / 0.065 | **yes** |
| folded generator, warmed Adam, lr 2 × 10⁻⁴ | 58.3 / 0 | 74.8 / 0.109 | 100.8 / 0.159 | 99.9 / 0.170 | 96.3 / 0.169 | no |

**Choice by the registered rule:** the folded generator at lr 2 × 10⁻⁵ (2 × 10⁻⁴ fails). Without BatchNorm, paper one's
learning rate degrades the generator within 500 steps; a tenth of it keeps quality (slightly better Fréchet distance) and
drifts less than resuming the BatchNorm generator. Warming Adam's second moments and training the discriminator removed
the pilot's runaway (B1 pilot: 0.12 change after one step at 2 × 10⁻⁴ with a cold optimiser and a fixed discriminator).

## D2. Exact vs approximate depth growth (2026-11-05; written up 2026-11-06)

Script `src/d2_depth_growth.py`; output `results/d2/`. TinySR-26 grown from 6 to 9 residual blocks, 3,000 steps, seeds 0–2. PSNR
change against the 6-block base (dB), bicubic / nearest-LR inputs, mean over seeds:

| arm | step 0 | step 1,000 | step 3,000 | inserted blocks' output conv, norm relative to the base blocks |
|---|---|---|---|---|
| (a) exact: three zero-output blocks inserted | 0.000 / 0.000 | +0.017 / +0.088 | +0.033 / +0.120 | 0.13, 0.15, 0.17 |
| (b) duplicate blocks 2–4 | −4.69 / −4.49 | +0.014 / +0.093 | **+0.040 / +0.124** | 0.96, 1.03, 1.01 |
| (c) control: keep training the 6-block base | 0.000 / 0.000 | −0.004 / +0.017 | +0.001 / +0.021 | — |
| (d) 9 blocks from scratch (reference) | −17.4 / −16.2 | −1.16 / −2.29 | −0.68 / −1.78 | — |

**Against the predictions.** (1) *(a) starts exactly at the base:* **confirmed** (0.000). *…and ends ≥ (b):* **wrong, narrowly**
(+0.033 vs +0.040 on bicubic, +0.120 vs +0.124 on nearest; differences of 1–3 standard deviations of the seed spread). (2) *(b)
loses PSNR at step 0 and recovers within 1,000 steps:* **confirmed** (−4.7 dB; within 0.05 dB of the base by step 250–500).
(3) *Neither ends more than 0.1 dB above the control:* **confirmed on bicubic** (+0.03, +0.04), **wrong on nearest** (+0.10 and
+0.10 above the control's +0.02, at the limit). (4) *(a)'s inserted blocks stay close to identity (relative norm < 0.1):* **wrong**:
they grow to 0.13–0.17 of a base block, so training does use the new depth.

**What it means.** Exact growth buys a damage-free start (the network never gets worse), but not a better endpoint: within 500
steps the crude duplicate has recovered, and after 3,000 they are equal. For depth, exactness matters for continuity, not for
where training ends. Both deeper networks gain a little over the control, mostly on nearest-LR inputs, so the extra depth is used.

## C1b. The detector's trade-off curve (2026-11-06)

Script `src/c1b_tradeoff.py`; output `results/c1b/` (table, every detector, figure `fig_c1b_tradeoff.png`); 45 Adam runs
and 6 closed-form thresholds, 4.7 hours on a shared GPU. Same frozen skeleton as C1; training now penalises exactly the
region evaluation measures (left half outside a 3-pixel dilation of the iris mask). Paper one's 300 test latents.

| closed form, threshold quantile | 0.995 | 0.99 | 0.98 | 0.97 | 0.95 | 0.90 |
|---|---|---|---|---|---|---|
| change outside the eyes (RGB levels) | 0.041 | 0.096 | 0.209 | 0.338 | 0.650 | 1.547 |
| left irises in band | 15.3% | 27.0% | 40.9% | 51.8% | 72.6% | 89.8% |

| Adam detector (seeds 0–2) | λ = 1 | λ = 10 | λ = 100 | λ = 1,000 | λ = 10,000 |
|---|---|---|---|---|---|
| 1 unit, from the closed form | 99.3% at 4.54 | 87.1% at 1.06 | **57.2% at 0.092** | **17.5% at 0.004** | 0.7% at 0.000 |
| 1 unit, random init | 99.3% at 4.86 | 86.7% at 1.19 | 59.1% at 0.116 | 11.4% at 0.004 | 0% (gate dead) |
| 8 units, random init | 97.6% at 2.78 | **92.8% at 0.86** | **66.3% at 0.122** | 7.8% at 0.006 | 0% (gates dead) |

(left irises in band at change outside the eyes, RGB levels; sd over seeds in `results/c1b/c1b_results.md`.)

**Against the predictions.**
1. *Collateral and iris share both fall as λ rises, in every detector type.* **Confirmed**, monotone in all three.
2. *At collateral ≤ 0.34, the best trained 1-unit detector beats the closed form at the same collateral.* **Confirmed,
   by a wide margin.** At ~0.09 RGB levels Adam puts 57–59% of left irises in band where the closed form manages ~26%
   (interpolated); at 0.004 Adam reaches 17.5%, which the closed form needs 10× the collateral (0.041) to approach. At
   ~1 level: 87% vs ~80%. Against paper one's patch itself (41.6% at 0.24), λ = 100 gives more irises at 2.6× less
   collateral.
3. *8 units lie above 1 unit at matched collateral.* **Confirmed for λ ≤ 100** (66% vs 57–59% at ~0.1; 93% at 0.86 vs
   87% at 1.06); **wrong at λ = 1,000** (7.8% vs 11–18%), where most of the 8 randomly started units die.
4. *Starting from the closed form beats random initialisation at high λ.* **Confirmed, weakly**: 17.5% vs 11.4% (sd
   10.6) at λ = 1,000; at λ = 10,000 random-init gates die in every seed (no pixel above zero), the warm-started gate
   keeps 0.7%.
5. *Right-half margin negative in every run; float64 right-half change ≤ k × 2 × 10⁻⁷.* **Confirmed**: largest gate
   input on the right half is ≥ 10.5 below the threshold in every run; float64 change ≤ 1.8 × 10⁻⁷ (1 unit) and
   ≤ 1.3 × 10⁻⁶ (8 units). (An apparent 0.0 in the run log was rounding to four decimals.)

**Where the collateral sits.** 1–12% of the change outside the eyes falls inside the left eye box; the rest is the
detector firing elsewhere on the left half. The trade-off is detection precision, not spill around the iris.

**What it means.** With position supplied by a frozen wire and the left/right guarantee by a frozen AND, gradient
descent fills in the detector better than the hand-fitted rule at every collateral level measured (0.004 to ~1 RGB
levels), from random initialisation too, and more units help except where they die. The guarantee never moves: the
right half is unchanged to rounding in all 51 detectors. Scaffold from the patch plus content from gradient descent beats
the hand-fitted content. Whether gradient descent alone (no scaffold) can produce the behaviour is A1's and D5's question,
not C1b's. Caveat: Adam optimised the evaluation's own objective (iris colour plus collateral), while the closed form fits
a classification proxy (ridge on iris vs background pixels) and a threshold; part of the gap may be the objective, not the
optimiser.

## C1c. Eight units, warm-started from the closed form (2026-11-06)

Script `src/c1c_warm_units.py`; output `results/c1c/`. 8 units, each the closed-form detector scaled by 1/8 (exactly the
closed form at install, checked to 2.4 × 10⁻⁷) plus symmetry-breaking noise; otherwise as C1b. Seeds 0–2.

| λ | left irises in band | change outside the eyes (RGB levels) | dead units of 8 (per seed) | C1b, 8 random units | C1b, 1 warm unit |
|---|---|---|---|---|---|
| 1 | 98.1 ± 0.6% | 3.16 | 4, 5, 3 | 97.6% at 2.78 | 99.3% at 4.54 |
| 10 | 90.5 ± 0.5% | 0.87 | 5, 4, 3 | 92.8% at 0.86 | 87.1% at 1.06 |
| 100 | **72.9 ± 2.7%** | 0.12 | 4, 4, 6 | 66.3% at 0.12 | 57.2% at 0.09 |
| 1,000 | **0%** | 0.00 | **8, 8, 8** | 7.8% at 0.006 | 17.5% at 0.004 |
| 10,000 | 0% | 0.00 | 8, 8, 8 | 0% | 0.7% |

**Against the predictions.**
1. *No unit dies at λ ≤ 1,000.* **Wrong**: 3–6 of 8 die even at λ ≤ 100; all 8 die at λ = 1,000 in every seed.
2. *≥ 17.5% at λ = 1,000.* **Wrong**: 0%.
3. *At λ ≤ 100, at least as many irises as 8 random units at no more collateral.* **Mixed**: confirmed at λ = 100 (72.9% vs
   66.3% at the same collateral, the best point on any curve at ~0.12 levels); wrong at λ = 10 (90.5% vs 92.8%); undecided
   at λ = 1 (more irises, more collateral).
4. *Right-half margin negative; float64 change ≤ 8 × 2 × 10⁻⁷.* **Confirmed** (≥ 10.9 below; ≤ 1.3 × 10⁻⁶).

**Why the warm units die.** The split scaled each unit's inputs (kernel and bias) by 1/8. Adam moves every parameter by about
the learning rate whatever its scale, so each copy's bias steps are 8× larger relative to its own pre-activation than the
single unit's; and the copies are nearly identical, so the preservation penalty pushes them down together and they die
together. The same split can be made on the output side instead (each unit keeps the full detector, output weight 1/8), also
exactly the closed form at install but with the single unit's scale: C1d.

## C1d. Eight units, split on the output side (2026-11-06)

Script `src/c1d_output_split.py`; output `results/c1d/`; combined figure `results/c1b/fig_c1_detectors.png` (C1b–C1d). Each of 8
units is the full closed-form detector plus noise, with fixed output weight 1/8 (exactly the closed form at install: 0.0 without
noise). 50 minutes.

| λ | left irises in band | change outside the eyes (RGB levels) | dead units of 8 (per seed) | right-half margin | float64 right half |
|---|---|---|---|---|---|
| 1 | 98.3 ± 0.6% | 3.11 | 0, 0, 1 | −7.0 | 1.9 × 10⁻⁷ |
| 10 | **91.7 ± 0.5%** | **0.74** | 0, 0, 0 | −6.7 | 2.0 × 10⁻⁷ |
| 100 | **78.0 ± 1.0%** | **0.14** | 0, 1, 1 | −7.1 | 2.1 × 10⁻⁷ |
| 1,000 | **40.1 ± 2.8%** | **0.016** | 1, 0, 1 | −7.9 | 1.9 × 10⁻⁷ |
| 10,000 | 0.7% | 0.001 | 0, 0, 0 | −9.5 | 1.8 × 10⁻⁷ |

**Against the predictions.**
1. *Fewer than 8 units die at λ = 1,000; none at λ ≤ 100.* **Confirmed for λ = 1,000** (at most 1 of 8 in any seed, against all
   8 in C1c); **narrowly wrong for λ ≤ 100** (1 of 8 died in 3 of 9 runs).
2. *≥ 17.5% at λ = 1,000.* **Confirmed**: 40.1%, at 0.016 levels; the 1-unit curve reaches ~35% at that collateral
   (log-interpolated), C1c 0%.
3. *At λ = 100, ≥ 66.3% at ≤ 0.12 levels.* **Literally not met** (78.0% but at 0.139); **confirmed at matched collateral**: C1b's
   8 random units reach ~68% at 0.139 (interpolated), the closed form ~32%.
4. *Margin negative; float64 ≤ 8 × 2 × 10⁻⁷.* **Confirmed** (≥ 6.7 below; ≤ 2.1 × 10⁻⁷, the 1-unit level, because the 1/8 output
   weights shrink each pair's rounding residual).

**What C1b–C1d show together.** Inside the frozen wire and gate, an 8-unit detector warm-started by an output-side split of the
closed form is the best at every collateral level from ~0.01 to ~1 RGB level: 78% of left irises at 0.14 levels, where paper
one's rule gets ~32% and paper one's patch 41.6% at 0.24. How the warm start is split decides whether it trains at all: the same
function split on the input side (C1c) dies under Adam, split on the output side it survives and wins. A third case of the
parameterisation lesson (B2/B2b, C1c/C1d): equal functions, different fates under training.

## A3. Distilling the union (2026-11-06)

Script `src/a3_union_distill.py` (table by `src/a3_write_table.py`); output `results/a3/`. 3,000 steps from TinySR-26 on paper
one's mixed batches; PSNR change against the base on paper one's 180 test photos. The 3×-wide arm was cut to one seed at the
person's request (it took 5.6 hours on the shared GPU).

| arm | seeds | bicubic inputs | noisy inputs | nearest inputs | meets spec | parameters |
|---|---|---|---|---|---|---|
| (a) joint retraining, ground truth | 3 | +0.023 ± 0.004 | +3.192 ± 0.002 | +0.081 ± 0.003 | 3/3 | 276,732 |
| (b) distillation from the union | 3 | **+0.108 ± 0.004** | +3.083 ± 0.014 | −0.079 ± 0.006 | 3/3 | 276,732 |
| (c) distillation + ground truth | 3 | +0.094 ± 0.005 | +3.143 ± 0.007 | −0.000 ± 0.014 | 3/3 | 276,732 |
| (d) 3× width, joint retraining | 1 | +0.089 | **+3.465** | **+0.136** | 1/1 | 2,447,580 |
| paper one: routed union (3 networks) | — | **+0.23** | +3.26 | +0.00 | yes | 830,196 + 33 |

**Against the predictions.** *(b) keeps ≥ +0.10 dB of the bicubic gain, more than (a), and does not reach +0.23; (d) also keeps
more than (a).* **Confirmed** on all counts: (b) +0.108 vs (a) +0.023; (d) +0.089 (one seed, still rising at step 3,000:
+0.075 → +0.089 over the last 500 steps).

**What it means.** The union's conflict is partly an optimisation problem: at equal size, training on the union's outputs keeps
4.7× the bicubic gain that training on ground truth keeps, about as much as making the network 3× wider (8.8× the parameters).
Neither reaches the union's +0.23 on clean bicubic inputs, so routing each input type to its own expert is still worth ~0.12 dB
there. The 3×-wide network spends its extra capacity mostly on noisy and nearest inputs, where it beats the union (+3.47 vs
+3.26, +0.14 vs 0.00). A patched network is a better teacher than the data, but at this budget not a perfect one.

## P1. A provable gap (2026-11-06)

Script `src/p1_separation.py`; output `results/p1/`. Test accuracy on 4,000 fresh images, mean ± sd over seeds 0–2:

| arm | N = 64 | N = 256 | N = 1,024 | N = 4,096 | N = 16,384 |
|---|---|---|---|---|---|
| (a) circular padding (exactly shift-equivariant) | 49.5 ± 0.2 | 50.1 ± 0.4 | 50.4 ± 0.0 | 50.1 ± 0.3 | 50.4 ± 0.0 |
| (b) zero padding | 71.7 ± 0.5 | 77.9 ± 0.3 | 79.3 ± 2.2 | 78.9 ± 0.6 | 82.1 ± 2.9 |
| (c) circular + coordinate channel | **97.9 ± 0.3** | 98.3 ± 0.5 | 98.8 ± 0.3 | 99.2 ± 0.3 | 98.4 ± 0.2 |
| (d) contract: wire + signed gate + frozen clamp | 96.9 ± 1.6 | 97.9 ± 0.3 | 97.9 ± 0.7 | 97.0 ± 1.8 | 98.1 ± 0.3 |
| (e) the same contract without the clamp | 97.3 ± 1.0 | 97.4 ± 1.0 | 98.2 ± 0.7 | 97.7 ± 0.6 | 98.5 ± 0.8 |

Right-half margin (largest |gate input| on the right half minus M): (d) −30.0 in all 15 runs (= B − M, by construction); (e)
**+4.5 to +44.5 in all 15 runs**. (a) memorises small training sets (67.7% training accuracy at N = 64) but is at chance on test
images; the circular network's output is identical for an image and its 16-column shift (difference 0.0).

**Against the predictions.** (1) *(a) at 50 ± 2% at every N:* **confirmed** (49.5–50.4%). (2) *(b) improves with N, stays below
(c), below 90% at 16,384:* **confirmed** (82.1%). (3) *(c) ≥ 95% at N ≥ 4,096:* **confirmed**, and far earlier than predicted
(97.9% at N = 64). (4) *(d) ≥ 95% at N ≤ 1,024, with the fewest samples of any arm:* **half right**: 96.9% at N = 64, but (c)
gets there as fast. (5) *(d) ≥ (c) > (b) > (a) at every N:* **wrong for (d) ≥ (c)** ((c) is 0.4–2.2 points higher at every N);
right for the rest. (6) *(d)'s margin −30 in every run:* **confirmed**. (7) *(e) learns as well as (d) and breaks the guarantee
in at least half its runs:* **confirmed, in all 15**.

**What it means.** The impossibility is real (no weights help the symmetric network) and is cured by *any* supply of the missing
information: a generic coordinate channel learns the rule from 64 examples, as fast as the full contract. From scratch, the
contract's gate and clamp buy no sample efficiency; they buy the guarantee — and only the clamped version keeps it, since
training pushed the unclamped detector past its margin in every run. Where the contract pays is with a pretrained host (P3).

## P3. A post hoc contract on a pretrained network (2026-11-06)

Script `src/p2_p4_contracts.py p3`; output `results/p3/`. Networks pretrained on "is there a red blob anywhere?" (100% accuracy,
all seeds), then tested on the left/right task on 4,000 images; mean over seeds 0–2 (per seed in the JSON).

| labelled left/right examples | 0 (zero-shot) | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| (i) contract installed post hoc on the pretrained symmetric network | **88.0%** (87.7–88.3) | 90.1% | 94.6% | 96.9% | 97.5% |
| (ii) pretrained coordinate network, fine-tuned | — | 71.3% | 81.0% | 81.9% | 81.8% |
| pretrained symmetric network, no contract | 49.8% | | | | |

(ii) is bimodal: seeds 0 and 1 reach 94–96% from 32 labels; seed 2 never learns to use its coordinate channel (51–55% at every N).
The contract's calibration (largest |score| on 512 unlabelled images) was 68–95 across seeds; after it, the largest score is 10.

**Against the predictions.** (1) *Zero-shot ≥ 85%:* **confirmed** (88.0%). (2) *≥ 95% from 32 labels:* **narrowly wrong** (94.6%;
1 of 3 seeds above). (3) *(ii) below 75% at N = 8 and needs ≥ 128 labels for 95%:* **confirmed** (71.3%; its mean never reaches
95% because one seed fails entirely).

**What it means.** A network that provably cannot tell left from right does so at 88% after a patch and no labels, because the
contract re-uses a detector the network already has and only restricts where it counts. The unconstrained route — a coordinate
channel the pretrained network ignored during pretraining — has to be discovered by fine-tuning, and in one seed of three it
never is. With P1: supplying the missing information makes the task learnable from scratch in any form; installing it as a
contract additionally transfers existing skills without training, reliably, and with a guarantee.

## P2. Does the contract learn the rule rather than the examples? (2026-11-06)

Script `src/p2_p4_contracts.py p2`; output `results/p2/`. Accuracy in distribution and under the shift, mean over seeds 0–2;
the change is shifted minus in-distribution (points).

| arm | position shift, N = 256 | N = 4,096 | colour shift, N = 256 | N = 4,096 |
|---|---|---|---|---|
| (a) circular | 49.7% (−0.1) | 50.3% (+0.1) | 49.7% (−0.1) | 50.3% (+0.1) |
| (b) zero padding | 77.2% (−0.5) | 83.3% (−1.8) | 74.0% (−6.2) | 75.3% (−8.6) |
| (c) circular + coordinate channel | 98.9% (+0.4) | 99.1% (+0.3) | 78.2% ± 10.9 (−19.4) | 93.3% ± 6.1 (−5.8) |
| (d) contract | 98.3% (+0.6) | 98.0% (+0.2) | 74.3% ± 3.1 (−23.5) | 87.9% ± 6.5 (−10.1) |

**Against the predictions.** (1) *Position shift: (d) loses ≤ 2 points; (b) and (c) lose ≥ 5 at N = 256.* **First half confirmed,
second wrong**: nobody loses (b −0.5, c +0.4). **The probe was uninformative**: circular networks are equivariant to vertical
shifts as well, so moving the blob from the top rows to the bottom rows is not out of distribution for (c) or (d). (2) *Colour shift:
(c) and (d) lose similar amounts (within 3 points).* **Wrong**: the contract loses more (23.5 vs 19.4 at N = 256, 10.1 vs 5.8 at
4,096). (3) *(a) at chance everywhere:* **confirmed**.

**What it means.** The contract fixes where evidence counts, not what counts as evidence: with new distractor colours (yellow has a
strong red channel) its detector fires on them, and the contract's readout, a plain average of the gated detector map, has nothing
else to fall back on. A contract adds no robustness along axes it does not constrain, and here it is slightly more brittle than a
network that learned its own use of position. The informative position probe is horizontal (train with the red blob away from
the edges, test near them), where the coordinate channel sees ramp values it never saw in training and the contract's binary wire
does not change; P2b.

## S3. Phantom limb: cut the bicubic shortcut, heal, re-attach (2026-11-06)

Script `src/s3_phantom_limb.py`; output `results/s3/` (figure `fig_s3_phantom_limb.png`). PSNR change on bicubic-LR test images
against paper one's TinySR-26 (32.53 dB), mean of seeds 0–2; the last column is the correlation of the network's own output (without
the shortcut) with bicubic(x), its "own copy" of the low frequencies.

| arm | step 0 | 300 | 1,000 | 2,000 | 3,000 | own copy at 3,000 |
|---|---|---|---|---|---|---|
| (a) shortcut cut, healing | −26.43 | −12.57 | −1.98 | −1.08 | −0.81 ± 0.04 | 0.998 |
| (b) control, shortcut kept | 0.00 | −0.01 | 0.00 | 0.00 | 0.00 | 0.062 |
| (c) newborn without shortcut | −25.23 | −6.18 | −1.78 | −1.34 | −1.19 ± 0.01 | 0.999 |

Then (a)'s shortcut is re-attached (its own copy now counts twice) and it trains 1,000 more steps:

| steps after re-attaching | 0 | 100 | 200 | 300 | 500 | 800 | 1,000 |
|---|---|---|---|---|---|---|---|
| PSNR change (dB) | −22.44 | −7.06 | −1.88 | −0.76 | −0.26 | −0.09 | −0.06 |
| own copy (correlation with bicubic) | 0.998 | 0.71 | 0.40 | 0.22 | 0.10 | 0.07 | 0.07 |

Nearest-LR inputs tell the same story, about twice as large: (a) −1.61, (c) −2.23 at 3,000; −0.12 after re-attaching.

**Against the predictions.** (1) *Cutting costs ≥ 10 dB:* **confirmed** (26.4 dB). (2) *(a) heals to within 0.5 dB of the control
by step 3,000 and is ahead of the newborn at every evaluation:* **both wrong.** It is 0.81 dB short (1.6 dB on nearest-LR), and it is
*behind* the newborn for the first 1,100–1,600 steps (at step 300: −12.6 vs −6.2 dB); it overtakes after that and ends 0.38 dB ahead.
(3) *Re-attaching costs ≥ 5 dB and heals to within 0.1 dB within 500 steps:* **first half confirmed** (21.6 dB below the healed
network), **second wrong**: 0.26 dB short at 500 steps, within 0.1 dB at 800 in all three seeds.

**What it means.**
- **A fixed exact module is worth real capacity.** Without the shortcut, TinySR-26 has to spend capacity reproducing bicubic itself
  (correlation 0.998), and after 3,000 steps it is still 0.8 dB worse. Give the module back and 300 steps beat 3,000 steps of healing.
- **No phantom limb: gradient descent deletes the redundant copy.** Once the shortcut returns, the network's own bicubic copy is
  gone within ~500 steps (0.998 → 0.10) and it settles at the control's 0.06. The network does not keep a skill that a fixed module
  supplies; it hands the job over. This is the mechanism S1 (domestication) asks about from the other side.
- **Prior training slows the early recovery.** The cut network recovers more slowly than a random one for ~1,000 steps, then overtakes
  it (savings appear late). A guess, not tested: its weights are tuned to emit small residuals, and producing a whole image needs
  the output scale to grow, which Adam's roughly fixed step size makes slow.

## P4. Robustness by construction (2026-11-06)

Script `src/p2_p4_contracts.py p4`; output `results/p4/`. N = 4,096, seeds 0–2, 1,000 test images. PGD (100 steps) may change
only columns 20–27, the part of the right half outside every left pixel's receptive field. Accuracy under attack, mean ± std over
seeds; the last two columns are the largest logit change over all images and seeds (random noise in the region; PGD at ε = 1).

| arm | clean | ε = 8/255 | ε = 32/255 | ε = 1 (repaint the region) | random noise: max logit change | ε = 1: max logit change |
|---|---|---|---|---|---|---|
| (b) zero padding | 80.9% | 47.0% ± 3.3 | 13.8% ± 7.4 | 0.8% ± 0.7 | 10.30 | 386.5 |
| (c) circular + coordinate channel | 98.6% | 98.2% ± 1.3 | 85.7% ± 7.8 | 66.6% ± 16.2 | 41.35 | 950.6 |
| (d) contract | 98.9% | 98.9% ± 0.6 | 98.9% ± 0.6 | 98.9% ± 0.6 | 0.00 | 0.0 |

**Against the predictions.** (1) *(d)'s accuracy and logits exactly unchanged under every attack:* **confirmed**: the largest logit
change is 0.0 in float32 across 3 radii × 1,000 images × 3 seeds, including ε = 1, where the attacker may paint the region freely.
(2) *(b) and (c) lose ≥ 30 points at ε = 32/255:* **confirmed for (b)** (−67 points), **wrong for (c)**: it loses 13 on average
(seed 0 only 0.3; seed 2, 22). At ε = 1, (c) loses 32 points, and its seeds range from 89% to 50%.

**What it means.** The contract's guarantee is a certificate that needs no adversarial training and no verification step: it holds
for all weights, so it holds for these trained ones, and the strongest attack in the region moves the logit by exactly zero. The
generic networks are attackable there because their global average reads every pixel; the coordinate channel learned partial
robustness on its own, but how much depends on the seed. The caveat is the other face of P2: a contract certifies exactly what it
encodes. Columns 20–27 cannot matter to the output because the contract says so, not because the data does; if the task's truth
depended on them, the same guarantee would be a guaranteed failure.

## P5–P7. Mapping the contract (2026-11-06)

Script `src/p5_p7_contracts.py`; output `results/p5/`, `results/p6/`, `results/p7/`. P1's task unless stated; test accuracy, mean ±
sd over seeds 0–2. (Initialisation in P2–P7 was not seeded per run, only the batches; fixed from P6b on.)

**P5. How much structure is enough?** The binary RIGHT wire given as a plain input channel (f), against P1's arms:

| arm | N = 64 | 256 | 1,024 | 4,096 |
|---|---|---|---|---|
| (c) coordinate ramp (P1) | 97.9 ± 0.3 | 98.3 ± 0.5 | 98.8 ± 0.3 | 99.2 ± 0.3 |
| (d) contract (P1) | 96.9 ± 1.6 | 97.9 ± 0.3 | 97.9 ± 0.7 | 97.0 ± 1.8 |
| (f) binary wire as an input channel | 97.6 ± 0.1 | **99.0 ± 0.1** | **98.9 ± 0.7** | **99.3 ± 0.2** |

*Predictions:* (f) beats (c) at every N: **wrong at N = 64** (−0.3), right but within noise elsewhere (+0.1 to +0.7). (f) needs ≥ 4×
the samples (d) needs for 95%: **wrong**: both are above 95% at the smallest N tested. The grid started too late to answer
sample-efficiency questions (P9b).

**P6. Wrong contracts.**

| contract | N = 256 | N = 4,096 |
|---|---|---|
| correct (boundary at column 16) | 97.1 ± 1.2 | 98.5 ± 0.6 |
| boundary at column 14 | 96.7 ± 1.0 | 98.7 ± 0.3 |
| boundary at column 12 | 97.4 ± 1.0 | 98.8 ± 0.3 |
| inverted halves (gate passes the right half) | 97.8 ± 0.1 | 99.3 ± 0.1 |

*Predictions:* a shifted boundary caps accuracy at 93.75% (column 14) and 87.5% (column 12): **wrong**: both are as accurate as the
correct contract. The inverted contract is as good as the correct one: **confirmed**. So a frozen boundary did not stay where it
was put. The gate is frozen, but the detector feeding it is not, and a shift-equivariant detector can still place its response
off-centre (up to its receptive-field radius plus the blob's extent): the effective boundary of gate ∘ detector can move back.
This does not touch P1/P4's guarantee, which was stated for pixels outside the detector's reach, but it means a contract's
*boundary* is soft by the reach of the trainable part in front of it. P6b measures how far it can move and checks the mechanism.

**P7. Two contracts composed** (red in the left half AND blue in the top half; 24% positives; balanced accuracy):

| arm | N = 256 | 1,024 | 4,096 |
|---|---|---|---|
| (b2) zero padding | 67.8 ± 2.9 | 67.7 ± 5.6 | 62.1 ± 9.3 |
| (c2) circular + x and y ramps | **96.9 ± 1.2** | **97.9 ± 0.8** | **97.4 ± 1.7** |
| (d2) two contracts, linear readout | 91.5 ± 0.9 | 93.3 ± 1.3 | 88.5 ± 5.3 |

*Predictions:* (d2) ≥ 95% at N ≤ 1,024: **wrong** (91.5, 93.3). (c2) below 90% at N = 1,024: **wrong** (97.9). The second contract's
sample cost is small (within 3 points of P1's (d)): **wrong** (93.3 vs 97.9). Composition did not come for free: two contracts with
a linear readout lose to generic coordinates. Hypothesis (P7b): a linear readout cannot threshold each graded evidence before the
AND (a blob straddling a boundary gives partial evidence). (d2) at N = 4,096 is also unstable (81–93% by seed).

**What P5–P7 mean together.** From scratch, the exact form of the supplied information matters little: a ramp, a binary channel or a
contract all solve P1's task from 64 examples, and the binary channel is the most accurate. A contract's distinctive value is its
guarantee (P1, P4) and zero-shot transfer (P3), not learning speed. Its boundary is softer than it looks (P6), and composing two
contracts needs care in the readout (P7).

## E1. A shielded edit in Pythia-160M (2026-11-06)

Script `src/e1_shielded_edit.py`; output `results/e1/`. Edit: "The Eiffel Tower is located in the city of" → " Rome", one unit
grown into layer 10's MLP (chosen on fitting data), exact at install. Then 1,000 steps of continued training of the whole model on
held-out pile-10k text (Adam 10⁻⁵, batch 8 × 256), seeds 0–2. "Fitting paraphrases" = the 20 the unit was fitted on (survival
criterion, amendment 2); Rome's probability is averaged over them.

| arm | fitting paraphrases with Rome top-1 at step 1,000 (per seed) | Rome probability, step 0 → 1,000 | steps with exactly zero unit gradient (w) | training tokens that fire the unit (of 2.05M) | largest unit weight change |
|---|---|---|---|---|---|
| unit, margin 0.5 | 4, 19, 12 | 0.99 → 0.52 | 0 | 3–10 | 2.8e-3 |
| unit, margin 2 | 18, 19, 17 | 0.99 → 0.87 | 0 | 13–19 | 1.1e-3 |
| unit, margin 6 | 20, 9, 16 | 0.99 → 0.72 | 683–745 | 13–20 | 9.9e-4 |
| unit, margin 12 | 17, 19, 6 | 0.99 → 0.69 | 915–918 | 14–16 | 1.1e-3 |
| unit, margin 30 | **20, 20, 19** | **0.99 → 0.98** | **964–973** | 14–18 | 9.4e-4 |
| fine-tuned edit (layer 10's MLP) | 12, 10, 12 | 1.00 → 0.29 | — | — | — |

Held-out paraphrases (10): 5 at install for every unit, 10 for the fine-tuned edit; at step 1,000, 1.0–4.7 (units) and 5.0
(fine-tuned). Pile validation loss: 3.064 at install (fine-tuned edit 3.136), **3.18 after 100 steps in every arm**, flat after.
Neighbouring prompts with the base model's answer: 7–8 of 8 at install (fine-tuned 2), about 4 of 8 after 1,000 steps in every arm.

**Against the predictions.** (1) *m = 30 (amended): the unit's gradient exactly zero in ≥ 95% of batches:* **confirmed** (96.4–97.3%).
(2) *m ≥ 12: weights end within 10⁻⁶ of their installed values:* **wrong**: they moved 6–8 × 10⁻⁴, as much as at m = 2. (3) *m ≤ 2:
weights move:* confirmed, but uninformative given (2). (4) *The edit survives (≥ 16/20) for m ≥ 12:* **confirmed for m = 30** (20, 20,
19; Rome probability still 0.98), **2 of 3 seeds for m = 12**. (5) *The fine-tuned edit decays to ≤ 10/20:* **narrowly wrong** (12, 10,
12), though its Rome probability fell from 1.00 to 0.29.

**Why the "shielded" weights moved anyway.** Two mechanisms, both measured:
- **The detector has false positives on new text.** 14–18 of 2.05M training tokens fire the unit in every run, whatever the margin.
  Raising the margin by scaling the detector scales those outliers too, so no margin removes them; it only empties the band of
  near-threshold tokens (2M tokens at m = 0.5, ~5,000 at m = 2, ~27 at m = 30).
- **Adam magnifies rare gradients.** Adam divides each parameter's step by its own running gradient size. A gradient that appears in
  3% of steps therefore still produces near full-size steps, and momentum carries each one on for about ten steps. "Exactly zero
  gradient most of the time" is not protection under Adam. Protection has to be exact on every input (as the frozen contracts in
  P1/P4 are) or not depend on the gradient at all.

**A confound in the setup.** Validation loss rose by 0.12 within the first 100 steps in every arm and then stayed flat, and every
arm lost about half its neighbouring-prompt answers. The continued training itself perturbed the whole model. That drift, not the
unit, dominates the fluctuations of the edit: margin 30's seed 0 dips to 8/20 at step 500 and is back at 20/20 by step 800, and
the differences between margins 2–12 are within seed noise. The likely cause is a cold Adam start (the first steps move every
parameter by about the learning rate whatever its gradient), which the GAN experiments avoided by warming Adam's second moments;
untested. E1b repeats the key arms with that warm-up and a no-edit control.

**Drift check (after the run; CPU, identical batches, 50 steps).** Validation-loss change: +0.099 at lr 10⁻⁵ (E1's setting),
+0.075 with Adam's moments warmed first, +0.010 at 3 × 10⁻⁶, −0.005 at 10⁻⁶. So the cold start is only a small part; the main cause
is a learning rate too high for this batch size (2,048 tokens per step against Pythia's 2M). E1b repeats the key arms at 10⁻⁶.

**What it means.** At the largest margin, a unit grown into an LLM kept its edit through 1,000 steps of continued training (Rome
probability 0.98, against 0.29 for the same edit made by fine-tuning), with its gradient exactly zero in 97% of steps. The
protection is statistical, not exact: in an LLM a closed-form detector cannot be made silent on all text, and Adam turns the rare
leaks into real updates.

## P6b. How far can training move a frozen boundary? (2026-11-06)

Script `src/p_ext.py p6b`; output `results/p6b/` (figure `fig_p6b_boundary.png`). Contract (d) with the gate's boundary at column
b (true boundary 16); N = 4,096; seeds 0–2. Probes: a lone red blob at every column. "Columns called left" should be 16; the
centroid and extents describe the detector's response around the blob (columns; negative = to the left).

| gate boundary b | shift 16 − b | accuracy | columns called left (per seed) | response centroid | extent left / right |
|---|---|---|---|---|---|
| 16 | 0 | 98.3 ± 0.8 | 16, 16, 16 | -0.02 | 4.3 / 4.3 |
| 14 | 2 | 98.5 ± 0.8 | 16, 16, 16 | -1.03 | 5.3 / 3.3 |
| 12 | 4 | 98.5 ± 0.4 | 16, 16, 16 | -2.06 | 6.3 / 2.7 |
| 10 | 6 | 98.1 ± 0.5 | 16, 15, 16 | -2.75 | 7.0 / 2.3 |
| 8 | 8 | 94.2 ± 0.7 | 14, 15, 15 | -2.23 | 7.0 / 4.7 |
| 4 | 12 | 82.3 ± 0.4 | 11, 11, 12 | +0.07 | 7.3 / 7.7 |

**Against the predictions.** (1) *Accuracy non-increasing in the shift; b = 10 within 2 points of b = 16; b = 4 below 90%:*
**confirmed** (shifts 0–6 all within 98.2–98.5%; 94.2% at 8; 82.3% at 12). (2) *For b ∈ {16, 14, 12}, 16 ± 1 columns called left:*
**confirmed**, exactly 16 in all nine runs. (3) *The response leans left: centroid for b = 12 at least 1 column further left than
for b = 16, and extent_left − extent_right at least 2 larger:* **confirmed** (centroid −2.06 vs −0.02; asymmetry 3.7 vs 0.0).

**What it means.** The gate is frozen, but the detector in front of it is not, and the detector learned to lean its response
toward the gate: one column for a 2-column shift, two for a 4-column shift, up to a leftward reach of 7 columns (the trunk's
receptive-field radius, 4, plus the blob's visible extent). Within that reach, training silently corrected a misplaced contract:
the effective boundary went back to 16 for shifts up to 6. Beyond it the contract caps accuracy, predictably: at b = 4 the
detector gives up leaning (centroid ≈ 0) and widens its response instead, and the effective boundary stops at about 11.3, i.e.
4 + 7.3. So **a frozen boundary has slack equal to the reach of the trainable part in front of it.** That is a feature for
misplaced contracts (they self-correct) and a design rule for guarantees: P4's guarantee held exactly because it was stated for
input columns beyond the detector's reach; the contract's *semantic* boundary is only fixed up to that reach.

## P9b and P9. Small-N learning curves (2026-11-06)

Script `src/p_ext.py p9b`; output `results/p9b/`. Test accuracy (4,000 images), mean ± sd over seeds 0–4 at N = 8–32; larger N
from P1 and P5 (seeds 0–2). N = 16 is below N = 8 in every arm: the same first 16 training images are used throughout, so this
is a property of that subset, not of the arms.

| arm | N = 8 | 16 | 32 | 64 | 256 | 4,096 | fit of test error ≈ a · N^(−b), N = 8–4,096 |
|---|---|---|---|---|---|---|---|
| (c) coordinate ramp | 89.4 ± 1.0 | 85.0 ± 1.0 | 91.3 ± 0.9 | 97.9 | 98.3 | 99.2 | a = 0.32, b = 0.48 |
| (d) contract | 92.4 ± 1.3 | 89.4 ± 1.8 | 93.0 ± 0.9 | 96.9 | 97.9 | 97.0 | a = 0.12, b = 0.23 |
| (f) binary wire as an input channel | **94.6 ± 0.6** | **91.6 ± 0.7** | **93.8 ± 0.8** | 97.6 | **99.0** | **99.3** | a = 0.16, b = 0.40 |
| (b) zero padding (P1; N ≥ 64) | — | — | — | 71.7 | 77.9 | 78.9 | a = 0.35, b = 0.07 (N = 64–4,096) |

**Against the predictions.** (1) *(d) beats (c) by ≥ 5 points at N = 8 and 16:* **wrong, narrowly**: +3.0 and +4.4. (2) *(f) lies
between (c) and (d) at N = 8:* **wrong**: (f) is the best arm (94.6). (3) *The contract's constant a is ≥ 10× smaller than zero
padding's at the smallest common N:* **narrowly wrong**: at N = 64 the error ratio is 9.1× (28.3% vs 3.1%).

**What it means.** All three position-aware networks learn the rule from a handful of examples (89–95% from 8 images). The order
is the same at every small N: binary channel > contract > ramp. Sample efficiency follows how exact the supplied information is
(a binary boundary beats a ramp), not the frozen gating: the contract's gate and clamp cost 1–2 points against the same binary
information given freely. The power-law fits are rough (the curves dip at N = 16 and saturate at a 1–3% floor), but they say the
same thing: the contract starts lowest (smallest a) and improves slowest (smallest b), and the others catch it by N = 64.

## P7b. Composing contracts with an MLP readout (2026-11-06)

Script `src/p_ext.py p7b`; output `results/p7b/`. Balanced accuracy, mean ± sd (per seed in brackets), seeds 0–2, seeded
initialisation; the same test images as P7 (checked).

| arm | N = 256 | 1,024 | 4,096 |
|---|---|---|---|
| (d2) two contracts, linear readout | 91.6 ± 0.6 (91.1, 91.5, 92.4) | 94.2 ± 0.4 (94.7, 93.6, 94.3) | 93.6 ± 1.5 (94.3, 91.5, 94.9) |
| (d2m) two contracts, MLP readout (2 → 16 → 1) | 89.8 ± 3.6 (94.8, 86.6, 87.9) | 95.2 ± 1.6 (97.6, 93.8, 94.3) | 95.3 ± 2.5 (98.5, 92.4, 95.1) |

Errors concentrate at the boundaries: 87% (linear) and 88% (MLP) of all errors have the red blob's column or the blue
blob's row within 1.5 of a boundary, against 31.5% of test images.

**Against the predictions.** (1) *(d2m) ≥ 95% at N = 1,024:* **confirmed, barely** (95.2). (2) *(d2m) beats (d2) by ≥ 2 points at every
N:* **wrong**: −1.9, +1.0, +1.7. With seeded initialisation the linear readout itself does better than in P7 (91.7 / 94.2 / 93.6
against 91.5 / 93.3 / 88.5), so part of P7's shortfall was unlucky initialisations.

**What it means.** The readout was not the main bottleneck. The composed contract's errors sit where a blob straddles one of the two
boundaries and gives partial evidence, and the variance between seeds is large (the MLP readout ranges from 92.4% to 98.5% at
N = 4,096). That points to the boundary slack measured in P6b, now present on two axes at once, rather than to the readout's
inability to threshold. Composition works (up to 98.5% in the best seed) but is less reliable than one contract or than generic
coordinates (97–98%).

## S5. A frozen content filter inside continued GAN training (2026-11-06; arms 1 and 3 only, see the amendment)

Script `src/s5_filter.py` (arm 1: `results/s5/s5_arm1_results.json`; arm 3: `s5_arm3_results.json`). CatGAN-26 (folded), continued
training at lr 2e-5 on all AFHQ cats, 2,000 steps, with a frozen filter that turns orange cats into flat grey through paper one's
exact mux. 300 test latents; "orange share" is measured on the generator's unfiltered images; Fréchet distance (FD) on unfiltered
images. At step 0: 14.3% orange, FD 58.3.

| arm | seed | outcome | peak orange share (step) | at step 2,000: orange blocked / non-orange blocked | FD at 2,000 |
|---|---|---|---|---|---|
| (1) feature filter (reads block 2's features) | 0 | collapse: everything blocked from step 750 | 70.7% (1,000) | 100% / 100% | 654 |
| | 1 | evasion: keeps making orange cats, the filter stops firing | 21.0% (500–1,000) | 4% / 1% | 60.0 |
| | 2 | collapse | 87.3% (1,250) | 100% / 100% | 563 |
| (3) pixel filter (reads the output's colour) | 0 | drifting into the filter | 70.7% (2,000) | 93% / 3% | 109 |
| | 1 | collapse | 84.0% (1,250) | — / 100% | 662 |
| | 2 | drifting into the filter | 46.7% (2,000) | 79% / 7% | 89 |

In all six runs the filter's weights did not change (0.0), and every image the filter passed was bit-identical to the unfiltered one.

**Against the predictions.** (2) *Frozen filters keep their weights exactly, unblocked images exact:* **confirmed** (0.0 in all six runs).
(3) *On all cats the frozen filters keep blocking ≥ 80% of orange images, and the generator makes fewer orange cats than the control:*
**the second half is wrong in the opposite direction**: the generator made *more* orange cats in all six runs (peaks of 21–87% from
14%), until it collapsed (three runs) or, with the feature filter, learned to make orange cats the filter no longer catches (one run).
Blocking stayed ≥ 79% in five runs, three of them only because everything was blocked. (1) and (4) were not tested (arms dropped).

**What it means.** A filter inside training is not neutral: it pulls the generator into the very category it filters. Blocked
images reach the discriminator as flat grey and give the generator exactly zero gradient, so a generator that drifts into the
filtered region cannot get back out. Why it drifts *toward* orange, consistently, is the interesting part. A hypothesis, registered
as S5b: the filter removes orange cats from the generated images the discriminator sees but not from the real ones, so the
discriminator can treat orange as a mark of a real image, and the generator, rewarded for looking real, chases orange into the
filter. The feature filter can be satisfied without stopping (seed 1: orange cats that no longer trigger its features); the pixel
filter cannot, so its runs drift or collapse. The safety reading is narrow but concrete: a hard output filter must not sit inside a
fine-tuning loss in this form; whether filtering the real images the same way removes the pull is S5b's test.

## A2. Distilling a guarantee (2026-11-06)

Script `src/a2_guard_distill.py`; output `results/a2/` (`a2_results.md` has every seed). Teacher: paper one's "colour fold + guard"
TinySR-18A, whose output is bounded by construction (out ≤ R + 1/255, R a smoothed copy of the input). Students: TinySR-18A's
original architecture, 4,000 Adam steps. Largest violation of the bound over 100 attack starts, in RGB levels, mean over seeds
0–2; in brackets, the share of starts with a violation above 1 level ("black regions": pixels whose reference is black).

| model | regression gate (PSNR, dB) | whole image, no attack | whole image, ε = 16/255 | black regions, no attack | black regions, ε = 16/255 |
|---|---|---|---|---|---|
| teacher: colour fold + guard | PASS (25.1) | **0.00 (0%)** | **0.00 (0%)** | **0.00 (0%)** | **0.00 (0%)** |
| (a) distilled on the 9 calibration images | PASS (24.8) | 112 (100%) | 215 (100%) | 40.5 (100%) | 56.6 (100%) |
| (b) distilled on a broad set (+2,000 crops, 500 noise images) | PASS (25.1) | 117 (100%) | 199 (100%) | 40.4 (100%) | 60.2 (100%) |
| (c) ground-truth fine-tune (paper one) | PASS (27.1) | 493 (100%) | 561 (100%) | 59.8 (100%) | 73.9 (100%) |
| base model (reference) | 21.8 | 272 (100%) | 384 (100%) | 85.5 (100%) | 133 (100%) |

**Against the predictions.** (1) *The teacher shows no violation above 10⁻⁶ under any attack:* **confirmed** (exactly 0.00 everywhere,
including unconstrained attacks). (2) *Every student violates by more than 1/255 from at least half the starts at ε = 16/255:*
**confirmed** (100% of starts, every student, every seed). (3) *(b) is harder to break than (a) but not unbreakable:* **not clearly**: over the
whole image its violations are 7% smaller at ε = 16/255, in black regions 6% larger; 100% of starts broken either way. Amendment: *without attack, students stay within 2 levels of
the bound in black regions on most starts:* **wrong**: they exceed it by about 40 levels on every start, attack or not.
*At ε = 16/255 the attack lifts black regions more than 4 levels above the bound in every student:* **confirmed** (55–78 levels).

**What it means.** Distillation copies behaviour, not guarantees. The distilled students match the teacher on the data (same
regression score, gate PASS), yet violate its bound on every single start, by 40 levels in black regions before any attack, while
the teacher's violation is exactly zero under every attack. Training on a much broader set (b), including noise images, barely
changes the violations (7% smaller over the whole image, 6% larger in black regions). A guarantee is a property of the structure that produces it, not of the outputs it produces on the
data; this is the converse of P4 (a contract is a certificate because of its structure) and fits E1 (a learned shield is only
statistical). For the teacher thread: a patched teacher can pass on skill (A3) but not a guarantee.

## L5. What a module is worth: the exchange rate (TinySR part, 2026-11-06; the toy part follows when it runs)

Script `src/l5_exchange.py`; output `results/l5/`. TinySR-26's architecture from random initialisation, 3,000 steps (S3's budget
and objective), seeds 0–1; PSNR change on bicubic-LR inputs against paper one's trained base (dB), mean over seeds (seeds agree within
0.06 dB). 48 channels without the shortcut is S3's newborn arm (same initialisation and objective). The last column: the width a network
*without* the shortcut needs to match this one *with* it (interpolated in log-width), as a multiple of this width.

| channels | with shortcut | without | the shortcut's gain | width needed without it, to match |
|---|---|---|---|---|
| 24 | -1.13 | -1.53 | **+0.41** | 2.2× |
| 48 | -0.81 | -1.19 | **+0.39** | 1.7× |
| 96 | -0.44 | -0.69 | **+0.25** | 1.4× (extrapolated) |

**Against the predictions.** (1) *The shortcut's gain shrinks as width grows:* **confirmed** (0.41 → 0.39 → 0.25 dB). (2) *96 channels
without the shortcut stay below 48 with it:* **wrong**: 96 without reaches -0.69 dB, above 48 with (-0.81). The shortcut is
worth about 1.7× the width at 48 channels (about 2.9× the parameters), not more than 2×.

**What it means.** A fixed module has a price in capacity, and the price falls as the network grows: worth about 2.2× the width to a
24-channel network, 1.7× to a 48-channel one, and roughly 1.4× (extrapolated) to a 96-channel one. Small networks lean on installed
structure most, which is also where S3 found the network deleting its own copy. At the other end, P1's proof says no width at all
substitutes for a contract when the task is impossible; L5's toy part checks that end directly.

**Toy part (the impossible end; `src/l_series.py l5toy`, `results/l5toy/`).** P1's circular network without a contract at 16, 64 and 256
channels (N = 4,096, seed 0): test accuracy 49.65% at every width, training accuracy 51.2% (the majority share): all three settled on a
constant answer. *Prediction: 50 ± 2% at every width:* **confirmed**. Sixteen times the width buys nothing, as the proof says; the same
structure that is worth 1.4–2.2× the width in TinySR is worth more than any width here.

## L1. The reach law on P1's task (2026-11-06)

Script `src/l_series.py l1`; output `results/l1/` (analysis `l1_analysis.json`, figure `fig_l1_reach_law.png`). 72 runs: trunk reach r ∈ {2, 4, 6}
(conv layers) × blob width σ ∈ {1, 2, 3} × gate boundary b ∈ {16, 12, 8, 4} × seeds 0–1.

**Guarantee side (the proposition).** On all 18 models with the boundary at 16, an attacker who may repaint one input column freely
moved the output by up to 45 logits at the boundary, by 0.014–8 logits at distance r − 1, and by **exactly 0.0 at every distance ≥ r**.
The edge of the guarantee is sharp and sits exactly at the trainable trunk's reach, as the proposition says it must.

**Renegotiation side.** How far training moved a misplaced boundary back (k*, columns; the largest correction over b = 12, 8, 4, mean of
seeds; "≥" where it came within one column of the shift itself, so the true reach may be larger):

| reach r | σ = 1 | σ = 2 | σ = 3 |
|---|---|---|---|
| 2 | 1.5 | 4.5 | 6.5 |
| 4 | 5.0 | 7.5 | ≥ 11.5 |
| 6 | 7.0 | 9.5 | ≥ 11.0 |

Fit k* = r + a·σ over the seven uncensored cells: a = 1.42; residuals −0.4 to +0.7 columns in six cells, −1.9 for r = 2, σ = 1.
Two of 72 runs failed outright (r = 6 with a large shift; the network output a constant, 49.7%) and are excluded.

**Against the predictions.** (1) *Exactly 0.0 at d ≥ r and > 10⁻³ at d = r − 1, every model:* **confirmed** (18 of 18). (2) *k* increases
with r at each σ and does not decrease with σ:* **confirmed** where measurable (σ = 3 reaches the cap at r ≥ 4). (3) *k* = r + a·σ with one
constant, every residual ≤ 1.5 columns, 1 ≤ a ≤ 2.5:* **wrong as stated**: a = 1.42 fits six of seven cells within 0.7 columns, but the
smallest network with the smallest blob falls 1.9 short, and wide blobs at r = 4 reach at least 3.3 further than the line. (4) *Where the
correction falls short, accuracy = 1 − (16 − b − k*)/32 within 3 points:* **wrong as stated**: 9 of 16 cells; the formula ignores the
2–5% error that remains even at the correct boundary, and accuracy sits on average 2.9 points below it. (5) *At b = 16, accuracy ≥ 93%:*
**confirmed** (95.0–98.9%).

**What it means.** The two halves of the law are different kinds of statement. The guarantee is a theorem: whatever training does,
nothing beyond the trainable parts' reach can move the output, and the experiment finds the edge exactly there, a drop from
logit changes of up to 45 to exactly zero within one column. Renegotiation is an empirical quantity bounded by the same reach: training
pulls a misplaced boundary back by roughly the reach plus 1.4 blob widths, i.e. by as far as the detector can still see the object.
It follows the bound closely for networks with reach ≥ 4 and falls short for the smallest one. So the design rule is: a contract is
exact beyond the reach of what is trainable in front of it, and anything inside that band, its boundary included, is open to
renegotiation by training, about as far as the trainable part can see.

## L2. The reach law on natural images (2026-11-06)

Script `src/l_series.py l2`; output `results/l2/`. AFHQ cats at 64×64, each cyclically shifted at random, with a red disc (σ = 2);
label = disc in the left half; trunk reach 4. Training backgrounds: 4,096 cats; test: 2,000 images on the other 1,057 cats.
Per seed (0, 1, 2): accuracy, and the correction measured with lone-disc probes ("failed": the network output a constant). The last column
is the accuracy ceiling 1 − (shift − 3)/64 if training moves the boundary back by 3 columns.

| shift | gate boundary | accuracy (%) | correction (columns) | ceiling at a correction of 3 (%) |
|---|---|---|---|---|
| 0 | 32 | 96.9, 97.4, 96.7 | 0, 0, -32 |  |
| 4 | 28 | 94.0, 95.9, 95.0 | 2, 2, 1 | 98.4 |
| 8 | 24 | 90.1, 89.0, 74.2 | 4, 2, -2 | 92.2 |
| 12 | 20 | 82.9, 51.2, 84.6 | 3, failed, 3 | 85.9 |
| 16 | 16 | 80.1, 75.7, 79.5 | 4, 2, 3 | 79.7 |

(Shift 0, seed 2: the probe called column 0 "right", so its correction reads −32; an artefact of the probe at the wrap-around edge, the run's accuracy is 96.7%.)

The circular network without a contract: 51.1%. Guarantee check (boundary at 32): an attacker repainting one input
column moved the output by 0.37–11 logits at distances 0–3 and by **exactly 0.0 at distances 4 and 5**, in all three models.

**Against the predictions.** (1) *The circular network without a contract is at chance:* **confirmed** (51.1%): the impossibility carries
over to natural images once they are cyclically shifted. (2) *The proposition holds exactly on natural images:* **confirmed** (3 of 3).
(3) *L1's law transfers: a correction within 1.5 columns of r + 1.42σ = 6.8, or below it but at least the reach (4):* **wrong**: training
moved the boundary back by only about 3 columns (2–4 per run at shifts 12–16), less than the reach. The accuracies agree: within a
point or two of the ceiling that a 3-column correction allows (after the ~3% error at the correct boundary). Two of 15 runs went wrong
(shift 12, seed 1: constant output; shift 8, seed 2: 74%, a boundary pulled the wrong way).

**What it means.** On natural images the two halves of the law separate further. The guarantee is untouched: exact beyond the reach, as on
synthetic data. Renegotiation shrinks from about 7.5 columns on clean backgrounds (L1, r = 4, σ = 2) to about 3 on cat fur: with clutter
around it, a detector that leans its response toward the gate also leans into false alarms, so training can use less of its reach. The
renegotiable band is set by what the trainable part can reliably see, not only by what it can reach; the guarantee depends on reach alone.

## E1b. The shielded edit under training that does not damage the model (2026-11-06)

Script `src/e1_shielded_edit.py e1b`; output `results/e1b/`. As E1, at lr 10⁻⁶ (from the drift check) with Adam's moments warmed for
100 steps at lr 0; 1,000 steps; seeds 0–2; evaluations at steps 20, 50 and every 100.

| arm | Rome top-1 on the 20 fitting paraphrases (every evaluation) | Rome probability, install → step 1,000 | validation loss, install → 1,000 | neighbouring prompts unchanged, install → 1,000 |
|---|---|---|---|---|
| no edit (control) | — | — | 3.064 → 3.058 | 8 → 6 (every seed) |
| unit, margin 30 | **20/20 at every evaluation, every seed** | 0.994 → **1.000** | 3.064 → 3.059 | 8 → 6 |
| fine-tuned edit (layer 10's MLP) | 20/20 at every evaluation, every seed | 1.000 → 0.975 (falling steadily) | 3.136 → 3.063 | 2 → 3 |

The unit's gradient was exactly zero in 96.0–97.3% of steps (14–26 of 2.05M training tokens fired it); its weights moved by at most
1.0–1.6 × 10⁻⁴, and its bias by exactly 0 (the updates fall below the float32 spacing at the bias's large magnitude).

**Against the predictions.** (1) *Control: validation loss within ±0.02, ≥ 7 of 8 neighbouring prompts keep the base answer:* **first half
confirmed** (−0.006), **second wrong**: 6 of 8 in every seed, so even gentle training flips two near-tie prompts; the edits do no worse.
(2) *Margin 30: gradient exactly zero in ≥ 95% of steps, Rome top-1 ≥ 18/20 at every evaluation, probability ≥ 0.95 at step 1,000:*
**confirmed** (20/20 throughout; 1.000). (3) *The fine-tuned edit keeps Rome probability ≥ 0.5:* **confirmed** (0.974–0.977). (4) *The
unit's weights still move (≥ 10⁻⁵):* **confirmed** (10⁻⁴), a tenth of E1's movement at a tenth of the learning rate.

**What it means, with E1.** Most of the fine-tuned edit's decay in E1 came from the damaging learning rate: under gentle training both
edits survive. What separates them is aggressiveness and side effects. Under aggressive training (E1) the shielded unit kept the edit
(0.98) while the fine-tuned edit faded (0.29); under gentle training (E1b) the unit does not decay at all while the fine-tuned edit
slides slowly (1.00 → 0.975). And the unit is clean from the start: it leaves all 8 neighbouring prompts and the validation loss as they
were, where the fine-tune changes 6 of 8 neighbours and costs 0.07 in validation loss (which continued training then repairs).
The shield remains statistical, not exact: its weights still move, in proportion to the learning rate.

## L3. Does the network's own position code wither once a contract supplies position? (2026-11-06)

Script `src/l_series.py l3`; output `results/l3/`. P1's zero-padding network, pretrained 2,000 steps (it learns position from the image
borders), then 2,000 more steps either as is (control) or with P1's contract installed on its score map. "Probe": a linear read-out of
left/right from the globally pooled trunk features, i.e. what the trunk still knows about position without the contract. Mean of seeds
0–2 (%).

| step after install | control: accuracy | control: probe | contract: accuracy | contract: probe |
|---|---|---|---|---|
| 0 | 81.2 | 82.8 | 55.9 | 82.8 |
| 100 | 80.3 | 83.5 | 98.2 | 78.6 |
| 250 | 82.5 | 84.2 | 97.7 | 78.4 |
| 500 | 83.1 | 83.9 | 98.8 | 78.5 |
| 1,000 | 83.8 | 84.8 | 98.8 | 78.8 |
| 2,000 | 84.5 | 85.8 | 99.3 | 78.3 |

**Against the predictions.** (1) *Control: probe ≥ 75% throughout:* **confirmed** (it even rises, 82 → 86%). (2) *Contract: accuracy ≥ 95%
by step 500:* **confirmed** (98% by step 100). (3) *Contract: the probe falls by ≥ 10 points by step 2,000:* **wrong**: it falls by
4.5 points (to 78.3%; 7.5 points below the control), almost all of it in the first 100 steps, and then stays flat.

**What it means, with S3.** The registration named the alternative outcome, and it is the one we got: an unused copy is not deleted.
When the contract is installed, the trunk reorganises quickly (the first 100 steps cost the own position code about 5 points while task
accuracy jumps from chance to 98%), and after that nothing pushes the old code either way, so it stays. In S3 the network's own bicubic
copy was deleted within ~500 steps, but there the copy was harmful: with the shortcut back, the low frequencies were counted twice. So
the reorganisation around installed structure has a specific trigger: gradient descent removes a redundant skill when keeping it costs
loss, and leaves it largely intact when it is merely unused. Dependence on the contract therefore grows only as far as the copy erodes:
here the network keeps a working fallback (~77–81% without the contract).

## L4. Adding a skill through a contract without forgetting the old one (2026-11-06)

Script `src/l_series.py l4`; output `results/l4/`. Circular networks pretrained on "is there a red blob?" (100% on that task), then
taught "is the red blob on the left?" from 32 labels in 300 steps, keeping the original head: (A) P3's contract on the shared score map;
(B) an unconstrained new head reading [trunk features, x-ramp] (1×1 conv → ReLU → 1×1 conv → mean). Per seed (0, 1, 2), in %:

| | new task (left) | old task after (red present; 100 before) |
|---|---|---|
| (A) contract | 96.7, 95.7, 74.7 | 100, 100, 73.4 |
| (B) unconstrained head | 50.3, 50.3, 50.3 | 100, 100, 50.2 |

**Against the predictions.** (1) *A's new-task accuracy ≥ B's + 5 points:* **confirmed** (B never learned the new task: a constant answer in
every seed). (2) *A's old-task accuracy drops by ≤ 2 points, and less than B's:* **wrong as stated**: no drop in seeds 0 and 1, but in seed 2
the contract run learned the new task only partly (74.7%) and the old task fell to 73.4%; B's seed 2 fell further (to chance) without learning
anything.

**What it means.** In two of three seeds the contract added the new skill from 32 labels with no forgetting at all, because it reuses the
existing detector instead of repurposing it. The third seed shows that this is not guaranteed: when fine-tuning through the contract goes
badly, the shared detector, and with it the old skill, is damaged too. The unconstrained head is a weak comparison here (it never learned the
task in 300 steps), so L4 supports the claim only modestly; P3 remains the stronger evidence that a contract transfers an existing skill.

## D5. Distilling the odd-eye patch into networks without it (2026-11-06)

Script `src/d5_distil_odd_eye.py`; output `results/d5/`. Teacher: paper one's left-eye patched CatGAN-26 (41.6% of left irises in band,
0% of right). Students start from the base generator and are trained 2,000 steps (lr 2 × 10⁻⁵, amended) to match the teacher's image for
fresh latents every batch. EMA students at step 2,000, mean ± sd over seeds 0–2, on paper one's 300 test latents.

| student | left irises in band | right irises in band | left only | right half vs teacher, mean (RGB levels) | dense layer's mirror asymmetry (0 at start) |
|---|---|---|---|---|---|
| (a) original architecture | 50.2 ± 0.2% | 5.9 ± 0.2% | 45.4% | 3.15 | 1.29 |
| (b) with room: 16 extra channels per layer | 45.9 ± 0.2% | **0.0%** | **46.1%** | **2.08** | 1.31 |
| (c) patched architecture, gate reset to zero, wire frozen | 50.2 ± 0.2% | 6.0 ± 0.0% | 45.4% | 2.97 | 1.29 |

**Against the predictions.** (1) *(a) puts ≥ 20% of left irises in band with ≤ 5% of right, and its right half drifts (> 0.5 levels):*
**left and drift confirmed** (50.2%; 3.15 levels), **right narrowly wrong** (5.9%). (2) *(b) ≥ (a):* **in part**: fewer left irises in
band (45.9 vs 50.2%) but more *left-only* ones (46.1 vs 45.4%), no right iris in band and a third less drift. (3) *(c) reaches 41.6% with
the right half exact to float rounding:* **first half confirmed, second wrong**: (c) matches (a) to within 0.2 points everywhere, with
the right half drifting by 3 levels: everything else in (c) trains too, so nothing protects the right half, and the reset gate did not
make (c) behave differently from (a).

**What it means.** The odd eyes, which gradient descent could not find from real cat images (the companion paper), are reachable with a
teacher: a plain generator learns them in 2,000 steps, putting more left irises in band than the teacher itself (50% against 42%), and the
dense layer's per-cell biases become left-right asymmetric (mirror asymmetry 0 → 1.3), the one place this generator can hold position.
What does not transfer is the teacher's exactness: no student keeps the right half unchanged (2–3 RGB levels of drift, against the
teacher's 1.6 × 10⁻⁷), and without room the student also tints 6% of right irises. With room (b) the student is the cleanest: no right
iris in band. As in A2, a patched teacher passes on a behaviour, not its guarantee.

## S1. Domestication: does the host adopt a grafted organ, and come to depend on it? (2026-11-06)

Script `src/s1_domestication.py` (interrupted by a GPU crash after 4 of 12 runs and resumed; seeded, so the remaining runs are as they
would have been); output `results/s1/`. Paper one's noisy-input expert, frozen, beside a TinySR-26 host, behind a gate computed from the
host's features (exactly 0 at install). 3,000 steps of paper one's mixed objective (clean bicubic, noisy bicubic and nearest inputs).
"Host alone" = the system with the gate forced to 0. PSNR change against paper one's base (dB), mean of seeds 0–2.

| arm | system: bicubic / noisy / nearest | host alone: bicubic / noisy / nearest | mean gate: bicubic / noisy / nearest |
|---|---|---|---|
| adult host + organ | +0.064 / +3.294 / +0.111 | +0.061 / **+0.973** / +0.108 | 0.09 / **0.94** / 0.06 |
| adult control (no organ) | +0.028 / +3.177 / +0.067 | same | — |
| young host + organ | −0.166 / +3.184 / −0.825 | **−4.66 / −0.57 / −1.89** | **0.91 / 0.92 / 0.92** |
| young control | −0.838 / +2.098 / −1.468 | same | — |

The organ alone: −0.20 / +3.26 / −1.01. Dependence on noisy inputs (system minus host alone) over training: adult 1.65 dB at step 250 →
2.32 at 3,000; young 0.35 → 3.75, both rising at every evaluation but one.

**Against the predictions.** (1) *The gate routes: ≥ 0.5 on noisy inputs, ≤ 0.2 on clean ones (adult):* **confirmed** (0.94, 0.09).
(2) *The host's own denoising withers: host-alone noisy PSNR ≥ 0.5 dB below the control's:* **confirmed** (+0.97 vs +3.18, 2.2 dB below).
(3) *Freed capacity: the system's clean-bicubic gain exceeds joint retraining's +0.03 dB by ≥ 0.05:* **narrowly wrong** (+0.064; this
experiment's own control +0.028). (4) *Dependence grows over training:* **confirmed** (adult 1.65 → 2.32, young 0.35 → 3.75 dB).
(5) *The young host withers more than the adult:* **confirmed**, far more: it hands *every* input type to the organ (gate 0.91–0.92 on
clean and nearest inputs too) and its own output falls to −4.7 dB on clean inputs.

**What it means.** The host hands over exactly the jobs the organ does better than it does. The adult host is better than the organ on
clean and nearest inputs, so it routes only noisy ones (0.94 vs 0.06–0.09); the young host is worse than the organ at everything (young
control: −0.84 / +2.10 / −1.47 against the organ's −0.20 / +3.26 / −1.01), so it routes everything. What it hands over, it loses, and
dependence grows step by step. With L3 this sharpens the deletion rule: here the host's skill is not harmful (the gate weights the host's
noisy-input output by only 1 − 0.94), but it competes for capacity with the jobs the host still does, and it fades; in L3 nothing
competed for the unused position code, and it stayed.

## L6. Does object-like clutter shrink renegotiation? (2026-11-06)

Script `src/l_series.py l6`; output `results/l6/`. L1's task and contract (reach 4, σ = 2) with 0, 2 or 6 orange distractor blobs per
image; gate boundary b ∈ {16, 12, 8, 4}; seeds 0–1. Effective boundary per seed (truth: 16), from clean probes:

| orange distractors | b = 16 | b = 12 | b = 8 | b = 4 | accuracy at b = 4 (%) | largest correction k* |
|---|---|---|---|---|---|---|
| 0 | 16, 17 | 16, 16 | 14, 15 | 12, 11 | 80.0, 82.2 | 7.5 |
| 2 | 16, 16 | 16, 16 | 14, 14 | 12, 11 | 78.0, 81.0 | 7.5 |
| 6 | 16, 15 | 0*, 16 | 14, 13 | 11, 25* | 82.0, 75.9 | ≈ 7 |

\* Probe artefacts: 0 = column 0 called "right" (the wrap-around edge, as in L2); 25 = a run whose response spread past the true boundary
(75.9% accuracy). Guarantee check at b = 16: 0.15–0.37 logits at distance 3, exactly 0.0 at distances 4 and 5, in all six models.

**Against the predictions.** (1) *k* decreases as orange distractors grow:* **wrong**: 7.5, 7.5, about 7. (2) *With none, within 1.5 of
L1's 7.5; with six, below the reach (4):* **first half confirmed** (7.5), **second wrong** (about 7). (3) *The guarantee is exact at every
clutter level:* **confirmed** (6 of 6).

**What it means.** Distractors of a different hue do not shrink the renegotiable band, so L2's shrinkage on cat fur is not explained by
"more things that look like the object". The orange blobs here are easy to tell from the red one (their green channel differs by 0.35),
so a detector can lean toward the gate without leaning into them. What natural backgrounds add is texture: large, spatially correlated
variation against which the faint tail of the disc is hard to see. The remaining hypothesis is visibility against background variation,
which L8 tests by raising the background noise.

## L7. Reach, not depth (2026-11-06)

Script `src/l_series.py l7`; output `results/l7/`. Three trunks with reach 4 on L1's task (σ = 2), boundary b ∈ {16, 12, 8, 4}, seeds 0–1.

| trunk (reach 4) | change at distance 3 (logits) | at distances 4, 5 | effective boundary at b = 8 / b = 4 (per seed) | k* |
|---|---|---|---|---|
| 2 layers, the second dilated by 3 | 2.3, 2.6 | **0.0, 0.0** | 13, 13 / 10, 10 | 6.0 |
| 4 layers (as L1) | 1.9, 0.37 | **0.0, 0.0** | 14, 14 / 12, 12 | 8.0 |
| 6 layers, the last two 1×1 | 0.68, 0.31 | **0.0, 0.0** | 14, 15 / 11, 10 | 6.5 |

For comparison, L1 at σ = 2: k* = 4.5 at reach 2, 7.5 at reach 4, 9.5 at reach 6.

**Against the predictions.** (1) *All three reach-4 trunks: exactly unchanged at distances ≥ 4, changed at distance 3:* **confirmed** (6 of 6
models). (2) *Each reach-4 trunk's k* within 1.5 of 7.5 and further from 4.5:* **confirmed for two trunks** (8.0; 6.5), **borderline for the
dilated one** (6.0: within 1.5 of 7.5, but exactly as far from 4.5).

**What it means.** The guarantee's edge follows reach, not depth or parameter count: three different architectures with reach 4 all have it
at exactly 4. Renegotiation is set by reach to within about a column: at reach 4 the three trunks move a misplaced boundary back by 6.0–8.0
columns, between L1's reach-2 (4.5) and reach-6 (9.5) values; the two-layer dilated trunk, which sees its outer pixels through only three
taps, uses a little less of its reach.

## D4. An exact switch with a branch trained by GAN: cat → anime (block 3 only; 2026-11-06)

Script `src/d4_switch.py 3 2e-4`; output `results/d4/` (grids `grid_k3_s*.png`). A copy of CatGAN-26's blocks 4–5 and output layer reads
block 3's features behind an exact switch on an extra latent bit; only the branch (and a discriminator) trains, on 63,565 anime faces,
3,000 steps, seeds 0–2. The block-2 variant was dropped (the person's decision, to save GPU time).

| seed | switch off vs CatGAN-26, float32 / float64 | Fréchet distance to anime faces, step 0 → 3,000 | layout correlation, matched vs shuffled | colour correlation, matched vs shuffled |
|---|---|---|---|---|
| 0 | 3.0 × 10⁻⁶ / 1.2 × 10⁻¹⁵ | 720 → 331 | 0.33 vs 0.04 | 0.03 vs −0.08 |
| 1 | 3.2 × 10⁻⁶ / 1.2 × 10⁻¹⁵ | 720 → 343 | 0.33 vs 0.05 | 0.04 vs −0.06 |
| 2 | 3.0 × 10⁻⁶ / 1.1 × 10⁻¹⁵ | 720 → 348 | 0.42 vs 0.04 | 0.08 vs −0.04 |

**Against the predictions.** (1) *Off is exact to float rounding:* **confirmed** (10⁻¹⁵ in float64). (2) *The branch keeps the cat's colours
and pose, correspondence well above shuffled pairs:* **confirmed for layout** (0.33–0.42 against 0.04), **weak for colour** (a gap of
0.09–0.12). (3) *The block-2 variant:* not run.

**What it means.** One network, two exact modes. With the switch off it is the cat generator to 10⁻¹⁵; with it on, a branch trained only
on unpaired anime faces turns each latent's cat into an anime face that keeps the cat's layout: face position, pose and, visibly, hair
colour following fur colour (a black cat becomes a dark-haired face). The frozen trunk's features shape what the new mode learns, without
paired data. At 3,000 steps the anime faces are recognisable but rough (Fréchet distance 331–348).

## L8. Renegotiation and visibility (2026-11-06)

Script `src/l_series.py l8` (and `l8g`, the guarantee recheck); output `results/l8/`, `results/l8g/`. L1's task and contract (reach 4,
σ = 2) with background noise of standard deviation 0.05, 0.15 or 0.3 in training, test and probe images; seeds 0–1. Effective boundary
per seed (truth: 16):

| background noise | b = 16 | b = 12 | b = 8 | b = 4 | largest correction k* |
|---|---|---|---|---|---|
| 0.05 (as L1) | 16, 16 | 16, 16 | 14, 15 | 12, 24* | 8 |
| 0.15 | 16, 17 | 16, 16 | 13, 14 | 12, 11 | 7.5 |
| 0.3 | 16, 17 | 16, 16 | 13, 13 | 10, 9 | 5.5 |

\* A run whose response spread past the true boundary (as one run in L6); with it, k* at 0.05 would read 10.

**Guarantee check, and an artefact found and fixed.** The first check showed changes of 0.02–0.86 logits at distances 4 and 5 for noise 0.15
and 0.3, identical at both distances. The attack clipped the *whole* image to [0, 1] after each step; with heavy noise, 0.07% (0.15) and
8.3% (0.3) of pixels lie outside that range, so the clipping changed pixels inside the gate's region. (At noise 0.05 no pixel lies outside
it, so L1, L2, L6, L7 and P4 are unaffected.) With clipping confined to the attacked column, retrained models (`l8g`, CPU, seeds 0–1)
give 0.20–0.44 logits at distance 3 and **exactly 0.0 at distances 4 and 5 at every noise level** (6 of 6).

**Against the predictions.** (1) *k* decreases as the noise grows:* **confirmed** (8 → 7.5 → 5.5). (2) *Within 1.5 of 7.5 at 0.05; below
the reach at 0.3:* **first half confirmed** (8), **second wrong** (5.5 > 4). (3) *The guarantee exact at every noise level:* **confirmed**
after the attack fix (6 of 6).

**What it means.** Visibility, not distractors, sets the renegotiable band: the noisier the background, the less of its reach the detector
can use to lean toward the gate (8 → 5.5 columns), while the guarantee stays exact. Independent pixel noise partly averages out over the
detector's 9×9 field, which is why even noise 0.3 leaves the band above the reach; the correlated texture of cat fur (L2, ~3 columns) does not
average out the same way.

## S4. Transplant and rejection (2026-11-06)

Script `src/s4_transplant.py`; output `results/s4/`. Block 2 of paper one's TinySR-26 replaced, then healed 1,000 steps on the D2
objective. PSNR change against the base (dB), bicubic / noisy / nearest inputs, mean of seeds 0–2 (every seed within 0.07 dB of the
mean at step 1,000).

| transplant | step 0 | step 200 | step 500 | step 1,000 |
|---|---|---|---|---|
| (a) same lineage: the bicubic expert's block 2 | −0.62 / −0.04 / −0.48 | +0.01 / +0.05 / −0.04 | 0.00 / +0.04 / 0.00 | 0.00 / +0.04 / 0.00 |
| (b) same lineage: the noisy expert's block 2 | −0.04 / **+0.28** / −1.04 | −0.01 / +0.11 / −0.09 | −0.01 / +0.11 / −0.01 | 0.00 / +0.06 / 0.00 |
| (c) foreign: block 4 of a 9-block network trained from scratch | −0.98 / −0.12 / −1.87 | −0.29 / +0.06 / −0.76 | −0.18 / +0.06 / −0.52 | **−0.12 / +0.04 / −0.31** |
| (d) foreign + closed-form 1×1 stitching adapters | **−0.55 / +0.04 / −0.94** | −0.26 / +0.09 / −0.70 | −0.19 / +0.07 / −0.53 | −0.14 / +0.13 / −0.45 |
| (e) control | 0 / 0 / 0 | 0.00 / +0.02 / 0.00 | 0.00 / +0.02 / 0.00 | 0.00 / −0.01 / −0.01 |

**Against the predictions.** (1) *Same-lineage transplants cost < 0.3 dB at step 0, are within 0.05 dB of the control by step 200, and (b)
carries ≥ +0.3 dB of noisy-input skill:* **mostly wrong**: (a) costs 0.62 dB and (b) 1.04 dB at step 0 (on bicubic and nearest inputs
respectively); (a) is within 0.05 dB by step 200, (b) by step 300; (b) carries +0.28 dB (narrowly short). (2) *The foreign transplant costs
> 3 dB and is not within 0.1 dB of the control by step 500:* **first half wrong** (it costs 1–1.9 dB), **second confirmed** (0.12–0.31 dB
short even at step 1,000). (3) *Stitching cuts the foreign transplant's damage below 1 dB:* **confirmed** (0.55 / 0.94).

**What it means.** A block from a sibling network (same pretraining) is accepted within 200–300 steps; a block of the same shape from an
independently trained network is "rejected" in the sense of slow integration: it costs 1–2 dB and is still 0.1–0.3 dB short after 1,000
steps. Closed-form stitching adapters halve the damage at install but do not speed up healing (on nearest inputs the stitched transplant
ends further behind, −0.45 vs −0.31 dB). And a skill carried in by the transplant is not kept unless the objective uses it: the noisy
expert's block brings +0.28 dB on noisy inputs, which fades to +0.06 dB under an objective without noise.

## S2. Lesion atlas: remove a part, watch it heal (2026-11-06)

Script `src/s2_lesion_atlas.py`; output `results/s2/`. Each lesion on paper one's TinySR-26, healed 1,000 steps on the D2 objective. PSNR
change against the base (dB), bicubic / nearest inputs, mean of seeds 0–2 (seeds agree within 0.06 dB at step 1,000). "Largest weight
changes": the two layers with the largest relative change (‖ΔW‖/‖W‖) during healing, named by their original block.

| lesion | step 0 | step 1,000 | largest weight changes |
|---|---|---|---|
| remove block 0 | −0.43 / −3.47 | −0.17 / −0.50 | blocks 1, 2 |
| remove block 1 | −0.56 / −3.94 | −0.22 / −0.65 | blocks 2, 0 |
| remove block 2 | −1.46 / −0.96 | −0.16 / −0.47 | blocks 3, 1 |
| remove block 3 | −0.34 / −2.01 | −0.13 / −0.23 | blocks 4, 2 |
| remove block 4 | −0.66 / −0.37 | −0.10 / −0.21 | blocks 3, 5 |
| remove block 5 | −0.91 / −1.65 | −0.14 / −0.26 | block 4, output conv |
| remove the tail conv | **−3.23 / −3.53** | −0.08 / −0.11 | output conv, block 5 |
| remove the head skip | −0.94 / −0.99 | **−0.01 / +0.01** (healed by step 300) | tail, output conv |
| 12 of 48 channels in block 2, crude | −0.23 / −0.11 | −0.01 / −0.01 (healed by step 200) | block 2 |
| the same, sutured (closed-form refit) | −0.02 / −0.07 | −0.01 / −0.02 (healed by step 100) | block 2 |

**Against the predictions.** (1) *Removing the head skip is the most damaging lesion:* **wrong**: removing the tail conv (−3.2 / −3.5 dB) or
an early block (−3.5 to −3.9 dB on nearest inputs) is worse; the head skip costs ~1 dB and heals completely. (2) *Every single-block removal
heals to within 0.1 dB of the control by step 1,000:* **wrong**: 0.10–0.22 dB short on bicubic inputs and 0.21–0.65 on nearest. (3) *Healing
concentrates next to the lesion:* **confirmed**: in every block removal the largest weight changes are in the blocks on either side of the
gap (for block 0, whose other neighbour is the head, in the next two blocks). (4) *Sutured channel removal starts ≥ 5× closer than crude and
is within 0.05 dB by step 100; crude needs ≥ 300 steps:* **partly**: 12.6× closer on bicubic inputs but only 1.6× on nearest; sutured within
0.05 dB by step 100 (confirmed), crude by step 200 (faster than predicted).

**What it means.** Healing is local: the network rebuilds around a missing block mostly in the blocks that border it. Removing whole blocks
leaves a lasting deficit at this budget (1,000 steps), while removing a quarter of one block's channels, or the head skip, is fully repaired;
a closed-form suture makes a channel lesion nearly free from the start. The early blocks matter most for nearest-neighbour inputs, the tail and
the middle block for bicubic ones.
