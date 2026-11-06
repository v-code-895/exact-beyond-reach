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
