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
