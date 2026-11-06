# Where patching meets gradient descent

Follow-up to *Beyond Fixed Architectures: An LLM Agent Rewires Trained Neural Networks with Exact,
Tested Patches* ([wsse](https://github.com/v-code-895/wsse), DOI
[10.5281/zenodo.23956114](https://doi.org/10.5281/zenodo.23956114)).

Paper one installs exact structure in trained networks: coordinate wires, AND gates, routed unions,
exception units, grown width and depth. This project asks one question: **what does gradient descent
do with structure it could not find on its own?** Every result is placed on a map: is the behaviour
*representable* in the original architecture, *reachable* by gradient descent unaided, or *reachable
with help* once a patch installs the structure?

Work in progress; nothing here is released yet.

## How the project is run

- **[PREREG.md](PREREG.md)**: arms, metrics and predictions, written before each experiment runs and
  never edited after it starts.
- **[RESULTS.md](RESULTS.md)**: what each run produced, checked against its predictions, wrong ones
  included.
- **[LOG.md](LOG.md)**: every proposal (person or agent), what stopped it and how it was resolved, as in
  paper one's Table 11.
- **[MAP.md](MAP.md)**: the representable / reachable / reachable-with-help table.

## Status

| experiment | question | status |
|---|---|---|
| B2 gradient shielding | which circuit weights can gradient descent not move? | done |
| B1 pilot | can the patched GAN keep training? | done (setup unusable; stability check running) |
| C1 frozen skeleton, trained detector | can SGD improve the detector inside a frozen wire and gate? | done |
| C1b detector trade-off curves | is SGD's detector better at equal collateral? | running |
| A3 distilling the union | was the merge conflict optimisation or capacity? | running |
| D2 exact vs approximate depth growth | does exact insertion heal better? | running |
| A1, B1, D4, E1 | odd-eye 2×2, continued GAN training, fused generators, one circuit in a small LLM | next |

## Layout

```
src/        one script per experiment; common.py (CatGAN-26, odd-eye patch, trainable folded
            generator) and sr_common.py (TinySR-26 training and evaluation)
results/    one folder per experiment (JSON, Markdown tables, logs); caches and checkpoints are not tracked
data/       datasets, not tracked
```

## Running

The scripts import paper one's code unchanged and expect a checkout of
[wsse](https://github.com/v-code-895/wsse) beside this folder (`../wsse`) with its data and models
(see its DATA.md and MODELS.md). Paper one's step-30,000 CatGAN-26 training state (`catgan26.pt`, a
v1.0.0 release asset) is read from `../release/v1.0.0/`. Use wsse's environment
(`requirements.txt`, `requirements-gpu.txt`); tested on an RTX 3050 8 GB with Python 3.9 and
PyTorch 2.8 (CUDA 12.6).

```
python src/b2_gradient_shielding.py
```

On Windows, set `KMP_DUPLICATE_LIB_OK=TRUE`.

## License

MIT (code). See [LICENSE](LICENSE).
