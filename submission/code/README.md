# Current model code

This directory contains the canonical training and evaluation programs for the
two models reported in the paper and served by the website.

## Scripts

- `train_convnextv2.py` — six-class, 384px ConvNeXt V2-Tiny; downloads and
  deduplicates OMA + RealWaste before group-aware training.
- `train_swin_b.py` — ten-class, 224px Swin-B with augmentation, held-out test,
  and TTA evaluation.
- `evaluate.py` — **separate testing entry point** for either model: loads a
  checkpoint, scores a labelled image set, and writes metrics, a confusion
  matrix, and error tables. Runs on CPU/MPS/CUDA.
- `verify_artifacts.py` — validates recorded metrics and repository checkpoints
  (identities, class orders, parameter counts, scores, SHA-256).

The scripts are submission-local copies of the canonical implementations in the
repository-level `model/` directory as of 2026-07-23. Checkpoints are not copied
here because they total roughly 438 MB; `../results/manifest.json` records their
canonical paths and exact hashes.

## Install

```bash
pip install -r requirements.txt          # ranges
# or, for the exact tested versions:
pip install -r requirements.lock.txt
```

## Evaluate a model

`evaluate.py` reads a folder of class sub-directories (`cardboard/`, `glass/`, …)
**or** an explicit CSV manifest (`path,label` per row), and is deterministic
(fixed seed, evaluation transforms only, no shuffling).

```bash
# ConvNeXt V2-Tiny on a labelled folder
python evaluate.py --model convnet \
    --data-root /path/to/images --out-dir out/convnet

# Swin-B, restricted to the six shared classes, from a manifest
python evaluate.py --model transformer --classes six \
    --split-manifest test.csv --out-dir out/swin

# Quick CPU smoke check on the first 32 images, no plot
python evaluate.py --model convnet --data-root imgs \
    --out-dir out --device cpu --limit 32 --no-plots
```

It writes to `--out-dir`: `metrics.json` (accuracy, macro-F1, per-class
precision/recall/F1, loss, checkpoint SHA-256, config echo), `predictions.csv`,
`confusion_matrix.png` + `.json`, and `errors_highest_loss.csv` /
`errors_confident_wrong.csv` for error analysis. It exits non-zero with a clear
message on a missing checkpoint, an architecture/class-order mismatch, or an
empty split.

> Note: the transformer's `--classes native` mode scores all ten of its trained
> classes; `--classes six` (the default) masks its logits to the six classes it
> shares with the ConvNet so the two models answer the same question.

## Run the tests

```bash
pytest tests -q
```

The suite covers evaluate.py's class-masking contract, deterministic transforms,
a CPU end-to-end run, and every failure mode; checkpoint parameter counts and
SHA-256 integrity (skipped automatically when the weights are absent); and the
group-disjoint, reproducible split invariant that ConvNeXt's validation number
depends on.
