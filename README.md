# Second Life AI — TrashNet Garbage Classifier

A 6-class garbage image classifier for the [TrashNet](https://github.com/garythung/trashnet)
dataset, built for a one-week NYU Shanghai AI course team competition (judged 50%
technical, 50% presentation). The goal isn't just a high number — it's **a high
number that survives scrutiny**, plus a poster/paper explaining *why* the 99–100%
accuracy commonly reported on this dataset is usually inflated by data leakage.

The trained model is the engine behind **Second Life AI**, a planned demo web app
that identifies a piece of trash, tells you how to dispose of it, and imagines the
object's "second life."

For the complete development story—including every major pipeline and website change,
experiment decisions, failures, fixes, limitations, and paper/presentation guidance—see
**[`PROJECT_PROCESS_README.md`](PROJECT_PROCESS_README.md)**.

- **Classes (2,527 images):** paper (594), glass (501), plastic (482), metal (410), cardboard (403), trash (137)
- **Current best:** 94.93% validation accuracy (ResNet-50 + 4-view TTA), leakage-free by construction
- **Status:** model pipeline complete and reproducible; stronger backbone + ensemble run are next, and the offline website is implemented

## Results

| Metric | Result |
|---|---:|
| Validation — ResNet-50 baseline | 404/434 = **93.09%** |
| Validation — + 4-view TTA | 412/434 = **94.93%** |
| Test — single pre-intervention measurement (now spent) | 322/361 = **89.20%** |

> The 89.20% test figure is a **single, pre-registered measurement** taken before
> any tuning and never used to select a model — always cite it with that caption.
> A manual audit of the worst test errors found roughly **half are mislabeled or
> genuinely ambiguous** (part of the poster story). Honest ceiling for clean
> methodology on this dataset: **~96–97%**. Anything at 99–100% almost always
> comes from the leakage this pipeline is designed to avoid — and to demonstrate.

## Repository structure

```
.
├── README.md            This file — project overview
├── GUIDE.md             Step-by-step training guide (start here to train)
├── model/               The ML pipeline (all code, data splits, results)
│   ├── src/                 pipeline modules (train, evaluate, kfold, leakage, analysis)
│   ├── configs/             one YAML per experiment (baseline.yaml is the reference)
│   ├── scripts/             setup + dataset download + preflight checks
│   ├── data/splits/         committed, seeded, group-aware fold indices
│   ├── reports/             committed figures, metrics, and the TEST_SPENT guard
│   ├── experiments.csv      append-only run log (every run, hashed + reproducible)
│   ├── demo/                offline Gradio demo (photo → class + confidence + Grad-CAM)
│   └── tests/               split/leakage unit tests
└── website/             Implemented offline Second Life AI web app
```

## Methodology — the rules enforced at runtime

These constraints are the whole point; they're what make the numbers trustworthy.

1. **The test set is touched exactly once.** All 361 test images are quarantined
   and every train/eval entry point asserts no test image — or its near-duplicate
   *group* — leaks into training. A guard file (`model/reports/TEST_SPENT.json`)
   physically blocks re-running the final evaluation.
2. **Group-aware splits.** TrashNet has many photos of the *same object* from
   different angles. Near-duplicates are detected and kept within a single split,
   so the model is never tested on an object it trained on. This is the exact leak
   that inflates published numbers.
3. **Split first, augment after** — augmentation applies to training data only.
4. **Everything is logged and reproducible** — fixed seeds, frozen dependencies,
   an append-only `experiments.csv`, and pre-registered acceptance gates decided
   *before* each experiment ran.

## Quickstart

```bash
cd model
bash scripts/setup_env.sh                 # venv + deps + GPU smoke test
source .venv/bin/activate
python scripts/download_data.py           # fetch data, verify counts, dedup, build splits
python -m src.train --config configs/baseline.yaml        # train the baseline
python -m src.evaluate --checkpoint checkpoints/baseline_resnet50/fold0/best.pth --tta
```

To train the full pipeline from scratch — baseline → backbone bake-off →
5-fold ensemble → the poster experiments — follow **[`GUIDE.md`](GUIDE.md)**.

### Offline demo

```bash
cd model
python demo/app.py --checkpoint checkpoints/<winner>/ensemble.json
```

Opens `http://127.0.0.1:7860` — upload a photo, get class + confidence + a
Grad-CAM overlay. Fully offline, safe to run in front of judges.

## What's next

- **Stronger model.** A backbone bake-off (EVA-02, ConvNeXtV2, SigLIP-2 — all far
  ahead of the 2015-era ResNet-50) → 5-fold cross-validation of the winner →
  a 10-model two-architecture ensemble. Calibrated expectation: **96–97.5%**
  out-of-fold accuracy; anything higher we treat as a bug until proven otherwise.
- **Poster experiments.** A leakage demo (honest split-first vs. leaky
  augment-then-split, same model), a cleanlab + manual label audit, and Grad-CAM
  figures.
- **The website.** Integrate the final trained ensemble into the implemented offline app
  (identify → disposal guidance → the object's "second life").

## Notes for the team

- **Reporting numbers:** validation numbers can be used freely; the 89.20% test
  number must always carry the pre-intervention caption above. Spending a *fresh*
  test measurement on the final ensemble is a one-shot group decision — no re-rolls.
- **Never edit `experiments.csv` by hand** — it's the audit trail.
- **Differences under ~1.5% (≈5 images) are noise** — don't chase them.
