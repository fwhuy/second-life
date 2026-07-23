# Second Life AI — Final Submission

**Track 1: Image Classification** · NYU Shanghai, Introduction to Artificial Intelligence
Six-class waste classifier (cardboard, glass, metal, paper, plastic, trash).

## Contents

```
submission/
├── README.md      this file — what is where, and how it maps to the rubric
├── GUIDE.md       step-by-step: environment → data → train → evaluate → demo
├── paper/         Second-Life-AI-paper.pdf (+ paper.html source)
├── poster/        posterv1.pdf
├── code/          src/, scripts/, configs/, tests/, data/splits, requirements, experiments.csv
├── figures/       the six figures, all regenerated from committed data
└── results/       per-run reports: confusion matrices, per-class metrics, curves
```

## Where each rubric band is answered

### Technical Work — 60 pts

**Project design & rationale (15).** Paper §1 states the problem and §2.2 the
pre-declared acceptance gate. Alternatives with reasoning: `code/configs/` holds
one YAML per experiment, and paper Table 4 compares four candidate backbones with
the reason each was or wasn't chosen. Five tuning experiments were run and four
were **rejected** — paper §2.2 and Figure 5 give the reasoning for each rejection.
Limitations are §7.

**Code standard & completeness (12).** `code/` runs end to end; `GUIDE.md` is the
step-by-step. Training and evaluation are **separate entry points**:

- train — `python -m src.train --config configs/baseline.yaml`
- evaluate — `python -m src.evaluate --checkpoint <path> --tta`
- large-corpus training — `python scripts/kaggle_train.py` (self-contained)

Every module carries a docstring explaining intent, not just mechanics. Parameter
cap violations, missing checkpoints, and absent data files all raise with an
actionable message rather than failing late.

**Experiments & analysis (23).**

- *train/val/test splits* — `code/data/splits/` (TrashNet) and
  `code/data/splits_unified/` (pooled corpus) hold committed, seeded, group-aware
  fold indices; Table 1 in the paper gives the arithmetic. Verify with
  `cd code && python -m pytest tests/ -q` — 11 tests, including split disjointness.
- *data augmentation* — paper §2.2 (rejected "modern augmentation" ablation) and
  `scripts/kaggle_train.py` (TrivialAugment, Mixup/CutMix, random erasing).
- *hyperparameter tuning* — `scripts/autotune.py`; ablations logged in
  `code/experiments.csv`, one row per run with a config hash.
- *visualised confusion matrix* — `figures/fig9_confusion_matrix.png`, plus
  per-run matrices in `results/`.
- *misclassified samples with actionable fixes* — `figures/fig6_per_class_recall.png`
  shows the failure concentrated in *trash*; the actionable fix is documented and
  taken: expanding that class from 137 to 1,033 images (paper §3, Figure 2).

**Limitations & future extensions (10).** Paper §7 — seven named limitations,
each specific (image-level metric resolution, thresholded duplicate detection,
unaudited cross-source overlap, OOD threshold from a single example).

### Poster & Defense — 40 pts

**Poster compliance (10).** `poster/Second-Life-AI-poster.pdf` — five sections
Problem → Data → Method → Result → Takeaways, portrait, 846.7 x 1074.3 mm
(prints to A0 scaled on width). `poster/poster.html` is the editable source.

**Information & visualisation (10).** All figures in `figures/` are labelled with
axis titles and captions and were generated at print resolution.

**Live defense (12) / Q&A (8).** Every number on the poster traces to a config
hash in `code/experiments.csv`. See "Defending the number" below.

### Tie-breaker bonuses

- **Multi-model comparison** — the web app compares a ConvNeXt V2-Tiny ConvNet
  and a Swin-B Transformer side-by-side on one photo.
- **Expanded dataset** — five public sources pooled to ~19,500 images from a
  2,527-image starting point.
- **Live demo of visualised model outputs** — the offline web app, below.

## Running the demo

```bash
cd website && .venv/bin/python app.py     # http://127.0.0.1:5001
```

Upload or use the camera. Fully offline — no network needed during the defense.
The model toggle switches between the ConvNet and Transformer. The live project
loads their canonical checkpoints directly from `model/`; no checkpoint copy is
needed.

## Defending the number

Ranking is on raw test accuracy, and our reported test figure is a **single,
never-repeated measurement of 89.20%** taken before any tuning. Two things worth
saying plainly if asked:

1. It is a snapshot, not the score of the final model. It was measured early and
   never used to select anything, which is why it is lower than our validation
   accuracy of 94.93%.
2. Our splits are group-aware — near-duplicate photographs of the same object are
   kept on one side of the split. Public waste datasets overlap heavily, and a
   random image-level split across them puts the same photograph in both training
   and evaluation. Our number is measured with that path closed.

If a number looks implausibly high for this task, the split is the first thing to
check, not the model.
