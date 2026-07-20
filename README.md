# TrashNet Image Classification — Project Overview

*Last updated: 2026-07-20. This is the group-facing summary; the code and full
audit trail live in `trashnet-current/`.*

## What we're building

A classifier for the **TrashNet dataset**: 2,527 photos of garbage in 6 classes
(paper 594, glass 501, plastic 482, metal 410, cardboard 403, trash 137). It's a
one-week team competition judged 50% on technical merit and 50% on presentation.
Every team gets the same task, so our angle is a **high accuracy number that
survives scrutiny**, plus poster findings about *why* published numbers on this
dataset (99–100%) are often inflated.

## The rules the whole pipeline is built around

1. **The test set (361 images) is touched exactly once, at the very end.** All
   decisions are made on validation only. A guard file (`reports/TEST_SPENT.json`)
   physically blocks re-running the test eval.
2. **Group-aware splits.** TrashNet contains many photos of the *same object*
   from different angles. Near-duplicates are detected and kept inside one split,
   so the model is never tested on an object it trained on. This is the leak that
   inflates the literature numbers.
3. **Split first, augment after** — augmentation is applied to training data only.
4. **Everything is logged and reproducible**: seeds fixed, every run appends to
   an append-only `experiments.csv`, every decision gate was pre-registered
   before the experiment ran.

## What has happened so far

**July 17 — codebase built.** Full pipeline: data download + dedup + splits,
training with resume, evaluation with TTA, 5-fold CV ensembling, leakage
experiment harness, label-audit tools, offline Gradio demo.

**July 18 — first training session (Windows GPU PC).** A ResNet-50 baseline was
trained, then three pre-registered improvement attempts were run. All three
failed their acceptance gate (+7 net validation images required):

| Experiment | Val result (TTA) | Net vs. baseline | Verdict |
|---|---:|---:|---|
| 4-view test-time augmentation (TTA) | 412/434 | **+8** | ✅ adopted |
| Progressive resize 224→384px | 412/434 | 0 | ❌ rejected |
| Modern augmentation (TrivialAugment etc.) | 415/434 | +3 | ❌ rejected (trash-class recall collapsed) |
| Weight decay 0.05→0.10 | 411/434 | −1 | ❌ rejected |

The session stopped itself after two consecutive failed steps, per its
pre-registered stopping rule. Key insight: recipe tweaks on a 2015-era backbone
just shuffle *which* images are wrong — the remaining headroom is in the
**backbone itself** and in **ensembling**, neither of which had been tried yet.

### Current official numbers

| Metric | Result |
|---|---:|
| Validation, ResNet-50 baseline | 404/434 = **93.09%** |
| Validation, + 4-view TTA | 412/434 = **94.93%** |
| Test (single pre-intervention measurement, now spent) | 322/361 = **89.20%** |

Context for the test number: a manual audit of the highest-loss test errors
found roughly **half look mislabeled or genuinely ambiguous** — part of our
poster story, not just an excuse. Honest ceiling estimate for this dataset with
clean methodology: **~96–97%**. (Published 99–100% claims come from the leakage
our pipeline is designed to avoid — and to demonstrate.)

**July 19 → 20 — first overnight optimization run failed silently.** The new
"autopilot" (backbone bake-off → 5-fold ensemble) hung at its very first step:
downloading pretrained weights from huggingface.co, which the training PC
apparently couldn't reach. Diagnosed and fixed: there is now a **`test.bat`
preflight** that checks the GPU, data, downloads *all* weights up front (with an
automatic mirror fallback), and runs the unit tests before any overnight run.

## What happens next

1. **Training PC:** unzip `trash-classification-v2-20260720.zip`, double-click
   **`test.bat`**, wait for `ALL CHECKS PASSED`, then double-click **`train.bat`**
   and leave it overnight (~8–16 h, resumable if interrupted).
   By default it runs the maximum-accuracy pipeline: a bake-off of modern
   backbones (EVA-02, ConvNeXtV2, SigLIP-2 — all far stronger than ResNet-50) →
   5-fold CV of the winner → a 10-model two-architecture ensemble.
   Results land in `reports/AUTOPILOT_SUMMARY.md`. Calibrated expectation:
   **96–97.5%** out-of-fold accuracy; anything above that we treat as a bug
   until proven otherwise.
2. **Poster experiments (not yet run):**
   - **Leakage demo:** identical model, two pipelines — split-first (honest) vs.
     augment-then-split (leaky) — to reproduce the mechanism behind 99–100%
     literature claims.
   - **Label audit:** cleanlab + manual review to quantify mislabeled images.
   - **Grad-CAM** figures showing what the model actually looks at.
3. **Paper + A0 poster**, plus the offline demo app (`demo/app.py`: upload a
   photo → class + confidence + Grad-CAM overlay).

## Folder map

| Path | Size | What it is |
|---|---:|---|
| `README.md` | — | This file |
| `trashnet-current/trash-classification/` | 51 MB | **The live codebase** — same content as the zip below. Start here to read code or run the demo/tests |
| `trash-classification-v2-20260720.zip` | 41 MB | Ready-to-send bundle for the Windows training PC (includes the dataset; run `test.bat` first, then `train.bat`) |
| `trashnet-current/archive-20260718-results/` | <1 MB | July 18 session results (`FINAL_SUMMARY.md`, `RESULTS.md`, `EXPERIMENT_LOG.md`, old reports) + original project spec (`IMPLEMENTATION_PROMPT.md`) — source of the numbers above |
| `trash-classification/` | 47 MB | Superseded July 17 first version of the code — historical only, use `trashnet-current` instead |

No Python environments are included anywhere — `test.bat` (Windows) or
`scripts/setup_env.sh` (Mac/Linux) rebuilds one automatically.

## Headline numbers policy (for whoever writes the poster)

Use the validation numbers freely. The **89.20% test number must always be
captioned** as the single pre-intervention measurement — it was never used to
pick a model. When the new ensemble is done we either (a) report its pooled
out-of-fold accuracy (n=2,166, leakage-free) and keep the old test caption, or
(b) consciously spend one fresh test measurement on the final ensemble. That's
a group decision — one shot, no re-rolls.
