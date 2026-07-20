# Training Guide (start here)

You got this as a zip. It's a complete, reproducible pipeline for the TrashNet
6-class garbage dataset. The dataset (2527 images) and the frozen train/test
splits are **already inside the zip** — you don't need Kaggle or any account.
You just need a machine with Python 3.10+ and ideally an NVIDIA GPU.

## 0. One-time setup (~5 min + downloads)

```bash
unzip trash-classification.zip && cd trash-classification
bash scripts/setup_env.sh
source .venv/bin/activate
```

The script creates `.venv`, installs everything, prints your GPU name, and
checks the model names. If it prints `WARNING: CPU only`, training will be very
slow — find a GPU machine.

Then verify the data + splits that shipped in the zip:

```bash
python scripts/download_data.py   # sees data already there, re-runs leak checks
pytest tests/ -q                  # split/leakage unit tests
```

## 1. Sanity gate (do NOT skip)

```bash
python -m src.train --config configs/baseline.yaml --overfit-batch
```

This trains on a single batch and must drive the loss to ~0 ("PASS"). If it
fails, the pipeline is broken — stop and debug before real training.

## 2. Baseline (the reference number, ~1-2 h on a decent GPU)

```bash
python -m src.train --config configs/baseline.yaml
```

Expect **~90–95% val accuracy**. Everything later is measured against this.
If it crashes mid-run, re-run with `--resume`. Curves and confusion data land
in `reports/baseline_resnet50/`, and every run appends a row to
`experiments.csv` — never edit that file, it's the audit trail.

```bash
python -m src.evaluate --checkpoint checkpoints/baseline_resnet50/fold0/best.pth
```

## 3. Improvements (in order; keep a change only if val gain > ~1.5%)

```bash
# 3a. backbone bake-off on one fold each (pick the winner by val acc)
python -m src.train --config configs/convnextv2_224.yaml --fold 0
python -m src.train --config configs/effnetv2_s_224.yaml --fold 0
# (copy a config to try eva02 / swinv2 / effnetv2_m — names are pre-verified)

# 3b. progressive resizing: edit configs/convnextv2_384.yaml so
#     init_checkpoint points at the 224 winner's best.pth, then:
python -m src.train --config configs/convnextv2_384.yaml --fold 0

# 3c. the final model: 5-fold ensemble of the winner (overnight job, resumable)
python -m src.kfold --config configs/<winner>.yaml
python -m src.evaluate --checkpoint checkpoints/<winner>/ensemble.json --tta
```

## 4. The poster findings

```bash
python -m src.leakage_experiment            # Arm A vs Arm B table → reports/
python -m src.analysis --audit    --checkpoint checkpoints/<winner>/ensemble.json
python -m src.analysis --gradcam  --checkpoint checkpoints/<winner>/fold0/best.pth
```

## 5. THE TEST SET — read this twice

The test set (361 images) is quarantined. **Nothing touches it until the very
end.** Choose the final model on validation numbers only, then run **exactly
once**, after code freeze:

```bash
python -m src.evaluate --final-eval --checkpoint checkpoints/<winner>/ensemble.json --tta
```

That writes `reports/<winner>/final_test_report.md`. That's the number for the
poster. Running it repeatedly to pick a better result = the exact cheating this
project is designed to expose.

## 6. Demo

```bash
python demo/app.py --checkpoint checkpoints/<winner>/ensemble.json
```

Opens http://127.0.0.1:7860 — upload a photo, get class + confidence + Grad-CAM.
Fully offline, safe in front of judges.

## Sanity expectations

| Result | Meaning |
|---|---|
| ~90–95% baseline val | healthy |
| 95–97% ensemble val | great, near the honest ceiling |
| >97% | suspect a bug/leak before celebrating |
| 100% | not real |

Differences under ~1.5% (≈5 images) are noise — don't chase them.

## Troubleshooting

- **CUDA OOM** → halve `batch_size` in the yaml (384 configs are the usual culprits).
- **Crashed overnight kfold** → same command again; it resumes per fold.
- **`data/raw` missing** (zip stripped it) → `python scripts/download_data.py`
  re-downloads; the committed `data/splits/*.csv` keep the splits identical.
- **Val accuracy collapses after unfreeze** → your `lr_backbone` is too high; the
  configs' values are safe, don't raise them.

Background reading in the repo: `paper/outline.md` for the story the numbers
feed into.
