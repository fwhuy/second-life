# TrashNet 6-Class Classification

Reproducible pipeline for the TrashNet garbage dataset (2527 images: paper, glass,
plastic, metal, cardboard, trash). Leakage-free by construction: split-before-augment,
group-aware splits (near-duplicate photos never span splits), test set touched exactly
once via a single `--final-eval` path.

## Quickstart

```bash
bash scripts/setup_env.sh          # venv + deps + GPU smoke test + freeze lockfile
source .venv/bin/activate
python scripts/download_data.py    # fetch dataset, verify counts, dedup, build splits
python -m src.train --config configs/baseline.yaml   # M1 baseline (one command retrain)
python -m src.evaluate --checkpoint checkpoints/baseline_resnet50/fold0/best.pth  # val metrics
```

## All commands

| Command | Purpose |
|---|---|
| `python scripts/download_data.py` | Fetch data, verify per-class counts, quarantine test, dedup, build committed splits |
| `python -m src.train --config configs/<name>.yaml [--resume] [--fold K] [--overfit-batch]` | Train + log one run to experiments.csv |
| `python -m src.kfold --config configs/<winner>.yaml` | 5-fold CV ensemble (resumable per fold) |
| `python -m src.evaluate --checkpoint <ckpt-or-ensemble.json> [--tta]` | Val metrics + confusion matrix + per-class table |
| `python -m src.evaluate --final-eval --checkpoint <ensemble.json> [--tta]` | **The single test-set run** |
| `python -m src.leakage_experiment` | Arm A (correct) vs Arm B (leaked) pipeline comparison |
| `python -m src.analysis --audit --gradcam` | cleanlab label audit + Grad-CAM figures |
| `python demo/app.py` | Offline Gradio demo (class + confidence + Grad-CAM) |
| `pytest tests/` | Split/leakage unit tests |

## Rules this repo enforces at runtime

1. **Test is quarantined.** `data/splits/test.csv` is written once by `download_data.py`.
   Every training/eval entry point asserts that no test image — and no near-duplicate
   *group* — appears in any train/val fold. Only `--final-eval` may read test images.
2. **Split before augmenting.** Augmentation lives inside the train `DataLoader` only.
   (`src/leakage_experiment.py` deliberately violates this in Arm B to measure the effect.)
3. **ImageNet normalization** everywhere (0.485/0.456/0.406, 0.229/0.224/0.225).
4. **Accuracy is never reported alone** — per-class recall + confusion matrix always.
5. **Every run** appends a row to `experiments.csv` (config hash + hyperparams + val
   metrics). Reported numbers must be traceable to a row.
6. Seeds fixed, deps frozen to `requirements.lock.txt`, checkpoints saved every epoch,
   crashed runs resume with `--resume`.

## Repo layout

```
configs/        one yaml per experiment (baseline.yaml is the reference)
data/raw/       images (gitignored)      data/splits/  committed seeded fold indices
src/            pipeline modules         scripts/      setup + download
experiments.csv append-only run log      reports/      figures + tables (committed)
checkpoints/    gitignored               demo/         offline Gradio app
paper/          short paper + figures    tests/        split/leakage unit tests
```

## Sanity targets

Honest ceiling ≈ 96–97%. Test is ~380 images, so 1% ≈ 4 images → differences under
~1.5% are noise. Above ~97% on an honest pipeline: suspect a bug before celebrating.
