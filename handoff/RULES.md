# Rules — non-negotiable

Breaking any of these invalidates every number the project produces. They are not style
preferences. Most are enforced in code; do not disable the enforcement.

## 1. The test set is never touched until the very end

`data/splits_unified/test.csv` (1,739 images) is quarantined. Do not evaluate on it, do not
look at it, do not use it to choose anything. Model selection uses validation folds only.
It gets measured **once**, at the end, on the final chosen configuration.

## 2. The 361 spent-test images stay in test, always

This project's headline claim is that published 99–100% accuracies on TrashNet are inflated
by data leakage. That claim dies if we leak.

The unified corpus **contains TrashNet inside another dataset**. 2,218 of 2,527 TrashNet
images are stored under `garbage_classification__*.jpg` filenames, including all 361 images
of the original spent test set. A filename check cannot find them.

`data/splits_unified/quarantine.csv` lists them. `load_split_frames` asserts on every load
that none are in training and all are in test. If that assertion fires, **stop and fix the
split** — never work around it.

## 3. Any new dataset goes through the same firewall

Public waste datasets recycle each other constantly. Assume a new one contains TrashNet
until proven otherwise. To add data:

```
# 1. extend scripts/download_unified_datasets.py (pin the revision SHA)
# 2. rebuild provenance and splits — both re-derive the quarantine automatically
python scripts/map_trashnet_provenance.py
python -m src.unified_data --build-splits
python scripts/preflight.py        # must pass before any training
```

Never hand-edit split CSVs. Paths inside them are always forward-slash regardless of OS —
the code normalises with `as_posix` so artifacts stay portable. Do not change that.

## 4. The pretrained backbone is capped at 30M parameters

Competition rule. Enforced in `src/model.py` (`MAX_PARAMS`). Verified good options:

| timm name | params |
|---|---:|
| `convnextv2_tiny.fcmae_ft_in22k_in1k` | 27.9M |
| `swinv2_tiny_window16_256.ms_in1k` | 27.6M |
| `caformer_s18.sail_in22k_ft_in1k` | 24.3M |
| `vit_small_patch14_reg4_dinov2.lvd142m` | 21.6M |
| `tf_efficientnetv2_s.in21k_ft_in1k` | 20.2M |
| `convnextv2_nano.fcmae_ft_in22k_in1k` | 15.0M |

An ensemble whose members are each ≤30M is a **judgment call**, not a certainty. Report the
best single model as the headline number and any ensemble as a clearly-labeled secondary
result. The headline must not depend on a reading a judge might reject.

## 5. Preflight before every training launch

```bash
python scripts/preflight.py
```

Exit 0 means safe. If it fails, fix that and nothing else. It catches the case that matters:
a regenerated corpus that no longer matches the committed splits.

## 6. Every run is logged, and the champion stays current

`train_run` appends to `experiments.csv` automatically. After each experiment:

```bash
python scripts/update_champion.py
```

This rewrites `reports/champion.json` and `reports/LEADERBOARD.md`. The project may be
stopped at any moment — champion.json must always point at a real, usable checkpoint.

## 7. Always resumable

Launch training with `--resume`. Every run writes `last.pth` each epoch and `best.pth` on
improvement, so a killed run continues with the identical command. Never start a long run
in a way that loses work when interrupted.

## 8. Report macro-F1, never accuracy alone

The corpus is imbalanced (`trash` is the smallest class at 1,034 images). Accuracy alone
hides minority-class collapse — an earlier experiment was rejected for exactly that.
