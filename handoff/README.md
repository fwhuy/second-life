# Waste classifier — training handoff

Everything needed to train and improve the model. **About 8MB** — the ~2GB image corpus is not
included, it downloads in one command.

## Quick start

Works on Windows, macOS and Linux. Needs Python 3.10+ and an NVIDIA GPU.

```
python setup.py     # env + corpus download + splits + verification
```

Then train — Windows:

```powershell
cd model
.venv\Scripts\Activate.ps1
python -m src.train --config configs\unified_convnextv2_tiny_224.yaml --fold 0 --resume
```

macOS / Linux:

```bash
cd model && source .venv/bin/activate
python -m src.train --config configs/unified_convnextv2_tiny_224.yaml --fold 0 --resume
```

Everything is resumable — rerun the same command after any interruption and it continues
from the last completed epoch.

### Getting full use of the GPU

Enabled automatically on CUDA, no config needed: TF32, `cudnn.benchmark`, channels-last
memory format, and bf16 autocast on Ampere or newer (bf16 needs no GradScaler and cannot
overflow; older cards fall back to fp16 + scaler). `num_workers: auto` sizes the loader
pool to the machine, with a deeper prefetch queue.

Batch size needs the real hardware to decide, so it is a separate step:

```
python scripts/autotune.py --config configs/unified_convnextv2_tiny_224.yaml --write configs/tuned.yaml
```

It probes real training steps until it OOMs, reports the largest batch that fits, and
scales the learning rates by the same factor (linear scaling rule) with a longer warmup.

> A larger batch is a **throughput** change, not a free accuracy win — fewer, less noisy
> gradient steps per epoch. Keeping the original LR at 4x the batch is the usual reason
> "I filled the GPU and accuracy dropped". autotune handles the LR, but you should still
> run the tuned config as an experiment and compare macro-F1 before adopting it.

System RAM needs no tuning. The corpus is ~2GB, so after the first epoch the OS page cache
holds all of it and image loading stops touching disk. Forcing more RAM use would not make
the model better.

> **The images are not in this handoff** (it is about 8MB, they are ~2GB). `python setup.py` downloads
> them from HuggingFace, at pinned revisions, and then **rebuilds the splits** — necessary
> because the shipped splits were built from 3 datasets and the download now pulls 5.
> The rebuild is deterministic and re-derives the test quarantine automatically, so it is
> safe and reproducible; it just must not be skipped. `preflight.py` catches you if it is.

## To hand this to a coding agent

Give it `CODEX_PROMPT.md`. That file is written to be pasted directly and tells the agent to
keep optimising until stopped, checkpointing as it goes. It assumes Windows/PowerShell.

## What's here

```
RULES.md            non-negotiable constraints — read this
STATUS.md           current state + prioritised backlog (the agent keeps this updated)
CODEX_PROMPT.md     paste-ready prompt for an autonomous agent
model/
├── src/            pipeline: data, splits, dedup, train, evaluate, kfold
├── configs/        one YAML per experiment; unified_*.yaml are the ready-to-run bake-off
├── scripts/        download, preflight, provenance, champion tracking
├── tests/          23 tests, all passing — run `pytest tests/ -q`
├── data/
│   ├── splits/           original TrashNet splits (needed to rebuild provenance)
│   ├── splits_unified/   the splits you train on — committed and verified
│   └── unified_waste/    manifest.csv + trashnet_provenance.csv (the audit trail)
├── experiments.csv append-only run log
├── requirements.txt       direct dependencies for deliberate lock refreshes
└── requirements.lock.txt  exact environment installed by setup
```

## The one thing to understand before touching splits

The corpus pools three public datasets — and one of them **contains TrashNet inside it**.
2,218 of 2,527 TrashNet images are stored under `garbage_classification__*.jpg` filenames,
including all 361 images of the project's original, already-spent test set.

`trashnet_provenance.csv` records which files those really are. The split builder forces
every near-duplicate group containing one into the test set, and `load_split_frames` asserts
this on every load. If that assertion ever fires, the split is broken — fix it, don't bypass
it. The project's entire argument is that leakage inflates published results on this dataset.

## Data

Five source datasets, all pinned to exact revisions in `scripts/download_unified_datasets.py`:

| source | HuggingFace repo | notes |
|---|---|---|
| garbage_classification | `omasteam/waste-garbage-management-dataset` | contains TrashNet inside it |
| realwaste | `shahzaibvohra/realwaste` | real landfill photos, different domain |
| trashnet | `garythung/trashnet` | the original benchmark |
| garbage_v2 | `steveharianto/waste-garbage-management-dataset` | **new**, ~19.8k, same 10-label schema |
| recycling11 | `viola77data/recycling-dataset` | **new**, ~3.1k, 11 material labels |

Only labels that map honestly to the six classes are used. `recycling11` contributes
aluminium→metal, hard/soft plastic→plastic, plus cardboard/glass/paper. Its composites
(takeaway cups, disposable plates, paper towel) and polystyrene straddle the
recyclable/non-recyclable line depending on local scheme, so they stay in `excluded/`
rather than being guessed into a class. Battery, biological, clothes and shoes are
excluded from every source for the same reason.

**Split sizes below are for the 3-source build shipped here.** After `python setup.py` pulls all
five, the corpus is larger and the splits are rebuilt — expect roughly 20-25k images. The
proportions and every guarantee hold; only the totals change.

| split | images |
|---|---:|
| train + val | 9,560 (5 folds of 1,912) |
| test (quarantined) | 1,739 |

| class | total | | source | total |
|---|---:|---|---|---:|
| plastic | 2,540 | | garbage_classification | 5,191 |
| glass | 2,259 | | realwaste | 3,587 |
| cardboard | 1,887 | | trashnet | 2,521 |
| paper | 1,854 | | | |
| metal | 1,726 | | | |
| trash | 1,033 | | | |

Note the `source` column reflects true origin, not filename — the 2,218 disguised TrashNet
files are counted as trashnet. Every fold contains all sources.

## Reproducing the splits from scratch

Only needed if you add a dataset or preflight reports a corpus mismatch:

```bash
python scripts/download_unified_datasets.py    # HF revisions are pinned
python scripts/map_trashnet_provenance.py      # self-asserts 2527/2527 match
python -m src.unified_data --build-splits
python scripts/preflight.py
```
