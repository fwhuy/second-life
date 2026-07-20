# Accuracy optimization plan (2026-07-18)

**TL;DR — on the Windows training PC, double-click `train.bat` (or run
`train.bat full --duo` for the maximum-accuracy variant) and let it run
overnight. Results land in `reports/AUTOPILOT_SUMMARY.md`. Everything is
resumable: rerun the same command after any crash and it continues.**

## Why this plan

The locked baseline (ResNet-50 @ 224, fold-0 val 93.09% / 94.93% with TTA) was
already squeezed dry: progressive 384px, modern augmentation, and higher weight
decay each failed the pre-registered +7-image gate. Those were all *recipe*
tweaks on the *same 2015-class backbone*. The two levers with real headroom were
deliberately never pulled (the plan stopped first):

1. **Backbone quality.** ResNet-50 (IN-1K, ~80% top-1) vs modern
   masked-image-modeling / web-scale backbones (88–90% top-1). On a 2,166-image
   training pool, pretrained representation quality dominates everything else.
   Published TrashNet results with small CNNs cluster at 94–95%; the repo's
   honest ceiling estimate is 96–97%.
2. **Ensembling.** 5-fold CV ensemble + TTA was built but never run.

## The bake-off candidates

| config | model (timm) | IN-1K top-1 | why |
|---|---|---|---|
| `opt_eva02b_448` | `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | 88.7% | best base-size transfer model; MIM pretraining shines on small data |
| `opt_convnextv2b_384` | `convnextv2_base.fcmae_ft_in22k_in1k_384` | 88.2% | best conv backbone; architectural diversity for the ensemble |
| `opt_siglip2b_384` | `vit_base_patch16_siglip_384.v2_webli` | n/a (contrastive) | web-scale SigLIP-2 pretraining — very different failure modes |
| `opt_eva02l_448` | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | 90.1% | strongest single model that fits a consumer GPU; auto-included when VRAM ≥ 11 GB |

All verified to exist in the installed `timm 1.0.28` and to build cleanly
through `src.model.build_model` (tested on this machine with the same
torch 2.13.0 / timm 1.0.28 versions as the training PC).

## What changed in the code

- **Layer-wise LR decay** (`layer_decay` + `lr_base` in configs):
  BEiT/EVA-style per-layer LRs via `timm.optim.param_groups_layer_decay`, the
  standard recipe for fine-tuning strong ViTs on scarce data. The old
  two-group scheme still works for old configs (`src/model.py: ft_param_groups`).
- **Fixed EMA** (`ema_warmup: true`): the baseline's flat 0.9998 decay has a
  ~5,000-step horizon but runs here take ~1,600 optimizer steps — EMA never
  converged, which is why it always lost to raw weights. `ModelEmaV3`'s decay
  warmup fixes that. With `selection_weights: best`, each epoch evaluates raw
  *and* EMA weights and checkpoints whichever wins (recorded per checkpoint).
- **Gradient accumulation** (`accum_steps`) + **gradient clipping**
  (`grad_clip`) + **gradient checkpointing** (`grad_checkpointing`): big
  models at 384–448px train in ~5 GB VRAM at effective batch 32.
- **Mixed-architecture ensembles**: `evaluate.py`/`oof.py` now evaluate each
  ensemble member at its own resolution with its own selected weights, so a
  ConvNeXt+EVA duo ensemble is one manifest away (`--mode duo` builds it).
- **`scripts/autopilot.py`**: bake-off → winner 5-fold → pooled OOF → report,
  every stage idempotent/resumable; `train.bat` is the Windows wrapper
  (creates the venv only if missing — your existing one is reused).
- Recipe guardrails kept from the experiment history: weighted sampler, label
  smoothing 0.1, **no mixup/cutmix** (step 2a collapsed trash recall), light
  TrivialAugment + random-erasing 0.1.

Nothing about the split/leakage machinery changed: group-aware splits, the
test quarantine, and `TEST_SPENT.json` guard are untouched, and all 11 repo
tests pass with the new code.

## What to run

```bat
train.bat              REM bake-off -> winner 5-fold -> report
train.bat full --duo   REM + runner-up 5-fold + 10-model ensemble
train.bat bakeoff      REM just the fold-0 bake-off
train.bat report       REM regenerate the summary from what exists
```

If a run OOMs: halve `batch_size` in the failing `configs/opt_*.yaml`, double
`accum_steps` (keeps effective batch = 32), rerun `train.bat`.

## Resource usage + ETA (RTX 5060, 128 GB RAM)

The opt configs are tuned for this box: `cache_images: true` holds the decoded
dataset in RAM (~1.5 GB, ~10 GB across 8 persistent workers — nothing on 128 GB)
so epochs never touch disk or JPEG decode; `num_workers: 8`;
`deterministic: false` re-enables the cuDNN autotuner and fast kernels (seeds
stay fixed; only bit-exact rerun reproducibility is traded). VRAM stays the
binding constraint: batch sizes are sized for 8 GB with AMP + gradient
checkpointing, and EVA-02 Large auto-joins only on a >= 11 GB card (5060 Ti
16 GB variant).

Rough wall-clock (8 GB 5060; treat as ±2x until you see real epoch times — the
console prints one line per epoch, so extrapolate after ~3 epochs):

| stage | estimate |
|---|---|
| bake-off: EVA02-B@448 fold 0 | ~1.5-3 min/epoch → 45-90 min |
| bake-off: ConvNeXtV2-B@384 / SigLIP2-B@384 fold 0 | ~1-2 min/epoch → 35-70 min each |
| bake-off TTA evals | ~2-5 min each |
| **bake-off total** | **~2-4 h** |
| winner 5-fold (4 more folds) | ~3-6 h |
| pooled OOF + report | ~15-40 min |
| **`train.bat` (full)** | **~5-10 h — one overnight run** |
| `--duo` add-on | +3-6 h |
| EVA02-L (only if 16 GB card) | +1.5-3 h bake-off; ~6-10 h if it wins 5-fold |

Two practicalities: the first run downloads pretrained weights (~350 MB per
base model) so the PC needs internet, and per-epoch resumable checkpoints are
large (`last.pth` ~1.4 GB for EVA02-B — keep ~30 GB free on an SSD).

## How to read the results

- **Headline honest number**: pooled out-of-fold accuracy (n=2,166, 4-view
  TTA) in `reports/AUTOPILOT_SUMMARY.md` — every image predicted by a model
  that never trained on its group. Compare against the baseline's fold-0
  94.93% TTA.
- **Deployable artifact**: `checkpoints/<winner>/ensemble.json` (5 models,
  works with `src.evaluate` and the Gradio demo), or
  `checkpoints/opt_duo_ensemble/ensemble.json` for the max-accuracy 10-model
  version.
- **Test set**: still spent (89.20% pre-intervention, guarded by
  `reports/TEST_SPENT.json`). The summary explains the one-shot procedure if
  you consciously decide to burn a fresh test measurement on the final
  ensemble; otherwise report OOF and caption the old test number as
  pre-intervention. That decision is yours, not the autopilot's.

## Expectations (calibrated, not promised)

Fold-0 val with TTA: baseline 412/434. The backbone jump (~80% → 88–90% IN-1K)
plus fixed EMA typically buys several points on small datasets; the 5-fold
ensemble + TTA adds ~0.5–1% more. Landing in the **96–97.5%** OOF band would
match the repo's honest-ceiling estimate; treat anything above ~97.5% with
suspicion (per README, suspect a bug before celebrating — ~half the audited
residual errors look like label noise/ambiguity).

## Sources

- EVA-02: [arxiv.org/abs/2303.11331](https://arxiv.org/abs/2303.11331) (MIM ViTs + layer-decay fine-tuning)
- ConvNeXt V2: [arxiv.org/abs/2301.00808](https://arxiv.org/abs/2301.00808)
- SigLIP 2: [arxiv.org/abs/2502.14786](https://arxiv.org/abs/2502.14786)
- timm model registry/results: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- TrashNet published results (94–95% band): [IJCT smart waste classification](https://ijctjournal.org/smart-waste-classification-sustainable-waste-management/), [Aral & Keskin, TrashNet deep models](https://www.semanticscholar.org/paper/Classification-of-TrashNet-Dataset-Based-on-Deep-Aral-Keskin/f5a380760b91393ad05bfc2063434f76935a428e), [Managing Household Waste through Transfer Learning](https://arxiv.org/pdf/2402.09437)
