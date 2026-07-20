# Final Summary

Stopped: 2026-07-18 18:34 AEST under stopping condition (b).

## Decision

Model work is finished. Step 1 failed, and both isolated Step 2 siblings failed, so Step 2 also failed. These are two consecutive failed numbered steps. The pre-registered rule therefore stops the sequence before five-fold CV or any backbone bake-off.

This is an evidence-based ceiling stop, not a crash or a time-limit stop. The spent test set was never measured again.

## Split arithmetic gate

The mandatory pre-training reconciliation passed:

| Quantity | Images |
|---|---:|
| Total inventory | 2,527 |
| Fold-0 train | 1,732 |
| Fold-0 validation | 434 |
| Quarantined test | 361 |
| Train + validation pool | 2,166 |
| Five validation folds | 434 + 433 + 433 + 433 + 433 = 2,166 |
| Expected five-fold OOF rows | 2,166 |

The raw inventory, path uniqueness, split union, train/validation/test disjointness, fold IDs, and OOF row arithmetic all passed. The canonical split identity is `b761576345b66207f1c8420eecca867dd669d3fc6af60e9b6736e1b5ffe9db96`.

## Best configuration

- Backbone and resolution: `resnet50.tv2_in1k` at 224px
- Weights: immutable raw epoch-44 baseline
- Checkpoint: `checkpoints/baseline_resnet50/fold0/best_raw.pth`
- Checkpoint SHA-256: `716C697AA0EA661BA933A62B2308CB42A555F13588367E45538DE1B92BFDD173`
- Seed: 42
- Training recipe: basic augmentation, weighted sampler, weight decay 0.05, label smoothing 0.1
- Adopted inference: deterministic four-view TTA
- Validation: 404/434 = 93.09% single-view; 412/434 = 94.93% with TTA
- TTA macro-F1: 0.9413997489

No accuracy-training experiment was accepted. The only accepted accuracy gain was the already-adopted deterministic TTA, worth +8 validation images over the same raw checkpoint.

## Accepted and rejected experiments

Every decision used the same 434-image fold-0 validation set. The gate was at least +7 net images over the current 412/434 reference, no macro-F1 drop, and no per-class recall loss greater than five percentage points.

| Step | Isolated variable | TTA result | Net images | Macro-F1 | Verdict |
|---|---|---:|---:|---:|---|
| 0 | Four-view TTA on locked raw baseline | 412/434 | +8 vs single-view | 0.9414 | Accepted before training experiments |
| 1 | Progressive resize, 224 to 384px | 412/434 | 0 | 0.9442 | Rejected: 10 fixes and 10 regressions; targeted cardboard/paper errors worsened 2 to 3 |
| 2a | Modern augmentation only | 415/434 | +3 | 0.9421 | Rejected: below +7; trash recall fell from 20/23 to 18/23 (-8.70 points) |
| 2b | Weight decay 0.05 to 0.10 only | 411/434 | -1 | 0.9390 | Rejected: 3 fixes and 4 regressions; macro-F1 also fell |

The three training runs consumed approximately 1.5 GPU-hours in total. Each ran detached with per-epoch resumable checkpoints and persistent logs.

## Honest ceiling assessment

The raw model is overfit: train accuracy is 99.60% versus 93.09% validation, a 6.51-point gap. TTA removes 8 of the 30 raw validation errors without retraining, but the targeted training changes only move which images are wrong:

- 384px produced 10 fixes and 10 regressions, net 0.
- Modern augmentation produced 12 fixes and 9 regressions, net +3, while collapsing trash recall.
- Higher weight decay produced 3 fixes and 4 regressions, net -1, while lowering macro-F1.

None approached the pre-registered +7-image threshold. That is strong evidence that the remaining gain available from these scoped changes is below the validation noise floor, not evidence for another round of speculative tuning.

The spent raw test result had 39 errors among 361 images. In the highest-loss audit, 29 of the top 30 were wrong predictions; manual review judged 14 as clearly ambiguous or mislabeled and one more as borderline. That is 48% clearly noisy, or 52% including the borderline case, among the audited wrong images. Ten lower-loss test errors were not manually audited. It would therefore be too strong to claim that most residual error is proven label noise, but it is equally wrong to treat all 39 as recoverable model failures.

Also, the real test set has 361 images, so one test point is 3.61 images, not approximately five. The acceptance rule was correctly applied to validation: `ceil(1.5% x 434) = 7` images.

Conclusion: stop model work. The marginal accuracy case is not worth sacrificing poster and paper quality. Five-fold OOF and the backbone bake-off were deliberately not run because the stopping condition fired first; inventing more experiments would violate the plan.

## Three poster numbers

| Poster metric | Result |
|---|---:|
| Test, pre-intervention raw checkpoint (spent) | **322/361 = 89.20%** |
| Validation, raw epoch-44 checkpoint | **404/434 = 93.09%** |
| Validation, same checkpoint + four-view TTA | **412/434 = 94.93%** |

The test number must be captioned as the single pre-intervention, spent measurement. It was not used for any experiment decision and was not rerun after the plan began.

## Audit trail

- Running status and process logs: `RESULTS.md`
- Human-readable decisions: `EXPERIMENT_LOG.md`
- Raw append-only metrics: `experiments.csv`
- Locked reference: `reports/locked_baseline/tta4/manifest.json`
- Split arithmetic: `reports/locked_baseline/split_arithmetic.json`
- Step 1 comparison: `reports/resnet50_384_progressive/comparison_vs_locked_tta4.json`
- Step 2a comparison: `reports/resnet50_224_aug_modern/comparison_vs_locked_tta4.json`
- Step 2b comparison: `reports/resnet50_224_wd010/comparison_vs_locked_tta4.json`
- Spent-test guard: `reports/TEST_SPENT.json`
- Final CPU hygiene suite: 11 passed; one unrelated pandas future warning
