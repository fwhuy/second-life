# Status

**You (the agent) own this file. Update it after every experiment. Keep it under ~70 lines.**
It is the project's memory. Anyone — human or agent — should be able to read only this file
and know exactly where things stand.

Last updated: 2026-07-21 — handoff created, nothing trained yet on the unified corpus.

## Current state

- Corpus: 5 datasets configured. The splits shipped here were built from the first 3
  (11,299 images); `python setup.py` downloads all 5 and rebuilds, expect roughly 20-25k images.
- Champion: **none yet.** Run something, then `python scripts/update_champion.py`.
- Test set: untouched. Quarantine holds all 361 spent-test images.

## Setup state

- [ ] `python setup.py` completed (env + ~2GB corpus + rebuilt splits + preflight)
- [ ] Record the post-rebuild corpus size here, replacing the estimate above
- [ ] First training run complete

## Done

_(nothing yet — append one line per finished experiment: name, macro-F1, verdict)_

## THE BUDGET IS ~13 HOURS OF GPU TIME — read this before planning anything

Train **one** model well: **ConvNeXtV2 Tiny**. Do not comparison-shop architectures.
It is the safest choice — strong pretrained features, 27.9M params under the 30M cap,
and predictable behaviour on a fixed budget.

| Stage | Budget |
|---|---|
| Setup + corpus download + split rebuild + preflight | 30–90 min |
| Batch-size autotune | 5–15 min |
| **ConvNeXtV2 Tiny @ 224px, fold 0** | 2–4 h |
| Progressive fine-tune @ 384px from that checkpoint | 3–6 h |
| Spare — one extra fold, or TTA/validation checks | remainder |
| Champion update + packaging + transfer | 15–30 min (reserved) |

Total ≈ 7–12 h, leaving deliberate slack. **Do not skip the packaging reservation.**

**Explicitly skipped — do not start these:** the four-model bake-off, full 5-fold CV,
two-architecture ensembles, any new dataset, manual label cleaning. Each is sound and
none fits. The other three configs stay in `configs/` for anyone with more time.

Because 5-fold is skipped, the result is a **single-fold** number. Caption it that way.
On a 1,912-image val fold, under ~0.5 macro-F1 points is noise — do not chase it.

## Backlog — in order

0. **Autotune the batch size** (`python scripts/autotune.py --config <cfg> --write <cfg>_tuned.yaml`).
   Do this first — every later stage runs faster. The GPU flags (TF32, cuDNN benchmark,
   bf16, channels-last) are already automatic; batch size is the one thing that needs the
   real hardware to decide. Verify the tuned config against the baseline on macro-F1 before
   adopting it: bigger batch + scaled LR is a throughput win, not automatically an accuracy
   one. Record both numbers here.
1. **ConvNeXtV2 Tiny, 224px, fold 0** — `configs/unified_convnextv2_tiny_224.yaml`.
   This is the primary model. From the moment it finishes there is a usable checkpoint,
   so everything after this point is upside rather than risk.
2. **Progressive resize to 384px** on that checkpoint. Set `init_checkpoint` to the 224 run's
   `best.pth`, `img_size: 384`, lower LRs, fewer epochs — copy the pattern in
   `configs/convnextv2_384.yaml`. Historically the single largest gain.
3. **Only if time remains:** one extra fold of the 224px config for a variance estimate, or
   a TTA check. Prefer whichever the clock actually allows; stop in time to package.

### Out of scope for this run (kept for the record)

- Label-noise cleaning: needs 5-fold out-of-fold predictions before the audit can start.
- 5-fold CV + ensemble: `src/kfold.py` writes `ensemble.json` and `src/evaluate.py` already
  averages softmax across members and TTA views. See RULES.md §4 on how to report it.
- More data: `garbage_v2` and `recycling11` are already wired up and arrive with
  `python setup.py`. For further additions follow RULES.md §3.
  **Avoid `griffinbholt/augmented_waste_classification`** — its 38k files are `_b`/`_h`/
  `_o`/`_v` augmented variants of an existing set. Synthetic duplicates add no information
  and, being near-duplicates of test images, would get forced into the test set and starve
  training.

## Notes and gotchas

- `trash` used to be the failure mode at 137 images; it is now 1,034 and much healthier.
  There is dormant per-class augmentation support (`aug_boost_classes` in configs, off by
  default) if it becomes the outlier again.
- `experiments.csv` spans both the old 2.5k TrashNet corpus and the new 11.3k one.
  `update_champion.py` filters by `splits_dir` so they never get compared — do not remove
  that filter.
- Old TrashNet results, for context only, on a different and now-superseded split:
  93.09% val / 89.20% test (single spent measurement).
