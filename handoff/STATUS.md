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

## Backlog — highest value first

0. **Autotune the batch size** (`python scripts/autotune.py --config <cfg> --write <cfg>_tuned.yaml`).
   Do this first — every later experiment runs faster. The GPU flags (TF32, cuDNN benchmark,
   bf16, channels-last) are already automatic; batch size is the one thing that needs the
   real hardware to decide. Verify the tuned config against the baseline on macro-F1 before
   adopting it: bigger batch + scaled LR is a throughput win, not automatically an accuracy
   one. Record both numbers here.
1. **Backbone bake-off, 224px, fold 0.** Four configs are ready and verified to build:
   `configs/unified_{convnextv2_tiny,caformer_s18,effnetv2_s,dinov2_vits}_224.yaml`.
   Run each, keep the top 2 by macro-F1. This is the first thing to do.
2. **Progressive resize to 384px** on the winner. Set `init_checkpoint` to the 224 run's
   `best.pth`, `img_size: 384`, lower LRs, fewer epochs — copy the pattern in
   `configs/convnextv2_384.yaml`. Historically the single largest gain.
3. **Label-noise cleaning.** An earlier audit found roughly half of the worst errors were
   mislabeled or genuinely ambiguous, and 3 pixel-identical pairs in this corpus carry
   contradictory labels. Get 5-fold out-of-fold predictions (`src/kfold.py`), rank by loss,
   review the top ~300, write decisions to `data/unified_waste/label_audit.csv`, retrain.
   Report before/after — the delta is a poster figure.
4. **5-fold CV + ensemble** on the top 2 backbones. `src/kfold.py` writes `ensemble.json`;
   `src/evaluate.py` already averages softmax across members and TTA views. See RULES.md §4
   on how to report this.
5. **More data.** Two sources were added on 2026-07-21 (`garbage_v2`, `recycling11`) and are
   already wired up — they arrive with `python setup.py`, nothing to do. For further additions
   follow RULES.md §3 exactly. Still-unexplored candidates: TACO, OpenLitterMap,
   `UdaraChamidu/Garbage-Classification-with-12-classes` and
   `Qween0fPandora/Garbage_Classification_Original_Dataset` (both ship as zips, so they need
   a custom loader rather than `import_hf_dataset`).
   **Avoid `griffinbholt/augmented_waste_classification`** — its 38k files are `_b`/`_h`/
   `_o`/`_v` augmented variants of an existing set. Synthetic duplicates add no information
   and, being near-duplicates of test images, would get forced into the test set and starve
   training.
6. **Tune the adopted TTA.** Currently a fixed 4-view scheme (`src/evaluate.py:52`). Worth
   one experiment, not more.

## Notes and gotchas

- `trash` used to be the failure mode at 137 images; it is now 1,034 and much healthier.
  There is dormant per-class augmentation support (`aug_boost_classes` in configs, off by
  default) if it becomes the outlier again.
- `experiments.csv` spans both the old 2.5k TrashNet corpus and the new 11.3k one.
  `update_champion.py` filters by `splits_dir` so they never get compared — do not remove
  that filter.
- Old TrashNet results, for context only, on a different and now-superseded split:
  93.09% val / 89.20% test (single spent measurement).
