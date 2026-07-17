# Paper / poster outline (fill numbers ONLY from experiments.csv rows)

## Title idea
"The ceiling on TrashNet is a data problem: a leakage-free 6-class pipeline
with error bars, a label audit, and a reproduction of the 99% mechanism"

## Narrative spine (from research context §10)
1. Published TrashNet accuracies span 22%–100%. Why?
2. Two clusters: transparent methodology (94–96%) vs exotic optimization
   claiming 99–100% with image counts (2527→10108 = exactly 4×) that imply
   augment-before-split.
3. We built a leakage-free pipeline: group-aware splits (near-duplicates never
   span splits), split-before-augment, test touched once.
4. Result: __._% ± _._ (5-fold), per-class recall + confusion matrix.
5. Label audit: cleanlab found N (_._%) suspected mislabels → retrain delta __.
6. Leakage experiment: same model/seed, Arm B (augment→split) scores __._% vs
   Arm A __._% — the mechanism reproduces the literature's inflated numbers.
7. Conclusion: the ceiling is label noise + dataset structure, not modelling.

## Sections
- Intro & related work (table from research-context §2, framed as two clusters)
- Method: pipeline diagram (dedup → group-aware stratified split → quarantine),
  two-phase fine-tuning recipe, 5-fold ensemble + TTA
- Results: main table (baseline vs winner vs ensemble±std vs final test),
  confusion matrix, per-class table (caveat: ~20 trash test images → noisy)
- Findings: leakage A/B table, label-audit examples grid, Grad-CAM panels
- Limitations: no external data, small test set (1% ≈ 4–5 images), test-set
  label noise reported but never corrected

## Claims discipline
- "This pipeline error reproduces those numbers" — NEVER "they cheated".
- Every number on the poster traces to a config hash in experiments.csv.
- Report the leakage null result honestly if Arm B fails to inflate.
