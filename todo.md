# Final Project Rubric Audit and TODO

Last updated: 2026-07-23

This document records the grading rubric, the current state of the submission,
known risks, and the work required before final submission.

## Important decision

Poster template compliance has been reviewed and approved externally. Do not
rebuild the poster solely to satisfy the official-template requirement unless
new feedback reverses that approval.

The poster still needs factual consistency checks, particularly its description
of the open-set guard and the distinction between validation and test results.

---

# NYU Shanghai AI Summer Camp Final Project — Detailed Grading Rubric

Total: 100 points

- Technical Work: 60 points
- Poster & Defense Presentation: 40 points

Groups are evaluated against other teams within the same track. Cross-track
comparison is not applicable.

## Part 1: Technical Work — 60 Points

### 1. Project Design & Rationale — 15 points

**14–15**

Precise problem definition. Selected models, features, and algorithms fit the
task. Clear justification of design choices. Alternate approaches are discussed
with reasoning for final selection. Project challenges and inherent limitations
are identified.

**11–13**

Reasonable overall solution. Model choices match the task with basic
explanations. Limited discussion of alternative methods. Minor oversights exist.

**7–10**

Viable baseline solution. Model selection lacks clear reasoning and is largely
adapted from sample code with limited independent adjustments.

**0–6**

Mismatched solution for the assigned task. Poor model selection. Direct copying
of templates without independent thinking.

### 2. Code Standard & Completeness — 12 points

**11–12**

Fully runnable code with thorough comments. Clean file structure and consistent
naming convention. Separate training and testing scripts. Basic error handling.
A README with run instructions is provided.

**8–10**

Executable code. Key logic contains comments. Organized structure; no README.

**5–7**

Barely runnable code. Redundant snippets exist. Sparse comments and disorganized
file arrangement.

**0–4**

Code fails to run. Almost no explanatory comments. Messy structure with low
reproducibility.

### 3. Experiments, Model Outcomes & Analysis — 23 points

General requirement: train, validation, and test splits must be implemented. Do
not report results only on training data. Stable reproducible outcomes are
required.

#### Track 1: Image Classification — Garbage Classification

Metric: **Test Accuracy**

**20–23**

Top-tier accuracy among peer groups. Data augmentation and hyperparameter tuning
implemented. Confusion matrix visualized. Misclassified samples analyzed with
actionable improvement ideas.

**15–19**

Moderate accuracy. Basic parameter tuning completed. Visualized results with
superficial analysis of error cases.

**9–14**

Trained baseline model with subpar accuracy. No systematic tuning. Only raw
numerical metrics without visualizations.

**0–8**

Unconverged model. Evaluation performed without an independent test set. No
result analysis.

#### Track 2: Task-Oriented Dialogue Bot

Metric: Task Completion Rate

**20–23**

Complete dialogue state machine covering most user scenarios. Edge cases and
abnormal inputs handled. Extensive test cases recorded. Root causes of failed
dialogues analyzed.

**15–19**

Core user scenarios can be resolved. Minor edge inputs lead to breakdowns. Basic
testing records without systematic optimization analysis.

**9–14**

Only minimal dialogue workflow implemented. Many scenarios cannot be handled.
No systematic testing.

**0–8**

Broken dialogue logic; unable to finish core user tasks.

#### Track 3: Second-hand Item Price Prediction

Metric: RMSLE

**20–23**

Comprehensive feature engineering and feature selection. Multiple models
compared. Error-stratified analysis for low-price versus high-price goods. Error
distribution visualized.

**15–19**

Basic feature engineering completed. Single model trained without cross-model
comparison. Only final metrics reported with no error breakdown.

**9–14**

Raw features fed directly into models without processing. Weak model
performance and no visual analysis.

**0–8**

Failed model training; predictions carry no practical value.

### 4. Limitation Discussion & Future Extensions — 10 points

**9–10**

Clear identification of model weaknesses. Specific, feasible optimization
proposals rather than vague statements. Concrete ideas for future model and data
improvements.

**6–8**

Existing problems are identified, but follow-up improvements are generalized.

**3–5**

Brief concluding remarks with no practical expansion plans.

**0–2**

No reflection or outlook.

## Part 2: Poster & Defense Presentation — 40 Points

### 1. Poster Compliance — 10 points

Critical requirement.

**9–10**

Strictly built on the official template. No full layout reconstruction.
Horizontal A0 format. All five required sections included in this order:
Problem → Data → Method → Result → Takeaways. Concise layout.

**6–8**

Template used, but with large-scale manual adjustment of the section layout.
Disordered section sequence or dense blocks of text.

**3–5**

Only the template background retained; core layout fully redesigned. Missing
required sections.

**0–2**

Official poster template not adopted.

Severe deduction if teams rebuild the layout from scratch.

**Project note:** The current poster format has been approved. Treat compliance
as accepted and focus further poster work on factual correctness and readability.

### 2. Poster Information & Visualization — 10 points

**9–10**

Clear charts and diagrams. Font size readable from two meters away. All visuals
have proper labels. Logical flow and no redundant text.

**6–8**

Basic visual elements included. Occasional dense text and partially annotated
charts.

**3–5**

Overwhelming text, insufficient figures, and fonts that are hard to read from a
distance.

**0–2**

Chaotic layout; information cannot be quickly understood.

### 3. Live Defense Performance — 12 points

Defense sequence:

1. One-minute elevator pitch
2. Five-minute poster walkthrough
3. Two-minute Q&A

**11–12**

Balanced speaking opportunities for every group member. Strict time control and
coherent narration. Elevator pitch highlights core innovations. Full mastery of
project content.

**8–10**

All members participate. Time constraints generally respected. Clear
explanation with occasional unfamiliar segments.

**5–7**

Only a few members speak. Severe overtime or premature ending. Frequent pauses.

**0–4**

Multiple members remain silent. Team cannot deliver a complete project
introduction.

### 4. Q&A Response — 8 points

**7–8**

Questions accurately understood. Answers supported by code and experimental
results. Unknowns are admitted honestly, with reasonable conjectures and
verification plans.

**5–6**

Most questions answered properly; several technical explanations are vague.

**2–4**

Unable to explain core model logic and experimental settings; questions are
evaded.

**0–1**

Cannot respond to judges' questions.

## General Rules and Adjustments

1. **Academic integrity:** Directly copied external code without citation causes
   a 10-point technical-score deduction. Widespread plagiarism disqualifies
   teams from awards.
2. **AI tool usage:** AI assistance is permitted. Every member must understand
   AI-generated code. Failure to explain code during questioning causes a
   5–10-point deduction per instance.
3. **Deadline:** 13:00, July 23 is firm. Late submission causes a 5–15-point
   deduction; extreme delays are not accepted.

## Bonus considerations

These do not raise the score cap and are used for tie-breaking:

- Ablation studies or comparisons across multiple models
- Self-collected expanded datasets or custom-designed extra test cases
- Live demo with visualized model outputs during the poster session

---

# Current Submission Assessment

## Current models

| Model | Role | Input | Labels | Parameters | Recorded result |
|---|---|---:|---:|---:|---:|
| ConvNeXt V2-Tiny | Cap-compliant primary CNN | 384×384 | 6 | 27,871,110 | 98.30% validation |
| Swin-B | Transformer comparison | 224×224 | 10 trained, 6 displayed | 86,753,474 | 97.61% test; 97.82% TTA |

The two scores are not directly comparable because the models use different
datasets, label spaces, and evaluation splits.

## 1. Project Design & Rationale

**Current estimate: 13–15 / 15**

Strengths:

- Clear six-class waste-classification problem.
- ConvNeXt V2-Tiny is justified by the 30M-parameter cap.
- Swin-B provides a distinct transformer comparison.
- Dataset breadth, duplicate leakage, class imbalance, and open-set behavior are
  discussed.
- The paper contains concrete limitations.

Remaining improvements:

- Make the final model-selection rationale easy to trace from experiments.
- Clearly distinguish measured claims from design hypotheses.
- Cite any externally adapted code or techniques.

## 2. Code Standard & Completeness

**Current estimate: 8–10 / 12**

Strengths:

- Current model training scripts are present.
- Code is heavily commented.
- Naming and folder structure are now clean.
- README and reproduction guide are present.
- Checkpoint hashes and metrics can be verified.

Gaps:

- There is no separate current testing/evaluation entry point.
- ConvNeXt training and evaluation remain coupled.
- Swin-B training and evaluation remain coupled.
- A clean smoke-test mode is needed.
- Dataset-path and missing-checkpoint errors should be tested.

## 3. Experiments, Outcomes & Analysis

**Current estimate: 9–15 / 23**

Strengths:

- Swin-B has validation, independent test, TTA, and per-class recall.
- ConvNeXt uses augmentation, EMA, MixUp, CutMix, and a group-disjoint
  validation fold.
- Two architectures are compared.
- The paper discusses data leakage and duplicate handling.

Critical gaps:

- ConvNeXt has no recorded independent test result.
- The submitted package does not currently demonstrate a complete
  train/validation/test split for ConvNeXt.
- No current confusion matrix is included.
- No current misclassified-sample gallery or error table is included.
- Actionable error analysis is not connected to current model predictions.
- Hyperparameter tuning and ablation evidence was removed during cleanup.
- The rubric's official metric is test accuracy, not validation accuracy.
- Swin-B uses ten classes while the deployed interface shows six; this
  distinction must be explained consistently.

## 4. Limitations & Future Extensions

**Current estimate: 9–10 / 10**

Strengths:

- The paper identifies limited test coverage, duplicate-detection thresholds,
  dataset overlap, label quality, domain shift, OOD behavior, and non-comparable
  model splits.
- Proposed improvements are generally concrete.

Remaining improvement:

- Tie each limitation to a feasible next experiment, required data, and success
  metric.

## 5. Poster Compliance

**Status: approved externally**

Do not prioritize template reconstruction.

Technical observations retained for reference:

- Poster is horizontal and uses the supplied template's 4:3 physical dimensions.
- All important concepts appear, though Data and Method are combined.
- The final source is HTML/PDF rather than an edited final PPTX.

These are not current action items because compliance has been approved.

## 6. Poster Information & Visualization

**Current estimate: 8–10 / 10**

Strengths:

- Strong hierarchy and large readable text.
- Clear parameter comparison and per-class recall chart.
- Logical model/data/result story.
- Live-demo callout is prominent.

Factual risks:

- ConvNeXt must always be labeled as validation accuracy, not test accuracy.
- Swin-B's test result must remain associated with its ten-class dataset.
- The open-set panel currently describes the retired feature-distance guard
  (`0.78 > 0.70`) rather than the active optional Swin-based guard.
- Claims about “fresh, real-world photos” need recorded evidence or more careful
  wording.

## 7. Defense & Q&A

**Not yet assessable**

Risks:

- Every member must be able to explain both architectures.
- Every member should understand MixUp, CutMix, EMA, group-aware splitting,
  TTA, label mapping, and the difference between validation and test accuracy.
- The team must be ready to explain why the two headline percentages are not a
  direct model ranking.
- The team must be ready to explain why ConvNeXt does not yet have an
  independent test score.

---

# Implementation TODO

## Session progress — 2026-07-23 (Claude)

Division of labor: the **teammate runs the ConvNeXt confusion matrix + accuracy
on Kaggle** (script ready at `model/convnextv2_tiny_cnn/eval_confusion_kaggle.py`);
Claude did everything else below. The poster is submitted — left untouched.

**Done + verified this session**
- **P1** — `submission/code/evaluate.py`: deterministic dual-model test entry
  point (accuracy, macro-F1, per-class P/R/F1, loss, predictions CSV, metrics
  JSON, labelled confusion matrix, highest-loss + confident-wrong exports; clear
  failures on missing/incompatible checkpoints and empty splits). Smoke-tested on
  both models on CPU.
- **P4** — ablation/comparison table added to the paper (Table 2) from the
  restored `experiments.csv` ledger; every number verified against the ledger.
- **P5** — `submission/code/tests/` (16 tests passing): split disjointness,
  class-order, checkpoint smoke + param count + SHA, deterministic transforms,
  CPU eval smoke, and error paths; `requirements.lock.txt` pinned.
- **P0/P6** — References section (13 citations) added to the paper; ConvNeXt kept
  labelled *validation*; the two headline numbers explicitly not comparable.
- **P8** — `submission/defense/` pack: elevator pitch, 5-min walkthrough (balanced
  speakers, no names), and a Q&A evidence sheet.

**Pending — waiting on the Kaggle run**
- Drop `convnext_confusion.png` + per-class + misclassified into
  `submission/figures/`.
- Update paper §5.1 finding box + §8 limitation (they currently say "no confusion
  matrix / per-class recall computed") once the validation matrix exists, and add
  a misclassification-analysis paragraph with actionable fixes.

**Human step**
- Re-export `paper.pdf` from the edited `paper.html` (Claude edits HTML only).

## Priority 0 — Preserve integrity

- [ ] Do not call ConvNeXt's 98.30% validation score “test accuracy.”
- [ ] Do not directly rank ConvNeXt 98.30% against Swin-B 97.61/97.82%.
- [ ] Preserve the original held-out Swin-B test split and result.
- [ ] Record hashes and configuration for every newly generated result.
- [ ] Add citations and attribution for external datasets, pretrained models,
      libraries, and adapted code.
- [ ] Ensure every team member can explain all submitted code.

## Priority 1 — Separate evaluation code

- [ ] Add `submission/code/evaluate.py`.
- [ ] Support ConvNeXt V2-Tiny checkpoint loading.
- [ ] Support Swin-B checkpoint loading.
- [ ] Require an explicit dataset root and split manifest.
- [ ] Keep evaluation deterministic.
- [ ] Calculate accuracy, macro-F1, per-class precision/recall/F1, and loss.
- [ ] Save predictions with image path, true label, predicted label, confidence,
      and correctness.
- [ ] Save a machine-readable metrics JSON.
- [ ] Generate a labeled confusion matrix.
- [ ] Export the highest-loss and highest-confidence mistakes.
- [ ] Fail clearly on missing checkpoints, class-order mismatches, empty splits,
      and incompatible state dictionaries.

## Priority 2 — ConvNeXt train/validation/test protocol

- [ ] Freeze a group-disjoint test set before further tuning.
- [ ] Store train, validation, and test manifests.
- [ ] Ensure no perceptual-duplicate group crosses any split.
- [ ] Document dataset sources and label mappings.
- [ ] Add assertions for split disjointness.
- [ ] Select the final ConvNeXt checkpoint using validation only.
- [ ] Run the independent ConvNeXt test exactly once after code/config freeze.
- [ ] Record the resulting test accuracy separately from validation accuracy.
- [ ] Never replace the recorded result merely because a later run scores
      better.

## Priority 3 — Current model analysis

- [ ] Generate ConvNeXt confusion matrix on its independent test set.
- [ ] Generate Swin-B confusion matrix on its held-out test set.
- [ ] Produce per-class metrics for both models.
- [ ] Create a misclassified-sample gallery.
- [ ] Analyze the most common directional confusion pairs.
- [ ] Analyze errors by source dataset.
- [ ] Analyze errors by confidence.
- [ ] Identify ambiguous or incorrectly labeled samples.
- [ ] Attach a specific improvement proposal to every major error cluster.
- [ ] Clearly separate six-class ConvNeXt analysis from ten-class Swin-B
      analysis.

## Priority 4 — Hyperparameter tuning and ablations

- [ ] Restore or reconstruct a compact experiment ledger.
- [ ] Include baseline architecture results where valid.
- [ ] Compare at least one augmentation ablation.
- [ ] Compare single-view versus TTA where applicable.
- [ ] Compare with and without MixUp/CutMix if recorded results exist.
- [ ] Compare relevant input resolutions if recorded results exist.
- [ ] Record seed, data split, model, image size, optimizer, learning rates,
      weight decay, epochs, and checkpoint hash for each run.
- [ ] Mark invalid or non-comparable experiments explicitly.
- [ ] Produce one concise ablation figure or table for the paper/poster/Q&A.

## Priority 5 — Tests and reproducibility

- [ ] Add tests for train/validation/test disjointness.
- [ ] Add tests for class-order consistency.
- [ ] Add checkpoint-loading smoke tests.
- [ ] Add deterministic transform tests.
- [ ] Add one-batch overfit or dry-run mode.
- [ ] Add a CPU-compatible evaluation smoke test.
- [ ] Pin a tested dependency set.
- [ ] Verify all commands from a clean environment.
- [ ] Ensure no code depends on files outside the submitted/repository paths
      documented in the README.

## Priority 6 — Paper updates

- [ ] Add the final ConvNeXt independent test result when available.
- [ ] Add current confusion matrices.
- [ ] Add current misclassification analysis.
- [ ] Add the compact experiment/ablation table.
- [ ] Ensure every number points to a committed metrics artifact.
- [ ] Regenerate and visually inspect the paper PDF after changes.
- [ ] Preserve explicit caveats where results use different datasets or splits.

## Priority 7 — Poster factual corrections

- [ ] Replace the retired feature-distance OOD panel with the current
      Swin-based guard, or label it clearly as a historical experiment.
- [ ] Keep ConvNeXt labeled as validation until a test result exists.
- [ ] Avoid implying the two headline scores share a test set.
- [ ] Replace unsupported “fresh, real-world photos” claims with measured demo or
      test evidence.
- [ ] Regenerate and visually inspect the poster PDF after factual edits.
- [ ] Do not redo the approved poster layout solely for compliance.

## Priority 8 — Defense preparation

- [ ] Write a one-minute elevator pitch.
- [ ] Write a five-minute poster walkthrough.
- [ ] Assign balanced speaking portions to every member.
- [ ] Rehearse the complete sequence under six minutes before Q&A.
- [ ] Prepare evidence-backed answers for model choice, data leakage, class
      imbalance, test protocol, TTA, augmentation, and OOD handling.
- [ ] Prepare an honest answer explaining non-comparable model scores.
- [ ] Prepare fallback verification plans for unknown questions.
- [ ] Ensure every member can locate the code or artifact supporting each poster
      claim.

## Priority 9 — Final packaging

- [ ] Run `python submission/code/verify_artifacts.py`.
- [ ] Run the full test suite in a clean environment.
- [ ] Check all JSON files parse.
- [ ] Check all PDF and HTML references resolve.
- [ ] Confirm no caches, virtual environments, duplicate checkpoints, or draft
      files are included.
- [ ] Confirm current checkpoints match `submission/results/manifest.json`.
- [ ] Review the submission folder against this rubric one final time.

---

# Recovery note

Legacy submission material removed during cleanup was moved to:

```text
/private/tmp/second-life-submission-legacy-20260723
```

It may contain useful historical split manifests, experiment logs, confusion
matrices, predictions, and analysis artifacts. Restore only evidence that is
valid, traceable, and relevant to the current models. Do not blindly restore the
entire old pipeline.
