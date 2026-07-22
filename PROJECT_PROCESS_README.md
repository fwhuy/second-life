# Second Life AI: Complete Project and Development Record

This document explains the full Second Life AI project, why it was built, how the
classifier and website work, what was changed during development, what went wrong, what
was learned, and which results are measured versus still planned. It is intended as source
material for the final paper, poster, and presentation.

## 1. Project overview

Second Life AI is a six-class waste-image classifier and educational web application. A
user photographs an object, the model predicts its material class, and the website explains
how to sort it and what may happen to it after correct or incorrect disposal.

The six model classes are:

| Class | Original TrashNet count | Website disposal mapping |
|---|---:|---|
| Cardboard | 403 | Recyclable waste |
| Glass | 501 | Recyclable waste |
| Metal | 410 | Recyclable waste |
| Paper | 594 | Recyclable waste |
| Plastic | 482 | Recyclable waste, subject to local rules |
| Trash | 137 | Residual/other waste |

The project has two connected goals:

1. Build an accurate and useful waste-identification system.
2. Show why extremely high reported results on TrashNet can be misleading when duplicate
   objects, augmented copies, or test images leak into training.

The project is therefore not only a classification demo. It is also an investigation of
dataset quality, leakage, honest evaluation, class imbalance, confidence, and the gap
between benchmark images and photographs taken in the real world.

## 2. The central research problem

TrashNet contains only 2,527 images. Many are multiple photographs of the same physical
object, often against a clean background. A random image-level split can place one view of
an object in training and another view of the same object in validation or test. A model can
then recognize the object rather than learn a general material concept.

Another common mistake is augmenting the dataset before splitting it. If rotated, flipped,
or recolored versions of one photograph land in different splits, validation accuracy is
artificially inflated. This can help explain some reported 99–100% results.

Our guiding question became:

> How accurately can TrashNet be classified when near-duplicate objects are grouped,
> augmentation happens only after splitting, and the test set is used exactly once?

This changed the project from “maximize a number” into “maximize a number that survives
methodological scrutiny.”

## 3. Repository structure

```text
.
├── README.md                         Short public project overview
├── PROJECT_PROCESS_README.md         This complete development record
├── GUIDE.md                          Original training instructions
├── issues.md                         Detailed internal problem log
├── model/
│   ├── src/                          Data, training, evaluation, deduplication, CV
│   ├── configs/                      Reproducible experiment configurations
│   ├── data/splits/                  Original TrashNet group-aware splits
│   ├── data/splits_unified/          Unified-corpus split metadata
│   ├── data/unified_waste/           Manifest and TrashNet provenance audit trail
│   ├── scripts/                      Setup, download, provenance, preflight, tuning
│   ├── tests/                        Leakage, split, and augmentation tests
│   ├── reports/                      Metrics, figures, guards, and audit artifacts
│   ├── experiments.csv               Append-only machine-readable experiment log
│   ├── FINAL_SUMMARY.md              Final report for the original TrashNet phase
│   └── paper/outline.md              Earlier paper/poster outline
├── website/                          Offline bilingual Second Life AI application
├── scripts/draft_poster.py           Poster draft generator
└── handoff/                          Small training package for another computer
    ├── README.md                     Setup and orientation for the receiving machine
    ├── RULES.md                      Non-negotiable methodological constraints
    ├── STATUS.md                     Living state and prioritised backlog
    ├── CODEX_PROMPT.md               Paste-ready prompt for an autonomous agent
    └── setup.py                      Cross-platform environment and corpus setup
```

The `model/` directory contains the original project and its measured TrashNet results.
The `handoff/` directory contains the newer unified-corpus training package. These phases
must be described separately because they use different datasets and splits.

## 4. Phase 1: the original TrashNet experiment

### 4.1 Leakage-resistant splitting

Before splitting, perceptual hashes and learned image embeddings were used to identify
near-duplicate photographs. Connected images were assigned a shared group identifier.
Groups, rather than individual files, were then assigned to five folds.

This provides three protections:

- Near-duplicate views cannot appear across training and validation.
- The quarantined test set is group-disjoint from training and validation.
- Augmentation is applied by the training dataset only after split assignment.

The split arithmetic was verified as follows:

| Quantity | Images |
|---|---:|
| Complete TrashNet inventory | 2,527 |
| Fold-0 training set | 1,732 |
| Fold-0 validation set | 434 |
| Quarantined test set | 361 |
| Complete train/validation pool | 2,166 |

The five validation folds contain 434, 433, 433, 433, and 433 images, totaling 2,166.

### 4.2 Training recipe

The reference model was an ImageNet-pretrained ResNet-50 trained in two phases:

1. Train the replacement classification head while the backbone is frozen.
2. Unfreeze the full network and fine-tune it with a much smaller backbone learning rate.

The pipeline used a fixed random seed, ImageNet normalization, a weighted sampler for class
imbalance, label smoothing, AdamW, learning-rate warmup and decay, early stopping,
checkpointing after every epoch, and append-only experiment logging.

Every experiment was configuration-driven. This made model, resolution, learning rates,
augmentation, weight decay, seed, and split identity recoverable from `experiments.csv`.

### 4.3 Measured results

These are completed measurements on the original 2,527-image TrashNet phase:

| Experiment | Validation result | Macro-F1 | Decision |
|---|---:|---:|---|
| ResNet-50, single view | 404/434 = **93.09%** | 0.9196 | Reference checkpoint |
| Same checkpoint, four-view TTA | 412/434 = **94.93%** | 0.9414 | Adopted |
| Progressive resize to 384px + TTA | 412/434 = **94.93%** | 0.9442 | Rejected: no net accuracy gain |
| Modern augmentation + TTA | 415/434 = **95.62%** | 0.9421 | Rejected: only +3 images and trash recall fell |
| Weight decay 0.10 + TTA | 411/434 = **94.70%** | 0.9390 | Rejected |

The original acceptance threshold was a gain of at least seven validation images, no
macro-F1 decrease, and no per-class recall loss larger than five percentage points. This
prevented the team from claiming tiny, noisy improvements as meaningful progress.

The best accepted configuration remained the raw ResNet-50 epoch-44 checkpoint with
deterministic four-view test-time augmentation.

### 4.4 The spent test result

The original test set was evaluated once before later tuning:

> **322/361 = 89.20% test accuracy, measured once before intervention and never used for
> model selection.**

This caption must accompany the number in the paper and presentation. The test result must
not be presented as a final optimized-model score. A guard file prevents accidental repeat
evaluation.

### 4.5 What the rejected experiments taught us

The 384px experiment corrected ten validation errors but introduced ten new ones. Modern
augmentation achieved three additional correct predictions, but trash recall fell from
20/23 to 18/23. Increased weight decay lost one correct image and lowered macro-F1.

These results demonstrate why accuracy alone is insufficient. The rare trash class can
worsen even when aggregate accuracy improves. They also show why results differing by only
a few images should not be overinterpreted on a 434-image validation fold.

## 5. Phase 2: unified-corpus expansion

The second phase expands beyond the small, studio-like TrashNet domain. Five public sources
are configured:

| Internal source name | Dataset | Role |
|---|---|---|
| `garbage_classification` | `omasteam/waste-garbage-management-dataset` | Six compatible waste labels |
| `realwaste` | `shahzaibvohra/realwaste` | More realistic landfill/background images |
| `trashnet` | `garythung/trashnet` | Original benchmark |
| `garbage_v2` | `steveharianto/waste-garbage-management-dataset` | Larger ten-label source |
| `recycling11` | `viola77data/recycling-dataset` | Additional material-level examples |

Only labels that map honestly to the six target classes are included. Batteries,
biological waste, clothing, shoes, composites, polystyrene, and other ambiguous labels are
retained in an excluded archive rather than forced into an incorrect class.

All upstream dataset revisions are pinned. Every stored image receives a source identifier,
source index, normalized pixel hash, dimensions, mapped label, and relative path in
`manifest.csv`.

### 5.1 Discovery of hidden TrashNet duplication

The most important finding during expansion was that one external dataset contains renamed
TrashNet images. A total of 2,218 of the 2,527 TrashNet files were recovered inside the
`garbage_classification` source, including all 361 images from the already-spent original
test set. Only 309 TrashNet files were unique enough to be stored under their own name.

Filename comparison could not detect this because the copied images had different names. Of
the 361 spent-test images, 318 sit in the corpus under `garbage_classification__*.jpg`
filenames and only 43 remain visibly named `trashnet`. Training on the corpus as downloaded
would therefore have meant training on 100% of the test set — precisely the leakage this
project exists to criticize.

#### How provenance was recovered

The mapping is exact rather than approximate, which matters because the entire firewall
rests on it. The downloader's `import_trashnet` step walked `sorted(zip members)`, so each
image's `source_index` is a deterministic function of its original TrashNet filename.
`model/scripts/map_trashnet_provenance.py` reconstructs that ordering from the 2,527 paths
in `data/splits/groups.csv` by re-sorting `'dataset-resized/' + label + '/' + filename`,
then joins to the manifest on `source_index`. Where an image was collapsed as a duplicate,
`duplicate_of` names the surviving unified file.

The script carries three hard assertions rather than warnings: every one of the 2,527 rows
must match, the counts must be equal, and every `original_label` must agree with the mapped
label. All three pass at 2,527/2,527 with complete label agreement. A warning would have
allowed a silent partial mapping, which would have left an unknown number of test images
loose in training.

The resulting `data/unified_waste/trashnet_provenance.csv` is committed and has columns
`original_path, source_index, original_label, status, unified_path, old_split`. It is the
single source of truth for the quarantine.

#### How the quarantine is enforced

The 361 quarantined paths are written to `data/splits_unified/quarantine.csv`. Any
near-duplicate group containing one of them is forced into the unified test split before the
remaining test images are chosen. `assert_old_test_quarantined` in `src/utils.py:211` then
re-checks this on every load, raising `LEAKAGE` if a spent-test image appears in any training
fold and `QUARANTINE BREACH` if one is missing from the test set.

The second failure mode is the important one. Path-equality and group-disjointness checks
already existed, but they compare splits against each other; neither can notice that a file
which *ought* to be quarantined was never quarantined at all. The new assertion was verified
by planting a spent-test image into a training fold and confirming that the existing checks
passed while the new one failed.

This is a major paper result: public datasets are not necessarily independent merely
because their names and filenames differ, and a benchmark can be re-imported into a project
without anyone involved intending it.

### 5.2 Scaling duplicate detection

The original pairwise duplicate computation was practical for 2,527 images but created
large full similarity matrices on the expanded corpus. `merge_by_phash` built two dense
(N, N) int32 matrices, and its comment still asserted "N=2527 → fits easily in memory". At
N = 11,359 each of those matrices is roughly 0.52 GB, and `merge_by_embedding` had the same
structure. Both were rewritten to process row-blocks of `BLOCK = 2048`
(`model/src/dedup.py:23`), which holds peak similarity-matrix memory near 0.1 GB and leaves
the grouping result identical.

This is a case where a stale comment was the actual warning sign. The code was correct when
written and became a scaling hazard only because the corpus grew 4.5x underneath it.

Paths written to CSV files were also standardized with forward slashes so regenerated
artifacts match on Windows, macOS, and Linux.

### 5.3 Unified split guarantees

The unified split builder:

- preserves source information;
- groups exact and near duplicates;
- keeps every group in one split;
- stratifies folds by class and source where possible;
- automatically re-derives the spent-test quarantine;
- rejects leakage at load time;
- retains ambiguous labels outside the training set.

Stratification uses a combined `label x source` stratum so that no fold can become
single-domain, with an automatic fallback to label-only when a stratum is too rare for
scikit-learn to place. Every fold in the current build contains all three sources.

### 5.4 Measured state of the three-source build

These figures are verified from the committed split metadata and describe the build shipped
in the handoff. They are a real measurement of the data, not of any model.

| Quantity | Images |
|---|---:|
| Files scanned in `included/` | 11,359 |
| Excluded by quality/consistency filters | 60 |
| Inventory entering the splits | 11,299 |
| Near-duplicate groups formed | 11,042 |
| Quarantined test set | 1,739 (15.39%) |
| Train/validation pool | 9,560 |
| Each of five validation folds | 1,912 |

The 60 exclusions divide into 54 images whose shorter side is under 128 pixels
(`MIN_SIDE = 128`; below this there is no usable detail at 224px training resolution) and
6 images forming 3 pixel-identical pairs that carry contradictory labels.

All three conflicting pairs trace back to TrashNet, which is a pointed result. Two are
labeled `glass` in TrashNet and `plastic` in the copy redistributed inside
`garbage_classification`, so the same photograph carries two different ground truths in two
public datasets. The third pair is labeled `glass` in one place and `metal` in another
**within TrashNet itself** — a single dataset disagreeing with its own copy of one image.

All copies were dropped rather than arbitrated, since choosing a side would mean asserting a
label the sources do not support. They are recorded in `data/splits_unified/excluded.csv`
with a `drop_reason`. This is also why the source table below shows 2,521 TrashNet images
rather than 2,527: the 6 dropped conflicts are exactly the difference, and all 54 size drops
came from `garbage_classification`.

The 128–224 pixel band was deliberately kept, since those images are small but still legible.

| Class | Images | | True source | Images |
|---|---:|---|---|---:|
| plastic | 2,540 | | garbage_classification | 5,191 |
| glass | 2,259 | | realwaste | 3,587 |
| cardboard | 1,887 | | trashnet | 2,521 |
| paper | 1,854 | | | |
| metal | 1,726 | | | |
| trash | 1,033 | | | |

The source column reflects recovered provenance rather than filename, so the 2,218 renamed
files are counted as TrashNet. The most consequential line is `trash`: 137 images in the
original phase, 1,033 here. The minority-class problem that dominated Phase 1 is largely
dissolved by data rather than by technique, which is itself a finding worth reporting.

All 361 quarantined images were verified to be a subset of the new test set. Because the
legacy 361 sit inside the new test split, a future final evaluation can report a
legacy-comparable number alongside the headline one.

The handoff contains metadata for this three-source, 11,299-image build. On the training PC,
setup is designed to obtain all five pinned sources and deterministically rebuild the final
splits. The expected final size is approximately 20,000–25,000 compatible images, but that
estimate must be replaced with the actual post-setup count before publication.

No unified-corpus performance result should be claimed until that remote training run
finishes and returns its logs, configuration, and checkpoint.

## 6. Model and training-system changes

The following capabilities were added or hardened:

### 6.1 Competition parameter cap

Model construction now counts parameters and refuses pretrained backbones over 30 million
parameters (`MAX_PARAMS` in `model/src/model.py:13`). The limit is overridable per config
via `max_params`, but defaults to the competition rule. This prevents hours of training from
producing an ineligible result, and it encodes the rule where it can fail loudly rather than
in a comment where it can be forgotten.

Four candidate configurations were written and each was verified to actually build under the
cap, rather than trusted from published parameter counts:

| Config | Backbone | Parameters |
|---|---|---:|
| `unified_convnextv2_tiny_224.yaml` | `convnextv2_tiny.fcmae_ft_in22k_in1k` | 27.9M |
| `unified_caformer_s18_224.yaml` | `caformer_s18.sail_in22k_ft_in1k` | 24.3M |
| `unified_effnetv2_s_224.yaml` | `tf_efficientnetv2_s.in21k_ft_in1k` | 20.2M |
| `unified_dinov2_vits_224.yaml` | `vit_small_patch14_reg4_dinov2.lvd142m` | 22.1M |

The DINOv2 entry required a `model_kwargs` passthrough so a config can set the ViT's
`img_size`, which is a constructor argument rather than a runtime one. All four enable
`aug: modern`, `mixup: 0.2`, `cutmix: 0.2`, and `random_erasing: 0.25` — settings that were
marginal at 2,527 images and are expected to pay off at four times the data.

Because only about 13 hours of remote GPU time are available, the current recommendation is
to train **ConvNeXtV2 Tiny** rather than spend the budget on a four-model bake-off.

A fifth candidate, EVA-02 Small, was dropped during preparation: the pretrained tag assumed
from memory did not exist in the installed timm release. Candidate tags are now confirmed
against the installed version before a config is written.

### 6.2 Hardware-aware performance

CUDA training now enables appropriate throughput features automatically, through
`configure_performance` in `model/src/utils.py:46`:

- TF32 matrix multiplication;
- cuDNN kernel benchmarking, which is safe here only because the input shape is fixed;
- channels-last memory format for both model and input tensors;
- bfloat16 autocast on Ampere-or-newer GPUs;
- fp16 plus gradient scaling on older CUDA hardware;
- automatically sized data-loader workers and deeper prefetch queues.

The bfloat16 path required correcting a real defect rather than adding a feature. The
training loop previously decided whether to autocast based on whether a gradient scaler
existed. That coupling is wrong for bfloat16: bfloat16 has the same exponent range as fp32
and must **not** be scaled, whereas fp16 must be. The scaler is now created only for fp16,
and the autocast dtype is passed explicitly. Left uncorrected, enabling bfloat16 would have
silently disabled mixed precision entirely.

Data-loader sizing accepts `num_workers: auto`, which resolves to one fewer than the
machine's core count, capped at 16 — beyond that the workers contend for memory bandwidth
and the returns on JPEG decoding go flat. A `prefetch_factor` of 4 keeps the queue deep
enough that the GPU does not wait on decoding at large batch sizes.

#### Batch size and the limits of "use the whole GPU"

Batch size cannot be set automatically, because activation memory depends on the
architecture and cannot be estimated reliably. `model/scripts/autotune.py` therefore probes
real forward, backward, and optimizer steps at increasing batch sizes until the GPU runs out
of memory, keeping the largest size that stays within 90% of total VRAM.

The important point for the paper is that this is a **throughput** change, not an accuracy
improvement. A larger batch means fewer and less noisy gradient steps per epoch. Holding the
learning rate constant while quadrupling the batch systematically underfits, and is the
ordinary explanation for "the GPU was fully utilized and accuracy fell". The script
therefore applies the linear scaling rule (Goyal et al., 2017), lengthens warmup by the
square root of the scale factor, and refuses by default to scale beyond 4x the tuned
baseline, since the rule degrades past that point. The tuned configuration is written with a
header instructing that it be compared against the baseline on macro-F1 before adoption.

#### System memory was deliberately left alone

An explicit decision was made not to increase system-RAM utilization. The corpus is roughly
2 GB, so after the first epoch the operating system's page cache holds all of it and image
loading stops touching disk; the benefit is already obtained without configuration. The
available mechanism for consuming more RAM would be caching pre-decoded images, and that
would *reduce* the diversity of `RandomResizedCrop` and cost accuracy.

Idle memory is not wasted memory. Resource utilization is not a project metric, and this is
recorded here as a design decision rather than an omission.

### 6.3 Class-specific augmentation support

The data loader can apply stronger geometric and occlusion augmentation to named minority
classes. This was designed for the original `trash` class, which had only 137 examples: the
weighted sampler already replayed each trash image roughly four times per epoch, and a
stronger transform is what stops those replays from being near-identical.

The boosted transform widens the crop scale, increases rotation, and adds vertical flip,
perspective, blur, and a floor on random erasing. It deliberately does **not** add per-class
grayscale or aggressive hue rotation. Color is genuine signal for the other five classes, and
stripping it from one class would teach the model a spurious cue — the intervention would
create the very artifact it is meant to correct for.

The per-class branch lives in the dataset rather than in the transform, so it is structurally
impossible for an evaluation dataset to pick up training augmentation. Six tests cover this.

It remains disabled by default because it is an experimental intervention, not an assumed
improvement, and with `trash` now at 1,033 images its original rationale is largely gone. It
should be revisited only if per-class recall shows `trash` as the outlier again.

### 6.4 Resumability and experiment memory

Each run writes `last.pth` every epoch and `best.pth` when validation improves. Resume state
includes model weights, optimizer, scheduler, phase, history, EMA state, and early-stopping
counter. Long remote runs can therefore survive interruption with at most one epoch lost.

`update_champion.py` derives a leaderboard and champion record from the append-only log and
only accepts checkpoints that exist on disk. Runs made on the old and unified split systems
are filtered so incomparable datasets cannot compete for the same champion title.

Five-fold orchestration was updated to aggregate macro-F1, accuracy, and per-class recall,
rather than reporting accuracy alone.

### 6.5 Test-set enforcement

The instruction “use the test set once” was converted into executable policy. A final
evaluation checks a split-specific guard before running (`assert_test_not_spent`) and writes
a spent-test record after completion (`mark_test_spent`), both in `model/src/utils.py`. The
original and unified tests have separate guard files, resolved by `test_spent_path`.

This section previously described a protection that did not exist. Project documentation
stated that the guard file "physically blocks re-running the final evaluation", but searching
the source for `TEST_SPENT` returned no matches outside the documentation itself: the file
was written and read by nothing. The enforcement described above was written in response.

**Lesson:** a documented guarantee is not an implemented one. Any claim that the pipeline
prevents something should be confirmed by finding the code that raises, not by finding the
sentence that says so.

### 6.6 Preflight and setup hardening

Preflight checks dependencies, accelerator availability, split arithmetic, path and group
disjointness, fold integrity, quarantine containment, corpus files, and model eligibility.

The handoff setup initially used a weak rule that treated any directory with more than
1,000 images as complete. This could mistake an interrupted or old three-source download
for the finished five-source corpus. It was corrected to require:

- a completed manifest;
- entries from all five expected sources;
- every stored manifest path to exist.

An interrupted rebuild now clears only generated image trees while retaining the download
cache, preventing stale images from mixing with the new corpus. Setup installs the locked
environment, and the previously missing Hugging Face `datasets` dependency was added.

The corrected handoff passes 23 offline tests: 5 on the original splits, 12 on the unified
splits and quarantine, and 6 on class-specific augmentation.

### 6.7 Design of the transfer package

Training happens on a rented GPU belonging to a collaborator, so the project had to be made
portable without shipping the data. Three constraints shaped the result.

**The corpus is not included.** An early version copied the images and reached 964 MB. That
was reverted in favour of metadata plus pinned Hugging Face revision hashes, so the receiving
machine regenerates a byte-identical corpus itself. The package is 1.5 MB across 65 files.
Reproducibility here comes from pinning, not from copying.

**The splits must be rebuilt, not trusted.** The shipped metadata was built from three
sources while setup downloads five, so the rebuild is mandatory. It is deterministic and
re-derives the quarantine automatically from provenance, and preflight refuses to let
training start if it was skipped.

**The return trip is a whitelist.** The receiving agent is instructed to send back only the
champion checkpoint, `reports/champion.json`, the leaderboard, `experiments.csv`, changed
configs and sources, `STATUS.md`, and a short results write-up — excluding the virtual
environment, the image corpus, the download cache, logs, and every checkpoint that is not the
champion or an ensemble member. It is further instructed to build the archive by copying that
list into a staging folder and zipping the folder, rather than zipping the project with
exclusions, because the latter makes it easy to sweep in a 2 GB corpus by accident.

The package targets Windows and assumes an autonomous coding agent will operate it, so every
entry point is a cross-platform Python script rather than a shell script, and the instructions
are written to be executed rather than read. The agent is told to babysit runs cheaply, to
change one variable per run, to treat differences under roughly 0.5 macro-F1 points on a
1,912-image validation fold as noise, and to check per-class recall so that a run which
improves the aggregate while collapsing a class is recorded as a regression.

## 7. The website

The original plan described a future website. It was later implemented as an offline,
bilingual web application using Flask and the repository's real PyTorch inference code.
React is stored locally under `website/vendor/`, allowing the interface to work without a
CDN during presentation.

Main features include:

- photo upload and clipboard paste;
- browser-camera capture and live recognition;
- real six-class probabilities from the local model;
- optional four-view accuracy mode;
- experimental activation-focused reclassification;
- model-derived localization/attention region;
- Chinese and English modes without mixed-language leakage;
- disposal-bin mapping with explanations;
- a material-based decision tree for uncertain cases;
- correct-versus-incorrect future simulations;
- local history of identifications and simulated choices;
- illustrative impact estimates with explicit assumptions;
- automatic checkpoint selection when a stronger model is supplied.

The website does not pretend that TrashNet covers every type of waste. Food and hazardous
waste are not model classes. The interface explains this limitation and uses the content
library and decision tree to provide safer educational guidance.

## 8. Confidence calibration and out-of-distribution images

The trained ResNet-50 was substantially under-confident, partly because label smoothing
discourages extreme output probabilities. Initial temperature scaling found an
NLL-optimal temperature near 0.22, but this made some incorrect predictions appear more
than 95% confident. A conservative temperature of 0.65 was chosen for the website instead.

Softmax confidence cannot solve the closed-set problem. A six-class softmax must assign an
unfamiliar image—such as a cat—to one of its six classes. Margin, entropy, and energy-score
guards were tested but failed: one real bottle appeared more uncertain than the cat.

The guard was therefore changed to feature-space nearest-neighbor distance. The uploaded
image's penultimate-layer embedding is compared with a bank of in-distribution embeddings.
In the small diagnostic set, real TrashNet images were at or below 0.63 cosine distance,
the bottle was 0.30, and the cat was 0.78. A threshold of 0.70 separated them.

This remains a limitation, not a solved research result. The threshold was informed by only
one explicit out-of-domain example, and the embedding bank must be rebuilt whenever the
served checkpoint changes.

## 9. Problems encountered and lessons learned

### 9.1 Untracked files were lost during reorganization

Results, documentation, and parts of a Windows training harness disappeared during an
earlier repository reorganization because they had never been committed. Copies were
recovered from archived downloads and conversation history.

**Lesson:** important results and operational scripts must be versioned before structural
changes. A clean-looking repository is not evidence that untracked work is safely stored.

### 9.2 A redundant frontend was initially built

A second SPA was created under `website/static/` before the existing `index.html` and
`support.js` application was identified. The duplicate frontend was removed, and changes
were integrated into the actual interface.

**Lesson:** inspect the application entry points and runtime architecture before adding a
parallel implementation.

### 9.3 Offline React assets were missing

The website referenced local React files that were not present, so it could not render
offline. React 18 production UMD files were added under `website/vendor/`.

**Lesson:** an offline claim must be tested with network access unavailable.

### 9.4 Launcher permissions failed on macOS

`start.sh` and `run.command` were not executable and produced a permission-denied error.
Their executable permissions were restored.

### 9.5 Bilingual modes leaked into each other

English subtitles appeared in Chinese mode, mixed labels appeared in both languages, and
Chinese punctuation remained in English mode. Rendering logic and translations were
cleaned so each language can stand alone.

### 9.6 Confidence was mistaken for correctness

The model could be correct with low confidence and confidently wrong after aggressive
calibration. This motivated conservative calibration, explicit uncertainty language, and
a separate feature-based OOD guard.

### 9.7 Real photographs exposed domain shift

TrashNet's clean backgrounds differ from cluttered phone photos. A model that performs well
on the benchmark can be only moderately reliable in deployment.

**Lesson:** benchmark accuracy and real-world usefulness are separate measurements. The
unified corpus was created partly to reduce this gap.

### 9.8 Platform-specific paths and temporary directories failed

Backslash/forward-slash differences threatened manifest joins, and Windows pytest failed
when it could not write to the account's temporary directory. Paths are now serialized with
forward slashes, and training preflight can use a repository-local pytest temporary folder.

The path problem was found by inspection rather than by failure. Four call sites serialized
with `str(path.relative_to(...))`, which on Windows would have emitted backslashes and broken
both the provenance script's filename parsing and the manifest-to-inventory join — on the
exact machine where training was to run, and as a confusing mismatch rather than an obvious
error. All four were changed to `as_posix()`, and the regenerated artifacts were confirmed
unchanged on macOS.

**Lesson:** when a pipeline is authored on one platform and executed on another, the
cross-platform review belongs before the handoff, not after the first failure report.

### 9.9 Data-loader workers failed for stdin scripts on macOS

Worker processes attempted to reload a script represented as `<stdin>` and crashed. The
diagnostic was moved to a real file and used `num_workers=0`.

### 9.10 Documentation and packaging drifted

The handoff referred to a nonexistent `setup.sh`, reported the wrong test count, described
three datasets while configuring five, and packaged a dependency lock it did not use.
These inconsistencies were corrected, and the handoff ZIP was rebuilt and verified.

### 9.11 Ignore rules were silently inert

`.gitignore` contained entries such as `data/raw/` and `data/unified_waste/included/`. A
leading path segment without a slash prefix anchors the pattern to the repository root, so
none of these ever matched the real locations under `model/`. The rules appeared to protect
the repository and did nothing. They were rewritten with `**/` prefixes and each one was
confirmed with `git check-ignore`.

This compounds the loss described in 9.1. Files were assumed to be deliberately excluded
when in fact the exclusion was not functioning.

**Lesson:** verify configuration that is supposed to have an effect. An ignore rule, like a
guard file, should be tested rather than read.

### 9.12 Incomparable experiments nearly competed for the same title

The champion-tracking script initially ranked every row of `experiments.csv` together. Four
completed TrashNet runs scored roughly 0.94 macro-F1 on a 434-image validation fold from the
old splits, and would have outranked any early unified-corpus run — declaring a model trained
on a superseded dataset the project champion. A filter on `splits_dir` was added so runs from
different split systems are never compared.

**Lesson:** an append-only log spanning two datasets is a hazard as well as an asset. Any
ranking over it must be scoped to a comparable population.

### 9.13 A test name claimed more than the test verified

A unit test named `test_stratification_falls_back_when_a_stratum_is_too_rare` did not
exercise the fallback at all; scikit-learn emits a warning in that situation rather than
raising, so the code path under test was never reached. The test passed and its name was
false. It was renamed to `test_a_rare_stratum_does_not_break_split_construction`, which is
what it actually checks.

**Lesson:** a test name is a claim about coverage. A green suite whose names overstate what
was verified is worse than a smaller honest one, because it discourages writing the missing
test.

## 10. Current remote-training plan under the 13-hour limit

The larger original backlog included four backbone comparisons, progressive resizing,
label auditing, five-fold training, ensembling, further datasets, and TTA tuning. That plan
is methodologically useful but does not fit a 13-hour GPU window.

The reduced plan is:

1. Complete setup and full preflight.
2. Autotune ConvNeXtV2 Tiny for the remote GPU.
3. Train ConvNeXtV2 Tiny at 224px on fold 0.
4. If enough time remains, progressively fine-tune the same model at 384px.
5. Use validation macro-F1 and per-class recall to select the checkpoint.
6. Reserve time to update the champion record, status, and transfer archive.
7. Do not spend the unified test set.

Estimated duration is approximately 7–12 hours, depending on GPU and download speed. A
usable 224px checkpoint should normally be available after roughly 3–5 hours. Full
five-fold training and a multi-architecture ensemble are intentionally deprioritized.

## 11. What five-fold training would have done

In five-fold cross-validation, five models are trained:

```text
Model 1: train on folds 1–4, validate on fold 0
Model 2: train on folds 0,2,3,4, validate on fold 1
Model 3: train on folds 0,1,3,4, validate on fold 2
Model 4: train on folds 0,1,2,4, validate on fold 3
Model 5: train on folds 0–3, validate on fold 4
```

Every train/validation image receives one out-of-fold prediction. This provides a more
stable mean and standard deviation, supports label-error analysis, and produces five models
whose probabilities can be averaged. It does not touch the quarantined test set.

Five-fold training was not completed in the measured TrashNet phase and is not currently
prioritized for the 13-hour unified-corpus run. It must therefore be described as planned
methodology, not a completed result.

## 12. Limitations

- Original results use a small benchmark and one 434-image validation fold.
- The original test has only 361 images; each error changes accuracy by about 0.277 points.
- The trash class has very few original examples, making its recall noisy.
- Many TrashNet images have clean backgrounds unlike real phone photographs.
- Dataset labels contain ambiguous or apparently incorrect examples.
- The error audit examined the highest-loss subset, not every test error.
- The website OOD threshold has not been validated on a large, diverse OOD benchmark.
- Disposal rules vary by city; plastic and composite items require local guidance.
- Impact values in the website are illustrative assumptions, not measured user outcomes.
- Unified-corpus performance is pending and cannot yet be compared fairly with the original
  TrashNet result.
- The original and unified phases use different data and splits, so their validation scores
  must never be placed in a table as if they were directly comparable.
- Duplicate detection is thresholded, not exhaustive. Perceptual-hash and embedding
  similarity find near-duplicates above a chosen cutoff; a visually similar pair below it
  will still be split apart. The firewall reduces leakage, it does not prove its absence.
- Provenance was recovered for TrashNet because the import ordering made it possible. There
  is no equivalent guarantee that the other four sources do not overlap each other, and no
  such audit has been performed.
- The unified corpus has been built and verified but never trained on. Every statement about
  it in this document concerns data, code, or method — none concerns performance.

## 13. Claims that are safe to make

- The original pipeline achieved 94.93% fold-0 validation accuracy with four-view TTA.
- The original test was measured once at 89.20% before later interventions.
- Group-aware splitting prevents detected near-duplicate groups from spanning splits.
- Augmentation is applied after splitting and only to training images.
- A public external dataset contained 2,218 renamed TrashNet images.
- All 361 images from the original spent test were found within the expanded corpus and
  quarantined by provenance. 318 of them carried names attributing them to another dataset.
- The provenance mapping is exact, not statistical: all 2,527 TrashNet images were matched
  with complete label agreement, and the script fails rather than warns if that is not true.
- The three-source unified build contains 11,299 usable images in 11,042 near-duplicate
  groups, split into 1,739 quarantined test images and 9,560 train/validation images across
  five folds of 1,912.
- Six images were dropped because three pixel-identical pairs carry contradictory labels.
  Two pairs are labeled glass by TrashNet and plastic by the dataset that redistributes it;
  the third is labeled glass and metal within TrashNet alone.
- Enlarging the corpus raised the `trash` class from 137 to 1,033 images, which addresses
  the minority-class problem through data rather than through technique.
- Small aggregate gains can hide a substantial minority-class recall loss.
- Softmax confidence alone did not reliably distinguish a real bottle from an OOD cat in
  the observed examples.

Claims to avoid:

- Do not call the 89.20% test result a final optimized score.
- Do not claim that all published 99–100% papers cheated. Say that leakage mechanisms can
  reproduce or help explain inflated results.
- Do not report estimated unified-corpus size or performance as measured fact.
- Do not claim the website reliably recognizes every type of waste.
- Do not claim that the feature-distance threshold is universally validated.
- Do not add results from different splits as though they were improvements on one shared
  benchmark.
- Do not describe a larger batch size, or any throughput setting, as an accuracy improvement.
- Do not present the three-source corpus figures as the final unified-corpus figures; the
  training machine builds from five sources.

A useful sanity rule for the eventual final evaluation: the legacy 361 images sit inside the
new test set, so their accuracy is directly comparable to 89.20%. It should be at least that.
If it comes back dramatically higher — above roughly 97% — that is the signature of surviving
contamination, not of a better model, and the split should be re-audited before the number is
reported.

## 14. Suggested paper structure

1. **Abstract:** task, leakage problem, method, measured result, and main lesson.
2. **Introduction:** waste sorting motivation and why headline accuracy can mislead.
3. **Dataset analysis:** TrashNet structure, imbalance, duplicate objects, and hidden
   cross-dataset reuse.
4. **Method:** provenance, deduplication, group-aware splits, quarantine, transfer learning,
   augmentation, TTA, and metrics.
5. **Experiments:** baseline, progressive resizing, modern augmentation, and weight decay.
6. **Results:** validation table, class recall, rejection criteria, and spent-test caption.
7. **Deployment:** offline website, uncertainty handling, OOD behavior, and education layer.
8. **Unified-corpus extension:** why it was built, safeguards, and pending remote training.
9. **Limitations and ethics:** label noise, domain shift, local recycling policies, and
   honest uncertainty.
10. **Conclusion:** reliable evaluation matters more than an unsupported near-perfect score.

## 15. Suggested presentation flow

1. **Problem:** people need accessible waste guidance.
2. **Surprise:** TrashNet's structure makes careless evaluation look nearly perfect.
3. **Leakage diagram:** same object or augmented image crossing a random split.
4. **Our firewall:** provenance → duplicate groups → split → train-only augmentation → test
   quarantine.
5. **Measured result:** 93.09% single-view validation and 94.93% with TTA.
6. **Why not claim more:** three intuitive training changes failed the predefined gate.
7. **Class-level lesson:** the augmentation run gained accuracy but harmed trash recall.
8. **Cross-dataset discovery:** 2,218 renamed TrashNet images in another public dataset.
9. **Product:** live bilingual website and future simulator.
10. **Real-world limitation:** bottle-versus-cat OOD example and domain shift.
11. **Next step:** time-bounded ConvNeXtV2 Tiny training on the unified corpus.
12. **Conclusion:** trustworthy data practice is part of model performance.

## 16. Key artifact references

- Raw experiment records: `model/experiments.csv`
- Original final report: `model/FINAL_SUMMARY.md`
- Human-readable decisions: `model/EXPERIMENT_LOG.md`
- Running history: `model/RESULTS.md`
- Spent-test guard: `model/reports/TEST_SPENT.json`
- Original split metadata: `model/data/splits/`
- Unified split metadata: `model/data/splits_unified/`
- Unified manifest and provenance: `model/data/unified_waste/manifest.csv` and
  `trashnet_provenance.csv`
- Quarantined spent-test paths: `model/data/splits_unified/quarantine.csv`
- Dropped images and reasons: `model/data/splits_unified/excluded.csv`
- Training source: `model/src/`
- Provenance reconstruction: `model/scripts/map_trashnet_provenance.py`
- Pre-training verification: `model/scripts/preflight.py`
- Batch-size and learning-rate tuning: `model/scripts/autotune.py`
- Champion and leaderboard derivation: `model/scripts/update_champion.py`
- Unified setup scripts: `model/scripts/`
- Website documentation: `website/README.md`
- Detailed development issues: `issues.md`
- Transfer package instructions: `handoff/README.md`, `handoff/RULES.md`,
  `handoff/STATUS.md`, and `handoff/CODEX_PROMPT.md`

## 17. Final project message

The most important result is not simply a percentage. The project demonstrates that model
quality depends on how the dataset was assembled, whether related images cross splits,
whether the test set influences decisions, how minority classes behave, and whether the
deployed interface communicates uncertainty honestly.

Second Life AI combines a functioning classifier and educational product with an audit
trail showing both successful and unsuccessful experiments. That complete process—the
constraints, failures, fixes, and limitations—is the strongest foundation for the final
paper and presentation.
