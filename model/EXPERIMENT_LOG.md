# Experiment Decision Log

This is the human-readable decision log. `experiments.csv` remains the append-only raw metric audit trail.

| Time (AEST) | Step | Experiment | Config hash | Reference | Result | Net images | Verdict |
|---|---:|---|---|---:|---:|---:|---|
| 2026-07-18 16:52 | 0 | Locked raw ResNet-50 epoch 44 + four-view TTA | `ab586f1a72` | 404/434 single | 412/434, macro-F1 0.9414 | +8 | Adopted before accuracy experiments |
| 2026-07-18 17:36 | 1 | ResNet-50 224 to 384 progressive resize | `bc27d87b89` | 412/434 TTA | 412/434, macro-F1 0.9442 | 0 | Rejected; 10 fixes/10 regressions, cardboard-paper 2 to 3 |
| 2026-07-18 18:03 | 2a | ResNet-50 modern augmentation only | `c3b7f8e2c9` | 412/434 TTA | 415/434, macro-F1 0.9421 | +3 | Rejected; below +7 and trash recall 20/23 to 18/23 |
| 2026-07-18 18:34 | 2b | ResNet-50 weight decay 0.10 only | `23aa1cc941` | 412/434 TTA | 411/434, macro-F1 0.9390 | -1 | Rejected; below +7 and macro-F1 dropped |
