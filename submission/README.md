# Second Life AI — Current Submission

NYU Shanghai · Introduction to Artificial Intelligence · Image Classification

**🔗 Live demo: [bit.ly/second-life-ai](https://bit.ly/second-life-ai)** — upload a
photo and classify it live with either model, side by side.

This package reflects the two models used by the current project and website:

| Track | Model | Input | Labels | Parameters | Recorded result |
|---|---|---:|---:|---:|---:|
| Cap-compliant ConvNet | TrashNeXt (ConvNeXt V2-Tiny) | 384×384 | 6 | 27,871,110 | 98.30% validation |
| Transformer comparison | Swin-B | 224×224 | 10 trained | 86,753,474 | 97.61% test; 97.82% TTA |

The models were trained on different datasets and splits, so their scores are
reported as separate design points rather than a head-to-head ranking.

## Contents

```text
submission/
├── README.md
├── GUIDE.md
├── code/
│   ├── README.md
│   ├── train_convnextv2.py             training — ConvNeXt V2-Tiny
│   ├── train_swin_b.py                 training — Swin-B
│   ├── evaluate.py                     testing — deterministic, both models
│   ├── verify_artifacts.py             checkpoint + metrics integrity check
│   ├── tests/                          16 tests (pytest tests -q)
│   ├── requirements.txt
│   └── requirements.lock.txt           exact tested versions
├── results/
│   ├── manifest.json
│   ├── convnextv2_tiny/
│   │   ├── best_convnextv2.pt          trained weights (106 MB)
│   │   └── metrics.json
│   └── swin_b/
│       ├── best_swin_b.pt              trained weights (331 MB)
│       └── metrics.json
├── figures/                    figures referenced by the paper
├── paper/
│   ├── Second-Life-AI-paper.pdf
│   └── paper.html
└── poster/
    ├── Second-Life-AI-poster.pdf
    ├── poster.html
    └── OFFICIAL-TEMPLATE-reference.pptx
```

The trained checkpoints ship inside `results/`, one per model, so the package is
self-contained. Each is identified by SHA-256 in `results/manifest.json`; run
`python code/verify_artifacts.py` to confirm the weights and metrics match.

The paper retains the ResNet-50 phase as historical motivation and comparison,
but the obsolete ResNet training pipeline, checkpoints, predictions, and
duplicate result exports are no longer part of the submitted code package.

See `GUIDE.md` for setup, training, verification, and website commands.
