# Second Life AI — Dual-Model Waste Classifier

Second Life AI compares two image classifiers on the same waste photo:

| UI option | Architecture | Input | Outputs | Parameters | Recorded result |
|---|---|---:|---:|---:|---:|
| ConvNet | ConvNeXt V2-Tiny | 384×384 | 6 classes | 27,871,110 | 98.30% validation |
| Transformer | Swin-B | 224×224 | 10 trained / 6 shown | 86,753,474 | 97.81% test TTA |

The shared website classes are `cardboard`, `glass`, `metal`, `paper`,
`plastic`, and `trash`. Swin-B was trained with four additional classes; the
website restricts its logits to the six shared labels and renormalizes them so
the comparison answers the same question.

## Project layout

```text
model/
  convnextv2_tiny_cnn/
    train_and_upload.py        ConvNet training program
    results/                   ConvNet checkpoint + metadata
  Swin B Transformer/
    train_swin_b.py            Transformer training program
    best_swin_b.pt             Transformer checkpoint
    swin_b_results.json        Transformer metrics
    garbage_vs_nongarbage.py   OOD-guard research source
website/
  app.py                       one inference API for both models
  index.html                   bilingual comparison UI
submission/                    frozen paper/poster/course-submission material
video/                         presentation video source
website/remotion/              website documentary video source
```

The old ResNet experiment pipeline is no longer the active runtime. Historical
ResNet results remain in `submission/` because they document how the project
arrived at the two current models.

## Run the dual-model app

```bash
cd website
./start.sh
```

Then open <(https://161-33-76-49.sslip.io/)>. Both committed checkpoint paths are resolved
directly from `model/`; no checkpoint copying or network access is required for
inference.

The API uses stable model keys:

- `convnet` — ConvNeXt V2-Tiny (default)
- `transformer` — Swin-B

Legacy `ours` and `swin` request values are accepted only for compatibility.
See [GUIDE.md](GUIDE.md) for training and verification commands.
