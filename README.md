# Second Life AI: Dual-Model Waste Classifier

<p align="center">
  <a href="#overview"><img alt="Overview" src="https://img.shields.io/badge/Overview-Second%20Life%20AI-2f855a"></a>
  <a href="#model-comparison"><img alt="Models" src="https://img.shields.io/badge/Models-ConvNeXt%20V2%20%7C%20Swin--B-2563eb"></a>
  <a href="#run-the-app"><img alt="App" src="https://img.shields.io/badge/App-Flask%20Inference-f59e0b"></a>
</p>

<p align="center">
  <img src="website/documentary-poster.jpg" alt="Second Life AI waste-classification interface preview" width="820">
</p>

## Overview

Second Life AI compares two image classifiers on the same waste photo through a
single bilingual web interface. The app reports normalized probabilities for the
six shared material categories:

`cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.

The ConvNet path uses ConvNeXt V2-Tiny at 384px. The Transformer path uses
Swin-B at 224px; it was trained with ten labels, and the website restricts its
logits to the same six public labels before renormalizing the prediction.

Second Life AI is designed to:

- compare CNN and Transformer predictions under one consistent API;
- keep checkpoint metadata close to the model that produced it;
- expose the same six probability keys for every supported model; and
- run locally without copying checkpoints into the website directory.

## Model Comparison

| UI option | Architecture | Input | Outputs | Parameters | Recorded result |
|---|---|---:|---:|---:|---:|
| ConvNet | ConvNeXt V2-Tiny | 384x384 | 6 classes | 27,871,110 | 98.30% validation |
| Transformer | Swin-B | 224x224 | 10 trained / 6 shown | 86,753,474 | 97.81% test TTA |

The shared runtime contract keeps the public model keys stable:

- `convnet` - ConvNeXt V2-Tiny, used as the default model.
- `transformer` - Swin-B, filtered to the six shared website labels.

Legacy request values `ours` and `swin` are still accepted for compatibility.

## Run the App

```bash
cd website
./start.sh
```

Then open <http://127.0.0.1:5001>.

Useful checks while the server is running:

```bash
curl http://127.0.0.1:5001/api/model
curl -F image=@/path/to/photo.jpg -F model=convnet \
  http://127.0.0.1:5001/api/identify
curl -F image=@/path/to/photo.jpg -F model=transformer \
  http://127.0.0.1:5001/api/identify
```

Both committed checkpoint paths are resolved directly from `model/`; no network
access is required for inference.

## Project Layout

```text
model/
  convnextv2_tiny_cnn/
    train_and_upload .py
    results/
  Swing B Transformer/
    train_swin_b.py
    best_swin_b.pt
    swin_b_results.json
website/
  app.py
  index.html
  documentary-poster.jpg
  start.sh
video/
  remotion.config.ts
```

See [GUIDE.md](GUIDE.md) for training, verification, and checkpoint contract
details.
