# Second Life AI — website

An offline web app around the **real** trained TrashNet classifier. Identify an
item, learn which bin it belongs in and what it's made of, then play out the two
futures waiting for it in the Future Simulator.

## Run it

**macOS:** double-click `run.command`
**Windows:** double-click `run.bat`
**Any terminal:**

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision timm scikit-learn pandas pyyaml pillow flask
.venv/bin/python app.py            # then open http://127.0.0.1:5001
```

First launch builds the environment (needs internet once). After that it runs
fully offline — no CDN, no analytics, no network calls during a demo.

## How it serves the model

`app.py` reuses the training repo's own inference code (`../model/src`), so the
predictions and confidences are identical to `evaluate.py` — never inflated.
It picks a checkpoint automatically (first match wins):

1. `checkpoints/opt_duo_ensemble/ensemble.json` — the strong ensemble (once trained)
2. `checkpoints/*/ensemble.json` — any ensemble manifest
3. `checkpoints/baseline_resnet50/fold0/best.pth` — today's ResNet-50 baseline

**Swapping in the better model tomorrow:** drop the trained `checkpoints/` folder
from the training run in here (or point at it with `--checkpoint`) and restart.
The About page and footer update to whatever is actually loaded.

```bash
.venv/bin/python app.py --checkpoint checkpoints/opt_duo_ensemble/ensemble.json
```

## Layout

```
app.py          Flask backend: /api/identify, /api/content, /api/spotlight, /api/model
content.json    offline content library (bin advice, second-life facts, the two futures)
static/         index.html · styles.css · app.js · img/ (sample photos)
checkpoints/    model weights (gitignored — placed locally, not committed)
```

## Honesty

Every confidence is the model's raw softmax output. The About page states the
real 94.93% validation accuracy and notes that the six material classes don't
cover everything real recycling does, and that bin advice varies by locality.
The Gradio demo at `../model/demo/app.py` is left untouched.
