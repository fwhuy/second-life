# Second Life AI — website

An offline web app around the **real** trained TrashNet classifier. Identify an
item, learn which bin it belongs in and what it's made of, then play out the two
futures waiting for it in the Future Simulator. Bilingual (中文 / English).

## Run it

**macOS:** double-click `run.command`  ·  **Windows:** double-click `run.bat`
**Terminal (macOS/Linux):** `./start.sh`

Or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision timm scikit-learn pandas pyyaml pillow flask
.venv/bin/python app.py            # then open http://127.0.0.1:5001
```

First launch builds the environment (needs internet once). After that it runs
fully offline — React is served from `vendor/`, and inference is local.

## How the pieces fit

- **`index.html`** — the whole UI (a React app in an `x-dc` template) plus the
  content library, stories, and the two-futures simulator. Its `onFile` handler
  posts the uploaded photo to `/api/identify`.
- **`support.js`** — the `x-dc`/React runtime that renders `index.html`.
- **`vendor/`** — React 18 UMD (production), so the page works with no CDN.
- **`app.py`** — Flask backend. Serves the page and answers `/api/identify`
  with `{ cls, conf, probs }` — the model's **real** softmax output, computed
  through the training repo's own inference code (`../model/src`), identical to
  `evaluate.py`. When the backend isn't running, the frontend falls back to
  labelled demo data and says so.

Upload a photo → the model's real class, confidence, and six-class probabilities
show on the result page (captioned "real softmax output"). Type an item name
instead and it uses the offline content library.

## Swapping in the stronger model

`app.py` auto-selects a checkpoint (first match wins):

1. `checkpoints/opt_duo_ensemble/ensemble.json` — the trained ensemble
2. `checkpoints/*/ensemble.json` — any ensemble manifest
3. `checkpoints/baseline_resnet50/fold0/best.pth` — today's ResNet-50 baseline

Drop the trained `checkpoints/` folder in here (or pass `--checkpoint`) and
restart. Nothing else changes.

```bash
.venv/bin/python app.py --checkpoint checkpoints/opt_duo_ensemble/ensemble.json
```

## Honesty

Every confidence shown for a photo is the model's raw output, never inflated.
The home page states the real 94.93% validation accuracy and maps the six model
classes to bins transparently, noting the dataset has no food-waste or hazardous
classes. The Gradio demo at `../model/demo/app.py` is left untouched.

`.venv/` and `checkpoints/` are gitignored; `vendor/` React is committed so the
site runs offline straight from a clone.
