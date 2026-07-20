"""Second Life AI — backend for the offline web app (index.html + support.js).

Serves the actually-trained TrashNet model behind the site's /api/identify hook.
Same inference path as src/evaluate.py, so predictions are the model's real
output. The frontend's onFile() posts an image and expects { cls, conf }.

Model selection (first match wins), override with --checkpoint:
  1. checkpoints/opt_duo_ensemble/ensemble.json   (tomorrow's strong model)
  2. checkpoints/*/ensemble.json                   (any ensemble)
  3. checkpoints/baseline_resnet50/fold0/best.pth  (today's baseline)

Fully offline: ResNet-50 weights come from the checkpoint, no downloads.

  python app.py [--checkpoint PATH] [--port 5001] [--device cpu|cuda|mps]
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model"
sys.path.insert(0, str(MODEL_DIR))  # reuse the training repo's inference code

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never phone home during a demo

import torch  # noqa: E402
from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
from PIL import Image  # noqa: E402

from src.data import build_eval_transform  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import CLASSES  # noqa: E402


def find_checkpoint() -> Path:
    root = HERE / "checkpoints"
    duo = root / "opt_duo_ensemble" / "ensemble.json"
    if duo.exists():
        return duo
    ensembles = sorted(root.glob("*/ensemble.json"))
    if ensembles:
        return ensembles[0]
    baseline = root / "baseline_resnet50" / "fold0" / "best.pth"
    if baseline.exists():
        return baseline
    raise SystemExit(
        f"No model found under {root}. Drop in a trained checkpoint "
        "(a fold's best.pth, or an ensemble.json manifest) and restart."
    )


def load_predictor(checkpoint: Path, device: torch.device):
    """Mirror src.evaluate.load_models, resolving ensemble members relative to
    the manifest so the site works wherever the checkpoints folder lands."""
    if checkpoint.suffix == ".json":
        manifest = json.loads(checkpoint.read_text())
        members = [Path(p) if Path(p).is_absolute() else checkpoint.parent.parent / p
                   for p in manifest["checkpoints"]]
    else:
        members = [checkpoint]

    models, cfg = [], None
    for p in members:
        if not p.exists():                    # manifest written on another machine
            p = checkpoint.parent / Path(p).name
        state = torch.load(p, map_location="cpu", weights_only=False)
        cfg = state["cfg"]
        model = build_model(cfg, pretrained=False)
        model.load_state_dict(state["ema"] if state.get("ema") is not None else state["model"])
        models.append(model.to(device).eval())

    return models, cfg, build_eval_transform(cfg["img_size"])


app = Flask(__name__)
MODELS = CFG = TRANSFORM = DEVICE = None


@torch.no_grad()
def predict(img: Image.Image):
    x = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.stack(
        [torch.softmax(m(x).float(), dim=1)[0] for m in MODELS]
    ).mean(0).cpu().tolist()
    return {c: p for c, p in zip(CLASSES, probs)}


def _image_from_request():
    if "image" in request.files:
        return Image.open(request.files["image"].stream)
    data = request.get_json(silent=True) or {}
    if data.get("image"):
        raw = data["image"].split(",", 1)[-1]
        return Image.open(io.BytesIO(base64.b64decode(raw)))
    return None


# ---- static frontend (the real site: index.html + support.js + vendor React) ----
@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/support.js")
def support_js():
    return send_from_directory(HERE, "support.js")


@app.get("/vendor/<path:filename>")
def vendor(filename):
    return send_from_directory(HERE / "vendor", filename)


# ---- API ----
@app.post("/api/identify")
def api_identify():
    img = _image_from_request()
    if img is None:
        return jsonify(error="No image received."), 400
    probs = predict(img)
    top = max(probs, key=probs.get)
    return jsonify(cls=top, conf=probs[top], probs=probs)  # frontend reads cls + conf


@app.get("/api/model")
def api_model():
    return jsonify(
        model=CFG["model"], img_size=CFG["img_size"], members=len(MODELS),
        device=str(DEVICE), classes=CLASSES, is_ensemble=len(MODELS) > 1,
        val_accuracy_tta="94.93%",
    )


def main():
    global MODELS, CFG, TRANSFORM, DEVICE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    DEVICE = torch.device(args.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    MODELS, CFG, TRANSFORM = load_predictor(checkpoint, DEVICE)
    tag = f"{len(MODELS)}-model ensemble" if len(MODELS) > 1 else CFG["model"]
    print(f"[Second Life AI] {tag} · {DEVICE} · http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
