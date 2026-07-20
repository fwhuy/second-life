"""Second Life AI — offline web app around the real TrashNet classifier.

Serves the actually-trained model (same inference path as src/evaluate.py) behind
a friendly site: identify an item, learn how to bin it, and play out its future.

Model selection (first match wins), override with --checkpoint:
  1. website/checkpoints/opt_duo_ensemble/ensemble.json   (tomorrow's strong model)
  2. website/checkpoints/*/ensemble.json                   (any ensemble)
  3. website/checkpoints/baseline_resnet50/fold0/best.pth  (today's baseline)

Runs fully offline: ResNet-50 weights come from the checkpoint, no downloads.

  python app.py [--checkpoint PATH] [--port 5001] [--device cpu|cuda|mps]
"""

import argparse
import base64
import io
import json
import os
import sys
from datetime import date
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

CONTENT = json.loads((HERE / "content.json").read_text())


def find_checkpoint() -> Path:
    ckpt_root = HERE / "checkpoints"
    duo = ckpt_root / "opt_duo_ensemble" / "ensemble.json"
    if duo.exists():
        return duo
    ensembles = sorted(ckpt_root.glob("*/ensemble.json"))
    if ensembles:
        return ensembles[0]
    baseline = ckpt_root / "baseline_resnet50" / "fold0" / "best.pth"
    if baseline.exists():
        return baseline
    raise SystemExit(
        f"No model found under {ckpt_root}. Drop in a trained checkpoint "
        "(a fold's best.pth, or an ensemble.json manifest) and restart."
    )


def load_predictor(checkpoint: Path, device: torch.device):
    """Mirror src.evaluate.load_models, but resolve ensemble members relative to
    the manifest so the site works wherever the checkpoints folder is dropped."""
    if checkpoint.suffix == ".json":
        manifest = json.loads(checkpoint.read_text())
        members = []
        for p in manifest["checkpoints"]:
            p = Path(p)
            members.append(p if p.is_absolute() else (checkpoint.parent.parent / p))
    else:
        members = [checkpoint]

    models, cfg = [], None
    for p in members:
        if not p.exists():  # tolerate manifest paths written on another machine
            p = checkpoint.parent / Path(p).name
        state = torch.load(p, map_location="cpu", weights_only=False)
        cfg = state["cfg"]
        model = build_model(cfg, pretrained=False)
        model.load_state_dict(state["ema"] if state.get("ema") is not None else state["model"])
        models.append(model.to(device).eval())

    transform = build_eval_transform(cfg["img_size"])
    return models, cfg, transform


app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="")

MODELS = None
CFG = None
TRANSFORM = None
DEVICE = None


@torch.no_grad()
def predict(img: Image.Image):
    x = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.stack(
        [torch.softmax(m(x).float(), dim=1)[0] for m in MODELS]
    ).mean(0).cpu().tolist()
    ranked = sorted(
        ({"class": c, "prob": p} for c, p in zip(CLASSES, probs)),
        key=lambda d: d["prob"], reverse=True,
    )
    return ranked


def _image_from_request():
    if "image" in request.files:
        return Image.open(request.files["image"].stream)
    data = request.get_json(silent=True) or {}
    if data.get("image"):
        raw = data["image"].split(",", 1)[-1]  # strip data-URL prefix
        return Image.open(io.BytesIO(base64.b64decode(raw)))
    return None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/identify")
def api_identify():
    img = _image_from_request()
    if img is None:
        return jsonify(error="No image received. Upload a photo or pick a sample."), 400
    ranked = predict(img)
    top = ranked[0]
    return jsonify(
        predictions=ranked,
        top=top["class"],
        confidence=top["prob"],
        content=CONTENT["classes"][top["class"]],
    )


@app.get("/api/content/<cls>")
def api_content(cls):
    entry = CONTENT["classes"].get(cls)
    if not entry:
        return jsonify(error=f"Unknown class '{cls}'."), 404
    return jsonify(entry)


@app.get("/api/spotlight")
def api_spotlight():
    pool = CONTENT["spotlights"]
    pick = pool[date.today().toordinal() % len(pool)]  # stable for the whole day
    return jsonify(spotlight=pick, item=CONTENT["classes"][pick["class"]])


@app.get("/api/model")
def api_model():
    return jsonify(
        model=CFG["model"],
        img_size=CFG["img_size"],
        members=len(MODELS),
        device=str(DEVICE),
        classes=CLASSES,
        val_accuracy_tta="94.93%",  # ResNet-50 baseline + 4-view TTA (see reports/)
        is_ensemble=len(MODELS) > 1,
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
    print(f"[Second Life AI] {CFG['model']} · {len(MODELS)} model(s) · "
          f"{DEVICE} · {checkpoint.relative_to(HERE) if checkpoint.is_relative_to(HERE) else checkpoint}")
    print(f"[Second Life AI] http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
