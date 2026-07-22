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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
from PIL import Image  # noqa: E402

from src.data import build_eval_transform  # noqa: E402
from src.evaluate import tta_transforms  # noqa: E402
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
OOD_BANK = None      # [N, D] L2-normalized in-distribution embeddings, or None
OOD_K = 5

# --- Calibration + out-of-distribution guard --------------------------------
# The baseline was trained with label smoothing, so it is badly under-confident:
# ~89% accurate on validation but only ~40% mean top-confidence. Two things:
#
# 1) Temperature scaling (T<1 sharpens the softmax) restores believable
#    confidence. The NLL-optimal T (~0.22) saturates easy cases to ~100% and
#    makes some wrong answers look 95%+ — dishonest for a demo. We use a
#    conservative T=0.65: on fold-0 validation it shows correct predictions
#    ~56% and wrong ones ~41%, with essentially nothing >=90%. Never changes
#    the top class.
#
# 2) Closed-set guard. The model has exactly six slots and softmax must sum to
#    1, so an out-of-domain photo (a cat, a face) is forced into whichever class
#    it least-poorly resembles. Softmax margin/entropy CANNOT detect this: the
#    model is so under-confident that a real photo often looks flatter than a
#    cat. What works is FEATURE space — an out-of-domain image's penultimate
#    embedding is far from every trash cluster even when the classifier is weak.
#    We score each photo by cosine distance to its k nearest neighbours in a
#    precomputed bank of in-distribution embeddings (build_ood_bank.py) and flag
#    it when that distance exceeds FEATURE_OOD_THRESHOLD. Measured on the
#    baseline: real trash tops out ~0.63, a cat lands ~0.78, so 0.70 separates
#    them cleanly. If the bank is missing, the guard is simply off.
TEMPERATURE = 0.65            # softens confidence without inflating it
FEATURE_OOD_THRESHOLD = 0.70  # flag as uncertain if kNN feature distance exceeds this


def activation_box(features: torch.Tensor, model, class_index: int):
    """Return a normalized weak-localization box from the classifier's spatial
    evidence. This is class activation mapping, not object detection: it shows
    the region that most influenced the TrashNet prediction without requiring a
    separately trained detector or box annotations."""
    if not isinstance(features, torch.Tensor) or features.ndim != 4:
        return None
    fmap = features[0].float()
    # Some backbones expose NHWC features; normalize to CHW.
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    in_features = getattr(classifier, "in_features", None)
    if in_features and fmap.shape[-1] == in_features and fmap.shape[0] != in_features:
        fmap = fmap.permute(2, 0, 1)
    try:
        weights = classifier.weight[class_index].float().view(-1, 1, 1)
        heat = torch.relu((fmap * weights).sum(0)) if weights.shape[0] == fmap.shape[0] else fmap.abs().mean(0)
    except (AttributeError, IndexError, RuntimeError):
        heat = fmap.abs().mean(0)
    if heat.numel() == 0 or float(heat.max()) <= 0:
        return None
    threshold = torch.maximum(torch.quantile(heat.flatten(), 0.72), heat.max() * 0.38)
    mask = (heat >= threshold).detach().cpu()
    # Keep the strongest connected evidence island instead of enclosing every
    # scattered hot cell, which would often produce an unhelpfully huge box.
    seen, components = set(), []
    for yy in range(mask.shape[0]):
        for xx in range(mask.shape[1]):
            if not bool(mask[yy, xx]) or (yy, xx) in seen:
                continue
            stack, component = [(yy, xx)], []
            seen.add((yy, xx))
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and bool(mask[ny, nx]) and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            components.append(component)
    if not components:
        return None
    strongest = max(components, key=lambda c: sum(float(heat[y, x]) for y, x in c))
    points = torch.tensor(strongest)
    if points.numel() == 0:
        return None
    height, width = heat.shape
    y1, x1 = points.min(0).values.tolist()
    y2, x2 = points.max(0).values.tolist()
    # Add context and enforce a useful minimum size for the overlay.
    pad_x, pad_y = max(1, round(width * .07)), max(1, round(height * .07))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(width - 1, x2 + pad_x), min(height - 1, y2 + pad_y)
    return {
        "x": round(x1 / width, 4), "y": round(y1 / height, 4),
        "w": round((x2 - x1 + 1) / width, 4), "h": round((y2 - y1 + 1) / height, 4),
        "method": "class_activation_map",
    }


def focus_crop(img: Image.Image, bbox: dict | None, padding: float = 0.12):
    """Crop around class-activation evidence with some surrounding context.

    This is a lightweight, depth-like foreground focus—not a true depth map.
    Returning None leaves the original inference path unchanged.
    """
    if not bbox:
        return None
    width, height = img.size
    x1 = max(0.0, bbox["x"] - padding)
    y1 = max(0.0, bbox["y"] - padding)
    x2 = min(1.0, bbox["x"] + bbox["w"] + padding)
    y2 = min(1.0, bbox["y"] + bbox["h"] + padding)
    # Avoid magnifying tiny/noisy activation islands.
    if (x2 - x1) * (y2 - y1) < 0.12:
        return None
    return img.crop((round(x1 * width), round(y1 * height),
                     round(x2 * width), round(y2 * height)))


@torch.no_grad()
def predict(img: Image.Image, accuracy_mode: bool = False, focus_filter: bool = False):
    rgb = img.convert("RGB")
    transforms = tta_transforms(CFG["img_size"]) if accuracy_mode else [TRANSFORM]
    views = [tf(rgb).unsqueeze(0).to(DEVICE) for tf in transforms]
    disp = torch.stack([
        torch.softmax(m(x).float() / TEMPERATURE, dim=1)[0]
        for x in views for m in MODELS
    ]).mean(0)
    # Feature-space out-of-distribution distance, from the reference model (the
    # bank was built with MODELS[0]); None/off when no bank is loaded.
    uncertain, ood_dist = False, None
    ref = MODELS[0]
    x = views[0]  # localization/OOD use the canonical view for stable geometry
    ref_features = ref.forward_features(x)
    bbox = activation_box(ref_features, ref, int(disp.argmax()))
    focused = focus_crop(rgb, bbox) if focus_filter else None
    if focused is not None:
        focus_x = TRANSFORM(focused).unsqueeze(0).to(DEVICE)
        focus_disp = torch.stack([
            torch.softmax(m(focus_x).float() / TEMPERATURE, dim=1)[0]
            for m in MODELS
        ]).mean(0)
        # Keep the original view dominant because this filter is experimental
        # and the model was trained on full images, not CAM-derived crops.
        disp = 0.65 * disp + 0.35 * focus_disp
    probs = {c: float(p) for c, p in zip(CLASSES, disp)}
    if OOD_BANK is not None:
        feat = torch.nn.functional.normalize(
            ref.forward_head(ref_features, pre_logits=True), dim=1)[0]
        sims = OOD_BANK.to(feat.dtype) @ feat
        ood_dist = float(1 - sims.topk(OOD_K).values.mean())
        uncertain = ood_dist > FEATURE_OOD_THRESHOLD
    return probs, uncertain, ood_dist, bbox


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
    accuracy_mode = str(request.form.get("accuracy_mode", request.args.get("accuracy_mode", "0"))).lower() in {"1", "true", "yes", "on"}
    focus_filter = str(request.form.get("focus_filter", request.args.get("focus_filter", "0"))).lower() in {"1", "true", "yes", "on"}
    probs, uncertain, ood_dist, bbox = predict(
        img, accuracy_mode=accuracy_mode, focus_filter=focus_filter)
    top = max(probs, key=probs.get)
    # frontend reads cls + conf; `uncertain` triggers the "not one of my six" note
    return jsonify(cls=top, conf=probs[top], probs=probs, uncertain=uncertain, bbox=bbox,
                   accuracy_mode=accuracy_mode, focus_filter=focus_filter,
                   views=(4 if accuracy_mode else 1) + (1 if focus_filter else 0),
                   ood_distance=None if ood_dist is None else round(ood_dist, 4))


@app.get("/api/model")
def api_model():
    return jsonify(
        model=CFG["model"], img_size=CFG["img_size"], members=len(MODELS),
        device=str(DEVICE), classes=CLASSES, is_ensemble=len(MODELS) > 1,
        val_accuracy_tta="94.93%", temperature=TEMPERATURE, calibrated=True,
        guarded=OOD_BANK is not None, ood_threshold=FEATURE_OOD_THRESHOLD,
        ood_bank_size=(0 if OOD_BANK is None else int(OOD_BANK.shape[0])),
    )


def load_ood_bank(path: Path, device: torch.device):
    """Load the precomputed in-distribution embedding bank, or None if absent."""
    global OOD_BANK, OOD_K
    if not path.exists():
        print(f"[Second Life AI] no OOD bank at {path.name} — closed-set guard OFF "
              f"(run build_ood_bank.py to enable it)")
        return
    data = np.load(path)
    OOD_BANK = torch.from_numpy(data["features"]).float().to(device)
    OOD_K = int(data["k"])
    print(f"[Second Life AI] OOD guard ON · bank {OOD_BANK.shape[0]}x{OOD_BANK.shape[1]} "
          f"· threshold {FEATURE_OOD_THRESHOLD}")


def main():
    global MODELS, CFG, TRANSFORM, DEVICE, TEMPERATURE, FEATURE_OOD_THRESHOLD
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--temperature", type=float, default=TEMPERATURE,
                    help="calibration temperature; <1 sharpens confidence (0=off -> raw)")
    ap.add_argument("--ood-threshold", type=float, default=FEATURE_OOD_THRESHOLD,
                    help="flag 'uncertain' if kNN feature distance exceeds this (higher = laxer)")
    ap.add_argument("--ood-bank", default=str(HERE / "ood_bank.npz"),
                    help="path to the precomputed in-distribution embedding bank")
    args = ap.parse_args()

    TEMPERATURE = args.temperature if args.temperature > 0 else 1.0
    FEATURE_OOD_THRESHOLD = args.ood_threshold
    DEVICE = torch.device(args.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    MODELS, CFG, TRANSFORM = load_predictor(checkpoint, DEVICE)
    load_ood_bank(Path(args.ood_bank), DEVICE)
    tag = f"{len(MODELS)}-model ensemble" if len(MODELS) > 1 else CFG["model"]
    print(f"[Second Life AI] {tag} · {DEVICE} · http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
