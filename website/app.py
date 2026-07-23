"""Second Life AI — backend for the offline web app (index.html + support.js).

Serves two trained models behind /api/identify so they can be compared on the
same photo. The frontend posts an image plus `model` and expects { cls, conf }.

  convnet      TrashNeXt (ConvNeXt V2-Tiny, 384), 27.9M params, six classes.
  transformer  Swin-B (224), 86.7M params, trained on ten classes and restricted
               here to the same six output classes for comparison.

Fully offline: all weights come from local checkpoints, no downloads.

  python app.py [--port 5001] [--device cpu|cuda|mps]
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model"

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never phone home during a demo

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import models as tvm, transforms as T  # noqa: E402

# Serving constants, inlined so this file has no dependency on the training repo.
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(img_size: int):
    """Resize the short side to ~1.14x, centre-crop, normalize — the same eval
    protocol the ConvNeXt was trained and validated under."""
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# The Swin model's label order is sorted(os.listdir(standardized_256)); six of those
# ten are ours, and we read only those logits so both models answer the same
# question. The other four (battery, biological, clothes, shoes) are dropped.
_TRANSFORMER_CANDIDATES = [
    MODEL_DIR / "Swin B Transformer" / "best_swin_b.pt",
    # The original training directory in this project contains this historical
    # "Swing" typo. Keep both paths valid so deployment does not silently hide
    # a checkpoint that is already present.
    MODEL_DIR / "Swing B Transformer" / "best_swin_b.pt",
]
TRANSFORMER_CKPT = next(
    (path for path in _TRANSFORMER_CANDIDATES if path.exists()),
    _TRANSFORMER_CANDIDATES[0],
)
SWIN_CLASSES = ["battery", "biological", "cardboard", "clothes", "glass",
                "metal", "paper", "plastic", "shoes", "trash"]
SWIN_KEEP = [SWIN_CLASSES.index(c) for c in CLASSES]

# Our current model: ConvNeXt V2-Tiny (384), trained by
# model/convnextv2_tiny_cnn/train_and_upload.py and saved as a bare state_dict.
CONVNET_DIR = MODEL_DIR / "convnextv2_tiny_cnn" / "results"
CONVNET_CKPT = CONVNET_DIR / "best_convnextv2.pt"
CONVNET_META = CONVNET_DIR / "best_convnextv2_metadata.json"
CONVNEXT_ARCH = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
CONVNEXT_IMG_SIZE = 384


class SwinB(nn.Module):
    """The training architecture verbatim, so the checkpoint's keys line up."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.swin = tvm.swin_b(weights=None)  # weights come from the checkpoint
        self.swin.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.swin.head.in_features, num_classes),
        )

    def forward(self, x):
        return self.swin(x)


def load_transformer(device: torch.device):
    if not TRANSFORMER_CKPT.exists():
        raise FileNotFoundError(TRANSFORMER_CKPT)
    net = SwinB()
    net.load_state_dict(torch.load(TRANSFORMER_CKPT, map_location="cpu", weights_only=True))
    net = net.to(device).eval()
    # Its eval transform: squash to 224x224, no centre crop (train_swin_b.py:156).
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return {"nets": [net], "cfg": {"model": "swin_b", "img_size": 224},
            "transform": transform, "label": "Swin-B Transformer",
            "params": sum(p.numel() for p in net.parameters()), "guarded": False}


def load_convnet(device: torch.device):
    """ConvNeXt V2-Tiny (384) trained by model/convnextv2_tiny_cnn/train_and_upload.py,
    saved as a bare state_dict rather than the training repo's checkpoint dict."""
    import timm
    net = timm.create_model(CONVNEXT_ARCH, pretrained=False, num_classes=len(CLASSES))
    net.load_state_dict(torch.load(CONVNET_CKPT, map_location="cpu", weights_only=True))
    net = net.to(device).eval()
    return {"nets": [net], "cfg": {"model": CONVNEXT_ARCH, "img_size": CONVNEXT_IMG_SIZE},
            "transform": build_eval_transform(CONVNEXT_IMG_SIZE), "label": "TrashNeXt",
            "params": sum(p.numel() for p in net.parameters()),
            # The bundled bank contains embeddings from a retired ResNet. It must
            # never be mixed with this model's feature space.
            "guarded": False}


app = Flask(__name__)
REGISTRY: dict = {}      # "convnet" / "transformer" -> loaded bundle, filled on demand
DEVICE = None

# Optional five-signal "is this trash at all?" guard (ood_guard.py), off by default and
# toggled per request with guard=on. Built lazily on first use because it loads a second
# ~87M ImageNet Swin-B. Independent of the convnet/transformer display choice.
OOD_GUARD = None
OOD_GUARD_CALIB = HERE / "ood_guard_calib.npz"


def get_bundle(key: str):
    key = {"ours": "convnet", "swin": "transformer"}.get(key, key)
    if key not in {"convnet", "transformer"}:
        raise ValueError(f"unknown model: {key}")
    if key not in REGISTRY:
        if key == "convnet" and not CONVNET_CKPT.exists():
            raise SystemExit(
                f"No ConvNet checkpoint at {CONVNET_CKPT}. Train it or restore the results file."
            )
        REGISTRY[key] = load_transformer(DEVICE) if key == "transformer" else load_convnet(DEVICE)
        b = REGISTRY[key]
        print(f"[Second Life AI] loaded '{key}': {b['label']} "
              f"· {b['params'] / 1e6:.1f}M params · guard {'on' if b['guarded'] else 'off'}")
    return REGISTRY[key]


def get_guard():
    """Build the OOD guard lazily. Reuses the already-loaded Swin-B for features and
    loads its calibration cache; raises FileNotFoundError if the Swin checkpoint is
    missing (the guard is Swin-native regardless of the display model)."""
    global OOD_GUARD
    if OOD_GUARD is None:
        from ood_guard import OODGuard
        bundle = get_bundle("transformer")  # loads best_swin_b.pt if not already
        OOD_GUARD = OODGuard(bundle["nets"][0], DEVICE)
        OOD_GUARD.load_cache(OOD_GUARD_CALIB)
    return OOD_GUARD


@torch.no_grad()
def run_guard(img: Image.Image):
    """Run the five-signal guard on one image, using the Swin-B's 224 eval transform."""
    guard = get_guard()
    tf = get_bundle("transformer")["transform"]
    x = tf(img.convert("RGB")).unsqueeze(0)
    return guard.classify(x)


@torch.no_grad()
def predict(img: Image.Image, model_key: str = "convnet"):
    bundle = get_bundle(model_key)
    nets, size = bundle["nets"], bundle["cfg"]["img_size"]
    rgb = img.convert("RGB")

    # Each model runs under its own eval transform, single view.
    views = [bundle["transform"](rgb).unsqueeze(0).to(DEVICE)]

    logits = [net(x).float() for x in views for net in nets]
    if model_key == "transformer":
        logits = [lg[:, SWIN_KEEP] for lg in logits]  # read only the six shared slots
    disp = torch.stack([torch.softmax(lg, dim=1)[0] for lg in logits]).mean(0)
    probs = {c: float(p) for c, p in zip(CLASSES, disp)}

    return probs


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


@app.get("/documentary.mp4")
def documentary_video():
    return send_from_directory(HERE, "documentary.mp4", mimetype="video/mp4", conditional=True)


@app.get("/documentary-poster.jpg")
def documentary_poster():
    return send_from_directory(HERE, "documentary-poster.jpg", mimetype="image/jpeg")


@app.get("/logo-mark.svg")
def logo_mark():
    return send_from_directory(HERE, "logo-mark.svg", mimetype="image/svg+xml")


@app.get("/logo.svg")
def logo():
    return send_from_directory(HERE, "logo.svg", mimetype="image/svg+xml")


@app.get("/hero-phoenix.png")
def hero_phoenix():
    return send_from_directory(HERE, "hero-phoenix.png", mimetype="image/png")


@app.get("/logo-lockup.svg")
def logo_lockup():
    return send_from_directory(HERE, "logo-lockup.svg", mimetype="image/svg+xml")


@app.get("/favicon.svg")
def favicon():
    return send_from_directory(HERE, "favicon.svg", mimetype="image/svg+xml")


@app.get("/vendor/<path:filename>")
def vendor(filename):
    return send_from_directory(HERE / "vendor", filename)


# ---- API ----
@app.post("/api/identify")
def api_identify():
    img = _image_from_request()
    if img is None:
        return jsonify(error="No image received."), 400
    requested = str(request.form.get("model", request.args.get("model", "convnet"))).lower()
    model_key = {"ours": "convnet", "swin": "transformer"}.get(requested, requested)
    if model_key not in {"convnet", "transformer"}:
        return jsonify(error="model must be 'convnet' or 'transformer'"), 400
    # Old/cached frontends may still request the optional Transformer even when
    # its checkpoint is not installed on this host. Keep identification usable
    # by falling back to the always-present ConvNeXt model instead of returning
    # a phone-visible 503 error.
    fallback_model = model_key == "transformer" and not TRANSFORMER_CKPT.exists()
    if fallback_model:
        model_key = "convnet"
    guard_on = str(request.form.get("guard", request.args.get("guard", "off"))).lower() \
        in {"on", "1", "true", "yes"}
    try:
        probs = predict(img, model_key)
    except FileNotFoundError:
        return jsonify(error=f"{TRANSFORMER_CKPT.name} not found — cannot serve that model."), 503

    # The five-signal guard is a standalone "is this trash at all?" verdict layered on
    # the chosen classifier; when it says non-garbage, that drives the `uncertain` note.
    guard, uncertain = None, False
    if guard_on:
        try:
            guard = run_guard(img)
            uncertain = not guard["is_garbage"]
        except FileNotFoundError:
            guard = {"error": f"{TRANSFORMER_CKPT.name} not found — guard unavailable."}

    top = max(probs, key=probs.get)
    # frontend reads cls + conf; `uncertain` triggers the "not one of my six" note
    return jsonify(cls=top, conf=probs[top], probs=probs, uncertain=uncertain,
                   model=model_key, requested_model=requested,
                   fallback_model=fallback_model, guarded=guard_on, guard=guard)


@app.get("/api/model")
def api_model():
    convnet = get_bundle("convnet")
    val_accuracy = None
    if CONVNET_META.exists():
        val_accuracy = json.loads(CONVNET_META.read_text())["validation_accuracy"]
    return jsonify(
        default_model="convnet", model=convnet["label"],
        img_size=convnet["cfg"]["img_size"], members=len(convnet["nets"]),
        device=str(DEVICE), classes=CLASSES, is_ensemble=False,
        # Single view under the model's own evaluation transform.
        val_accuracy=val_accuracy, guarded=False,
        available=["convnet", "transformer"] if TRANSFORMER_CKPT.exists() else ["convnet"],
        # Five-signal guard: available if the Swin checkpoint exists; calibrated only if
        # its .npz cache is present (otherwise it runs the weaker uncalibrated fallback).
        guard_available=TRANSFORMER_CKPT.exists(),
        guard_calibrated=OOD_GUARD_CALIB.exists(),
    )

def main():
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__)
    # Hosted containers (Hugging Face Spaces, Fly, Render) inject PORT and need
    # the server bound to all interfaces; locally both stay loopback-only.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5001)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    DEVICE = torch.device(args.device)
    convnet = get_bundle("convnet")
    extra = "" if TRANSFORMER_CKPT.exists() else "  (transformer checkpoint missing)"
    shown = "127.0.0.1" if args.host in {"127.0.0.1", "localhost"} else args.host
    print(f"[Second Life AI] {convnet['label']} · {DEVICE} · http://{shown}:{args.port}{extra}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
