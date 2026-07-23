"""Second Life AI — backend for the offline web app (index.html + support.js).

Serves two trained models behind /api/identify so they can be compared on the
same photo. The frontend posts an image plus `model` and expects { cls, conf }.

  convnet      ConvNeXt V2-Tiny (384), 27.9M params, six classes.
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

import numpy as np  # noqa: E402
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
TRANSFORMER_CKPT = MODEL_DIR / "Swing B Transformer" / "best_swin_b.pt"
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
            "transform": build_eval_transform(CONVNEXT_IMG_SIZE), "label": "ConvNeXt V2-Tiny ConvNet",
            "params": sum(p.numel() for p in net.parameters()),
            # The bundled bank contains embeddings from a retired ResNet. It must
            # never be mixed with this model's feature space.
            "guarded": False}


app = Flask(__name__)
REGISTRY: dict = {}      # "convnet" / "transformer" -> loaded bundle, filled on demand
DEVICE = None
OOD_BANK = None          # [N, D] L2-normalized in-distribution embeddings, or None
OOD_K = 5

# Closed-set guard. The model has exactly six slots and softmax must sum to 1, so
# an out-of-domain photo (a cat, a face) is forced into whichever class it
# least-poorly resembles. Softmax margin/entropy CANNOT detect this: the model is
# under-confident enough that a real photo often looks flatter than a cat. What
# works is FEATURE space — an out-of-domain image's penultimate embedding is far
# from every trash cluster even when the classifier is weak. We score each photo
# by cosine distance to its k nearest neighbours in a precomputed bank of
# in-distribution embeddings (build_ood_bank.py) and flag it past the threshold.
# Measured on the baseline: real trash tops out ~0.63, a cat lands ~0.78, so 0.70
# separates them cleanly. The bundled bank holds retired ResNet embeddings, so
# both current loaders report the guard off rather than mixing feature spaces.
FEATURE_OOD_THRESHOLD = 0.70


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

    uncertain, ood_dist = False, None
    if bundle["guarded"] and OOD_BANK is not None:
        ref = nets[0]
        feat = torch.nn.functional.normalize(
            ref.forward_head(ref.forward_features(views[0]), pre_logits=True), dim=1)[0]
        ood_dist = float(1 - (OOD_BANK.to(feat.dtype) @ feat).topk(OOD_K).values.mean())
        uncertain = ood_dist > FEATURE_OOD_THRESHOLD
    return probs, uncertain, ood_dist


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
    try:
        probs, uncertain, ood_dist = predict(img, model_key)
    except FileNotFoundError:
        return jsonify(error=f"{TRANSFORMER_CKPT.name} not found — cannot serve that model."), 503
    top = max(probs, key=probs.get)
    # frontend reads cls + conf; `uncertain` triggers the "not one of my six" note
    return jsonify(cls=top, conf=probs[top], probs=probs, uncertain=uncertain,
                   model=model_key, guarded=REGISTRY[model_key]["guarded"],
                   ood_distance=None if ood_dist is None else round(ood_dist, 4))


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
        # single view under the model's own eval transform, matching what this
        # server actually runs. The OOD guard's bank holds retired ResNet embeddings,
        # so it is off: report the served bundle's real state, not the
        # mere presence of a bank it can't use.
        val_accuracy=val_accuracy, guarded=convnet["guarded"] and OOD_BANK is not None,
        ood_threshold=FEATURE_OOD_THRESHOLD,
        ood_bank_size=(0 if OOD_BANK is None else int(OOD_BANK.shape[0])),
        available=["convnet", "transformer"] if TRANSFORMER_CKPT.exists() else ["convnet"],
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
    global DEVICE, FEATURE_OOD_THRESHOLD
    ap = argparse.ArgumentParser(description=__doc__)
    # Hosted containers (Hugging Face Spaces, Fly, Render) inject PORT and need
    # the server bound to all interfaces; locally both stay loopback-only.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5001)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--ood-threshold", type=float, default=FEATURE_OOD_THRESHOLD,
                    help="flag 'uncertain' if kNN feature distance exceeds this (higher = laxer)")
    ap.add_argument("--ood-bank", default=str(HERE / "ood_bank.npz"),
                    help="path to the precomputed in-distribution embedding bank")
    args = ap.parse_args()

    FEATURE_OOD_THRESHOLD = args.ood_threshold
    DEVICE = torch.device(args.device)
    load_ood_bank(Path(args.ood_bank), DEVICE)
    convnet = get_bundle("convnet")
    extra = "" if TRANSFORMER_CKPT.exists() else "  (transformer checkpoint missing)"
    shown = "127.0.0.1" if args.host in {"127.0.0.1", "localhost"} else args.host
    print(f"[Second Life AI] {convnet['label']} · {DEVICE} · http://{shown}:{args.port}{extra}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
