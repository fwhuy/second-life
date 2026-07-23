"""Calibrate the OOD guard — one-time, run on the prep machine, not the server.

Extracts the garbage Swin-B's 1024-d penultimate features over a set of real waste
images and writes them to website/ood_guard_calib.npz (features + labels). ood_guard.py
fits the Mahalanobis + prototype detectors from that cache at serve time, so the VM
never needs the image dataset.

Two image sources:
  --hf garythung/trashnet   pull TrashNet from HuggingFace (6 classes: cardboard, glass,
                            metal, paper, plastic, trash — exactly the six shared classes)
  --dir path/to/garbage     a folder of class subdirectories (the original's format)

  python build_ood_guard.py --hf garythung/trashnet --per-class 120
"""

import argparseyes
import ssl
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

ssl._create_default_https_context = ssl._create_unverified_context  # HF/torch cert wall

from app import SwinB, TRANSFORMER_CKPT  # noqa: E402  reuse the deployed architecture
from ood_guard import swin_forward, SWIN_CLASSES  # noqa: E402

HERE = Path(__file__).resolve().parent
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The garbage Swin-B's eval transform (app.py load_transformer): squash to 224, no crop.
EVAL_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def collect_from_hf(name: str, per_class: int):
    """Yield (PIL image, swin_class_index) from a HuggingFace image-classification set."""
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    label_names = ds.features["label"].names  # e.g. ['cardboard','glass',...]
    print(f"  {name}: {len(ds)} images, classes={label_names}")

    counts = {i: 0 for i in range(len(label_names))}
    for ex in ds:
        li = ex["label"]
        cls_name = label_names[li]
        if cls_name not in SWIN_CLASSES:
            continue  # skip classes the Swin head doesn't have
        if counts[li] >= per_class:
            continue
        counts[li] += 1
        yield ex["image"], SWIN_CLASSES.index(cls_name)


# Fold assorted dataset folder names (TrashNet, RealWaste, the 10-class set) into
# the Swin's ten classes so several sources can be combined into one calibration.
CLASS_ALIASES = {
    "cardboard": "cardboard", "glass": "glass", "metal": "metal",
    "paper": "paper", "plastic": "plastic", "trash": "trash",
    "miscellaneous trash": "trash", "textile trash": "clothes",
    "food organics": "biological", "vegetation": "biological",
    "biological": "biological", "clothes": "clothes",
    "battery": "battery", "shoes": "shoes",
}


def collect_from_dir(root: str, per_class: int):
    root = Path(root)
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = CLASS_ALIASES.get(cls_dir.name.strip().lower())
        if cls is None:
            continue  # a folder whose name we can't map to a Swin class
        idx = SWIN_CLASSES.index(cls)
        imgs = [p for p in cls_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        for p in imgs[:per_class]:
            yield Image.open(p), idx


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", default="garythung/trashnet",
                    help="HuggingFace image-classification dataset id")
    ap.add_argument("--dir", action="append", default=None,
                    help="local folder of class subdirectories (repeatable to combine sources)")
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--out", default=str(HERE / "ood_guard_calib.npz"))
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device or (
        "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"  device: {device}")

    net = SwinB()
    net.load_state_dict(torch.load(TRANSFORMER_CKPT, map_location="cpu", weights_only=True))
    net = net.to(device).eval()
    swin = net.swin

    if args.dir:
        import itertools
        source = itertools.chain.from_iterable(
            collect_from_dir(d, args.per_class) for d in args.dir)
    else:
        source = collect_from_hf(args.hf, args.per_class)

    feats_all, labels_all = [], []
    batch_imgs, batch_lbls = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        _, feats = swin_forward(swin, x)
        feats_all.append(feats.cpu().numpy())
        labels_all.extend(batch_lbls)
        batch_imgs.clear()
        batch_lbls.clear()

    n = 0
    for img, lbl in source:
        try:
            batch_imgs.append(EVAL_TF(img.convert("RGB")))
        except Exception as e:
            print(f"    skip: {e}")
            continue
        batch_lbls.append(lbl)
        n += 1
        if len(batch_imgs) >= args.batch:
            flush()
            print(f"    {n} images…", end="\r")
    flush()

    features = np.vstack(feats_all).astype(np.float32)
    labels = np.array(labels_all, dtype=np.int64)
    present = sorted({int(x) for x in np.unique(labels)})
    print(f"\n  extracted {features.shape[0]} x {features.shape[1]} features")
    print(f"  classes present: {[SWIN_CLASSES[i] for i in present]}")

    np.savez(args.out, features=features, labels=labels)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
