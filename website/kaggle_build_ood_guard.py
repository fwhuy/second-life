"""Kaggle calibration for the Second Life OOD guard — run in a Kaggle GPU notebook.

Your local wifi is slow, so we keep the big datasets on Kaggle and only bring back the
tiny output. Paste this whole file into ONE Kaggle notebook cell and run it.

ADD THESE NOTEBOOK INPUTS (right sidebar → "Add Input"):
  1. Your Swin-B checkpoint `best_swin_b.pt` — as a private Kaggle dataset (one upload),
     OR set SWIN_CKPT below to wherever it lives under /kaggle/input.
  2. One or more waste datasets that have class subfolders, e.g.:
       - RealWaste       (search Kaggle: "realwaste")   → messy, real-facility photos
       - TrashNet        (search Kaggle: "trashnet")    → clean single items
     Add both — the more varied the "waste" images, the better the guard tells a real
     bin bag (waste) apart from a cat (not waste).

The script walks /kaggle/input, finds every folder whose name maps to one of the Swin's
ten classes, extracts the Swin's 1024-d features, and writes:
       /kaggle/working/ood_guard_calib.npz
Download that file (a few MB) and hand it back — it drops straight into website/.
"""

import glob
import os
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tvm, transforms as T

ssl._create_default_https_context = ssl._create_unverified_context  # UCI/HF cert wall

# ── config ───────────────────────────────────────────────────────────────────
SWIN_CKPT = None          # None → auto-find best_swin_b.pt under /kaggle/input
PER_CLASS = 300           # cap images per class (across all datasets)
BATCH = 32
OUT = "/kaggle/working/ood_guard_calib.npz"

# If no waste folders are attached under /kaggle/input, download these automatically
# (needs the notebook's Internet toggle ON: Settings → Internet → On).
WORK_WASTE = "/kaggle/working/_waste"
DATASET_URLS = {
    "realwaste.zip": "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
    "trashnet.zip": "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
}
SCAN_ROOTS = ["/kaggle/input", WORK_WASTE]

SWIN_CLASSES = ["battery", "biological", "cardboard", "clothes", "glass",
                "metal", "paper", "plastic", "shoes", "trash"]
# Fold assorted dataset folder names into the Swin's ten classes.
CLASS_ALIASES = {
    "cardboard": "cardboard", "glass": "glass", "metal": "metal",
    "paper": "paper", "plastic": "plastic", "trash": "trash",
    "miscellaneous trash": "trash", "textile trash": "clothes",
    "food organics": "biological", "vegetation": "biological",
    "biological": "biological", "clothes": "clothes",
    "battery": "battery", "shoes": "shoes",
}
EVAL_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ── the deployed Swin-B architecture (verbatim, so the checkpoint keys line up) ─
class SwinB(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.swin = tvm.swin_b(weights=None)
        self.swin.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(self.swin.head.in_features, num_classes))


def swin_features(swin, x):
    f = swin.features(x)
    f = swin.norm(f)
    f = f.permute(0, 3, 1, 2)
    f = swin.avgpool(f)
    return torch.flatten(f, 1)


def find_ckpt():
    if SWIN_CKPT and os.path.exists(SWIN_CKPT):
        return SWIN_CKPT
    hits = glob.glob("/kaggle/input/**/best_swin_b.pt", recursive=True)
    if not hits:
        raise SystemExit("No best_swin_b.pt found under /kaggle/input — add it as an input "
                         "dataset, or set SWIN_CKPT at the top of this file.")
    return hits[0]


def discover_class_dirs():
    """Map every subfolder (under /kaggle/input or the downloaded set) whose name is a
    known class → its image paths."""
    by_class = {}  # class_idx -> [paths]
    for root in SCAN_ROOTS:
        for d in glob.glob(os.path.join(root, "**", ""), recursive=True):
            cls = CLASS_ALIASES.get(Path(d.rstrip("/")).name.strip().lower())
            if cls is None:
                continue
            imgs = [p for p in glob.glob(os.path.join(d, "*"))
                    if p.lower().endswith(IMG_EXT)]
            if imgs:
                by_class.setdefault(SWIN_CLASSES.index(cls), []).extend(imgs)
    return by_class


def download_datasets():
    """Fetch RealWaste + TrashNet zips and extract them under WORK_WASTE (Kaggle has
    fast internet, so this is quick). Needs the notebook Internet toggle ON."""
    os.makedirs(WORK_WASTE, exist_ok=True)
    for fname, url in DATASET_URLS.items():
        zpath = os.path.join(WORK_WASTE, fname)
        try:
            if not os.path.exists(zpath):
                print(f"  downloading {fname} …")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(zpath, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            print(f"  extracting {fname} …")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(os.path.join(WORK_WASTE, fname[:-4]))
        except Exception as e:
            print(f"  ! {fname} failed: {e}  (is the notebook's Internet toggle ON?)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_ckpt()
    print(f"device={device}  checkpoint={ckpt}")

    net = SwinB()
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net = net.to(device).eval()
    swin = net.swin

    by_class = discover_class_dirs()
    if not by_class:
        print("No waste folders attached under /kaggle/input — downloading datasets…")
        download_datasets()
        by_class = discover_class_dirs()
    if not by_class:
        raise SystemExit("Still no class folders. Turn the notebook's Internet toggle ON "
                         "(Settings → Internet → On) and re-run, or attach RealWaste/TrashNet manually.")

    # Balance: up to PER_CLASS per class, shuffled.
    rng = np.random.default_rng(42)
    paths, labels = [], []
    for idx, ps in sorted(by_class.items()):
        ps = list(ps)
        rng.shuffle(ps)
        ps = ps[:PER_CLASS]
        paths += ps
        labels += [idx] * len(ps)
        print(f"  {SWIN_CLASSES[idx]:<11}: {len(ps)} images")
    print(f"  total: {len(paths)} images across {len(by_class)} classes")

    feats_all, labels_all = [], []
    batch_imgs, batch_lbls = [], []

    @torch.no_grad()
    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        feats_all.append(swin_features(swin, x).cpu().numpy())
        labels_all.extend(batch_lbls)
        batch_imgs.clear()
        batch_lbls.clear()

    for i, (p, lbl) in enumerate(zip(paths, labels)):
        try:
            batch_imgs.append(EVAL_TF(Image.open(p).convert("RGB")))
        except Exception as e:
            print("  skip", p, e)
            continue
        batch_lbls.append(lbl)
        if len(batch_imgs) >= BATCH:
            flush()
            if i % (BATCH * 10) == 0:
                print(f"    {i}/{len(paths)}…")
    flush()

    features = np.vstack(feats_all).astype(np.float32)
    labels_arr = np.array(labels_all, dtype=np.int64)
    present = [SWIN_CLASSES[i] for i in sorted(set(labels_arr.tolist()))]
    print(f"\nextracted {features.shape[0]} x {features.shape[1]} features; classes: {present}")

    np.savez(OUT, features=features, labels=labels_arr)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)  — download this and hand it back")


if __name__ == "__main__":
    main()
