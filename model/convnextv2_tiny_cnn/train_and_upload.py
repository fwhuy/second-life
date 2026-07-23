"""
Train a ConvNeXt-V2 Tiny image classifier for 6-class waste sorting, then
upload the best checkpoint to the Hugging Face Hub.

Big picture, top to bottom:
  1. ensure_dependencies() installs any missing pip packages so the file runs
     in a fresh environment (e.g. a clean Kaggle notebook).
  2. Configuration constants define the model, hyperparameters, and file paths.
  3. Data preparation downloads two public waste datasets, normalizes and
    perceptually de-duplsicates the images, and caches them to disk.
  4. Model and training fine-tunes the pretrained backbone in two phases and
     saves the best-performing weights.
  5. Upload pushes the checkpoint + metadata to a Hugging Face model repo.

Typical usage:
  python train_and_upload.py                 # train, then prompt for HF upload
  python train_and_upload.py --no-upload     # train only, no upload
  python train_and_upload.py --upload-only   # upload an existing checkpoint
  python train_and_upload.py --repo-id you/waste-convnextv2 --private

A CUDA GPU is strongly recommended; CPU training would take days.
"""

import argparse
import getpass
import io
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path


def ensure_dependencies():
    """Install missing packages so a clean Python environment can run the file."""
    required = {
        "imagehash": "ImageHash",
        "numpy": "numpy",
        "requests": "requests",
        "timm": "timm",
        "torch": "torch",
        "torchvision": "torchvision",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "datasets": "datasets",
        "huggingface_hub": "huggingface_hub",
    }
    missing = sorted(
        {package for module, package in required.items()
         if importlib.util.find_spec(module) is None}
    )
    if not missing:
        return

    print("Installing missing packages: " + ", ".join(missing), flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--quiet",
            *missing,
        ]
    )


ensure_dependencies()

import imagehash
import numpy as np
import requests
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# A fixed random seed makes the run reproducible: same data split, same weight
# initialization, same augmentation order every time.
SEED = 42

# The six waste categories, in a fixed order. The index of a class in this list
# IS its integer label (cardboard=0, glass=1, ... trash=5).
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Official pretrained checkpoint: 28.6M parameters, under the 30M cap.
# The name encodes: architecture (convnextv2_tiny), pretraining recipe
# (fcmae + fine-tuned on ImageNet-22k then 1k), at 384px input resolution.
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
MAX_PARAMS = 30_000_000  # Project rule: the backbone must stay under 30M params.
IMG_SIZE = 384           # Height/width the model expects, in pixels.

# BATCH_SIZE is how many images go through the GPU at once. ACCUM_STEPS lets us
# simulate a larger batch on limited VRAM: we accumulate gradients over several
# mini-batches before updating the weights (see the training loop). The
# "effective" batch size is BATCH_SIZE * ACCUM_STEPS = 64 here.
# Conservative default; local GPU VRAM detection below adjusts smaller cards.
BATCH_SIZE = 16
ACCUM_STEPS = 4
NUM_WORKERS = min(8, max(2, (os.cpu_count() or 4) // 2))  # parallel data loaders

# Training happens in two phases (see make_optimizer / train_model):
#   Phase 1 "head": freeze the backbone, train only the new classifier head so
#           it stops producing random outputs before we touch the backbone.
#   Phase 2 "finetune": unfreeze everything and fine-tune the whole network.
# Two classifier-only epochs plus 58 full fine-tuning epochs = 60 maximum.
EPOCHS_HEAD = 2
EPOCHS_FT = 58
LR_HEAD_WARMUP = 1e-3  # learning rate for the head during phase 1
LR_HEAD_FT = 2e-4      # head learning rate during phase 2
LR_BACKBONE = 3e-5     # much smaller LR for the pretrained backbone, so we
                       # nudge it rather than destroy what it already learned
WEIGHT_DECAY = 0.05    # L2-style regularization to discourage overfitting

# Regularization / augmentation knobs that make the model generalize better:
LABEL_SMOOTHING = 0.05   # soften one-hot targets so the model is less overconfident
MIXUP = 0.2              # blend two images + their labels together
CUTMIX = 0.5             # paste a patch of one image onto another
MIX_PROB = 0.8           # probability a given batch gets mixup/cutmix applied
NO_MIX_LAST_EPOCHS = 3   # turn mixing off near the end so we train on real images
EMA_DECAY = 0.999        # exponential moving average of weights (see below)


# Hard wall-clock limit. If training reaches this many hours it stops early and
# keeps the best checkpoint found so far (useful on time-limited Kaggle GPUs).
TIME_BUDGET_H = 10.0

# Detect a Kaggle notebook so we can use its special working/temp folders.
IS_KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working") if IS_KAGGLE else Path(".")
TEMP = (
    Path("/kaggle/temp")
    if Path("/kaggle/temp").exists()
    else Path(tempfile.gettempdir())
)
CACHE_DIR = TEMP / "waste_training_cache"
CKPT = OUT / "best_convnextv2.pt"                 # trained weights we upload
META = OUT / "best_convnextv2_metadata.json"      # accuracy/class info alongside them

# The two public datasets we combine for training.
OMASTEAM_REPO = "omasteam/waste-garbage-management-dataset"           # Hugging Face
REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"  # UCI

# Both datasets use their own category names. This maps each source label onto
# one of our six CLASSES; names not listed here are simply dropped (e.g. the
# OMA dataset has battery/clothes/shoes classes we don't train on).
MAPPING = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "aluminium": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "hard plastic": "plastic",
    "soft plastics": "plastic",
    "trash": "trash",
    "miscellaneous trash": "trash",
}

# Seed every random number generator we use so results are reproducible.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True  # let cuDNN auto-tune the fastest kernels

# Train on the GPU if one is available, otherwise fall back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# AMP = automatic mixed precision: run parts of the math in 16-bit for a big
# speed/memory win on GPUs. Only safe/beneficial on CUDA, so gate it on that.
AMP_ENABLED = DEVICE.type == "cuda"

# Adapt to smaller local GPUs while preserving an effective batch size of 64.
if AMP_ENABLED and not IS_KAGGLE:
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb < 7.5:
        BATCH_SIZE, ACCUM_STEPS = 4, 16
    elif gpu_memory_gb < 15.5:
        BATCH_SIZE, ACCUM_STEPS = 8, 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-V2 Tiny and upload its best checkpoint."
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="upload an existing best_convnextv2.pt without training",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="train locally without configuring a Hugging Face upload",
    )
    parser.add_argument(
        "--repo-id",
        help="Hugging Face repository, for example alice/waste-convnextv2",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="create a private repository (public is the default)",
    )
    return parser.parse_args()


def confirm_hardware():
    """Make an accidental multi-day CPU training run difficult."""
    if DEVICE.type == "cuda":
        name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"GPU: {name} ({memory_gb:.1f} GB) | mini-batch={BATCH_SIZE} "
            f"accumulation={ACCUM_STEPS} | effective batch="
            f"{BATCH_SIZE * ACCUM_STEPS}",
            flush=True,
        )
        return

    warning = (
        "CUDA was not detected. Training on the CPU could take days. "
        "Install an NVIDIA-enabled PyTorch build or use Kaggle."
    )
    if IS_KAGGLE:
        raise RuntimeError(warning + " Enable the notebook GPU accelerator.")
    print("\nWARNING: " + warning)
    answer = input("Continue on CPU anyway? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Stopped before training; no files were changed.")


def configure_hf_upload(args):
    """Create/access a Hub model repository without persisting the token."""
    if IS_KAGGLE or args.no_upload:
        return None

    from huggingface_hub import HfApi

    print(
        "\nAutomatic upload setup\n"
        "Create a WRITE token at https://huggingface.co/settings/tokens\n"
        "The token input is hidden and stays in this Python process only."
    )
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token = getpass.getpass("Hugging Face WRITE token: ").strip()
    if not token:
        raise SystemExit("No token supplied; rerun with --no-upload if desired.")

    api = HfApi(token=token)
    try:
        identity = api.whoami()
    except Exception as exc:
        raise SystemExit(
            "Hugging Face rejected the token or could not be reached: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    username = identity.get("name") or identity.get("fullname")
    if not username:
        raise SystemExit("Could not determine the Hugging Face account name.")

    default_repo = f"{username}/waste-convnextv2"
    repo_id = args.repo_id or os.environ.get("HF_REPO_ID", "").strip()
    if not repo_id:
        entered = input(f"Model repository [{default_repo}]: ").strip()
        repo_id = entered or default_repo
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    private = args.private
    if not args.private and "HF_REPO_PRIVATE" not in os.environ:
        choice = input(
            "Make the result PUBLIC so it can be opened without an account? "
            "[Y/n]: "
        ).strip().lower()
        private = choice in {"n", "no"}
    elif os.environ.get("HF_REPO_PRIVATE", "").lower() in {"1", "true", "yes"}:
        private = True

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
    except Exception as exc:
        raise SystemExit(
            "Could not create/access the repository. Confirm that the token has "
            f"write permission: {type(exc).__name__}: {exc}"
        ) from exc

    visibility = "private" if private else "public"
    print(f"Upload destination ({visibility}): https://huggingface.co/{repo_id}")
    return api, repo_id


def upload_results(upload_config):
    """Upload the checkpoint and metadata, with bounded retry backoff."""
    if upload_config is None:
        return False
    if not CKPT.is_file() or CKPT.stat().st_size < 1_000_000:
        print(f"\nNo usable checkpoint found at {CKPT.resolve()}; nothing uploaded.")
        return False

    api, repo_id = upload_config
    upload_paths = [CKPT]
    if META.is_file():
        upload_paths.append(META)

    for path in upload_paths:
        uploaded = False
        for attempt in range(1, 7):
            try:
                print(f"Uploading {path.name} (attempt {attempt}/6)...", flush=True)
                api.upload_file(
                    path_or_fileobj=str(path.resolve()),
                    path_in_repo=path.name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {path.name} from training run",
                )
                uploaded = True
                break
            except Exception as exc:
                print(
                    f"  upload failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt < 6:
                    time.sleep(min(60, 2 ** attempt))
        if not uploaded:
            print(
                f"Could not upload {path.name}. It is safe locally at "
                f"{path.resolve()}. Rerun with --upload-only later."
            )
            return False

    print("\nUPLOAD COMPLETE")
    print(f"Model page: https://huggingface.co/{repo_id}")
    print(
        "Direct checkpoint: "
        f"https://huggingface.co/{repo_id}/resolve/main/{CKPT.name}"
    )
    return True


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def open_hf_image(payload):
    """Open an image returned by datasets.Image(decode=False)."""
    if payload.get("bytes") is not None:
        return Image.open(io.BytesIO(payload["bytes"]))
    if payload.get("path"):
        return Image.open(payload["path"])
    raise ValueError("Hugging Face image has neither bytes nor a path")


def add_image(records, seen, img, label_name, source, stats):
    """Normalize, perceptually deduplicate and cache one labelled image.

    `records` is the growing list of usable images; `seen` maps a perceptual
    fingerprint to the label we already stored for it, so we can skip duplicates.
    `stats` is a Counter that tallies added/duplicate/unmapped/bad images.
    """
    # Translate the source's class name to one of our six; skip if unknown.
    mapped = MAPPING.get(label_name.strip().lower())
    if mapped is None:
        stats["unmapped"] += 1
        return

    try:
        # Respect the photo's EXIF orientation, then force 3-channel RGB.
        img = ImageOps.exif_transpose(img).convert("RGB")

        # Very large source images waste RAM and I/O. 768 px still leaves ample
        # detail for a 384 px random crop while keeping the local cache compact.
        img.thumbnail((768, 768), Image.Resampling.LANCZOS)

        # A "perceptual hash" is a short fingerprint of what an image LOOKS like,
        # so near-identical photos (resized, re-saved, lightly edited) get the
        # same digest even though their bytes differ. Combining two hash types
        # (phash + dhash) makes an accidental collision extremely unlikely.
        # We use this to drop duplicates AND to keep copies of one image from
        # landing on both sides of the train/validation split (data leakage).
        digest = (
            f"{imagehash.phash(img, hash_size=16)}:"
            f"{imagehash.dhash(img, hash_size=16)}"
        )
        label = CLASSES.index(mapped)

        # Already seen this image? Skip it, and note if two sources disagree on
        # its label (a sign of noisy data).
        if digest in seen:
            stats["duplicates"] += 1
            if seen[digest] != label:
                stats["label_conflicts"] += 1
            return
        seen[digest] = label

        # Save the cleaned image to the on-disk cache and remember where it is.
        path = CACHE_DIR / f"img_{len(records):06d}.jpg"
        img.save(path, format="JPEG", quality=93, optimize=False)
        records.append(
            {
                "path": str(path),    # where the cached JPEG lives
                "label": label,        # integer class 0-5
                "group": digest,       # perceptual fingerprint, used to keep
                                       # look-alikes together in one split
                "source": source,      # which dataset it came from
            }
        )
        stats["added"] += 1
    except Exception as exc:
        stats["bad_images"] += 1
        if stats["bad_images"] <= 5:
            print(f"  ignored unreadable image: {type(exc).__name__}: {exc}")


def load_omasteam(records, seen):
    """Load the six relevant classes from the labelled OMA dataset."""
    from datasets import Image as HFImage
    from datasets import load_dataset

    stats = Counter()
    print(f"loading {OMASTEAM_REPO} ...", flush=True)
    ds = load_dataset(OMASTEAM_REPO, split="train")

    if "image" not in ds.features or "label" not in ds.features:
        raise RuntimeError(
            f"{OMASTEAM_REPO} schema changed; found columns {ds.column_names}"
        )
    names = ds.features["label"].names

    # Avoid decoding images belonging to battery/clothes/etc. unnecessarily.
    ds = ds.cast_column("image", HFImage(decode=False))
    for i, row in enumerate(ds):
        label_name = names[int(row["label"])]
        if MAPPING.get(label_name.strip().lower()) is None:
            stats["unmapped"] += 1
            continue
        try:
            with open_hf_image(row["image"]) as img:
                add_image(records, seen, img, label_name, "omasteam", stats)
        except Exception as exc:
            stats["bad_images"] += 1
            if stats["bad_images"] <= 5:
                print(f"  ignored row {i}: {type(exc).__name__}: {exc}")

        if (i + 1) % 3000 == 0:
            print(f"  processed {i + 1}/{len(ds)} rows", flush=True)

    print(
        f"  OMA: added {stats['added']}, duplicates {stats['duplicates']}, "
        f"unmapped {stats['unmapped']}, bad {stats['bad_images']}"
    )


def download_file(url, destination):
    """Stream a URL to disk without retaining the whole archive in RAM."""
    if destination.exists() and destination.stat().st_size > 600_000_000:
        print(f"using cached {destination.name}")
        return

    print(f"downloading official RealWaste archive to {destination} ...", flush=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        last_report = 0
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 100 * 1024 * 1024:
                    if total:
                        print(f"  {downloaded / total * 100:5.1f}%", flush=True)
                    else:
                        print(f"  {downloaded / 1e6:.0f} MB", flush=True)
                    last_report = downloaded
    os.replace(partial, destination)


def load_realwaste(records, seen):
    """Load RealWaste from the official UCI archive, whose folders are labels."""
    stats = Counter()
    archive = TEMP / "realwaste.zip"
    extracted = TEMP / "realwaste_uci"

    print("loading official UCI RealWaste ...", flush=True)
    download_file(REALWASTE_URL, archive)
    extracted.mkdir(parents=True, exist_ok=True)

    roots = [
        p for p in extracted.rglob("RealWaste")
        if p.is_dir() and (p / "Cardboard").is_dir()
    ]
    if not roots:
        print("  extracting RealWaste ...", flush=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        roots = [
            p for p in extracted.rglob("RealWaste")
            if p.is_dir() and (p / "Cardboard").is_dir()
        ]
    if not roots:
        raise RuntimeError("Could not locate the labelled RealWaste folders")

    root = roots[0]
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label_name = class_dir.name
        if MAPPING.get(label_name.strip().lower()) is None:
            stats["unmapped"] += sum(
                p.suffix.lower() in extensions for p in class_dir.rglob("*")
            )
            continue

        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in extensions:
                continue
            try:
                with Image.open(path) as img:
                    add_image(records, seen, img, label_name, "realwaste", stats)
            except Exception as exc:
                stats["bad_images"] += 1
                if stats["bad_images"] <= 5:
                    print(f"  ignored {path.name}: {type(exc).__name__}: {exc}")

    print(
        f"  RealWaste: added {stats['added']}, duplicates {stats['duplicates']}, "
        f"unmapped {stats['unmapped']}, bad {stats['bad_images']}"
    )


def load_corpus():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    seen = {}

    # OMA is required; it supplies the largest labelled portion of the corpus.
    load_omasteam(records, seen)

    # RealWaste materially improves real-world backgrounds. If UCI is briefly
    # unavailable, continue with OMA rather than throwing away the whole run.
    try:
        load_realwaste(records, seen)
    except Exception as exc:
        print(f"  SKIPPED RealWaste: {type(exc).__name__}: {exc}")

    # Sanity check: every class must have at least one image, or training the
    # 6-way classifier makes no sense.
    counts = Counter(CLASSES[r["label"]] for r in records)
    missing = [name for name in CLASSES if counts[name] == 0]
    if missing:
        raise RuntimeError(f"Corpus is missing required classes: {missing}")

    print(f"\ncorpus: {len(records)} perceptually unique images")
    for name in CLASSES:
        print(f"  {name:<11} {counts[name]:5d}")
    return records


# A PyTorch Dataset: given an index, load that cached image from disk, apply the
# transform pipeline, and return (image_tensor, label). The DataLoader uses this
# to feed batches to the model.
class WasteDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["path"]) as img:
            image = self.transform(img.convert("RGB"))
        return image, record["label"]


# ImageNet channel statistics. The backbone was pretrained on images normalized
# with these values, so we must normalize ours the same way.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Training transforms deliberately add randomness so the model sees a slightly
# different picture each epoch and learns to generalize instead of memorize.
train_tf = T.Compose(
    [
        T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),  # random zoom/crop
        T.RandomHorizontalFlip(),                          # random left-right flip
        T.TrivialAugmentWide(),                            # random color/geometry ops
        T.ToTensor(),                                      # PIL image -> tensor [0,1]
        T.Normalize(MEAN, STD),                            # standardize channels
        T.RandomErasing(p=0.20, scale=(0.02, 0.15)),       # randomly blank a patch
    ]
)

# Evaluation transforms are deterministic: resize a bit larger, center-crop to
# the target size, and normalize. No randomness, so validation is consistent.
eval_tf = T.Compose(
    [
        T.Resize(int(IMG_SIZE * 1.14)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]
)


# -----------------------------------------------------------------------------
# Model and training
# -----------------------------------------------------------------------------
def build_model():
    # Load the pretrained ConvNeXt-V2 Tiny, but replace its final layer with a
    # fresh 6-output classifier (num_classes) matching our waste categories.
    model = timm.create_model(MODEL, pretrained=True, num_classes=len(CLASSES))
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count > MAX_PARAMS:
        raise RuntimeError(
            f"{MODEL} has {parameter_count / 1e6:.2f}M parameters, over the cap"
        )
    print(
        f"model={MODEL} parameters={parameter_count / 1e6:.2f}M "
        f"(cap={MAX_PARAMS / 1e6:.0f}M)"
    )
    return model.to(DEVICE)


def make_optimizer(model, phase):
    """Build an AdamW optimizer for the current training phase.

    Split the parameters into the classifier "head" and the pretrained
    "backbone" so we can treat them differently.
    """
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]

    # In phase 1 the backbone is frozen (requires_grad=False) so only the head
    # learns; in phase 2 the whole network is trainable.
    for parameter in backbone:
        parameter.requires_grad = phase == "finetune"

    if phase == "head":
        # Only optimize the head, at a relatively high warmup learning rate.
        groups = [{"params": head, "lr": LR_HEAD_WARMUP}]
    else:
        # Optimize both, but move the delicate backbone much more slowly than
        # the head (discriminative / layer-wise learning rates).
        groups = [
            {"params": head, "lr": LR_HEAD_FT},
            {"params": backbone, "lr": LR_BACKBONE},
        ]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


@torch.no_grad()  # disable gradient tracking: we're only measuring, not learning
def evaluate(model, loader):
    """Run the model over a data loader and return (overall_accuracy, per_class).

    per_class is a [num_classes, 2] tensor holding [correct, total] counts for
    each class, so we can print per-class accuracy at the end.
    """
    model.eval()  # switch off dropout / use running batch-norm stats
    correct = 0
    total = 0
    per_class = torch.zeros(len(CLASSES), 2, dtype=torch.long)

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):
            predictions = model(images).argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.numel()
        for class_index in range(len(CLASSES)):
            mask = targets == class_index
            per_class[class_index, 0] += (predictions[mask] == class_index).sum().cpu()
            per_class[class_index, 1] += mask.sum().cpu()

    return correct / max(total, 1), per_class


def save_best(model, accuracy, epoch):
    """Write the model weights to CKPT and a small JSON summary to META."""
    # Copy every weight tensor to CPU before saving so the file loads anywhere,
    # not just on a machine with the same GPU.
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(state, CKPT)
    with open(META, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": MODEL,
                "classes": CLASSES,
                "image_size": IMG_SIZE,
                "validation_accuracy": accuracy,
                "epoch": epoch,
                "parameter_count": sum(p.numel() for p in model.parameters()),
            },
            handle,
            indent=2,
        )


def train_model():
    started = time.time()
    records = load_corpus()

    # Split into training and validation sets. StratifiedGroupKFold does two
    # important things at once:
    #   - "Stratified": keeps each class's proportion roughly equal in both sets.
    #   - "Group": never lets images sharing a `group` (perceptual duplicates)
    #     land on both sides. Otherwise the model could "cheat" by memorizing a
    #     picture in training and seeing a near-copy in validation (data leakage,
    #     which inflates the score). We take the first of the 5 folds as our
    #     single train/val split.
    labels = np.array([record["label"] for record in records])
    groups = np.array([record["group"] for record in records])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, val_indices = next(
        splitter.split(np.zeros(len(records)), labels, groups)
    )

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]
    # Belt-and-suspenders: prove no group ended up in both sets.
    assert not (
        set(groups[train_indices]) & set(groups[val_indices])
    ), "duplicate group leaked across the split"
    print(
        f"train={len(train_records)} val={len(val_records)} "
        f"groups-disjoint=OK device={DEVICE}"
    )

    loader_options = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(
        WasteDataset(train_records, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        WasteDataset(val_records, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_options,
    )

    model = build_model()

    # Mixup/CutMix blend images and their labels, so the loss must compare
    # against "soft" (non-integer) targets. When mixing is off we fall back to
    # ordinary cross-entropy on the real integer labels.
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy

    mixup_fn = Mixup(
        mixup_alpha=MIXUP,
        cutmix_alpha=CUTMIX,
        prob=MIX_PROB,
        switch_prob=0.5,
        label_smoothing=LABEL_SMOOTHING,
        num_classes=len(CLASSES),
    )
    soft_criterion = SoftTargetCrossEntropy()  # loss when mixup/cutmix is active
    hard_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)  # otherwise

    # The gradient scaler is part of mixed-precision training: it scales the loss
    # up before backprop so tiny 16-bit gradients don't underflow to zero, then
    # unscales before the optimizer step. It's a no-op when AMP is disabled.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    except TypeError:  # Compatibility with older Kaggle PyTorch images.
        scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    total_epochs = EPOCHS_HEAD + EPOCHS_FT
    best_accuracy = 0.0
    best_per_class = None
    phase = None
    optimizer = None
    scheduler = None
    ema = None

    for epoch in range(total_epochs):
        # Decide which phase this epoch belongs to. On the transition from head
        # to finetune we rebuild the optimizer and learning-rate schedule.
        requested_phase = "head" if epoch < EPOCHS_HEAD else "finetune"
        if requested_phase != phase:
            phase = requested_phase
            optimizer = make_optimizer(model, phase)
            phase_epochs = EPOCHS_HEAD if phase == "head" else EPOCHS_FT
            # Cosine schedule smoothly decays the learning rate to near zero
            # across the phase, which usually gives a better final model.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=phase_epochs, eta_min=1e-6
            )
            if phase == "finetune" and ema is None:
                # EMA keeps a slowly-updated running average of the weights.
                # It's typically more accurate and less noisy than the raw model,
                # so we evaluate and save the EMA copy. We start it here so it
                # never averages in the random, untrained initial head.
                ema = timm.utils.ModelEmaV2(model, decay=EMA_DECAY)
            print(f"\n--- phase={phase} ---")

        model.train()  # enable dropout / batch-norm updates
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen_examples = 0
        # Turn augmentation mixing off for the final few epochs so the model
        # finishes by learning on clean, un-blended images.
        use_mix = epoch < total_epochs - NO_MIX_LAST_EPOCHS

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # Optionally blend the batch (mixup/cutmix) and pick the matching loss.
            if use_mix:
                images, training_targets = mixup_fn(images, targets)
                criterion = soft_criterion
            else:
                training_targets = targets
                criterion = hard_criterion

            # Forward pass under autocast (mixed precision). Divide the loss by
            # ACCUM_STEPS because we sum gradients over that many mini-batches
            # before a single optimizer step (gradient accumulation).
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=AMP_ENABLED,
            ):
                loss = criterion(model(images), training_targets)
                scaled_loss = loss / ACCUM_STEPS

            # Backward pass accumulates gradients (scaled for mixed precision).
            scaler.scale(scaled_loss).backward()

            # Only update the weights once we've accumulated ACCUM_STEPS batches
            # (or reached the last batch of the epoch).
            should_step = (
                (step + 1) % ACCUM_STEPS == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)  # undo loss scaling before clipping
                # Clip gradients to a max norm to keep a rare huge update from
                # destabilizing training.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)   # apply the update
                scaler.update()          # adjust the scale factor for next time
                optimizer.zero_grad(set_to_none=True)  # reset accumulated grads
                if ema is not None:
                    ema.update(model)    # fold the new weights into the EMA copy

            # Track loss weighted by batch size for an accurate epoch average.
            running_loss += loss.item() * targets.size(0)
            seen_examples += targets.size(0)

        scheduler.step()  # advance the learning-rate schedule once per epoch
        # Evaluate the EMA weights once we're in the finetune phase, else the
        # live model. Validation accuracy is what we use to pick the best epoch.
        eval_model = ema.module if ema is not None else model
        accuracy, per_class = evaluate(eval_model, val_loader)
        elapsed_h = (time.time() - started) / 3600

        print(
            f"epoch {epoch + 1:02d}/{total_epochs} phase={phase:<8} "
            f"loss={running_loss / max(seen_examples, 1):.4f} "
            f"val_acc={accuracy:.4f} best={max(best_accuracy, accuracy):.4f} "
            f"elapsed={elapsed_h:.2f}h",
            flush=True,
        )

        # Keep only the best checkpoint seen so far (early-stopping by saving).
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_per_class = per_class.clone()
            save_best(eval_model, accuracy, epoch + 1)
            print(f"  saved {CKPT} @ {accuracy:.4f}")

        if elapsed_h >= TIME_BUDGET_H:
            print(
                f"time budget of {TIME_BUDGET_H:.1f}h reached; "
                "stopping with the best checkpoint"
            )
            break

    print(f"\nbest validation accuracy={best_accuracy:.4f}")
    print(f"weights: {CKPT}")
    print(f"metadata: {META}")
    if best_per_class is not None:
        for class_index, name in enumerate(CLASSES):
            correct, total = best_per_class[class_index].tolist()
            accuracy = 100.0 * correct / max(total, 1)
            print(f"  {name:<11} {accuracy:5.1f}% ({correct}/{total})")


def main():
    args = parse_args()
    if args.upload_only and args.no_upload:
        raise SystemExit("Choose either --upload-only or --no-upload, not both.")

    # Shortcut: skip training and just upload an existing checkpoint.
    if args.upload_only:
        upload_results(configure_hf_upload(args))
        return

    confirm_hardware()  # refuse/confirm slow CPU runs before doing real work
    # Set up the Hugging Face upload BEFORE training so a bad token fails fast,
    # not after hours of GPU time.
    upload_config = configure_hf_upload(args)
    try:
        train_model()
    except KeyboardInterrupt:
        # Ctrl-C still uploads whatever best checkpoint we already saved.
        print("\nTraining interrupted. Preserving and uploading the best checkpoint...")
    finally:
        # `finally` guarantees we attempt the upload whether training finished,
        # was interrupted, or errored out.
        upload_results(upload_config)


if __name__ == "__main__":
    main()
