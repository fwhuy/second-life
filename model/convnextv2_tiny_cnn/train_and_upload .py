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
SEED = 42
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Official pretrained checkpoint: 28.6M parameters, under the 30M cap.
MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
MAX_PARAMS = 30_000_000
IMG_SIZE = 384

# Conservative default; local GPU VRAM detection below adjusts smaller cards.
BATCH_SIZE = 16
ACCUM_STEPS = 4
NUM_WORKERS = min(8, max(2, (os.cpu_count() or 4) // 2))

# Two classifier-only epochs plus 58 full fine-tuning epochs = 60 maximum.
EPOCHS_HEAD = 2
EPOCHS_FT = 58
LR_HEAD_WARMUP = 1e-3
LR_HEAD_FT = 2e-4
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 0.05

LABEL_SMOOTHING = 0.05
MIXUP = 0.2
CUTMIX = 0.5
MIX_PROB = 0.8
NO_MIX_LAST_EPOCHS = 3
EMA_DECAY = 0.999


TIME_BUDGET_H = 10.0

IS_KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working") if IS_KAGGLE else Path(".")
TEMP = (
    Path("/kaggle/temp")
    if Path("/kaggle/temp").exists()
    else Path(tempfile.gettempdir())
)
CACHE_DIR = TEMP / "waste_training_cache"
CKPT = OUT / "best_convnextv2.pt"
META = OUT / "best_convnextv2_metadata.json"

OMASTEAM_REPO = "omasteam/waste-garbage-management-dataset"
REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    """Normalize, perceptually deduplicate and cache one labelled image."""
    mapped = MAPPING.get(label_name.strip().lower())
    if mapped is None:
        stats["unmapped"] += 1
        return

    try:
        img = ImageOps.exif_transpose(img).convert("RGB")

        # Very large source images waste RAM and I/O. 768 px still leaves ample
        # detail for a 384 px random crop while keeping the local cache compact.
        img.thumbnail((768, 768), Image.Resampling.LANCZOS)

        # Combining two perceptual hashes catches resized/re-encoded copies while
        # making accidental collisions extremely unlikely.
        digest = (
            f"{imagehash.phash(img, hash_size=16)}:"
            f"{imagehash.dhash(img, hash_size=16)}"
        )
        label = CLASSES.index(mapped)

        if digest in seen:
            stats["duplicates"] += 1
            if seen[digest] != label:
                stats["label_conflicts"] += 1
            return
        seen[digest] = label

        path = CACHE_DIR / f"img_{len(records):06d}.jpg"
        img.save(path, format="JPEG", quality=93, optimize=False)
        records.append(
            {
                "path": str(path),
                "label": label,
                "group": digest,
                "source": source,
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

    counts = Counter(CLASSES[r["label"]] for r in records)
    missing = [name for name in CLASSES if counts[name] == 0]
    if missing:
        raise RuntimeError(f"Corpus is missing required classes: {missing}")

    print(f"\ncorpus: {len(records)} perceptually unique images")
    for name in CLASSES:
        print(f"  {name:<11} {counts[name]:5d}")
    return records


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


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_tf = T.Compose(
    [
        T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.20, scale=(0.02, 0.15)),
    ]
)

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
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]

    for parameter in backbone:
        parameter.requires_grad = phase == "finetune"

    if phase == "head":
        groups = [{"params": head, "lr": LR_HEAD_WARMUP}]
    else:
        groups = [
            {"params": head, "lr": LR_HEAD_FT},
            {"params": backbone, "lr": LR_BACKBONE},
        ]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
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

    labels = np.array([record["label"] for record in records])
    groups = np.array([record["group"] for record in records])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, val_indices = next(
        splitter.split(np.zeros(len(records)), labels, groups)
    )

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]
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
    soft_criterion = SoftTargetCrossEntropy()
    hard_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

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
        requested_phase = "head" if epoch < EPOCHS_HEAD else "finetune"
        if requested_phase != phase:
            phase = requested_phase
            optimizer = make_optimizer(model, phase)
            phase_epochs = EPOCHS_HEAD if phase == "head" else EPOCHS_FT
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=phase_epochs, eta_min=1e-6
            )
            if phase == "finetune" and ema is None:
                # Initializing here avoids averaging in the random initial head.
                ema = timm.utils.ModelEmaV2(model, decay=EMA_DECAY)
            print(f"\n--- phase={phase} ---")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen_examples = 0
        use_mix = epoch < total_epochs - NO_MIX_LAST_EPOCHS

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            if use_mix:
                images, training_targets = mixup_fn(images, targets)
                criterion = soft_criterion
            else:
                training_targets = targets
                criterion = hard_criterion

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=AMP_ENABLED,
            ):
                loss = criterion(model(images), training_targets)
                scaled_loss = loss / ACCUM_STEPS

            scaler.scale(scaled_loss).backward()
            should_step = (
                (step + 1) % ACCUM_STEPS == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            running_loss += loss.item() * targets.size(0)
            seen_examples += targets.size(0)

        scheduler.step()
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

    if args.upload_only:
        upload_results(configure_hf_upload(args))
        return

    confirm_hardware()
    upload_config = configure_hf_upload(args)
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Preserving and uploading the best checkpoint...")
    finally:
        upload_results(upload_config)


if __name__ == "__main__":
    main()
