"""
evaluate.py — deterministic, standalone evaluation for the two waste models.

This is the separate *testing* entry point (training lives in
``train_convnextv2.py`` / ``train_swin_b.py``). Given a checkpoint and a labelled
image set it computes accuracy, macro-F1, per-class precision/recall/F1 and mean
loss, writes a per-image prediction table, renders a labelled confusion matrix,
and exports the worst mistakes for error analysis. It is self-contained: it does
not import the training scripts, so it stays runnable even if those change.

Two models are supported through one interface:

  * ``convnet``      — ConvNeXt V2-Tiny, 6 classes, 384px centre-crop eval.
  * ``transformer``  — Swin-B, trained on 10 classes at 224px. Use
                       ``--classes six`` (default) to mask its logits to the six
                       shared classes and compare it on the same question as the
                       ConvNet, or ``--classes native`` to score all ten.

Determinism: fixed seed, deterministic algorithms, evaluation transforms only,
no shuffling — the same inputs always produce the same numbers.

Usage
-----
    # ConvNeXt on a folder of class sub-directories (cardboard/, glass/, ...)
    python evaluate.py --model convnet \
        --data-root /path/to/test_images --out-dir out/convnet_test

    # Swin-B, six shared classes, from an explicit manifest (path,label per row)
    python evaluate.py --model transformer --classes six \
        --split-manifest test.csv --out-dir out/swin_test

    # Force CPU and cap the number of images (handy for a quick smoke check)
    python evaluate.py --model convnet --data-root imgs \
        --out-dir out --device cpu --limit 32 --no-plots

Exit status is non-zero with a clear message on missing checkpoints,
architecture / class-order mismatches, empty splits, or an unreadable manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T


# -----------------------------------------------------------------------------
# Fixed vocabulary and constants
# -----------------------------------------------------------------------------
# The six shared classes, in the ConvNeXt training order (index == label).
CLASSES_6 = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# Swin-B's ten classes, in the sorted order its training script produced
# (``sorted(os.listdir(DATA_ROOT))`` over the standardized_256 folders).
CLASSES_10 = ["battery", "biological", "cardboard", "clothes", "glass",
              "metal", "paper", "plastic", "shoes", "trash"]
# Column indices of the six shared classes inside the ten-class logit vector.
SHARED_IN_10 = [CLASSES_10.index(c) for c in CLASSES_6]  # -> [2, 4, 5, 6, 7, 9]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Repository layout: this file lives at submission/code/evaluate.py, so the
# checkpoints sit alongside it under submission/results/.
HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
RESULTS = SUBMISSION / "results"
CANONICAL_CKPT = {
    "convnet": RESULTS / "convnextv2_tiny" / "best_convnextv2.pt",
    "transformer": RESULTS / "swin_b" / "best_swin_b.pt",
}
# Expected parameter counts, used only as a sanity warning after loading.
EXPECTED_PARAMS = {"convnet": 27_871_110, "transformer": 86_753_474}


def die(message: str) -> "NoReturn":  # type: ignore[valid-type]
    """Print a clear error and exit non-zero. Used for all handled failures."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


# -----------------------------------------------------------------------------
# Model construction (mirrors the two training scripts exactly)
# -----------------------------------------------------------------------------
def build_convnet() -> nn.Module:
    import timm  # imported lazily so --help works without the dependency
    return timm.create_model(
        "convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=6
    )


class SwinBClassifier(nn.Module):
    """Swin-B with a 10-way head — the exact module ``train_swin_b.py`` saved."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        from torchvision import models
        self.swin = models.swin_b(weights=None)
        self.swin.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.swin.head.in_features, num_classes),
        )

    def forward(self, x):  # noqa: D401
        return self.swin(x)


def convnet_transform() -> T.Compose:
    # Resize 1.14x then centre-crop to 384 — identical to training's eval_tf.
    return T.Compose([
        T.Resize(int(384 * 1.14)),
        T.CenterCrop(384),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def swin_transform() -> T.Compose:
    # Squash to 224x224 (no centre crop) — identical to train_swin_b's val_transform.
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(model: nn.Module, path: Path, model_key: str) -> nn.Module:
    """Load weights, failing loudly on a missing file or a shape mismatch."""
    if not path.is_file():
        die(f"checkpoint not found: {path}\n"
            f"       pass --checkpoint or place the file at the canonical path.")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older-format checkpoints (rare) may need the unrestricted loader.
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        die(f"checkpoint is incompatible with the {model_key} architecture — the "
            f"state dict does not match (wrong model, class count, or class order?).\n"
            f"       torch reported: {exc}")
    got = sum(p.numel() for p in model.parameters())
    want = EXPECTED_PARAMS.get(model_key)
    if want is not None and got != want:
        print(f"WARNING: {model_key} has {got:,} parameters, expected {want:,}.",
              file=sys.stderr)
    return model


# -----------------------------------------------------------------------------
# Building the labelled sample list
# -----------------------------------------------------------------------------
def samples_from_dir(root: Path, classes: list[str]) -> list[tuple[str, int]]:
    """Collect (path, label) from a folder whose sub-dirs are class names."""
    if not root.is_dir():
        die(f"--data-root is not a directory: {root}")
    index = {c.lower(): i for i, c in enumerate(classes)}
    samples: list[tuple[str, int]] = []
    ignored: list[str] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        label = index.get(child.name.lower())
        if label is None:
            ignored.append(child.name)
            continue
        for f in sorted(child.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                samples.append((str(f), label))
    if ignored:
        print(f"note: ignored {len(ignored)} non-target sub-dir(s): "
              f"{', '.join(ignored[:8])}{' ...' if len(ignored) > 8 else ''}")
    return samples


def samples_from_manifest(manifest: Path, classes: list[str],
                          data_root: Path | None) -> list[tuple[str, int]]:
    """Collect (path, label) from a CSV with a path column and a label column."""
    if not manifest.is_file():
        die(f"--split-manifest not found: {manifest}")
    index = {c.lower(): i for i, c in enumerate(classes)}
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            die(f"manifest has no header row: {manifest}")
        fields = {name.lower(): name for name in reader.fieldnames}
        path_col = next((fields[k] for k in ("path", "filepath", "image", "file")
                         if k in fields), None)
        label_col = next((fields[k] for k in ("label", "class", "target", "y")
                          if k in fields), None)
        if path_col is None or label_col is None:
            die(f"manifest must have a path column (path/filepath/image/file) and a "
                f"label column (label/class/target/y); found {reader.fieldnames}")
        base = data_root if data_root is not None else manifest.parent
        samples: list[tuple[str, int]] = []
        for row_no, row in enumerate(reader, start=2):
            raw_path = (row.get(path_col) or "").strip()
            raw_label = (row.get(label_col) or "").strip()
            if not raw_path:
                continue
            p = Path(raw_path)
            resolved = p if p.is_absolute() else base / p
            if raw_label.lower() in index:
                label = index[raw_label.lower()]
            elif raw_label.lstrip("-").isdigit() and 0 <= int(raw_label) < len(classes):
                label = int(raw_label)
            else:
                die(f"manifest row {row_no}: label {raw_label!r} is not one of "
                    f"{classes} nor a valid index 0..{len(classes) - 1}")
            samples.append((str(resolved), label))
    return samples


# -----------------------------------------------------------------------------
# Device + inference
# -----------------------------------------------------------------------------
def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            die("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            die("--device mps requested but MPS is not available")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_inference(model, samples, transform, device, shared_idx, batch_size, limit):
    """Return aligned (paths, labels, active-space logits) plus any skipped files."""
    model.eval()
    paths: list[str] = []
    labels: list[int] = []
    logits_chunks: list[torch.Tensor] = []
    skipped: list[tuple[str, str]] = []
    batch_imgs: list[torch.Tensor] = []
    batch_meta: list[tuple[str, int]] = []

    def flush() -> None:
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        out = model(x)
        if shared_idx is not None:            # mask ten logits down to the six shared
            out = out[:, shared_idx]
        logits_chunks.append(out.float().cpu())
        for pth, lab in batch_meta:
            paths.append(pth)
            labels.append(lab)
        batch_imgs.clear()
        batch_meta.clear()

    for i, (pth, lab) in enumerate(samples):
        if limit is not None and i >= limit:
            break
        try:
            with Image.open(pth) as im:
                tensor = transform(im.convert("RGB"))
        except Exception as exc:
            skipped.append((pth, f"{type(exc).__name__}: {exc}"))
            continue
        batch_imgs.append(tensor)
        batch_meta.append((pth, lab))
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    if not logits_chunks:
        die("no images could be evaluated — the split is empty or every file was "
            "unreadable. Check --data-root / --split-manifest.")
    return paths, np.asarray(labels), torch.cat(logits_chunks), skipped


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compute_metrics(labels, logits, classes):
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(1)
    confidence = probs.max(1)
    n = len(labels)
    per_sample_loss = -np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0))

    k = len(classes)
    confusion = np.zeros((k, k), dtype=int)   # confusion[true, pred]
    for t, p in zip(labels, preds):
        confusion[t, p] += 1

    per_class = {}
    for i, c in enumerate(classes):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1,
                        "support": int(confusion[i, :].sum())}

    return {
        "accuracy": float(confusion.trace() / max(confusion.sum(), 1)),
        "macro_f1": float(np.mean([per_class[c]["f1"] for c in classes])),
        "mean_loss": float(per_sample_loss.mean()),
        "confusion": confusion,
        "per_class": per_class,
        "preds": preds,
        "confidence": confidence,
        "per_sample_loss": per_sample_loss,
    }


def plot_confusion(confusion, classes, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    green = "#2C5842"
    cmap = LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#CFE0D6", green])
    k = len(classes)
    labs = [c.capitalize() for c in classes]
    fig, ax = plt.subplots(figsize=(0.9 * k + 2, 0.9 * k + 1.6), dpi=200)
    vmax = max(int(confusion.max()), 1)
    ax.imshow(confusion, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(k)); ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(k)); ax.set_yticklabels(labs, fontsize=11)
    ax.set_xlabel("predicted", color="#5A6560"); ax.set_ylabel("true", color="#5A6560")
    ax.set_title(title, color="#5A6560", fontsize=11, pad=10)
    thr = vmax * 0.5
    for i in range(k):
        for j in range(k):
            v = int(confusion[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                    color=("white" if v > thr else ("#B9C2BC" if v == 0 else "#20302a")),
                    fontweight=("bold" if i == j else "normal"))
    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------
def write_predictions(path, paths, labels, metrics, classes):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "true", "pred", "confidence", "correct", "loss"])
        for p, t, pred, conf, loss in zip(
            paths, labels, metrics["preds"], metrics["confidence"],
            metrics["per_sample_loss"],
        ):
            w.writerow([p, classes[t], classes[pred], f"{conf:.4f}",
                        int(t == pred), f"{loss:.4f}"])


def write_error_tables(out_dir, paths, labels, metrics, classes, top_k):
    errors = [i for i in range(len(labels)) if labels[i] != metrics["preds"][i]]
    by_loss = sorted(errors, key=lambda i: metrics["per_sample_loss"][i], reverse=True)
    by_conf = sorted(errors, key=lambda i: metrics["confidence"][i], reverse=True)

    def dump(name, order):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "true", "pred", "confidence", "loss"])
            for i in order[:top_k]:
                w.writerow([paths[i], classes[labels[i]], classes[metrics["preds"][i]],
                            f"{metrics['confidence'][i]:.4f}",
                            f"{metrics['per_sample_loss'][i]:.4f}"])

    dump("errors_highest_loss.csv", by_loss)
    dump("errors_confident_wrong.csv", by_conf)
    return len(errors)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["convnet", "transformer"])
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="checkpoint path (defaults to the canonical model path)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-root", type=Path, help="folder of class sub-directories")
    src.add_argument("--split-manifest", type=Path,
                     help="CSV with a path column and a label column")
    ap.add_argument("--data-root-for-manifest", type=Path, default=None,
                    help="base dir to resolve relative manifest paths against")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--classes", choices=["six", "native"], default="six",
                    help="transformer only: score the six shared classes or all ten")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate at most N images (smoke tests)")
    ap.add_argument("--top-k", type=int, default=40, help="rows per error table")
    ap.add_argument("--no-plots", action="store_true", help="skip the confusion PNG")
    return ap.parse_args()


def main():
    args = parse_args()

    # Determinism first, before anything touches an RNG.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Resolve the model contract: architecture, active class list, masking, transform.
    if args.model == "convnet":
        model = build_convnet()
        classes, shared_idx, transform = CLASSES_6, None, convnet_transform()
        image_size, class_mode = 384, "six"
    else:
        model = SwinBClassifier(num_classes=10)
        if args.classes == "six":
            classes, shared_idx, class_mode = CLASSES_6, SHARED_IN_10, "six"
        else:
            classes, shared_idx, class_mode = CLASSES_10, None, "native"
        transform, image_size = swin_transform(), 224

    checkpoint = args.checkpoint or CANONICAL_CKPT[args.model]
    checkpoint = Path(checkpoint)
    load_checkpoint(model, checkpoint, args.model)
    device = pick_device(args.device)
    model.to(device)

    # Build the labelled sample list from a folder or an explicit manifest.
    if args.data_root is not None:
        samples = samples_from_dir(args.data_root, classes)
        data_source = {"data_root": str(args.data_root)}
    else:
        samples = samples_from_manifest(
            args.split_manifest, classes, args.data_root_for_manifest)
        data_source = {"split_manifest": str(args.split_manifest)}
    if not samples:
        die("no labelled images found — check that sub-directory / label names match "
            f"the expected classes {classes}")

    paths, labels, logits, skipped = run_inference(
        model, samples, transform, device, shared_idx, args.batch_size, args.limit)
    metrics = compute_metrics(labels, logits, classes)

    # Write everything to the output directory.
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / "predictions.csv", paths, labels, metrics, classes)
    n_errors = write_error_tables(out_dir, paths, labels, metrics, classes, args.top_k)
    with open(out_dir / "confusion_matrix.json", "w", encoding="utf-8") as fh:
        json.dump({"classes": classes, "confusion": metrics["confusion"].tolist()},
                  fh, indent=2)
    title = f"{args.model} ({class_mode} classes)"
    if not args.no_plots:
        plot_confusion(metrics["confusion"], classes,
                       out_dir / "confusion_matrix.png", title)

    report = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "class_mode": class_mode,
        "classes": classes,
        "image_size": image_size,
        "device": str(device),
        "seed": args.seed,
        "deterministic": True,
        "n_images": int(len(labels)),
        "n_skipped": len(skipped),
        "n_errors": n_errors,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mean_loss": metrics["mean_loss"],
        "per_class": metrics["per_class"],
        **data_source,
        "torch_version": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Human-readable summary.
    print(f"\n{args.model} · {class_mode} classes · {len(labels)} images "
          f"({len(skipped)} skipped) · {device}")
    print(f"accuracy = {metrics['accuracy']*100:.2f}%   "
          f"macro-F1 = {metrics['macro_f1']:.4f}   loss = {metrics['mean_loss']:.4f}")
    print(f"{'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>7}")
    for c in classes:
        pc = metrics["per_class"][c]
        print(f"{c:<12}{pc['precision']*100:>7.1f}%{pc['recall']*100:>7.1f}%"
              f"{pc['f1']*100:>7.1f}%{pc['support']:>7}")
    print(f"\nwrote metrics.json, predictions.csv, confusion_matrix.(json|png), "
          f"and error tables to {out_dir}")
    if skipped:
        print(f"note: {len(skipped)} file(s) were unreadable and skipped; "
              f"first was {skipped[0][0]}")


if __name__ == "__main__":
    main()
