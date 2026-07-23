"""Evaluation: metrics, confusion matrix, TTA, ensembles, and the single
--final-eval path (the only code allowed to read test images — Rule 1).

Usage:
  python -m src.evaluate --checkpoint checkpoints/<run>/fold0/best.pth [--tta]
  python -m src.evaluate --checkpoint checkpoints/<run>/ensemble.json [--tta]
  python -m src.evaluate --final-eval --checkpoint <...> [--tta]   # ONCE, at the end
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from torchvision import transforms as T

from .data import TrashDataset, build_eval_transform
from .model import build_model
from .utils import (
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    REPO_ROOT,
    assert_test_not_spent,
    config_hash,
    get_device,
    load_split_frames,
    log_experiment,
    mark_test_spent,
    train_val_from_folds,
)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    """Softmax probabilities + targets for one model over one loader."""
    model.eval()
    probs, targets = [], []
    for imgs, tgt in loader:
        out = model(imgs.to(device, non_blocking=True))
        probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        targets.append(tgt.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def tta_transforms(img_size: int):
    """Original + hflip + a couple of crops, averaged (M2.6)."""
    resize = int(img_size * 1.14)
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return [
        build_eval_transform(img_size),
        T.Compose([T.Resize(resize), T.CenterCrop(img_size),
                   T.RandomHorizontalFlip(p=1.0), *norm]),
        T.Compose([T.Resize((img_size, img_size)), *norm]),          # full image, no crop
        T.Compose([T.Resize(int(img_size * 1.35)), T.CenterCrop(img_size), *norm]),
    ]


def predict_frame(models, frame, cfg, device, tta=False, batch_size=64):
    """Average softmax over all models (ensemble) and all TTA views."""
    tfs = tta_transforms(cfg["img_size"]) if tta else [build_eval_transform(cfg["img_size"])]
    acc_probs, targets = None, None
    for tf in tfs:
        loader = torch.utils.data.DataLoader(
            TrashDataset(frame, tf), batch_size=batch_size, shuffle=False,
            num_workers=cfg.get("num_workers", 4))
        for model in models:
            probs, targets = predict(model, loader, device)
            acc_probs = probs if acc_probs is None else acc_probs + probs
    return acc_probs / (len(tfs) * len(models)), targets


# ---------------------------------------------------------------------------
# Metrics + reporting (Rule 5: never accuracy alone)
# ---------------------------------------------------------------------------

def compute_metrics(probs, targets) -> dict:
    preds = probs.argmax(1)
    recalls = recall_score(targets, preds, average=None, labels=range(len(CLASSES)),
                           zero_division=0)
    return {
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "per_class_recall": {c: round(float(r), 4) for c, r in zip(CLASSES, recalls)},
    }


def full_report(probs, targets) -> str:
    return classification_report(targets, probs.argmax(1), target_names=CLASSES,
                                 digits=4, zero_division=0)


def plot_confusion(probs, targets, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(targets, probs.argmax(1), labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


# ---------------------------------------------------------------------------
# Checkpoint loading (single .pth or ensemble .json manifest)
# ---------------------------------------------------------------------------

def load_models(checkpoint: str, device):
    path = Path(checkpoint)
    ckpt_paths = [path]
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        ckpt_paths = [REPO_ROOT / p for p in manifest["checkpoints"]]

    models, cfg, val_fold = [], None, 0
    for p in ckpt_paths:
        state = torch.load(p, map_location="cpu", weights_only=False)
        cfg = state["cfg"]
        val_fold = state.get("val_fold", 0)
        m = build_model(cfg, pretrained=False)
        m.load_state_dict(state["ema"] if state.get("ema") is not None else state["model"])
        m.to(device).eval()
        models.append(m)
    print(f"loaded {len(models)} model(s) from {checkpoint}")
    return models, cfg, val_fold


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help=".pth or ensemble .json")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--final-eval", action="store_true",
                    help="THE single test-set run. Do this once, at the very end.")
    args = ap.parse_args()

    device = get_device()
    models, cfg, val_fold = load_models(args.checkpoint, device)
    folds, test = load_split_frames(REPO_ROOT / cfg["splits_dir"])  # asserts no leakage

    if args.final_eval:
        assert_test_not_spent(cfg["splits_dir"])  # Rule 1, enforced not just documented
        frame, split_name = test, "test"
        print("\n*** FINAL EVAL: this is the one and only test-set run. ***")
        print(f"*** Leak assertions passed: {len(test)} test images, "
              f"{test['group'].nunique()} groups, disjoint from all folds. ***\n")
    else:
        _, frame = train_val_from_folds(folds, val_fold)
        split_name = f"val_fold{val_fold}"

    probs, targets = predict_frame(models, frame, cfg, device, tta=args.tta)
    metrics = compute_metrics(probs, targets)
    report = full_report(probs, targets)

    run = cfg["run_name"] + ("_ensemble" if len(models) > 1 else "")
    tag = f"{run}_{split_name}" + ("_tta" if args.tta else "")
    out_dir = REPO_ROOT / "reports" / cfg["run_name"]
    plot_confusion(probs, targets, out_dir / f"confusion_{tag}.png", tag)
    (out_dir / f"report_{tag}.txt").write_text(
        f"accuracy: {metrics['acc']:.4f}\nmacro F1: {metrics['macro_f1']:.4f}\n\n{report}")

    print(f"split={split_name} n={len(frame)} models={len(models)} tta={args.tta}")
    print(f"accuracy {metrics['acc']:.4f} | macro F1 {metrics['macro_f1']:.4f}")
    print(report)

    stage = "FINAL_EVAL" if args.final_eval else "eval"
    log_experiment(cfg, stage=stage, metrics=metrics, fold=val_fold,
                   notes=f"ckpt={args.checkpoint} tta={args.tta} n_models={len(models)}")

    if args.final_eval:
        md = out_dir / "final_test_report.md"
        md.write_text(
            f"# Final test-set result — {run}\n\n"
            f"- date: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"- checkpoint: `{args.checkpoint}` ({len(models)} model(s), tta={args.tta})\n"
            f"- config hash: `{config_hash(cfg)}` (see experiments.csv)\n"
            f"- test accuracy: **{metrics['acc']:.4f}** (n={len(frame)}; "
            f"~1% ≈ {max(1, round(len(frame) / 100))} images — gaps under ~1.5% are noise)\n"
            f"- macro F1: {metrics['macro_f1']:.4f}\n\n"
            f"```\n{report}\n```\n\n"
            f"![confusion](confusion_{tag}.png)\n")
        guard = mark_test_spent(cfg["splits_dir"], checkpoint=args.checkpoint,
                                metrics=metrics, n_images=len(frame))
        print(f"wrote {md}")
        print(f"wrote {guard} — this test set is now spent and further "
              f"--final-eval runs on it will be refused")


if __name__ == "__main__":
    main()
