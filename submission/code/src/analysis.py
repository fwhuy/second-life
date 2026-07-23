"""Error analysis: cleanlab label audit, Grad-CAM figures, confusion clusters,
and an optional glass/plastic/metal specialist head (M3).

Usage:
  python -m src.analysis --audit   --checkpoint checkpoints/<run>/ensemble.json
  python -m src.analysis --gradcam --checkpoint checkpoints/<run>/fold0/best.pth
  python -m src.analysis --audit --make-clean-splits --checkpoint <ensemble.json>
  python -m src.analysis --specialist --checkpoint checkpoints/<run>/fold0/best.pth

The ensemble manifest gives proper out-of-fold predictions for the audit
(each fold's model predicts only its own held-out fold). With a single
checkpoint the audit falls back to in-sample predictions — noisier, flagged.
Test labels are NEVER audited or corrected (report test noise as a limitation).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .data import TrashDataset, build_eval_transform
from .evaluate import compute_metrics, load_models, predict, predict_frame
from .utils import (
    CLASS_TO_IDX,
    CLASSES,
    REPO_ROOT,
    get_device,
    load_split_frames,
    seed_everything,
    train_val_from_folds,
)

SPECIALIST_CLASSES = ["glass", "metal", "plastic"]  # the dominant confusion cluster


# ---------------------------------------------------------------------------
# Out-of-fold predictions
# ---------------------------------------------------------------------------

def oof_predictions(checkpoint: str, device):
    """probs + labels for every train+val image, out-of-fold when possible."""
    models, cfg, _ = load_models(checkpoint, device)
    folds, _ = load_split_frames(REPO_ROOT / cfg["splits_dir"])

    if len(models) == folds["fold"].nunique():
        # ensemble manifest: model i was validated on fold i → true OOF
        parts = []
        for fold_id, model in enumerate(models):
            _, val_df = train_val_from_folds(folds, fold_id)
            probs, _ = predict_frame([model], val_df, cfg, device)
            parts.append(val_df.assign(**{f"p_{c}": probs[:, i] for i, c in enumerate(CLASSES)}))
        merged = pd.concat(parts, ignore_index=True)
        in_sample = False
    else:
        print("WARNING: single model → in-sample audit (train an ensemble for a clean one)")
        probs, _ = predict_frame(models, folds, cfg, device)
        merged = folds.assign(**{f"p_{c}": probs[:, i] for i, c in enumerate(CLASSES)})
        in_sample = True

    probs = merged[[f"p_{c}" for c in CLASSES]].to_numpy()
    labels = merged["label"].map(CLASS_TO_IDX).to_numpy()
    return merged, probs, labels, cfg, in_sample


# ---------------------------------------------------------------------------
# Label audit (cleanlab + highest-loss inspection)
# ---------------------------------------------------------------------------

def run_audit(checkpoint: str, device, make_clean_splits: bool):
    from cleanlab.filter import find_label_issues

    merged, probs, labels, cfg, in_sample = oof_predictions(checkpoint, device)
    issue_idx = find_label_issues(labels, probs, return_indices_ranked_by="self_confidence")
    frac = len(issue_idx) / len(merged)
    print(f"\ncleanlab: {len(issue_idx)} suspected mislabels / {len(merged)} images "
          f"({frac:.1%}){' [IN-SAMPLE]' if in_sample else ''}")

    suspects = merged.iloc[issue_idx].copy()
    suspects["predicted"] = [CLASSES[i] for i in probs[issue_idx].argmax(1)]
    suspects["pred_conf"] = probs[issue_idx].max(1)
    out_dir = REPO_ROOT / "reports" / "label_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "predicted", "pred_conf"]
    suspects[cols].to_csv(out_dir / "suspects.csv", index=False)
    print(suspects[cols].head(15).to_string(index=False))

    grid_path = out_dir / "top_suspects.png"
    save_image_grid(suspects.head(24), grid_path,
                    lambda r: f"label:{r.label} → pred:{r.predicted} ({r.pred_conf:.2f})")
    print(f"wrote {out_dir}/suspects.csv and {grid_path}")

    # confusion-cluster summary from the same predictions
    preds = probs.argmax(1)
    wrong = preds != labels
    pairs = pd.Series(
        [f"{CLASSES[t]}→{CLASSES[p]}" for t, p in zip(labels[wrong], preds[wrong])]
    ).value_counts()
    print("\nTop confusion pairs (true→pred):")
    print(pairs.head(8).to_string())

    if make_clean_splits:
        clean_dir = REPO_ROOT / "data" / "splits_clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        folds, _ = load_split_frames(REPO_ROOT / cfg["splits_dir"])
        clean = folds[~folds["path"].isin(suspects["path"])].reset_index(drop=True)
        clean.to_csv(clean_dir / "folds.csv", index=False)
        # test set copied UNMODIFIED — never correct test labels
        shutil.copy2(REPO_ROOT / cfg["splits_dir"] / "test.csv", clean_dir / "test.csv")
        print(f"\nwrote {clean_dir} with {len(folds) - len(clean)} suspect train images removed.")
        print("Retrain with a config whose splits_dir is data/splits_clean and compare "
              "val accuracy to the original row in experiments.csv.")


def save_image_grid(frame: pd.DataFrame, out_path: Path, caption_fn, cols: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frame)
    rows = max(1, -(-n // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, frame.itertuples()):
        ax.imshow(Image.open(REPO_ROOT / r.path).convert("RGB"))
        ax.set_title(caption_fn(r), fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grad-CAM (poster figures + "what does it attend to" finding)
# ---------------------------------------------------------------------------

def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("no Conv2d layer found — Grad-CAM here supports CNN backbones; "
                           "for ViT-style models add a reshape_transform")
    return last

def run_gradcam(checkpoint: str, device, per_class: int = 2, n_errors: int = 8):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    models, cfg, val_fold = load_models(checkpoint, device)
    model = models[0]
    folds, _ = load_split_frames(REPO_ROOT / cfg["splits_dir"])
    _, val_df = train_val_from_folds(folds, val_fold)

    probs, targets = predict_frame([model], val_df, cfg, device)
    preds = probs.argmax(1)
    correct_sel = [val_df[(val_df["label"] == c) & (preds == CLASS_TO_IDX[c])].head(per_class)
                   for c in CLASSES]
    wrong_df = val_df[preds != targets].copy()
    wrong_df["predicted"] = [CLASSES[p] for p in preds[preds != targets]]
    picks = pd.concat(correct_sel + [wrong_df.head(n_errors)], ignore_index=True)

    tf = build_eval_transform(cfg["img_size"])
    cam = GradCAM(model=model, target_layers=[find_last_conv(model)])
    out_dir = REPO_ROOT / "reports" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r in picks.itertuples():
        img = Image.open(REPO_ROOT / r.path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        gray = cam(input_tensor=x)[0]
        vis_base = np.asarray(
            img.resize((cfg["img_size"], cfg["img_size"]))).astype(np.float32) / 255
        overlay = show_cam_on_image(vis_base, gray, use_rgb=True)
        pred_note = getattr(r, "predicted", None)
        title = f"{r.label}" + (f" → pred {pred_note}" if pred_note else " (correct)")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(overlay); ax.set_title(title, fontsize=8); ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{Path(r.path).stem}_{'err' if pred_note else 'ok'}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {len(picks)} Grad-CAM overlays to {out_dir}")
    print("Poster check: does the heat sit on the OBJECT, or on background/shadows? "
          "Background attention is itself a finding.")


# ---------------------------------------------------------------------------
# Glass/plastic/metal specialist head
# ---------------------------------------------------------------------------

def run_specialist(checkpoint: str, device):
    """Fine-tune a 3-way head on the confusion cluster; report the cluster delta."""
    import timm
    from torch import nn
    from torch.utils.data import DataLoader

    from .data import build_loaders, build_train_transform
    from .train import build_phase, train_one_epoch

    models, cfg, val_fold = load_models(checkpoint, device)
    main_model = models[0]
    folds, _ = load_split_frames(REPO_ROOT / cfg["splits_dir"])
    train_df, val_df = train_val_from_folds(folds, val_fold)
    sub_train = train_df[train_df["label"].isin(SPECIALIST_CLASSES)].reset_index(drop=True)
    sub_val = val_df[val_df["label"].isin(SPECIALIST_CLASSES)].reset_index(drop=True)

    # main model restricted to the cluster = the baseline to beat
    probs, targets = predict_frame([main_model], sub_val, cfg, device)
    cluster_idx = [CLASS_TO_IDX[c] for c in SPECIALIST_CLASSES]
    main_acc = float((probs[:, cluster_idx].argmax(1) ==
                      np.array([cluster_idx.index(t) for t in targets])).mean())
    print(f"main model on {'/'.join(SPECIALIST_CLASSES)} val subset: {main_acc:.4f}")

    class SubsetDS(torch.utils.data.Dataset):
        def __init__(self, frame, tf):
            self.frame, self.tf = frame, tf
        def __len__(self):
            return len(self.frame)
        def __getitem__(self, i):
            r = self.frame.iloc[i]
            img = Image.open(REPO_ROOT / r["path"]).convert("RGB")
            return self.tf(img), SPECIALIST_CLASSES.index(r["label"])

    seed_everything(cfg["seed"])
    spec = timm.create_model(cfg["model"], pretrained=False, num_classes=len(SPECIALIST_CLASSES))
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    spec.load_state_dict(state["ema"] or state["model"], strict=False)  # head shape differs
    spec.to(device)

    scfg = {**cfg, "epochs_head": 2, "epochs_ft": 8, "warmup_epochs": 1}
    train_loader = DataLoader(SubsetDS(sub_train, build_train_transform(cfg)),
                              batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg.get("num_workers", 4), drop_last=True)
    val_loader = DataLoader(SubsetDS(sub_val, build_eval_transform(cfg["img_size"])),
                            batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 4))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler() if (cfg.get("amp") and device.type == "cuda") else None
    phase = opt = sched = None
    for epoch in range(scfg["epochs_head"] + scfg["epochs_ft"]):
        wanted = "head" if epoch < scfg["epochs_head"] else "ft"
        if wanted != phase:
            phase = wanted
            opt, sched = build_phase(scfg, spec, phase)
        loss = train_one_epoch(spec, train_loader, criterion, opt, device, scaler, None, None)
        sched.step()
        print(f"  specialist epoch {epoch} [{phase}] loss {loss:.4f}")

    sprobs, stargets = predict(spec, val_loader, device)
    spec_acc = float((sprobs.argmax(1) == stargets).mean())
    print(f"\nspecialist 3-way accuracy: {spec_acc:.4f} vs main-model {main_acc:.4f} "
          f"(delta {100 * (spec_acc - main_acc):+.2f} pts on {len(sub_val)} val images)")
    print("Use the specialist only if the delta clears noise (~1.5%).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--specialist", action="store_true")
    ap.add_argument("--make-clean-splits", action="store_true")
    args = ap.parse_args()

    device = get_device()
    if args.audit or args.make_clean_splits:
        run_audit(args.checkpoint, device, args.make_clean_splits)
    if args.gradcam:
        run_gradcam(args.checkpoint, device)
    if args.specialist:
        run_specialist(args.checkpoint, device)
    if not (args.audit or args.gradcam or args.specialist or args.make_clean_splits):
        ap.print_help()


if __name__ == "__main__":
    main()
