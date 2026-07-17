"""Training loop: two-phase fine-tune, EMA, AMP, checkpoint/resume, CSV logging.

Phase "head": backbone frozen, only the new classifier trains (epochs_head).
Phase "ft":   everything unfreezes at discriminative LRs (backbone << head),
              cosine schedule with warmup, early stopping on val accuracy.

Usage:
  python -m src.train --config configs/baseline.yaml [--fold K] [--resume] [--overfit-batch]
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .data import build_loaders, get_train_val_frames
from .evaluate import compute_metrics, predict
from .model import build_model, freeze_backbone, param_groups
from .utils import (
    REPO_ROOT,
    config_hash,
    get_device,
    load_config,
    log_experiment,
    seed_everything,
)


def make_ema(model, decay):
    from timm import utils as timm_utils

    if hasattr(timm_utils, "ModelEmaV3"):
        return timm_utils.ModelEmaV3(model, decay=decay)
    return timm_utils.ModelEmaV2(model, decay=decay)


def warmup_cosine(total_epochs: int, warmup_epochs: int):
    def fn(epoch):
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return fn


def build_phase(cfg, model, phase: str):
    """(Re)build optimizer + scheduler at a phase boundary."""
    if phase == "head":
        freeze_backbone(model, frozen=True)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _e: 1.0)
    else:
        freeze_backbone(model, frozen=False)
        opt = torch.optim.AdamW(
            param_groups(model, cfg["lr_head"], cfg["lr_backbone"], cfg["weight_decay"]))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, warmup_cosine(cfg["epochs_ft"], cfg.get("warmup_epochs", 0)))
    return opt, sched


def train_one_epoch(model, loader, criterion, opt, device, scaler, ema, mixup_fn):
    model.train()
    total_loss, steps = 0.0, 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            imgs, targets = mixup_fn(imgs, targets)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            out = model(imgs)
            loss = criterion(out, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def overfit_a_batch(cfg, device):
    """M1 acceptance gate: a healthy pipeline drives one batch's loss to ~0."""
    seed_everything(cfg["seed"])
    train_df, _ = get_train_val_frames(cfg, val_fold=0)
    loader, _ = build_loaders({**cfg, "class_weighting": "none", "num_workers": 0}, train_df, train_df)
    imgs, targets = next(iter(loader))
    imgs, targets = imgs.to(device), targets.to(device)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(imgs), targets)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == 199:
            print(f"  step {step:3d}  loss {loss.item():.4f}")
        if loss.item() < 0.01:
            break
    final = loss.item()
    verdict = "PASS" if final < 0.05 else "FAIL — pipeline is broken, do not proceed"
    print(f"overfit-a-batch final loss {final:.4f}: {verdict}")
    return final < 0.05


def train_run(cfg: dict, val_fold: int = 0, resume: bool = False) -> dict:
    seed_everything(cfg["seed"])
    device = get_device()
    print(f"run={cfg['run_name']} fold={val_fold} device={device.type} "
          f"config_hash={config_hash(cfg)}")

    train_df, val_df = get_train_val_frames(cfg, val_fold=val_fold)
    print(f"train={len(train_df)} val={len(val_df)} (group-disjoint, test untouched)")
    train_loader, val_loader = build_loaders(cfg, train_df, val_df)

    model = build_model(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    ema = make_ema(model, cfg.get("ema_decay", 0.9998)) if cfg.get("ema") else None
    scaler = torch.amp.GradScaler() if (cfg.get("amp") and device.type == "cuda") else None

    mixup_fn = None
    use_mix = cfg.get("mixup", 0) > 0 or cfg.get("cutmix", 0) > 0
    if use_mix:
        from timm.data import Mixup
        mixup_fn = Mixup(mixup_alpha=cfg.get("mixup", 0), cutmix_alpha=cfg.get("cutmix", 0),
                         label_smoothing=cfg.get("label_smoothing", 0.0), num_classes=6)
        train_criterion = __import__("timm").loss.SoftTargetCrossEntropy()
    else:
        weight = None
        if cfg.get("class_weighting") == "loss_weights":
            from .data import class_weights
            weight = class_weights(train_df).to(device)
        train_criterion = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=cfg.get("label_smoothing", 0.0))
    val_criterion = nn.CrossEntropyLoss()

    ckpt_dir = REPO_ROOT / "checkpoints" / cfg["run_name"] / f"fold{val_fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pth", ckpt_dir / "best.pth"

    total_epochs = cfg["epochs_head"] + cfg["epochs_ft"]
    start_epoch, best_acc, patience_left = 0, 0.0, cfg.get("early_stop_patience", 10)
    history = []
    phase = None
    opt = sched = None

    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        best_acc = state["best_acc"]
        patience_left = state["patience_left"]
        history = state["history"]
        phase = state["phase"]
        opt, sched = build_phase(cfg, model, phase)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        if ema is not None and state.get("ema") is not None:
            ema.module.load_state_dict(state["ema"])
        print(f"resumed from epoch {start_epoch} (phase={phase}, best_acc={best_acc:.4f})")

    for epoch in range(start_epoch, total_epochs):
        wanted_phase = "head" if epoch < cfg["epochs_head"] else "ft"
        if wanted_phase != phase:
            phase = wanted_phase
            opt, sched = build_phase(cfg, model, phase)
            print(f"--- phase {phase} ---")

        train_loss = train_one_epoch(model, train_loader, train_criterion, opt,
                                     device, scaler, ema, mixup_fn)
        sched.step()

        eval_model = ema.module if ema is not None else model
        probs, targets = predict(eval_model, val_loader, device)
        metrics = compute_metrics(probs, targets)
        val_loss = val_criterion(torch.log(torch.tensor(probs).clamp_min(1e-9)),
                                 torch.tensor(targets)).item()
        history.append({"epoch": epoch, "phase": phase, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": metrics["acc"]})
        print(f"epoch {epoch:3d} [{phase}] train_loss {train_loss:.4f} "
              f"val_loss {val_loss:.4f} val_acc {metrics['acc']:.4f} "
              f"macro_f1 {metrics['macro_f1']:.4f}")

        improved = metrics["acc"] > best_acc
        if improved:
            best_acc = metrics["acc"]
            patience_left = cfg.get("early_stop_patience", 10)
            torch.save({"model": model.state_dict(),
                        "ema": ema.module.state_dict() if ema else None,
                        "cfg": cfg, "val_fold": val_fold, "epoch": epoch,
                        "metrics": metrics}, best_path)
        elif phase == "ft":
            patience_left -= 1

        torch.save({"model": model.state_dict(),
                    "ema": ema.module.state_dict() if ema else None,
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "cfg": cfg, "val_fold": val_fold, "epoch": epoch, "phase": phase,
                    "best_acc": best_acc, "patience_left": patience_left,
                    "history": history}, last_path)

        if phase == "ft" and patience_left <= 0:
            print(f"early stop at epoch {epoch} (best val_acc {best_acc:.4f})")
            break

    save_curves(cfg, history, val_fold)
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    result = {**best_state["metrics"], "best_val_acc": best_acc,
              "epochs_ran": len(history), "checkpoint": str(best_path)}
    log_experiment(cfg, stage="train", metrics=result, fold=val_fold)
    print(f"best val_acc {best_acc:.4f} → {best_path}")
    return result


def save_curves(cfg, history, val_fold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = REPO_ROOT / "reports" / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(out_dir / f"history_fold{val_fold}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].plot(df["epoch"], df["val_acc"])
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val accuracy")
    if cfg["epochs_head"] > 0:
        for ax in axes:
            ax.axvline(cfg["epochs_head"] - 0.5, ls="--", c="gray", lw=0.8)
    fig.suptitle(f"{cfg['run_name']} fold{val_fold}")
    fig.tight_layout()
    fig.savefig(out_dir / f"curves_fold{val_fold}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0, help="which CV fold is validation")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit-batch", action="store_true",
                    help="sanity gate: drive one batch's loss to ~0, then exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.overfit_batch:
        ok = overfit_a_batch(cfg, get_device())
        raise SystemExit(0 if ok else 1)
    train_run(cfg, val_fold=args.fold, resume=args.resume)


if __name__ == "__main__":
    main()
