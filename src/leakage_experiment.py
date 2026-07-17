"""Arm A vs Arm B: does augment-before-split reproduce the literature's 99%+?

Same model, same seed, same hyperparameters, same images. Only the pipeline
order changes:

  Arm A (correct): group-aware stratified split FIRST, then augment train only.
  Arm B (leaked):  "augment the full dataset 4x and save copies" (simulated by
                   deterministic per-copy augmentation), THEN random split at
                   the copy level — so augmented copies of the same source
                   photo land on both sides.

Both arms run on the train+val pool only; the quarantined test set is never
touched. Framing rule: this reproduces the *mechanism* consistent with
published 99-100% claims ("their image counts imply this ordering"), it does
not prove any specific paper did it. A null result is reported honestly.

Usage: python -m src.leakage_experiment [--config configs/baseline.yaml]
"""

import argparse
import random
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import TrashDataset, build_eval_transform, build_train_transform
from .evaluate import compute_metrics, predict
from .model import build_model
from .train import build_phase, train_one_epoch
from .utils import (
    CLASS_TO_IDX,
    REPO_ROOT,
    get_device,
    load_config,
    load_split_frames,
    log_experiment,
    seed_everything,
)

AUG_COPIES = 4          # 2527 → 10108 in the criticized papers is exactly 4x
EVAL_FRACTION = 0.2
OVERRIDES = {"epochs_head": 2, "epochs_ft": 8, "warmup_epochs": 1,
             "class_weighting": "none", "mixup": 0.0, "cutmix": 0.0,
             "aug": "basic", "ema": False}


class FixedAugCopies(Dataset):
    """Simulates an offline-augmented dataset: copy k of an image is always the
    SAME augmented image (seeded transform), as if it had been saved to disk."""

    def __init__(self, items, img_size):
        self.items = items  # (path, label_idx, copy_seed)
        self.transform = build_train_transform({"img_size": img_size, "aug": "basic"})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, copy_seed = self.items[i]
        img = Image.open(REPO_ROOT / path).convert("RGB")
        # freeze all randomness for this copy → deterministic augmented image
        state_py, state_torch = random.getstate(), torch.get_rng_state()
        random.seed(copy_seed)
        torch.manual_seed(copy_seed)
        out = self.transform(img)
        random.setstate(state_py)
        torch.set_rng_state(state_torch)
        return out, label


def run_arm(cfg, device, train_ds, eval_ds, arm_name):
    seed_everything(cfg["seed"])
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler() if (cfg.get("amp") and device.type == "cuda") else None

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg.get("num_workers", 4), drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg.get("num_workers", 4))

    total = cfg["epochs_head"] + cfg["epochs_ft"]
    phase = opt = sched = None
    for epoch in range(total):
        wanted = "head" if epoch < cfg["epochs_head"] else "ft"
        if wanted != phase:
            phase = wanted
            opt, sched = build_phase(cfg, model, phase)
        loss = train_one_epoch(model, train_loader, criterion, opt, device, scaler,
                               ema=None, mixup_fn=None)
        sched.step()
        print(f"  [{arm_name}] epoch {epoch} [{phase}] train_loss {loss:.4f}")

    probs, targets = predict(model, eval_loader, device)
    return compute_metrics(probs, targets)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = {**load_config(args.config), **OVERRIDES}
    cfg["run_name"] = "leakage_experiment"
    device = get_device()
    seed_everything(cfg["seed"])

    folds, _test = load_split_frames(REPO_ROOT / cfg["splits_dir"])  # test never touched
    pool = folds.reset_index(drop=True)
    print(f"pool = {len(pool)} train+val images (quarantined test untouched)")

    # ---- Arm A: split first (group-aware, stratified), augment train only ----
    sgkf = StratifiedGroupKFold(n_splits=int(1 / EVAL_FRACTION), shuffle=True,
                                random_state=cfg["seed"])
    tr_idx, ev_idx = next(sgkf.split(pool, pool["label"], pool["group"]))
    arm_a_train = TrashDataset(pool.iloc[tr_idx], build_train_transform(cfg))
    arm_a_eval = TrashDataset(pool.iloc[ev_idx], build_eval_transform(cfg["img_size"]))
    print(f"\nArm A (correct): {len(tr_idx)} train / {len(ev_idx)} eval sources, group-disjoint")
    metrics_a = run_arm(cfg, device, arm_a_train, arm_a_eval, "Arm A")

    # ---- Arm B: augment everything 4x, then random split over the copies ----
    # crc32, not hash(): Python string hashing is salted per process, and the
    # augmented "copies" must be identical across runs
    items = [(row.path, CLASS_TO_IDX[row.label], zlib.crc32(f"{row.path}#{k}".encode()))
             for row in pool.itertuples() for k in range(AUG_COPIES)]
    rng = np.random.RandomState(cfg["seed"])
    order = rng.permutation(len(items))
    n_eval = int(len(items) * EVAL_FRACTION)
    ev_items = [items[i] for i in order[:n_eval]]
    tr_items = [items[i] for i in order[n_eval:]]
    n_shared = len({p for p, _, _ in ev_items} & {p for p, _, _ in tr_items})
    print(f"\nArm B (leaked): {len(tr_items)} train / {len(ev_items)} eval copies; "
          f"{n_shared}/{len({p for p, _, _ in ev_items})} eval source images also in train")
    metrics_b = run_arm(cfg, device, FixedAugCopies(tr_items, cfg["img_size"]),
                        FixedAugCopies(ev_items, cfg["img_size"]), "Arm B")

    # ---- report ----
    gap = metrics_b["acc"] - metrics_a["acc"]
    table = pd.DataFrame([
        {"arm": "A (split→augment, group-aware)", "accuracy": metrics_a["acc"],
         "macro_f1": metrics_a["macro_f1"]},
        {"arm": "B (augment 4x→random split)", "accuracy": metrics_b["acc"],
         "macro_f1": metrics_b["macro_f1"]},
    ])
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nleakage inflation: {gap * 100:+.2f} accuracy points")

    out = REPO_ROOT / "reports" / "leakage_experiment.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "# Leakage experiment: augment-before-split vs split-before-augment\n\n"
        f"Same model ({cfg['model']}), seed ({cfg['seed']}), hyperparameters and images "
        f"(train+val pool only, n={len(pool)}); only the pipeline order differs.\n\n"
        + table.to_markdown(index=False, floatfmt=".4f")
        + f"\n\n**Inflation from leakage: {gap * 100:+.2f} points.**\n\n"
        f"Arm B's eval copies share {n_shared} source images with its train set — the model "
        "is tested on augmented copies of images it trained on. This is the mechanism "
        "consistent with published 99-100% TrashNet claims whose image counts (2527 → 10108, "
        "exactly 4x) imply augmentation before splitting. It does not prove any specific "
        "paper did this.\n")
    print(f"wrote {out}")

    log_experiment(cfg, stage="leakage_armA", metrics=metrics_a)
    log_experiment(cfg, stage="leakage_armB", metrics=metrics_b,
                   notes=f"inflation {gap * 100:+.2f} pts")


if __name__ == "__main__":
    main()
