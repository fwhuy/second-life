"""Seeding, config handling, experiment logging, and leakage assertions.

Every entry point in this repo funnels through `assert_no_leakage` /
`load_split_frames` so a leaked split fails loudly instead of inflating a number.
"""

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
# Rule 4: ImageNet normalization everywhere.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_COUNTS = {
    "paper": 594, "glass": 501, "plastic": 482,
    "metal": 410, "cardboard": 403, "trash": 137,
}
EXPERIMENTS_CSV = REPO_ROOT / "experiments.csv"

EXPERIMENT_COLUMNS = [
    "timestamp", "run_name", "config_hash", "seed", "model", "img_size", "fold",
    "stage", "epochs_ran", "best_val_acc", "val_macro_f1", "val_loss",
    "per_class_recall", "notes", "config_json",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def log_experiment(cfg: dict, *, stage: str, metrics: dict, fold=None, notes: str = "") -> None:
    """Append one row to the append-only experiments.csv (Rule 7)."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": cfg.get("run_name", "?"),
        "config_hash": config_hash(cfg),
        "seed": cfg.get("seed"),
        "model": cfg.get("model"),
        "img_size": cfg.get("img_size"),
        "fold": "" if fold is None else fold,
        "stage": stage,
        "epochs_ran": metrics.get("epochs_ran", ""),
        "best_val_acc": metrics.get("best_val_acc", metrics.get("acc", "")),
        "val_macro_f1": metrics.get("macro_f1", ""),
        "val_loss": metrics.get("loss", ""),
        "per_class_recall": json.dumps(metrics.get("per_class_recall", {})),
        "notes": notes,
        "config_json": json.dumps(
            {k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True, default=str
        ),
    }
    write_header = not EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Split loading + leakage assertions (Rules 1-3)
# ---------------------------------------------------------------------------

def load_split_frames(splits_dir: str | Path):
    """Return (folds_df, test_df). Both carry path, label, group columns.

    folds_df additionally has a `fold` column in 0..4. The default single-run
    validation split is fold 0; kfold rotates it.
    """
    splits_dir = Path(splits_dir)
    folds = pd.read_csv(splits_dir / "folds.csv")
    test = pd.read_csv(splits_dir / "test.csv")
    assert_no_leakage(folds, test)
    return folds, test


def assert_no_leakage(folds_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Hard-fail if any image OR near-duplicate group spans test and train/val."""
    overlap_paths = set(folds_df["path"]) & set(test_df["path"])
    if overlap_paths:
        raise AssertionError(
            f"LEAKAGE: {len(overlap_paths)} test images appear in train/val folds, "
            f"e.g. {sorted(overlap_paths)[:3]}"
        )
    overlap_groups = set(folds_df["group"]) & set(test_df["group"])
    if overlap_groups:
        raise AssertionError(
            f"LEAKAGE: {len(overlap_groups)} near-duplicate groups span test and "
            f"train/val, e.g. groups {sorted(overlap_groups)[:5]}"
        )


def assert_folds_group_disjoint(folds_df: pd.DataFrame) -> None:
    """Each near-duplicate group must live entirely inside one fold."""
    spread = folds_df.groupby("group")["fold"].nunique()
    bad = spread[spread > 1]
    if len(bad):
        raise AssertionError(
            f"LEAKAGE: {len(bad)} groups span multiple CV folds, e.g. {list(bad.index[:5])}"
        )


def train_val_from_folds(folds_df: pd.DataFrame, val_fold: int):
    val = folds_df[folds_df["fold"] == val_fold].reset_index(drop=True)
    train = folds_df[folds_df["fold"] != val_fold].reset_index(drop=True)
    assert set(train["group"]).isdisjoint(set(val["group"])), (
        f"LEAKAGE: train/val share near-duplicate groups (val_fold={val_fold})"
    )
    return train, val
