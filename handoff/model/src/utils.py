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


def configure_performance(cfg: dict, device: torch.device):
    """Turn on the throughput settings that cost nothing in accuracy.

    Returns the autocast dtype. bfloat16 on Ampere+ (compute capability >= 8) needs no
    GradScaler and cannot overflow; older cards fall back to fp16 with a scaler.

    TF32 and cudnn.benchmark are pure wins here: TF32 only affects matmul/conv internals,
    and benchmark autotunes kernels for a fixed input shape, which is exactly our case.
    """
    if device.type != "cuda":
        return None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    props = torch.cuda.get_device_properties(0)
    bf16 = props.major >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    if not cfg.get("amp", True):
        dtype = None

    total_gb = props.total_memory / 1024**3
    print(f"gpu={props.name} vram={total_gb:.0f}GB sm={props.major}.{props.minor} "
          f"tf32=on cudnn.benchmark=on amp={'off' if dtype is None else str(dtype).split('.')[-1]}")
    return dtype


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
    # Present only for the unified corpus, which absorbed the old TrashNet test
    # set under disguised filenames. Absent for the original TrashNet splits.
    quarantine = splits_dir / "quarantine.csv"
    if quarantine.exists():
        assert_old_test_quarantined(folds, test, pd.read_csv(quarantine)["path"])
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


def test_spent_path(splits_dir: str | Path) -> Path:
    """Where the one-shot test-set guard lives, per split set.

    Keyed by split set so spending the old TrashNet test does not block the unified
    corpus, and vice versa. The original file keeps its historic name.
    """
    name = Path(splits_dir).name
    filename = "TEST_SPENT.json" if name == "splits" else f"TEST_SPENT_{name}.json"
    return REPO_ROOT / "reports" / filename


def assert_test_not_spent(splits_dir: str | Path) -> None:
    """Hard-fail if this test set has already had its single measurement.

    Rule 1 is that the test set is measured exactly once. That was documented but
    never enforced, which is a real hazard when an agent runs unattended for hours.
    To deliberately re-spend, delete the guard file by hand — it should take a
    conscious act, not a stray flag.
    """
    guard = test_spent_path(splits_dir)
    if not guard.exists():
        return
    record = json.loads(guard.read_text())
    raise AssertionError(
        f"TEST SET ALREADY SPENT for {splits_dir}. It was measured on "
        f"{record.get('timestamp', 'an earlier date')} at "
        f"{record.get('accuracy', '?')} accuracy using {record.get('checkpoint', '?')}. "
        f"A second measurement is model selection on test data. See {guard}."
    )


def mark_test_spent(splits_dir: str | Path, *, checkpoint: str, metrics: dict,
                    n_images: int) -> Path:
    guard = test_spent_path(splits_dir)
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text(json.dumps({
        "spent": True,
        "splits_dir": str(splits_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": checkpoint,
        "test_images": n_images,
        "accuracy": round(float(metrics.get("acc", 0)), 6),
        "macro_f1": round(float(metrics.get("macro_f1", 0)), 6),
        "reason": "Single pre-registered measurement. Unavailable for any further "
                  "model selection or confirmatory evaluation.",
    }, indent=2) + "\n")
    return guard


def assert_old_test_quarantined(folds_df: pd.DataFrame, test_df: pd.DataFrame,
                                quarantine_paths) -> None:
    """Hard-fail unless every image from the spent TrashNet test set sits in test.

    The unified corpus contains all 2,527 TrashNet images, 2,218 of them renamed
    to `garbage_classification__*.jpg` by exact-pixel deduplication, so a path
    check against data/splits/test.csv would miss them entirely. These paths come
    from scripts/map_trashnet_provenance.py, which recovers the true identity.
    """
    quarantine = set(quarantine_paths)
    leaked = quarantine & set(folds_df["path"])
    if leaked:
        raise AssertionError(
            f"LEAKAGE: {len(leaked)} images from the spent test set appear in train/val "
            f"folds, e.g. {sorted(leaked)[:3]}"
        )
    absent = quarantine - set(test_df["path"])
    if absent:
        raise AssertionError(
            f"QUARANTINE BREACH: {len(absent)} spent-test images are in neither test nor "
            f"folds — they must be held in test, not silently dropped, "
            f"e.g. {sorted(absent)[:3]}"
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
