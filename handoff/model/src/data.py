"""Datasets, transforms, and the seeded group-aware stratified splits.

Split design (Rules 1-3):
- Test (~15%) is carved out first with StratifiedGroupKFold, so no
  near-duplicate group spans test and the rest. Written once to
  data/splits/test.csv and never read again outside `evaluate --final-eval`.
- The remaining images get a `fold` column (0-4) via StratifiedGroupKFold.
  Single runs use fold 0 as validation; kfold.py rotates it.
- Augmentation exists only inside the train transform — split happens first,
  always (leakage_experiment.py Arm B deliberately breaks this to measure it).
- cfg['aug_boost_classes'] gives the named classes a harder train transform
  (default off). Intended for `trash`: 137 images vs paper's 594, so the
  weighted sampler already replays each trash image ~4x per epoch and a
  stronger transform is what stops those replays being near-identical.

CLI: python -m src.data --build-splits
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

from .utils import (
    CLASS_TO_IDX,
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    REPO_ROOT,
    assert_folds_group_disjoint,
    assert_no_leakage,
    seed_everything,
    train_val_from_folds,
)

TEST_KFOLD_SPLITS = 7  # 1/7 ≈ 14.3% of groups → test
N_FOLDS = 5
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def build_splits(splits_dir: Path, seed: int = SPLIT_SEED) -> None:
    groups_csv = splits_dir / "groups.csv"
    if not groups_csv.exists():
        raise FileNotFoundError(f"{groups_csv} missing — run `python -m src.dedup` first")
    df = pd.read_csv(groups_csv)

    y = df["label"].to_numpy()
    g = df["group"].to_numpy()

    # 1) quarantine test: first fold of a 7-way stratified group split
    sgkf = StratifiedGroupKFold(n_splits=TEST_KFOLD_SPLITS, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(df, y, g))
    test = df.iloc[test_idx].reset_index(drop=True)
    trainval = df.iloc[trainval_idx].reset_index(drop=True)

    # 2) assign 5 CV folds over the remainder
    sgkf5 = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    trainval["fold"] = -1
    for k, (_, val_idx) in enumerate(sgkf5.split(trainval, trainval["label"], trainval["group"])):
        trainval.loc[val_idx, "fold"] = k
    assert (trainval["fold"] >= 0).all()

    assert_no_leakage(trainval, test)
    assert_folds_group_disjoint(trainval)

    splits_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(splits_dir / "test.csv", index=False)
    trainval.to_csv(splits_dir / "folds.csv", index=False)

    print(f"\nSplit summary (seed={seed}):")
    print(f"  test:      {len(test):4d} images, {test['group'].nunique()} groups (QUARANTINED)")
    print(f"  train+val: {len(trainval):4d} images across {N_FOLDS} folds")
    print("\nPer-class counts:")
    table = pd.crosstab(df["label"], df["path"].isin(test["path"]).map({True: "test", False: "trainval"}))
    print(table.to_string())
    print("\nAssertions passed: no image and no near-duplicate group spans splits.")


# ---------------------------------------------------------------------------
# Transforms (Rule 4: ImageNet normalization everywhere)
# ---------------------------------------------------------------------------

def build_train_transform(cfg: dict, *, boost: bool = False) -> T.Compose:
    """Train-side transform. `boost=True` is the harder variant for the classes
    named in cfg['aug_boost_classes'] (see BOOST_NOTE below)."""
    size = cfg["img_size"]
    aug = cfg.get("aug", "basic")
    scale = (0.4, 1.0) if boost else (0.6, 1.0)
    if aug == "basic":
        ops = [
            T.RandomResizedCrop(size, scale=scale),
            T.RandomHorizontalFlip(),
            T.RandomRotation(30 if boost else 15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08) if boost
            else T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    elif aug == "modern":
        ops = [
            T.RandomResizedCrop(size, scale=scale),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
        ]
    else:
        raise ValueError(f"unknown aug mode {aug!r}")
    if boost:
        # Geometric/occlusion only. Deliberately no per-class grayscale or hue
        # rotation beyond the above: colour is a real signal for the other five
        # classes, and stripping it from one class invents a spurious cue.
        ops[1:1] = [
            T.RandomVerticalFlip(),
            T.RandomPerspective(distortion_scale=0.3, p=0.5),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        ]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    erasing = cfg.get("random_erasing", 0)
    if boost:
        erasing = max(erasing, 0.25)
    if erasing > 0:
        ops.append(T.RandomErasing(p=erasing))
    return T.Compose(ops)


def build_eval_transform(img_size: int) -> T.Compose:
    resize = int(img_size * 1.14)  # standard ~256/224 crop ratio
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset + loaders
# ---------------------------------------------------------------------------

class TrashDataset(Dataset):
    """Images + labels. Optionally applies `boost_transform` to `boost_classes`.

    The per-class branch lives here rather than in the transform so it is
    structurally impossible for an eval dataset (built with one transform and no
    boost) to pick up training augmentation.
    """

    def __init__(self, frame: pd.DataFrame, transform, boost_transform=None, boost_classes=()):
        self.paths = [REPO_ROOT / p for p in frame["path"]]
        self.targets = torch.tensor([CLASS_TO_IDX[c] for c in frame["label"]])
        self.transform = transform
        self.boost_transform = boost_transform
        boost_idx = {CLASS_TO_IDX[c] for c in boost_classes}
        self.boost = (
            torch.tensor([int(t) in boost_idx for t in self.targets], dtype=torch.bool)
            if boost_idx and boost_transform is not None
            else None
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        transform = self.boost_transform if (self.boost is not None and self.boost[i]) else self.transform
        return transform(img), self.targets[i]


def class_weights(frame: pd.DataFrame) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex(CLASSES).to_numpy(dtype=np.float64)
    w = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32)


def resolve_workers(cfg: dict) -> int:
    """`num_workers: auto` sizes the pool to the machine actually running the job.

    Capped at 16: beyond that the workers contend for memory bandwidth and the returns
    on JPEG decode go flat. One core is left for the main process.
    """
    requested = cfg.get("num_workers", 4)
    if requested != "auto":
        return int(requested)
    return max(0, min(16, (os.cpu_count() or 4) - 1))


def build_loaders(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame):
    boost_classes = [c for c in cfg.get("aug_boost_classes") or [] if c in CLASSES]
    unknown = set(cfg.get("aug_boost_classes") or []) - set(CLASSES)
    if unknown:
        raise ValueError(f"aug_boost_classes names unknown classes: {sorted(unknown)}")
    train_ds = TrashDataset(
        train_df, build_train_transform(cfg),
        boost_transform=build_train_transform(cfg, boost=True) if boost_classes else None,
        boost_classes=boost_classes)
    val_ds = TrashDataset(val_df, build_eval_transform(cfg["img_size"]))
    if boost_classes:
        n = int(train_ds.boost.sum())
        print(f"aug boost on {boost_classes}: {n}/{len(train_ds)} train images get the harder transform")

    sampler = None
    shuffle = True
    if cfg.get("class_weighting") == "weighted_sampler":
        w = class_weights(train_df)
        sample_w = w[train_ds.targets]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)
        shuffle = False

    workers = resolve_workers(cfg)
    common = dict(num_workers=workers,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers > 0)
    if workers > 0:
        # Deeper queue so the GPU never waits on JPEG decode at large batch sizes.
        common["prefetch_factor"] = cfg.get("prefetch_factor", 4)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=shuffle, sampler=sampler,
        drop_last=True, **common)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, **common)
    return train_loader, val_loader


def get_train_val_frames(cfg: dict, val_fold: int = 0):
    from .utils import load_split_frames

    folds, _test = load_split_frames(REPO_ROOT / cfg["splits_dir"])
    return train_val_from_folds(folds, val_fold)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-splits", action="store_true")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    if args.build_splits:
        seed_everything(args.seed)
        build_splits(REPO_ROOT / args.splits_dir, seed=args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
