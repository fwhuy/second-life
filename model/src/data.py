"""Datasets, transforms, and the seeded group-aware stratified splits.

Split design (Rules 1-3):
- Test (~15%) is carved out first with StratifiedGroupKFold, so no
  near-duplicate group spans test and the rest. Written once to
  data/splits/test.csv and never read again outside `evaluate --final-eval`.
- The remaining images get a `fold` column (0-4) via StratifiedGroupKFold.
  Single runs use fold 0 as validation; kfold.py rotates it.
- Augmentation exists only inside the train transform — split happens first,
  always (leakage_experiment.py Arm B deliberately breaks this to measure it).

CLI: python -m src.data --build-splits
"""

import argparse
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

def build_train_transform(cfg: dict) -> T.Compose:
    size = cfg["img_size"]
    aug = cfg.get("aug", "basic")
    if aug == "basic":
        ops = [
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    elif aug == "modern":
        ops = [
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
        ]
    else:
        raise ValueError(f"unknown aug mode {aug!r}")
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if cfg.get("random_erasing", 0) > 0:
        ops.append(T.RandomErasing(p=cfg["random_erasing"]))
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
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = [REPO_ROOT / p for p in frame["path"]]
        self.targets = torch.tensor([CLASS_TO_IDX[c] for c in frame["label"]])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.targets[i]


def class_weights(frame: pd.DataFrame) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex(CLASSES).to_numpy(dtype=np.float64)
    w = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32)


def build_loaders(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_ds = TrashDataset(train_df, build_train_transform(cfg))
    val_ds = TrashDataset(val_df, build_eval_transform(cfg["img_size"]))

    sampler = None
    shuffle = True
    if cfg.get("class_weighting") == "weighted_sampler":
        w = class_weights(train_df)
        sample_w = w[train_ds.targets]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)
        shuffle = False

    common = dict(num_workers=cfg.get("num_workers", 4),
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=cfg.get("num_workers", 4) > 0)
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
