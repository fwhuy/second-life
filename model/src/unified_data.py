"""Unified three-source corpus: inventory, filters, and leakage-free splits.

data/unified_waste/included holds 11,359 six-class images pooled from three public
datasets. It silently contains all 2,527 TrashNet images — 2,218 of them renamed to
`garbage_classification__*.jpg` by exact-pixel deduplication — including every image
of the spent 361-image test set. Splitting this corpus naively trains on the test set.

Three guarantees, in the order they are enforced:

1. `scripts/map_trashnet_provenance.py` recovers the true identity of every TrashNet
   file. Its output drives everything below.
2. Every near-duplicate group containing a spent-test image is placed in the new test
   set *first*; the rest of test is topped up around it.
3. The surviving spent-test paths are written to quarantine.csv, and
   `load_split_frames` refuses to return a split that violates it — so the guarantee
   holds at every entry point, not just here.

Splits go to data/splits_unified/. The committed TrashNet splits in data/splits/ are
never read for training and never modified.

CLI: python -m src.unified_data --build-splits
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import N_FOLDS, SPLIT_SEED
from .dedup import group_frame
from .utils import (
    CLASSES,
    REPO_ROOT,
    assert_folds_group_disjoint,
    assert_no_leakage,
    assert_old_test_quarantined,
    seed_everything,
)

UNIFIED_ROOT = REPO_ROOT / "data" / "unified_waste"
INCLUDED_DIR = UNIFIED_ROOT / "included"
MANIFEST = UNIFIED_ROOT / "manifest.csv"
PROVENANCE = UNIFIED_ROOT / "trashnet_provenance.csv"
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits_unified"

TEST_FRACTION = 0.15
MIN_SIDE = 128  # below this an image carries no usable detail at 224px+ training


# ---------------------------------------------------------------------------
# Inventory + filters
# ---------------------------------------------------------------------------

def scan_unified(root: Path = INCLUDED_DIR) -> pd.DataFrame:
    """Inventory included/<class>/<source>__<index>__<digest>.jpg."""
    rows = []
    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {cls_dir}")
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                # as_posix, not str: on Windows str() yields backslashes, which would
                # not match the committed CSVs and would break path parsing downstream.
                rows.append({"path": p.relative_to(REPO_ROOT).as_posix(),
                             "label": cls,
                             "source": p.name.split("__")[0]})
    return pd.DataFrame(rows)


def conflicting_label_paths(manifest: pd.DataFrame) -> set:
    """Files whose exact pixels also appear under a different label.

    download_unified_datasets.py deduplicates within a label but deliberately keeps
    cross-label collisions visible. Those are genuine contradictions — one of the two
    labels is wrong and we cannot tell which — so every copy is dropped rather than
    guessed at.
    """
    stored = manifest[(manifest["included"] == 1) & (manifest["status"] == "stored")]
    per_hash = stored.groupby("pixel_sha256")["mapped_label"].nunique()
    bad = set(per_hash[per_hash > 1].index)
    return set(stored.loc[stored["pixel_sha256"].isin(bad), "relative_path"])


def build_inventory(splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scanned corpus with true `source`, minus filtered images. Returns (kept, dropped)."""
    df = scan_unified()
    manifest = pd.read_csv(MANIFEST)
    provenance = pd.read_csv(PROVENANCE)

    stored = manifest[(manifest["included"] == 1) & (manifest["status"] == "stored")]
    dims = stored.set_index("relative_path")[["width", "height"]]
    df = df.join(dims, on="path")
    if df["width"].isna().any():
        raise AssertionError(
            f"{int(df['width'].isna().sum())} images on disk are absent from manifest.csv "
            f"— the corpus and its manifest have drifted apart"
        )

    # A file that is pixel-identical to a TrashNet image IS a TrashNet image, whatever
    # the filename says. Correcting `source` here makes both the domain stratification
    # and the per-source accuracy report mean what they claim to.
    trashnet_paths = set(provenance["unified_path"].dropna())
    df.loc[df["path"].isin(trashnet_paths), "source"] = "trashnet"

    df["drop_reason"] = ""
    too_small = df[["width", "height"]].min(axis=1) < MIN_SIDE
    df.loc[too_small, "drop_reason"] = f"min_side<{MIN_SIDE}"
    conflicts = df["path"].isin(conflicting_label_paths(manifest))
    df.loc[conflicts & (df["drop_reason"] == ""), "drop_reason"] = "conflicting_label"

    dropped = df[df["drop_reason"] != ""].copy()
    kept = df[df["drop_reason"] == ""].drop(columns=["drop_reason"]).reset_index(drop=True)

    splits_dir.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(splits_dir / "excluded.csv", index=False)
    print(f"Inventory: {len(df)} images, dropped {len(dropped)} "
          f"({dropped['drop_reason'].value_counts().to_dict()}), kept {len(kept)}")
    print(f"  by source: {kept['source'].value_counts().to_dict()}")
    return kept, dropped


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _split_indices(frame: pd.DataFrame, n_splits: int, seed: int):
    """StratifiedGroupKFold on label x source, falling back to label alone.

    Fine-grained strata keep all three photo domains present in every fold, but a rare
    combination (trashnet/metal has 11 images) can be too small to support n_splits.
    """
    for column in ("stratum", "label"):
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = list(sgkf.split(frame, frame[column], frame["group"]))
            if column == "label":
                print(f"  note: stratified on label alone (n_splits={n_splits} too large "
                      f"for the rarest label x source combination)")
            return folds
        except ValueError:
            continue
    raise ValueError(f"cannot build {n_splits} stratified group splits from {len(frame)} rows")


def build_unified_splits(splits_dir: Path = DEFAULT_SPLITS_DIR, seed: int = SPLIT_SEED,
                         use_embeddings: bool = True) -> None:
    df, _dropped = build_inventory(splits_dir)
    provenance = pd.read_csv(PROVENANCE)

    print("\nGrouping near-duplicates across all three sources...")
    df = group_frame(df, use_embeddings=use_embeddings)
    df["stratum"] = df["label"] + "|" + df["source"]

    # 1) Quarantine first. Any group touching a spent-test image goes to test whole,
    #    so no near-duplicate of a test image can reach training either.
    spent = set(provenance.loc[provenance["old_split"] == "test", "unified_path"].dropna())
    surviving = spent & set(df["path"])
    forced_groups = set(df.loc[df["path"].isin(surviving), "group"])
    is_forced = df["group"].isin(forced_groups)
    print(f"\nSpent-test images surviving filters: {len(surviving)}/{len(spent)}")
    print(f"  their near-duplicate groups pull in {int(is_forced.sum())} images total")

    # 2) Top up to TEST_FRACTION from what is left, if the forced set falls short.
    target = int(round(TEST_FRACTION * len(df)))
    rest = df[~is_forced].reset_index(drop=True)
    shortfall = target - int(is_forced.sum())
    if shortfall > 0 and len(rest) > shortfall:
        n_splits = max(2, round(len(rest) / shortfall))
        _, extra_idx = _split_indices(rest, n_splits, seed)[0]
        extra_mask = rest.index.isin(extra_idx)
        test = pd.concat([df[is_forced], rest[extra_mask]], ignore_index=True)
        trainval = rest[~extra_mask].reset_index(drop=True)
    else:
        print(f"  forced set already covers the {TEST_FRACTION:.0%} target — no top-up")
        test, trainval = df[is_forced].reset_index(drop=True), rest

    # 3) Five CV folds over the remainder.
    trainval = trainval.copy()
    trainval["fold"] = -1
    for k, (_, val_idx) in enumerate(_split_indices(trainval, N_FOLDS, seed)):
        trainval.loc[val_idx, "fold"] = k
    assert (trainval["fold"] >= 0).all()

    assert_no_leakage(trainval, test)
    assert_folds_group_disjoint(trainval)
    assert_old_test_quarantined(trainval, test, surviving)

    quarantine = pd.DataFrame({"path": sorted(surviving)})
    splits_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(splits_dir / "groups.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)
    trainval.to_csv(splits_dir / "folds.csv", index=False)
    quarantine.to_csv(splits_dir / "quarantine.csv", index=False)

    report(df, trainval, test, surviving)
    print(f"\nWrote groups.csv, folds.csv, test.csv, quarantine.csv to {splits_dir}")


def report(df, trainval, test, surviving) -> None:
    print(f"\nSplit summary ({len(df)} images, {df['group'].nunique()} groups):")
    print(f"  test:      {len(test):5d} images ({len(test) / len(df):.1%}), "
          f"{test['group'].nunique()} groups (QUARANTINED)")
    print(f"  train+val: {len(trainval):5d} images across {N_FOLDS} folds")
    print("\nPer-class counts:")
    print(pd.crosstab(df["label"], df["path"].isin(set(test["path"])).map(
        {True: "test", False: "trainval"})).to_string())
    print("\nPer-source counts:")
    print(pd.crosstab(df["source"], df["path"].isin(set(test["path"])).map(
        {True: "test", False: "trainval"})).to_string())
    print("\nSource mix per fold (each fold must see all three domains):")
    print(pd.crosstab(trainval["fold"], trainval["source"]).to_string())
    print(f"\nAssertions passed: no image or near-duplicate group spans splits; "
          f"all {len(surviving)} surviving spent-test images are held in test.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-splits", action="store_true")
    ap.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--no-embeddings", action="store_true",
                    help="phash-only grouping; much faster, misses same-object-different-angle")
    args = ap.parse_args()
    if args.build_splits:
        seed_everything(args.seed)
        build_unified_splits(Path(args.splits_dir), seed=args.seed,
                             use_embeddings=not args.no_embeddings)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
