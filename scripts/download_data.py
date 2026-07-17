"""Fetch TrashNet, verify per-class counts, dedup, quarantine test, build splits.

Sources, tried in order:
1. kagglehub: asdasdasasdas/garbage-classification (the course-referenced copy)
2. GitHub LFS zip of the original repo (garythung/trashnet, dataset-resized)

Ends by running near-duplicate grouping (src.dedup) and building the committed
seeded splits (src.data), including the leak assertions. Idempotent: skips any
stage whose output already exists.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import CLASSES, EXPECTED_COUNTS, REPO_ROOT  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
TRASHNET_ZIP_URL = (
    "https://media.githubusercontent.com/media/garythung/trashnet/master/data/dataset-resized.zip"
)


def current_counts() -> dict:
    return {
        c: sum(1 for p in (RAW_DIR / c).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if (RAW_DIR / c).is_dir() else 0
        for c in CLASSES
    }


def copy_class_dirs(source_root: Path) -> bool:
    """Find the 6 class dirs anywhere under source_root and copy into data/raw."""
    found = {}
    for d in source_root.rglob("*"):
        if d.is_dir() and d.name.lower() in CLASSES and d.name.lower() not in found:
            found[d.name.lower()] = d
    if len(found) != len(CLASSES):
        print(f"  source only contains class dirs: {sorted(found)}")
        return False
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for cls, src in found.items():
        dst = RAW_DIR / cls
        dst.mkdir(exist_ok=True)
        for f in src.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(f, dst / f.name)
    return True


def fetch_kagglehub() -> bool:
    try:
        import kagglehub
        print("Trying kagglehub: asdasdasasdas/garbage-classification ...")
        path = Path(kagglehub.dataset_download("asdasdasasdas/garbage-classification"))
        return copy_class_dirs(path)
    except Exception as e:
        print(f"  kagglehub failed: {e}")
        return False


def fetch_github_zip() -> bool:
    import urllib.request
    try:
        print(f"Trying GitHub LFS zip: {TRASHNET_ZIP_URL} ...")
        zip_path = REPO_ROOT / "data" / "dataset-resized.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TRASHNET_ZIP_URL, zip_path)
        extract_dir = REPO_ROOT / "data" / "_extract"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        ok = copy_class_dirs(extract_dir)
        shutil.rmtree(extract_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        return ok
    except Exception as e:
        print(f"  GitHub download failed: {e}")
        return False


def verify_counts(allow_mismatch: bool) -> None:
    counts = current_counts()
    print("\nPer-class counts (found / expected):")
    ok = True
    for c in CLASSES:
        exp = EXPECTED_COUNTS[c]
        mark = "OK" if counts[c] == exp else "MISMATCH"
        ok &= counts[c] == exp
        print(f"  {c:9s} {counts[c]:4d} / {exp:4d}  {mark}")
    print(f"  total     {sum(counts.values()):4d} / {sum(EXPECTED_COUNTS.values()):4d}")
    if not ok and not allow_mismatch:
        sys.exit(
            "Counts do not match canonical TrashNet. If your course provided a "
            "different variant, re-run with --allow-count-mismatch and note it in the paper."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="phash-only dedup (skip backbone embedding stage)")
    ap.add_argument("--force-splits", action="store_true",
                    help="rebuild groups.csv + splits even if they exist "
                         "(WARNING: changes the committed test quarantine)")
    args = ap.parse_args()

    if all(v > 0 for v in current_counts().values()):
        print("data/raw already populated — skipping download.")
    elif not (fetch_kagglehub() or fetch_github_zip()):
        sys.exit(
            "All download sources failed. Manual fallback: download "
            "https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification "
            "and place the 6 class folders under data/raw/<class>/."
        )

    verify_counts(args.allow_count_mismatch)

    if (SPLITS_DIR / "test.csv").exists() and not args.force_splits:
        print("\nCommitted splits already exist — leaving the test quarantine untouched.")
        from src.utils import load_split_frames
        load_split_frames(SPLITS_DIR)  # re-run leak assertions
        print("Leak assertions passed.")
        return

    from src.dedup import build_groups
    from src.data import build_splits
    build_groups(RAW_DIR, SPLITS_DIR, use_embeddings=not args.no_embeddings)
    build_splits(SPLITS_DIR)
    print("\nDone. Commit data/splits/*.csv so every machine uses identical splits.")


if __name__ == "__main__":
    main()
