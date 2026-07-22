"""Recover which unified_waste files are TrashNet images — including the old test set.

The unified corpus silently absorbed TrashNet. `omasteam/waste-garbage-management-dataset`
contains it, so download_unified_datasets.py collapsed 2,218 of 2,527 TrashNet images as
exact pixel duplicates and kept only the copy named `garbage_classification__*.jpg`. All 361
images quarantined as the old test set are in there. Training on data/unified_waste/included
without this map means training on the test set.

Recovery is exact, not heuristic. `import_trashnet` walked `sorted(zip members)`, so
source_index is a deterministic function of the TrashNet filename. This script rebuilds that
ordering from the committed data/splits/groups.csv and refuses to write anything unless all
2,527 rows match with agreeing labels.

Output: data/unified_waste/trashnet_provenance.csv
CLI: python scripts/map_trashnet_provenance.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import REPO_ROOT  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "unified_waste" / "manifest.csv"
OUTPUT = REPO_ROOT / "data" / "unified_waste" / "trashnet_provenance.csv"
GROUPS_CSV = REPO_ROOT / "data" / "splits" / "groups.csv"
TEST_CSV = REPO_ROOT / "data" / "splits" / "test.csv"


def zip_member_order(groups: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the sorted zip-member ordering that produced each source_index.

    import_trashnet iterated `sorted(name for name in bundle.namelist() ...)` over
    members under `dataset-resized/`, incrementing an index per image. Reproducing
    that sort over the same 2,527 filenames reproduces the indices exactly.
    """
    filenames = groups["path"].str.rsplit("/", n=1).str[-1]
    members = "dataset-resized/" + groups["label"] + "/" + filenames
    order = pd.DataFrame({"member": members, "path": groups["path"], "label": groups["label"]})
    order = order.sort_values("member", kind="mergesort").reset_index(drop=True)
    order["source_index"] = order.index
    return order


def build_provenance() -> pd.DataFrame:
    groups = pd.read_csv(GROUPS_CSV)
    test = pd.read_csv(TEST_CSV)
    manifest = pd.read_csv(MANIFEST)

    order = zip_member_order(groups)
    trashnet = manifest[manifest["source"] == "trashnet"]

    merged = order.merge(trashnet, on="source_index", how="outer", indicator=True)

    # These three assertions ARE the correctness proof. A silent mismatch here puts
    # test images into training, so none of them may be downgraded to a warning.
    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched):
        raise AssertionError(
            f"{len(unmatched)} rows failed to match between groups.csv and the manifest "
            f"— the zip ordering assumption is broken, do not trust this map"
        )
    if len(merged) != len(groups):
        raise AssertionError(f"expected {len(groups)} rows, matched {len(merged)}")
    disagree = merged[merged["label"] != merged["original_label"]]
    if len(disagree):
        raise AssertionError(
            f"{len(disagree)} rows have a label that disagrees with the manifest, "
            f"e.g. {disagree[['path', 'label', 'original_label']].head(3).to_dict('records')}"
        )

    # A duplicate row's pixels live in the file it was deduplicated against.
    merged["unified_path"] = merged["relative_path"].where(
        merged["status"] == "stored", merged["duplicate_of"])
    test_paths = set(test["path"])
    merged["old_split"] = merged["path"].map(
        lambda p: "test" if p in test_paths else "trainval")

    out = merged[["path", "source_index", "label", "status", "unified_path", "old_split"]]
    return out.rename(columns={"path": "original_path", "label": "original_label"})


def main() -> None:
    provenance = build_provenance()
    provenance.to_csv(OUTPUT, index=False)

    quarantined = provenance[provenance["old_split"] == "test"]
    disguised = quarantined["unified_path"].str.contains("garbage_classification").sum()
    print(f"Matched {len(provenance)}/{len(provenance)} TrashNet images, labels all agree.")
    print(f"\nOld quarantined test images: {len(quarantined)}")
    print(f"  present in unified_waste/included: {quarantined['unified_path'].notna().sum()}")
    print(f"  disguised as garbage_classification: {disguised}")
    print(f"  still named trashnet: {len(quarantined) - disguised}")
    print(f"\nAll TrashNet images found in the unified corpus: "
          f"{provenance['unified_path'].notna().sum()}/{len(provenance)}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
