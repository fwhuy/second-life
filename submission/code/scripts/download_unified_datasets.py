"""Download and organize three public waste datasets into a six-class archive.

Compatible labels are stored in data/unified_waste/included/<class>. Labels
that cannot be mapped honestly are retained under excluded/<original-label>.
Exact duplicate pixel content is stored once and recorded in manifest.csv.
Existing TrashNet splits are never modified by this script.
"""

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps


MODEL_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = MODEL_ROOT / "data" / "unified_waste"
HF_CACHE = MODEL_ROOT / "data" / ".hf_cache"
MANIFEST = OUTPUT_ROOT / "manifest.csv"

# Pinned so the corpus is byte-reproducible. Filenames encode the enumeration order
# of each dataset (`<source>__<index>__<digest>.jpg`), and the committed splits in
# data/splits_unified reference those filenames — so an upstream reorder would leave
# every split path dangling. Only change these deliberately, then rebuild the splits.
REVISIONS = {
    "garbage_classification": "1022529d853bd0b86025876e3c2daee5d2357235",
    "realwaste": "c626504f80c98ffe0d437e8ebf65a642e8d9fdba",
    "trashnet": "94cd17ebfd6702bf62281d3c89f22ff649815b46",
    "garbage_v2": "30af9027084bc44f66011e7e37bf852392d2f34d",
    "recycling11": "e2e03c91c385e8d1a758389cdb20cf9c024f6cbf",
}

CANONICAL = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
MAPS = {
    "garbage_classification": {
        "cardboard": "cardboard", "glass": "glass", "metal": "metal",
        "paper": "paper", "plastic": "plastic", "trash": "trash",
    },
    "realwaste": {
        "cardboard": "cardboard", "glass": "glass", "metal": "metal",
        "paper": "paper", "plastic": "plastic",
        "miscellaneous trash": "trash",
    },
    "trashnet": {label: label for label in CANONICAL},
    # Same ten-label schema as garbage_classification but a larger export; heavy overlap
    # is expected and handled by pixel-hash dedup plus the near-duplicate grouping.
    "garbage_v2": {
        "cardboard": "cardboard", "glass": "glass", "metal": "metal",
        "paper": "paper", "plastic": "plastic", "trash": "trash",
    },
    # Eleven material-level labels. Only the six that map without inventing a claim are
    # taken: composites (takeaway cups, disposable plates, paper towel) and polystyrene
    # sit across the recyclable/non-recyclable line depending on the local scheme, so
    # they stay in excluded/ rather than being guessed into a class.
    "recycling11": {
        "aluminium": "metal", "cardboard": "cardboard", "glass": "glass",
        "paper": "paper", "hard plastic": "plastic", "soft plastics": "plastic",
    },
}

FIELDS = [
    "source", "source_index", "original_label", "mapped_label", "included",
    "relative_path", "pixel_sha256", "width", "height", "status",
    "duplicate_of",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def normalized_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.load()
    return image


def pixel_digest(image: Image.Image) -> str:
    h = hashlib.sha256()
    h.update(f"{image.width}x{image.height}:{image.mode}".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def store(records, seen, source, index, original_label, image):
    original = original_label.strip()
    mapped = MAPS[source].get(original.lower())
    included = mapped in CANONICAL
    bucket = OUTPUT_ROOT / ("included" if included else "excluded")
    label_dir = bucket / (mapped if included else safe_name(original))
    image = normalized_image(image)
    digest = pixel_digest(image)
    # Deduplicate only within the same final label. The same pixels carrying
    # conflicting labels must remain visible for the later label audit.
    dedup_key = (digest, mapped if included else f"excluded:{safe_name(original)}")
    duplicate_of = seen.get(dedup_key, "")
    status = "duplicate" if duplicate_of else "stored"
    relative = duplicate_of
    if not duplicate_of:
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{source}__{index:06d}__{digest[:12]}.jpg"
        path = label_dir / filename
        image.save(path, "JPEG", quality=95, optimize=True)
        # as_posix: the manifest is joined against split CSVs, so separators must be
        # identical no matter which OS built the corpus.
        relative = path.relative_to(MODEL_ROOT).as_posix()
        seen[dedup_key] = relative
    records.append({
        "source": source,
        "source_index": index,
        "original_label": original,
        "mapped_label": mapped or "",
        "included": int(included),
        "relative_path": relative,
        "pixel_sha256": digest,
        "width": image.width,
        "height": image.height,
        "status": status,
        "duplicate_of": duplicate_of,
    })


def class_name(dataset, raw_label):
    feature = dataset.features["label"]
    return feature.int2str(raw_label) if hasattr(feature, "int2str") else str(raw_label)


def import_hf_dataset(repo_id, source, records, seen):
    print(f"Downloading {repo_id} ...", flush=True)
    dataset = load_dataset(repo_id, split="train", cache_dir=str(HF_CACHE),
                           revision=REVISIONS[source])
    for index, row in enumerate(dataset):
        store(records, seen, source, index, class_name(dataset, row["label"]), row["image"])
        if (index + 1) % 1000 == 0:
            print(f"  {source}: {index + 1}/{len(dataset)}", flush=True)


def import_trashnet(records, seen):
    print("Downloading garythung/trashnet ...", flush=True)
    archive = hf_hub_download(
        repo_id="garythung/trashnet", repo_type="dataset",
        filename="dataset-resized.zip", cache_dir=str(HF_CACHE),
        revision=REVISIONS["trashnet"])
    index = 0
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(
            name for name in bundle.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
            and name.startswith("dataset-resized/")
        )
        for name in members:
            label = Path(name).parent.name
            with bundle.open(name) as handle:
                image = Image.open(io.BytesIO(handle.read()))
                store(records, seen, "trashnet", index, label, image)
            index += 1
    print(f"  trashnet: {index}/{index}", flush=True)


def write_manifest(records):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    included = sum(r["included"] and r["status"] == "stored" for r in records)
    excluded = sum(not r["included"] and r["status"] == "stored" for r in records)
    duplicates = sum(r["status"] == "duplicate" for r in records)
    print(f"\nStored compatible images: {included}")
    print(f"Stored excluded images:   {excluded}")
    print(f"Exact duplicates skipped: {duplicates}")

    # A source contributing zero usable images means its label names did not match the
    # MAPS keys — a silent schema drift upstream. Fail loudly here rather than let it
    # look like a smaller-than-expected corpus three steps later.
    print("\nPer-source stored/included:")
    dead = []
    for source in MAPS:
        rows = [r for r in records if r["source"] == source]
        kept = sum(r["included"] and r["status"] == "stored" for r in rows)
        print(f"  {source:24s} {kept:6d} included of {len(rows)} seen")
        if rows and not kept:
            labels = sorted({r["original_label"] for r in rows})[:8]
            dead.append(f"{source} (saw labels: {labels})")
    if dead:
        raise SystemExit(
            "\nERROR: these sources mapped nothing — their upstream label names likely "
            f"changed:\n  " + "\n  ".join(dead) +
            "\nFix the MAPS entry in this file, then re-run."
        )
    print(f"\nManifest: {MANIFEST}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records, seen = [], {}
    import_hf_dataset(
        "omasteam/waste-garbage-management-dataset",
        "garbage_classification", records, seen)
    import_hf_dataset("shahzaibvohra/realwaste", "realwaste", records, seen)
    import_hf_dataset("steveharianto/waste-garbage-management-dataset",
                      "garbage_v2", records, seen)
    import_hf_dataset("viola77data/recycling-dataset", "recycling11", records, seen)
    import_trashnet(records, seen)
    write_manifest(records)


if __name__ == "__main__":
    main()
