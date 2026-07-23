"""Sort the manually downloaded waste archives into one audited collection."""

import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_unified_datasets import OUTPUT_ROOT, store, write_manifest  # noqa: E402


DOWNLOADS = Path.home() / "Downloads"
SOURCES = [
    {
        "name": "garbage_classification",
        "archive": DOWNLOADS / "archive (1).zip",
        # The other two trees are resized copies of these same 12,259 images.
        "prefix": "original/",
    },
    {
        "name": "trashnet",
        "archive": DOWNLOADS / "archive (2).zip",
        # train/test/val each contain the same complete 2,527-image dataset.
        "prefix": "Garbage classification/train/",
    },
    {
        "name": "realwaste",
        "archive": DOWNLOADS / "realwaste.zip",
        "prefix": "realwaste-main/RealWaste/",
    },
]


def main():
    records, seen = [], {}
    for source in SOURCES:
        archive = source["archive"]
        if not archive.exists():
            raise FileNotFoundError(archive)
        print(f"Sorting {archive.name} as {source['name']} ...", flush=True)
        with zipfile.ZipFile(archive) as bundle:
            members = sorted(
                name for name in bundle.namelist()
                if name.startswith(source["prefix"])
                and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )
            for index, name in enumerate(members):
                with bundle.open(name) as handle:
                    image = Image.open(handle)
                    store(records, seen, source["name"], index,
                          Path(name).parent.name, image)
                if (index + 1) % 1000 == 0:
                    print(f"  {index + 1}/{len(members)}", flush=True)
        print(f"  {len(members)}/{len(members)}", flush=True)
    write_manifest(records)
    print(f"Sorted folders: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
