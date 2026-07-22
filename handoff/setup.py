"""One-command setup: environment, corpus, splits, verification. Windows/macOS/Linux.

    python setup.py

Safe to re-run — every step is idempotent, and the pipeline is seeded, so it produces the
same splits on any machine.

The splits shipped in this handoff were built from 3 datasets; the download now pulls 5,
so the splits MUST be rebuilt. This script does that for you rather than leaving it as a
step to forget.
"""

import csv
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"
VENV = MODEL / ".venv"
WINDOWS = platform.system() == "Windows"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")


def step(title: str) -> None:
    print(f"\n{'=' * 70}\n==> {title}\n{'=' * 70}", flush=True)


def run(args, **kw) -> None:
    """Run a step, streaming output. Any failure stops setup with a clear message."""
    printable = " ".join(str(a) for a in args)
    print(f"$ {printable}", flush=True)
    result = subprocess.run(args, cwd=MODEL, **kw)
    if result.returncode:
        raise SystemExit(
            f"\nFAILED: {printable}\n"
            f"Fix the error above, then re-run `python setup.py` — completed steps are "
            f"skipped automatically."
        )


EXPECTED_SOURCES = {
    "garbage_classification", "realwaste", "trashnet", "garbage_v2", "recycling11",
}


def corpus_is_complete() -> tuple[bool, str]:
    """Trust only a finished five-source manifest whose stored files still exist.

    Counting images is unsafe: an interrupted download or the older three-source corpus
    can both contain thousands of images. The downloader writes the manifest only after
    all sources finish, so it doubles as the completion marker.
    """
    manifest = MODEL / "data" / "unified_waste" / "manifest.csv"
    if not manifest.is_file():
        return False, "manifest missing (new or interrupted setup)"
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sources = {row["source"] for row in rows}
    missing_sources = EXPECTED_SOURCES - sources
    if missing_sources:
        return False, f"manifest missing sources: {', '.join(sorted(missing_sources))}"
    stored = [row["relative_path"] for row in rows if row["status"] == "stored"]
    missing_files = [path for path in stored if not (MODEL / path).is_file()]
    if missing_files:
        return False, f"{len(missing_files)} manifest files missing"
    return True, f"complete five-source manifest ({len(stored)} stored images)"


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10+ required, found {platform.python_version()}")
    print(f"{platform.system()} | Python {platform.python_version()} | {MODEL}")

    step("Python environment")
    if not venv_python().exists():
        print(f"creating venv at {VENV}")
        venv.EnvBuilder(with_pip=True).create(VENV)
    else:
        print("venv already exists")
    py = str(venv_python())
    run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-q", "-r", "requirements.lock.txt"])
    run([py, "-c", "import torch; print('torch', torch.__version__,"
                   "'| cuda', torch.cuda.is_available())"])

    step("Download corpus (~2GB across 5 datasets — this is the slow part)")
    complete, detail = corpus_is_complete()
    if complete:
        print(f"corpus already present: {detail}; skipping download")
    else:
        print(f"corpus rebuild required: {detail}")
        run([py, "scripts/download_unified_datasets.py"])

    step("Recover TrashNet provenance (identifies the disguised spent-test images)")
    run([py, "scripts/map_trashnet_provenance.py"])

    step("Rebuild group-aware splits with the spent-test quarantine")
    run([py, "-m", "src.unified_data", "--build-splits"])

    step("Verify")
    run([py, "-m", "pytest", "tests/", "-q"])
    run([py, "scripts/preflight.py"])

    activate = r".venv\Scripts\activate" if WINDOWS else "source .venv/bin/activate"
    print(f"""
{'=' * 70}
Setup complete. Start training:

  cd model
  {activate}
  python -m src.train --config configs/unified_convnextv2_tiny_224.yaml --fold 0 --resume

Read RULES.md before changing anything about the splits.
{'=' * 70}""")


if __name__ == "__main__":
    main()
