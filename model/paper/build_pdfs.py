"""Render the paper and the context document to PDF, then mirror context.pdf to the repo root.

    python3 model/paper/build_pdfs.py            # figures are NOT regenerated
    python3 model/paper/build_pdfs.py --figures  # regenerate figures first

Chrome's headless print engine is used because the machine has no LaTeX or pandoc.
The root-level context.pdf is a copy, not the source of truth -- it is rewritten from
model/paper/context.pdf on every build, so the two can never drift apart.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent.parent

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
CHROME_FALLBACKS = [
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
]

# source .html -> output .pdf (relative to this directory)
DOCUMENTS = {
    "paper.html": "Second-Life-AI-paper.pdf",
    "tech-spec.html": "context.pdf",
}

# outputs copied to the repo root for convenience
MIRROR_TO_ROOT = ["context.pdf"]


def find_chrome() -> Path:
    for candidate in [CHROME, *CHROME_FALLBACKS]:
        if candidate.exists():
            return candidate
    sys.exit("No Chrome/Chromium found — needed to render HTML to PDF.")


def render(chrome: Path, source: Path, out: Path) -> None:
    subprocess.run(
        [str(chrome), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={out}", "--virtual-time-budget=10000",
         source.as_uri()],
        check=True, capture_output=True,
    )
    if not out.exists() or out.stat().st_size == 0:
        sys.exit(f"Chrome produced no output for {source.name}")
    print(f"  {out.relative_to(REPO)}  ({out.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", action="store_true",
                    help="regenerate figures from the committed data first")
    args = ap.parse_args()

    if args.figures:
        print("Regenerating figures:")
        subprocess.run([sys.executable, str(PAPER / "make_figures.py")], check=True)

    chrome = find_chrome()
    print("Rendering:")
    for src, dst in DOCUMENTS.items():
        render(chrome, PAPER / src, PAPER / dst)

    print("Mirroring to repo root:")
    for name in MIRROR_TO_ROOT:
        shutil.copy2(PAPER / name, REPO / name)
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
