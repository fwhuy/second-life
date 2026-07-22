#!/usr/bin/env bash
# Assemble and push the Hugging Face Space.
#
#   ./deploy/hf-space/push.sh <owner>/<space-name>
#
# The Space is a *build output*, not a second source tree: this script stages a
# fresh copy of the site, the inference code, and the checkpoint, then force-
# pushes it. Nothing is ever edited on the Space side, so the staging tree can
# be thrown away and rebuilt from this repo at any time.
#
# Prerequisites: git-lfs (`brew install git-lfs`), and a Hugging Face write
# token — run `huggingface-cli login`, or paste the token when git prompts for
# a password (username is your HF handle).

set -euo pipefail

SPACE="${1:-${HF_SPACE:-}}"
if [[ -z "$SPACE" ]]; then
  echo "usage: $0 <owner>/<space-name>   (e.g. fwhuy/second-life)" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CHECKPOINT="$ROOT/website/checkpoints/baseline_resnet50/fold0/best.pth"

command -v git-lfs >/dev/null 2>&1 || { echo "git-lfs not installed: brew install git-lfs" >&2; exit 1; }

# The weights are gitignored in this repo (too large for GitHub) and are the one
# input that cannot be regenerated from source. Fail loudly rather than shipping
# a Space that falls back to demo data.
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  echo "The Space needs the trained weights. Restore them before pushing." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
echo "Staging in $STAGE"

cp "$HERE/Dockerfile" "$HERE/requirements.txt" "$HERE/README.md" "$STAGE/"

# The site, minus the local virtualenv and bytecode caches.
rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
      --exclude '.gitignore' "$ROOT/website/" "$STAGE/website/"

# The training repo's inference code, which app.py imports as `src.*`.
mkdir -p "$STAGE/model"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/model/src/" "$STAGE/model/src/"

cd "$STAGE"
git init -q -b main
git lfs install --local >/dev/null
git lfs track "*.pth" "*.npz" >/dev/null
git add -A
git -c user.email="deploy@local" -c user.name="deploy" \
    commit -qm "Deploy Second Life AI ($(date -u +%Y-%m-%dT%H:%MZ))"

echo "Pushing to https://huggingface.co/spaces/$SPACE"
git remote add origin "https://huggingface.co/spaces/$SPACE"
git push --force origin main

echo
echo "Done. Build logs: https://huggingface.co/spaces/$SPACE  (first build ~5-10 min)"
