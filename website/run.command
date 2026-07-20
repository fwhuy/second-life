#!/bin/bash
# Second Life AI — double-click to run on macOS. Builds the environment the
# first time (needs internet once), then serves the site and opens it.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[setup] first run — building the Python environment (a few minutes)…"
  python3 -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q torch torchvision timm scikit-learn pandas pyyaml pillow flask
fi
( sleep 2 && open http://127.0.0.1:5001 ) &
.venv/bin/python app.py "$@"
