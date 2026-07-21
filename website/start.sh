#!/usr/bin/env bash
# Second Life AI — start the website (macOS / Linux).
#   ./start.sh                # build env if needed, serve on :5001
#   ./start.sh --port 8000    # any app.py flags pass straight through
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "[setup] first run — building the Python environment (a few minutes)…"
  python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q torch torchvision timm scikit-learn pandas pyyaml pillow flask
fi

URL="http://127.0.0.1:5001"
# open a browser once the server is up, without blocking the server itself
( for _ in $(seq 1 30); do
    curl -sf "$URL/api/model" >/dev/null 2>&1 && break; sleep 1
  done
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  fi ) &

echo "[Second Life AI] serving at $URL  (Ctrl+C to stop)"
exec "$PY" app.py "$@"
