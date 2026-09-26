#!/usr/bin/env bash
# Start the FastAPI backend with auto-reload (dev mode).
set -euo pipefail
cd "$(dirname "$0")/../backend"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  ./.venv/bin/pip install --quiet -r requirements.txt
fi
exec ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${BACKEND_PORT:-8000}" --reload
