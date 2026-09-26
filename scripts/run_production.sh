#!/usr/bin/env bash
# Build the SPA and run everything on a single port (8000).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend" && npm run build
cd "$ROOT/backend"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
exec $PY -m uvicorn app.main:app --host 0.0.0.0 --port "${BACKEND_PORT:-8000}"
