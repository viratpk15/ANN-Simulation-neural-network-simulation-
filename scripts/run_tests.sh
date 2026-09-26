#!/usr/bin/env bash
# Run the complete test suite: backend pytest + frontend vitest + builds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "== backend tests =="
cd "$ROOT/backend"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
$PY -m pytest tests -q

echo "== frontend tests =="
cd "$ROOT/frontend"
npx vitest run

echo "== frontend production build =="
npm run build

echo "ALL GREEN ✅"
