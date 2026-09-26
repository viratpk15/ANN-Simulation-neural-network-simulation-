#!/usr/bin/env python3
"""Produce a clean, portable ZIP of the project (no node_modules, venvs,
build artifacts, runtime data, caches or secrets)."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "neural-network-simulator-final.zip"

EXCLUDE_DIRS = {
    "node_modules", "dist", "__pycache__", ".pytest_cache", ".venv", "venv",
    ".git", ".idea", ".vscode", ".mypy_cache", ".ruff_cache", "uploads", "runs",
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".db", ".pt", ".tsbuildinfo", ".DS_Store"}
EXCLUDE_EXACT = {".env", ".env.local"}


def main() -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / ".gitkeep").touch()
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
            for fn in sorted(filenames):
                if fn in EXCLUDE_EXACT or Path(fn).suffix in EXCLUDE_EXTS:
                    continue
                full = Path(dirpath) / fn
                rel = full.relative_to(ROOT.parent)
                if rel.parts[:2] == ("neural-network-simulator", "data") and fn != ".gitkeep":
                    continue
                z.write(full, rel)
                count += 1
    print(f"wrote {OUT} ({count} files)")


if __name__ == "__main__":
    main()
