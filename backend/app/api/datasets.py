"""Dataset endpoints: built-ins, previews, CSV upload & analysis."""
from __future__ import annotations

import re

import pandas as pd
from fastapi import APIRouter, UploadFile, File

from ..ml import datasets as ds
from .deps import err

router = APIRouter()

# Upload IDs are minted in `ml.datasets.save_upload` as `uuid.uuid4().hex[:12]`,
# i.e. exactly 12 lowercase hex characters. Validating against that exact shape
# is what makes the filesystem path below safe *by construction* rather than by
# accident of how the router happens to normalise URLs.
#
# This deliberately matches the current generator instead of a full 32-char
# `uuid4().hex`: existing on-disk uploads (and the tests) use the 12-char form,
# and a stricter pattern would 404 every one of them.
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _upload_path(upload_id: str, suffix: str = ".csv"):
    """Resolve `upload_id` to a path inside the upload dir, or raise HTTP 400.

    Two independent layers, because either alone would be a single point of
    failure:

    1. A strict allowlist on the identifier format. This rejects `..`, absolute
       paths, path separators, URL-encoded traversal and any other unexpected
       character *before* a path is ever constructed.
    2. A containment check on the resolved path, so that even if the pattern
       above were ever loosened, a path escaping `upload_dir` still cannot be
       read or unlinked.
    """
    if not _UPLOAD_ID_RE.match(upload_id):
        raise err(400, "Invalid upload id.")
    root = ds.settings.upload_dir.resolve()
    path = (ds.settings.upload_dir / f"{upload_id}{suffix}").resolve()
    if not path.is_relative_to(root):
        # Unreachable while the regex holds; kept as defence in depth.
        raise err(400, "Invalid upload id.")
    return path


@router.get("")
def list_datasets():
    """All available datasets: built-ins + previously uploaded CSVs."""
    out = [s.model_dump() for s in ds.builtin_summaries()]
    out.extend(ds.upload_summaries())
    return {"datasets": out}


@router.get("/builtin/{name}/preview")
def builtin_preview(name: str):
    try:
        df, target, task, classes = ds._load_builtin_raw(name)
    except KeyError as exc:
        # Preserve the chain so a genuine KeyError from the dataset loader is
        # still visible in the server log rather than silently becoming a 404.
        raise err(404, f"Unknown builtin dataset '{name}'.") from exc
    meta = ds._analyse_df(df, f"builtin:{name}", name)
    meta["task"] = task
    meta["class_names"] = classes
    meta["suggested_target"] = target
    preview = df.head(8).copy()
    return {
        "analysis": meta,
        "preview": preview.where(pd.notna(preview), None).to_dict(orient="records"),
    }


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    try:
        meta = ds.save_upload(content, file.filename or "dataset.csv")
    except ValueError as exc:
        raise err(400, str(exc)) from exc
    return meta


@router.get("/upload/{upload_id}")
def upload_analysis(upload_id: str):
    # Validate the id *before* touching the filesystem. `upload_meta`/`load_upload`
    # signal "absent" with a bare KeyError, which is indistinguishable from a
    # KeyError raised by a genuine bug inside the analysis code. Validating up
    # front means the 404 below can only mean "no such upload", so an internal
    # failure can no longer masquerade as a missing file.
    _upload_path(upload_id)
    try:
        meta = ds.upload_meta(upload_id)
        df = ds.load_upload(upload_id)
    except KeyError as exc:
        # `from exc` keeps the original traceback in the server log; the client
        # still only sees the 404.
        raise err(404, "Upload not found.") from exc
    df = df.head(8)
    meta["preview"] = df.where(pd.notna(df), None).to_dict(orient="records")
    return meta


@router.delete("/upload/{upload_id}")
def delete_upload(upload_id: str):
    ok = True
    for ext in (".csv", ".json"):
        # Resolved through the same allowlist + containment check, so `unlink()`
        # can only ever target a file inside upload_dir.
        p = _upload_path(upload_id, ext)
        if p.exists():
            p.unlink()
        else:
            ok = ok and ext == ".json"
    return {"deleted": ok}
