"""HTTP API tests using FastAPI's TestClient (real end-to-end flow)."""
import re
import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.storage import init_db

from .conftest import make_train_request, make_xor_spec

client = TestClient(app)
init_db()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    # LLM_ENABLED defaults to true: the layer is "on" even with no keys set,
    # and each provider reports itself unavailable at request time.
    assert body["status"] == "ok" and body["llm_enabled"] is True
    assert body["llm_provider_order"] == ["groq", "nvidia", "ollama"]


def test_llm_status_reports_the_chain_without_keys(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "nvidia_api_key", "")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    r = client.get("/api/diagnostics/llm-status")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["chain"] == ["groq", "nvidia", "ollama"]
    states = {p["name"]: p for p in body["providers"]}
    assert set(states) == {"groq", "nvidia", "ollama"}
    # No keys are set in the test environment.
    assert states["groq"]["state"] == "not_configured"
    assert states["nvidia"]["state"] == "not_configured"
    # Ollama needs no key, so it is configuration-available.
    assert states["ollama"]["state"] == "available"
    # Model names are exposed; credentials never are.
    assert body["models"]["groq"]
    assert "api_key" not in str(body).lower()


def test_dataset_listing_and_preview():
    r = client.get("/api/datasets")
    assert r.status_code == 200
    builtins = [d for d in r.json()["datasets"] if d["kind"] == "builtin"]
    assert len(builtins) == 12  # 11 tabular + the 8x8 synthetic image set
    assert {"xor", "iris", "shapes8"} <= {d["id"] for d in builtins}
    r = client.get("/api/datasets/builtin/iris/preview")
    assert r.status_code == 200
    assert len(r.json()["preview"]) == 8


def test_upload_csv_and_analysis():
    csv = b"a,b,y\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n"
    r = client.post("/api/datasets/upload", files={"file": ("t.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    uid = r.json()["upload_id"]
    r2 = client.get(f"/api/datasets/upload/{uid}")
    assert r2.status_code == 200
    assert r2.json()["suggested_target"] == "y"


# ---------------------------------------------------------------------------
# upload_id hardening
#
# `upload_id` is interpolated into a filesystem path, and DELETE unlinks it, so
# the handler validates the id against the exact shape the app mints
# (uuid4().hex[:12]). These tests pin that behaviour.
# ---------------------------------------------------------------------------

# Ids that must never be accepted. Each is rejected before any path is built.
BAD_UPLOAD_IDS = [
    "..",                                  # parent traversal
    "../..",                               # multi-level traversal
    "../../etc/passwd",                    # traversal to a real file
    "..%2F..%2Fetc%2Fpasswd",              # URL-encoded traversal
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",      # fully percent-encoded traversal
    "/etc/passwd",                         # absolute path
    "/tmp/evil",                           # absolute path, writable location
    "..\\..\\windows",                     # backslash separator
    "abcdefghi",                           # too short (11 chars)
    "abcdefghijk",                         # too short (11 chars)
    "ABCDEF012345",                        # uppercase hex, not the minted form
    "abcdef0123456",                       # 13 chars, too long
    "abcdef01234g",                        # non-hex character
    "abcdef 01234",                        # embedded space
    "abcdef01234/../../x",                 # traversal after a valid prefix
    "",                                    # empty
]


@pytest.mark.parametrize("bad", BAD_UPLOAD_IDS)
def test_upload_analysis_rejects_bad_ids(bad):
    # Called directly rather than over HTTP: httpx/starlette normalise "../"
    # segments out of the URL *before* routing, so the traversal string would
    # never reach the handler and the test would prove nothing. This is the
    # layer where the validation actually happens.
    from fastapi import HTTPException

    from app.api.datasets import upload_analysis

    with pytest.raises(HTTPException) as ei:
        upload_analysis(bad)
    assert ei.value.status_code == 400
    assert ei.value.detail == "Invalid upload id."


@pytest.mark.parametrize("bad", BAD_UPLOAD_IDS)
def test_delete_upload_rejects_bad_ids(bad):
    from fastapi import HTTPException

    from app.api.datasets import delete_upload

    with pytest.raises(HTTPException) as ei:
        delete_upload(bad)
    assert ei.value.status_code == 400
    assert ei.value.detail == "Invalid upload id."


@pytest.mark.parametrize("bad", ["../..", "../../etc/passwd", "/etc/passwd",
                                 "..%2F..%2Fetc%2Fpasswd", "..\\..\\windows",
                                 "%2e%2e%2f%2e%2e%2fetc%2fpasswd"])
def test_traversal_over_http_never_reaches_the_filesystem(bad):
    """End-to-end guard: no traversal id can read or delete anything.

    Percent-encoded forms are the interesting ones: they survive URL
    normalisation and reach the handler verbatim, so this asserts the *validator*
    (not the router) is what stops them. The outcome is either our 400 or a 404
    against an unrelated route -- never a successful read or delete.

    A bare ".." is excluded because HTTP clients collapse it to
    `/api/datasets/upload`, the harmless dataset-list route. The validator itself
    is covered for ".." by the direct-call tests above.
    """
    for method, path in (("get", f"/api/datasets/upload/{bad}"),
                         ("delete", f"/api/datasets/upload/{bad}")):
        r = getattr(client, method)(path)
        assert r.status_code >= 400, f"{method.upper()} {bad!r} -> {r.status_code} {r.text}"
        # Guard against a traversal ever being answered with upload contents.
        assert "preview" not in r.text
        assert "suggested_target" not in r.text


def test_upload_traversal_cannot_delete_files_outside_upload_dir():
    """A decoy file outside upload_dir must survive a traversal delete attempt.

    The canary is a sibling of upload_dir, addressed three ways: relative
    escape, deeper escape, and absolute path. Assertions are on the canary still
    existing, so this does not depend on OS-specific path handling.
    """
    from fastapi import HTTPException

    from app.api.datasets import delete_upload
    from app.ml import datasets as ds

    canary = ds.settings.upload_dir.parent / "CANARY.csv"
    canary.write_text("a,b\n1,2\n")
    try:
        for probe in ("../CANARY", "../../CANARY", str(canary)):
            with pytest.raises(HTTPException) as ei:
                delete_upload(probe)
            assert ei.value.status_code == 400
            assert canary.exists(), f"canary deleted via {probe!r}"
    finally:
        canary.unlink(missing_ok=True)


def test_upload_roundtrip_and_delete():
    """Valid id still works end to end: upload -> read -> delete -> gone."""
    csv = b"a,b,y\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n"
    r = client.post("/api/datasets/upload", files={"file": ("rt.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    uid = r.json()["upload_id"]
    # The minted id must satisfy the validator we now enforce.
    assert re.fullmatch(r"[0-9a-f]{12}", uid), uid

    r2 = client.get(f"/api/datasets/upload/{uid}")
    assert r2.status_code == 200 and "preview" in r2.json()

    assert client.delete(f"/api/datasets/upload/{uid}").json()["deleted"] is True
    # Files are gone, so it is now a genuine 404 rather than a 400.
    assert client.get(f"/api/datasets/upload/{uid}").status_code == 404
    # Deleting an already-removed upload reports deleted=False. That is the
    # pre-existing contract of the `ok = ok and ext == ".json"` logic, asserted
    # here so the hardening is provably behaviour-preserving.
    assert client.delete(f"/api/datasets/upload/{uid}").json()["deleted"] is False


def test_missing_but_wellformed_upload_id_is_404():
    """A well-formed id that simply does not exist is 404, not 400."""
    r = client.get("/api/datasets/upload/0123456789ab")
    assert r.status_code == 404
    assert r.json()["detail"] == "Upload not found."
    # A well-formed but absent id is also accepted by the delete route (it just
    # finds nothing to remove) rather than being rejected as malformed.
    assert client.delete("/api/datasets/upload/0123456789ab").status_code == 200


def test_validate_network_ok_and_errors():
    r = client.post("/api/networks/validate", json=make_xor_spec().model_dump())
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.json()["total_params"] == 114
    bad = make_xor_spec().model_dump()
    bad["edges"] = []
    r = client.post("/api/networks/validate", json=bad)
    assert r.json()["ok"] is False


def test_build_network():
    r = client.post("/api/networks/build", json=make_xor_spec().model_dump())
    assert r.json()["ok"] is True
    assert r.json()["parameter_tensors"] > 0


def test_presets():
    r = client.get("/api/networks/presets")
    ids = [p["id"] for p in r.json()["presets"]]
    assert "xor-demo" in ids and "iris-research" in ids


def test_activation_validation_and_library():
    ok = client.post("/api/activations/validate", json={"formula": "x / (1 + exp(-x))"})
    assert ok.status_code == 200 and ok.json()["ok"] is True
    assert len(ok.json()["xs"]) == len(ok.json()["ys"]) == len(ok.json()["dys"])
    evil = client.post("/api/activations/validate", json={"formula": "__import__('os').system('id')"})
    assert evil.json()["ok"] is False
    save = client.post("/api/activations", json={"name": "swishy", "formula": "x / (1 + exp(-x))"})
    assert save.status_code == 200
    lib = client.get("/api/activations/library")
    assert any(a["id"] == "relu" for a in lib.json()["builtin"])
    assert any(a["name"] == "swishy" for a in lib.json()["custom"])
    bad = client.post("/api/activations", json={"name": "bad", "formula": "open('x')"})
    assert bad.status_code == 400


def test_projects_crud_export_import():
    payload = {
        "name": "Test Project",
        "network": make_xor_spec().model_dump(),
        "custom_activations": {},
    }
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 200, r.text
    pid = r.json()["project_id"]
    got = client.get(f"/api/projects/{pid}")
    assert got.status_code == 200 and got.json()["payload"]["name"] == "Test Project"
    exp = client.post("/api/projects/export", json=payload)
    assert exp.status_code == 200
    doc = exp.json()
    assert doc["format"] == "neurosim-lab/project"
    imp = client.post("/api/projects/import", json=doc)
    assert imp.status_code == 200
    listing = client.get("/api/projects")
    assert any(p["id"] == pid for p in listing.json()["projects"])
    assert client.delete(f"/api/projects/{pid}").json()["deleted"] is True


def test_full_training_flow():
    req = make_train_request()
    req.config.epochs = 200
    req.experiment_name = "api-test-exp"
    r = client.post("/api/training/start", json=req.model_dump())
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # wait for completion via status polling
    t0 = time.time()
    while True:
        s = client.get(f"/api/training/{job_id}/status").json()
        if s["state"] in ("finished", "failed", "stopped"):
            break
        assert time.time() - t0 < 90, "training took too long"
        time.sleep(0.3)
    assert s["state"] == "finished", s.get("error")
    assert s["final"]["test_metrics"]["accuracy"] >= 0.85

    m = client.get(f"/api/training/{job_id}/metrics").json()
    assert len(m["history"]) == 200 and m["snapshots"]

    # prediction via saved run
    xor_point = [1, 0]  # XOR(1,0) = class 1
    p = client.post("/api/predict", json={"run_id": job_id, "features": xor_point})
    assert p.status_code == 200, p.text
    pred = p.json()
    assert pred["prediction"]["type"] == "classification"
    assert pred["probabilities"]
    assert abs(sum(pred["probabilities"]) - 1.0) < 1e-3
    assert pred["probabilities"][1] > 0.5, "XOR(1,0) should be class 1"

    # forward trace (math view)
    t = client.post("/api/inspect/forward-trace", json={"run_id": job_id, "features": xor_point})
    assert t.status_code == 200
    layers = t.json()["trace"]
    assert layers[0]["kind"] == "input"
    dense = [l for l in layers if l["kind"] in ("dense", "output")]
    neuron = dense[0]["neurons"][0]
    w = neuron["weights"]
    xi = dense[0]["inputs"]
    wx = sum(float(a) * float(b) for a, b in zip(w, xi[: len(w)]))
    assert neuron["z"] == pytest.approx(wx + neuron["bias"] + sum(neuron["weighted_inputs"][len(w):]), abs=2e-4)

    # backprop inspection
    bp = client.post("/api/inspect/backprop", json={"run_id": job_id, "features": xor_point, "target": 1})
    assert bp.status_code == 200, bp.text
    body = bp.json()
    assert body["w_new_actual"] == pytest.approx(body["w_old"] - body["learning_rate"] * body["gradient"], abs=1e-6)

    # weights endpoint
    w = client.get(f"/api/inspect/weights/{job_id}")
    assert w.status_code == 200 and w.json()["layers"]

    # diagnostics
    d = client.post("/api/diagnostics/analyze", json={"run_id": job_id, "use_llm": True})
    assert d.status_code == 200, d.text
    diag = d.json()
    assert diag["findings"]
    # No LLM provider is usable in tests, so the explanation is unavailable
    # while the deterministic findings are still returned in full.
    assert diag["llm"]["available"] is False
    assert diag["llm"]["explanation"] is None
    assert diag["llm"]["outcome"] == "unavailable"
    assert "Deterministic diagnosis is still available" in diag["llm"]["message"]
    assert diag["llm"]["chain"] == ["groq", "nvidia", "ollama"]

    # experiments recorded
    exps = client.get("/api/experiments").json()["experiments"]
    exp = next(e for e in exps if e["name"] == "api-test-exp")
    assert exp["runs"], "experiment should list the finished run"

    # second run to compare
    r2 = client.post("/api/training/start", json=req.model_dump())
    j2 = r2.json()["job_id"]
    t0 = time.time()
    while True:
        s2 = client.get(f"/api/training/{j2}/status").json()
        if s2["state"] in ("finished", "failed", "stopped"):
            break
        assert time.time() - t0 < 90
        time.sleep(0.3)
    cmp = client.get(f"/api/experiments/compare?run_ids={job_id},{j2}")
    assert cmp.status_code == 200
    runs = cmp.json()["runs"]
    assert len(runs) == 2 and all(r["history"] for r in runs)


def test_training_start_rejects_incompatible():
    req = make_train_request().model_dump()
    req["network"]["layers"][0]["params"]["features"] = 9  # xor has 2
    r = client.post("/api/training/start", json=req)
    assert r.status_code == 400
    assert "features" in r.json()["detail"]


def test_training_start_rejects_bad_loss():
    req = make_train_request().model_dump()
    req["config"]["loss"] = "mae"
    r = client.post("/api/training/start", json=req)
    assert r.status_code == 400


def test_unknown_routes_404():
    assert client.get("/api/datasets/builtin/nope/preview").status_code == 404
    assert client.post("/api/predict", json={"run_id": "nope", "features": [1, 2]}).status_code == 404
