#!/usr/bin/env python3
"""End-to-end smoke test of the complete NeuroSim Lab workflow against a
running backend (default http://localhost:8000). Exercises the exact flow a
user follows in the UI: preset → validate → train (WS streaming) → predict →
trace → backprop → weights → diagnose → custom activation → experiment compare
→ project save/export/import. Exits non-zero on any failure."""
from __future__ import annotations

import json
import os
import sys
import time

# pyrefly: ignore [missing-import]
import httpx

BASE = os.environ.get("NEUROSIM_API", "http://localhost:8000")
C = httpx.Client(base_url=BASE, timeout=60)
FAILURES = []


def check(name, cond, detail=""):
    mark = "✓" if cond else "✗"
    print(f"  {mark} {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def wait_job(job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = C.get(f"/api/training/{job_id}/status").json()
        if s["state"] in ("finished", "stopped", "failed"):
            return s
        time.sleep(0.4)
    raise TimeoutError("job did not finish")


def main():
    print(f"[1] Health — {BASE}")
    h = C.get("/api/health").json()
    check("health ok", h["status"] == "ok", h)

    print("[2] Datasets")
    ds = C.get("/api/datasets").json()["datasets"]
    builtins = [d for d in ds if d["kind"] == "builtin"]
    check("12 builtin datasets", len(builtins) == 12, [d["id"] for d in builtins])

    print("[3] Presets + validation")
    presets = C.get("/api/networks/presets").json()["presets"]
    xor = next(p for p in presets if p["id"] == "xor-demo")
    val = C.post("/api/networks/validate", json=xor["network"]).json()
    check("xor preset validates", val["ok"], val["errors"])
    check("xor chain order", val["order"] == ["input", "hidden1", "hidden2", "output"], val["order"])

    print("[4] Train XOR preset")
    cfg = dict(xor["config"]); cfg["epochs"] = 300
    req = {"network": xor["network"], "dataset": xor["dataset"], "config": cfg,
           "experiment_name": "e2e-xor"}
    r = C.post("/api/training/start", json=req)
    check("training starts", r.status_code == 200, r.text)
    job_id = r.json()["job_id"]
    st = wait_job(job_id)
    check("training finished", st["state"] == "finished", st.get("error"))
    acc = st["final"]["test_metrics"]["accuracy"]
    check(f"test accuracy >= 0.9 (got {acc:.3f})", acc >= 0.9)

    print("[5] Metrics & snapshots")
    m = C.get(f"/api/training/{job_id}/metrics").json()
    check("history recorded", len(m["history"]) == cfg["epochs"])
    check("snapshots recorded", len(m["snapshots"]) >= 5)
    snap = m["snapshots"][-1]
    check("activation stats present", len(snap["activations"]) >= 2)
    check("gradient stats present", len(snap["gradients"]["layers"]) >= 2)

    print("[6] Prediction + explanation")
    p = C.post("/api/predict", json={"run_id": job_id, "features": [1, 0]}).json()
    check("predicts XOR(1,0)=1", p["prediction"]["prediction"] == 1, p["prediction"])
    check("probabilities sum to 1", abs(sum(p["probabilities"]) - 1) < 1e-3)
    check("top neurons listed", len(p["top_neurons"]) > 0)

    print("[7] Forward trace math")
    t = C.post("/api/inspect/forward-trace", json={"run_id": job_id, "features": [1, 0]}).json()
    first = next(l for l in t["trace"] if l["kind"] == "dense")
    n0 = first["neurons"][0]
    manual = sum(i_ * w for i_, w in zip(first["inputs"], n0["weights"])) + n0["bias"]
    check("z matches manual dot product", abs(manual - n0["z"]) < 5e-4, f"{manual} vs {n0['z']}")

    print("[8] Backprop inspection")
    bp = C.post("/api/inspect/backprop", json={"run_id": job_id, "features": [1, 0], "target": 1}).json()
    check("w_new = w_old - lr*grad",
          abs(bp["w_new_actual"] - (bp["w_old"] - bp["learning_rate"] * bp["gradient"])) < 1e-6)

    print("[9] Weights")
    w = C.get(f"/api/inspect/weights/{job_id}").json()
    check("3 weight layers", len(w["layers"]) == 3)

    print("[10] Diagnosis")
    d = C.post("/api/diagnostics/analyze", json={"run_id": job_id, "use_llm": True}).json()
    check("findings returned", len(d["findings"]) >= 1)
    llm_ok = (
        (d["llm"]["available"] is False and bool(d["llm"].get("error")))
        or (d["llm"]["available"] is True and bool(d["llm"].get("explanation")))
    ) and ("api_key" not in str(d["llm"]).lower())
    check("llm disabled gracefully", llm_ok, d["llm"])

    print("[11] Custom activation lifecycle")
    av = C.post("/api/activations/validate", json={"formula": "x * tanh(log(1 + exp(x)))"}).json()  # mish
    check("mish validates", av["ok"], av.get("error"))
    evil = C.post("/api/activations/validate", json={"formula": "__import__('os').system('touch /tmp/pwn')"}).json()
    check("injection rejected", not evil["ok"])
    check("nothing executed", not os.path.exists("/tmp/pwn"))
    C.post("/api/activations", json={"name": "mishy", "formula": "x * tanh(log(1 + exp(x)))"})
    custom_net = json.loads(json.dumps(xor["network"]))
    custom_net["layers"][1]["params"]["activation"] = "custom::mishy"
    custom_net["layers"][2]["params"]["activation"] = "custom::mishy"
    custom_net["custom_activations"] = {"mishy": "x * tanh(log(1 + exp(x)))"}
    r2 = C.post("/api/training/start", json={
        "network": custom_net, "dataset": xor["dataset"],
        "config": {**cfg, "epochs": 200}, "experiment_name": "e2e-xor-custom"})
    check("custom-activation training starts", r2.status_code == 200, r2.text)
    st2 = wait_job(r2.json()["job_id"])
    acc2 = st2["final"]["test_metrics"]["accuracy"]
    check(f"custom activation trains to >=0.9 (got {acc2:.3f})", st2["state"] == "finished" and acc2 >= 0.9, st2.get("error"))

    print("[12] Experiment compare (standard vs custom activation)")
    cmp_ = C.get(f"/api/experiments/compare?run_ids={job_id},{r2.json()['job_id']}").json()
    check("compare returns both runs with history",
          len(cmp_["runs"]) == 2 and all(len(r_["history"]) > 0 for r_ in cmp_["runs"]))

    print("[13] Projects save/export/import")
    payload = {"name": "E2E Project", "network": xor["network"], "dataset": xor["dataset"],
               "training_config": cfg, "run_id": job_id,
               "custom_activations": {"mishy": "x * tanh(log(1 + exp(x)))"}}
    sp = C.post("/api/projects", json=payload).json()
    gp = C.get(f"/api/projects/{sp['project_id']}").json()
    check("project round-trip", gp["payload"]["name"] == "E2E Project")
    xp = C.post("/api/projects/export", json=payload)
    check("export has attachment header + format",
          "attachment" in xp.headers.get("content-disposition", "") and xp.json()["format"] == "neurosim-lab/project")
    ip = C.post("/api/projects/import", json=xp.json())
    check("import accepted", ip.status_code == 200)

    print("[14] Iris research preset (spec §37), brief run")
    iris = next(p for p in presets if p["id"] == "iris-research")
    r3 = C.post("/api/training/start", json={
        "network": iris["network"], "dataset": iris["dataset"],
        "config": {**iris["config"], "epochs": 120}, "experiment_name": "e2e-iris"})
    st3 = wait_job(r3.json()["job_id"], timeout=90)
    acc3 = st3["final"]["test_metrics"]["accuracy"]
    check(f"iris accuracy >= 0.85 (got {acc3:.3f})", acc3 >= 0.85, st3.get("error"))
    check("f1 present", "f1_macro" in st3["final"]["test_metrics"])
    check("confusion matrix 3x3", len(st3["final"]["confusion_matrix"]) == 3)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("ALL E2E CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
