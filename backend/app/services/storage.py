"""SQLite persistence layer.

Tables: projects, runs, experiments, activations. Plain ``sqlite3`` with a
thin functional API so it can later be swapped for SQLAlchemy/PostgreSQL
without touching routers.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from ..config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    network TEXT NOT NULL,
    network_summary TEXT,
    dataset_meta TEXT,
    config TEXT,
    history TEXT,
    snapshots TEXT,
    data_summary TEXT,
    final TEXT,
    weights_path TEXT,
    experiment_id TEXT,
    duration_s REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS activations (
    name TEXT PRIMARY KEY,
    formula TEXT NOT NULL,
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _conn():
    con = sqlite3.connect(settings.database_path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    settings.ensure_dirs()
    with _conn() as con:
        con.executescript(_SCHEMA)


# ------------------------- projects ---------------------------------------
def save_project(payload: dict, project_id: str | None = None) -> dict:
    pid = project_id or uuid.uuid4().hex[:12]
    now = _now()
    with _conn() as con:
        existing = con.execute("SELECT id, created_at FROM projects WHERE id=?", (pid,)).fetchone()
        if existing:
            con.execute("UPDATE projects SET name=?, payload=?, updated_at=? WHERE id=?",
                        (payload.get("name", "Untitled"), json.dumps(payload), now, pid))
            created = existing["created_at"]
        else:
            con.execute("INSERT INTO projects(id,name,payload,created_at,updated_at) VALUES(?,?,?,?,?)",
                        (pid, payload.get("name", "Untitled"), json.dumps(payload), now, now))
            created = now
    return {"id": pid, "created_at": created, "updated_at": now}


def get_project(pid: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not row:
        return None
    out = dict(row)
    out["payload"] = json.loads(out["payload"])
    return out


def list_projects() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT id,name,created_at,updated_at FROM projects ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_project(pid: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM projects WHERE id=?", (pid,))
    return cur.rowcount > 0


# ------------------------- runs -------------------------------------------
def save_run(run: dict) -> None:
    keys = ["id", "created_at", "status", "network", "network_summary", "dataset_meta",
            "config", "history", "snapshots", "data_summary", "final", "weights_path",
            "experiment_id", "duration_s", "error"]
    row = {k: run.get(k) for k in keys}
    for k in ("network", "network_summary", "dataset_meta", "config", "history",
              "snapshots", "data_summary", "final"):
        row[k] = json.dumps(row[k]) if row[k] is not None else None
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[k] for k in keys))


def get_run(rid: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    if not row:
        return None
    out = dict(row)
    for k in ("network", "network_summary", "dataset_meta", "config", "history",
              "snapshots", "data_summary", "final"):
        out[k] = json.loads(out[k]) if out[k] else None
    return out


def list_runs(experiment_id: str | None = None) -> list[dict]:
    with _conn() as con:
        if experiment_id:
            rows = con.execute(
                "SELECT id,created_at,status,duration_s,experiment_id,final,error "
                "FROM runs WHERE experiment_id=? ORDER BY created_at DESC",
                (experiment_id,)).fetchall()
        else:
            rows = con.execute(
                "SELECT id,created_at,status,duration_s,experiment_id,final,error "
                "FROM runs ORDER BY created_at DESC LIMIT 100").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["final"] = json.loads(d["final"]) if d["final"] else None
        out.append(d)
    return out


# ------------------------- experiments -------------------------------------
def save_experiment(name: str, config: dict, exp_id: str | None = None) -> dict:
    eid = exp_id or uuid.uuid4().hex[:12]
    with _conn() as con:
        existing = con.execute("SELECT id FROM experiments WHERE id=?", (eid,)).fetchone()
        if existing:
            con.execute("UPDATE experiments SET name=?, config=? WHERE id=?", (name, json.dumps(config), eid))
        else:
            con.execute("INSERT INTO experiments(id,name,created_at,config) VALUES(?,?,?,?)",
                        (eid, name, _now(), json.dumps(config)))
    return {"id": eid}


def get_experiment(eid: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
    if not row:
        return None
    out = dict(row)
    out["config"] = json.loads(out["config"])
    out["runs"] = list_runs(eid)
    return out


def list_experiments() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        runs = list_runs(r["id"])
        out.append({"id": r["id"], "name": r["name"], "created_at": r["created_at"],
                    "config": json.loads(r["config"]), "runs": runs})
    return out


def delete_experiment(eid: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM experiments WHERE id=?", (eid,))
    return cur.rowcount > 0


# ------------------------- activations library ------------------------------
def save_activation(name: str, formula: str, notes: str = "") -> dict:
    with _conn() as con:
        con.execute("INSERT OR REPLACE INTO activations(name,formula,notes,created_at) VALUES(?,?,?,?)",
                    (name, formula, notes, _now()))
    return {"name": name}


def list_activations() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM activations ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_activation(name: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM activations WHERE name=?", (name,))
    return cur.rowcount > 0
