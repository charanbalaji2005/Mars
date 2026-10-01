"""Persistent experiment store (SQLite).

Every trial is committed in its own transaction, so an interrupted run never
leaves a half-written record: either a trial is fully stored or it is absent.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import StoreError
from ..search.space import KernelConfig, Shape

SCHEMA_VERSION = 1
EXPERIMENT_STATUSES = ("running", "complete", "interrupted", "failed")
TRIAL_STATUSES = ("ok", "incorrect", "compile_error", "out_of_resources", "oom", "runtime_error")
REFERENCE_KEY = "torch.matmul"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_fingerprint TEXT,
    config_json TEXT NOT NULL,
    hardware_json TEXT NOT NULL,
    software_json TEXT NOT NULL,
    notes_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    strategy TEXT NOT NULL,
    seed INTEGER NOT NULL,
    trial_index INTEGER NOT NULL,
    m INTEGER NOT NULL, n INTEGER NOT NULL, k INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    config_key TEXT NOT NULL,
    config_json TEXT,
    status TEXT NOT NULL,
    error TEXT,
    first_call_s REAL,
    wall_s REAL,
    median_ms REAL, mean_ms REAL, std_ms REAL, min_ms REAL, p25_ms REAL, p75_ms REAL,
    n_samples INTEGER,
    samples_json TEXT,
    max_abs_err REAL,
    max_rel_err REAL,
    predicted_ms REAL,
    device_name TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (experiment_id, strategy, seed, m, n, k, dtype, config_key)
);
CREATE INDEX IF NOT EXISTS idx_trials_lookup ON trials (m, n, k, dtype, status, device_name);
CREATE INDEX IF NOT EXISTS idx_trials_experiment ON trials (experiment_id, strategy, seed);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TrialRecord:
    experiment_id: str
    strategy: str
    seed: int
    trial_index: int
    shape: Shape
    dtype: str
    config_key: str
    status: str
    config: dict | None = None
    error: str | None = None
    first_call_s: float | None = None
    wall_s: float | None = None
    median_ms: float | None = None
    mean_ms: float | None = None
    std_ms: float | None = None
    min_ms: float | None = None
    p25_ms: float | None = None
    p75_ms: float | None = None
    n_samples: int | None = None
    samples_ms: list[float] | None = None
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    predicted_ms: float | None = None
    device_name: str | None = None
    created_at: str = field(default_factory=_now)
    id: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def kernel_config(self) -> KernelConfig | None:
        return KernelConfig.from_dict(self.config) if self.config else None

    def latency(self, statistic: str = "median") -> float | None:
        return {"median": self.median_ms, "mean": self.mean_ms, "min": self.min_ms}[statistic]


class ExperimentStore:
    def __init__(self, path: str | Path, *, create: bool = True):
        self.path = Path(path)
        if not create and not self.path.is_file():
            raise StoreError(f"experiment database not found: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        with self.conn:
            self.conn.executescript(_SCHEMA)
            row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self.conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row["value"]) != SCHEMA_VERSION:
                raise StoreError(f"{self.path} uses schema v{row['value']}, this code expects v{SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----------------------------------------------------------------- experiments
    def create_experiment(self, kind: str, name: str, config: dict, hardware: dict, software: dict,
                          fingerprint: str | None = None, notes: dict | None = None) -> str:
        exp_id = f"{kind}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        now = _now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO experiments (id, kind, name, status, created_at, updated_at, config_fingerprint,"
                " config_json, hardware_json, software_json, notes_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (exp_id, kind, name, "running", now, now, fingerprint, json.dumps(config),
                 json.dumps(hardware, default=str), json.dumps(software, default=str),
                 json.dumps(notes or {}, default=str)),
            )
        return exp_id

    def get_experiment(self, exp_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        if row is None:
            raise StoreError(f"experiment '{exp_id}' not found in {self.path}")
        return self._experiment_row(row)

    def list_experiments(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM experiments ORDER BY created_at").fetchall()
        return [self._experiment_row(r) for r in rows]

    def set_status(self, exp_id: str, status: str, note: str | None = None) -> None:
        if status not in EXPERIMENT_STATUSES:
            raise ValueError(f"unknown experiment status {status!r}")
        with self.conn:
            self.conn.execute("UPDATE experiments SET status=?, updated_at=? WHERE id=?", (status, _now(), exp_id))
        if note:
            self.update_notes(exp_id, {"last_status_note": note})

    def update_notes(self, exp_id: str, updates: dict) -> None:
        notes = self.get_experiment(exp_id)["notes"]
        notes.update(updates)
        with self.conn:
            self.conn.execute("UPDATE experiments SET notes_json=?, updated_at=? WHERE id=?",
                              (json.dumps(notes, default=str), _now(), exp_id))

    @staticmethod
    def _experiment_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "kind": row["kind"], "name": row["name"], "status": row["status"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "config_fingerprint": row["config_fingerprint"],
            "config": json.loads(row["config_json"]), "hardware": json.loads(row["hardware_json"]),
            "software": json.loads(row["software_json"]), "notes": json.loads(row["notes_json"]),
        }

    # ---------------------------------------------------------------------- trials
    def record_trial(self, rec: TrialRecord) -> int:
        try:
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO trials (experiment_id, strategy, seed, trial_index, m, n, k, dtype, config_key,"
                    " config_json, status, error, first_call_s, wall_s, median_ms, mean_ms, std_ms, min_ms,"
                    " p25_ms, p75_ms, n_samples, samples_json, max_abs_err, max_rel_err, predicted_ms,"
                    " device_name, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rec.experiment_id, rec.strategy, rec.seed, rec.trial_index, *rec.shape, rec.dtype,
                     rec.config_key, json.dumps(rec.config) if rec.config else None, rec.status, rec.error,
                     rec.first_call_s, rec.wall_s, rec.median_ms, rec.mean_ms, rec.std_ms, rec.min_ms,
                     rec.p25_ms, rec.p75_ms, rec.n_samples,
                     json.dumps(rec.samples_ms) if rec.samples_ms is not None else None,
                     rec.max_abs_err, rec.max_rel_err, rec.predicted_ms, rec.device_name, rec.created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise StoreError(f"duplicate or invalid trial for {rec.config_key} {rec.shape}: {exc}") from exc
        rec.id = int(cur.lastrowid)
        return rec.id

    def trials(self, *, experiment_ids: Sequence[str] | None = None, strategy: str | None = None,
               seed: int | None = None, shape: Shape | None = None, status: str | None = None,
               dtype: str | None = None, device_name: str | None = None,
               kinds: Iterable[str] | None = None, exclude_strategies: Iterable[str] = ()) -> list[TrialRecord]:
        clauses, params = [], []
        if experiment_ids is not None:
            ids = list(experiment_ids)
            if not ids:
                return []
            clauses.append(f"t.experiment_id IN ({','.join('?' * len(ids))})")
            params += ids
        for column, value in (("t.strategy", strategy), ("t.seed", seed), ("t.status", status),
                              ("t.dtype", dtype), ("t.device_name", device_name)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if shape is not None:
            clauses.append("t.m = ? AND t.n = ? AND t.k = ?")
            params += list(shape)
        if kinds is not None:
            kinds = list(kinds)
            clauses.append(f"e.kind IN ({','.join('?' * len(kinds))})")
            params += kinds
        excluded = list(exclude_strategies)
        if excluded:
            clauses.append(f"t.strategy NOT IN ({','.join('?' * len(excluded))})")
            params += excluded
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(
            f"SELECT t.* FROM trials t JOIN experiments e ON e.id = t.experiment_id {where}"
            " ORDER BY t.experiment_id, t.strategy, t.seed, t.m, t.n, t.k, t.trial_index", params).fetchall()
        return [self._trial_row(r) for r in rows]

    def best_known(self, shape: Shape, dtype: str, device_name: str | None,
                   statistic: str = "median") -> float | None:
        """Best measured Triton latency for a shape across all experiments on this device."""
        column = {"median": "median_ms", "mean": "mean_ms", "min": "min_ms"}[statistic]
        query = (f"SELECT MIN({column}) AS best FROM trials WHERE status='ok' AND m=? AND n=? AND k=?"
                 " AND dtype=? AND config_key != ?")
        params: list = [*shape, dtype, REFERENCE_KEY]
        if device_name is not None:
            query += " AND device_name = ?"
            params.append(device_name)
        row = self.conn.execute(query, params).fetchone()
        return row["best"] if row and row["best"] is not None else None

    @staticmethod
    def _trial_row(row: sqlite3.Row) -> TrialRecord:
        return TrialRecord(
            id=row["id"], experiment_id=row["experiment_id"], strategy=row["strategy"], seed=row["seed"],
            trial_index=row["trial_index"], shape=(row["m"], row["n"], row["k"]), dtype=row["dtype"],
            config_key=row["config_key"],
            config=json.loads(row["config_json"]) if row["config_json"] else None,
            status=row["status"], error=row["error"], first_call_s=row["first_call_s"], wall_s=row["wall_s"],
            median_ms=row["median_ms"], mean_ms=row["mean_ms"], std_ms=row["std_ms"], min_ms=row["min_ms"],
            p25_ms=row["p25_ms"], p75_ms=row["p75_ms"], n_samples=row["n_samples"],
            samples_ms=json.loads(row["samples_json"]) if row["samples_json"] else None,
            max_abs_err=row["max_abs_err"], max_rel_err=row["max_rel_err"],
            predicted_ms=row["predicted_ms"], device_name=row["device_name"], created_at=row["created_at"],
        )
