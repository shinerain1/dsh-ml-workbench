"""Process lifecycle ledger. Metrics and experiment tracking remain in MLflow."""

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

from .files import canonical
from .spec import WorkbenchError

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root / "control.lock"), timeout=30)
        self.path = self.root / "lifecycle.sqlite3"
        with self.lock, self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "run_id TEXT NOT NULL, time REAL NOT NULL, state TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, "
                       "run_id TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    def get(self, run_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise WorkbenchError("INVALID_RUN_ID", "run_id must be an MLflow run ID returned by create")
        with self.connect() as db:
            row = db.execute("SELECT record FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise WorkbenchError("RUN_NOT_FOUND", "Experiment was not found in this workbench root")
        return json.loads(row[0])

    def save(self, record: dict, *, request_key: str | None = None, payload_hash: str = "") -> None:
        record["updated_at"] = time.time()
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET record=excluded.record",
                       (record["run_id"], canonical(record)))
            db.execute("INSERT INTO events(run_id,time,state) VALUES (?,?,?)",
                       (record["run_id"], record["updated_at"], record["state"]))
            if request_key is not None:
                db.execute("INSERT INTO requests VALUES (?,?,?)", (request_key, payload_hash, record["run_id"]))

    def previous_request(self, key: str | None, payload_hash: str) -> dict | None:
        if key is None:
            return None
        if not key or len(key) > 256:
            raise WorkbenchError("INVALID_REQUEST_KEY", "request_key must contain 1 to 256 characters")
        with self.connect() as db:
            row = db.execute("SELECT payload_hash,run_id FROM requests WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if row[0] != payload_hash:
            raise WorkbenchError("IDEMPOTENCY_CONFLICT", "request_key was already used with different arguments")
        return self.get(row[1])

    def list_ids(self, limit: int, offset: int = 0, state: str | None = None,
                 model: str | None = None, parent_run_id: str | None = None) -> list[str]:
        clauses, values = [], []
        for field, value in [("state", state), ("model", model), ("parent_run_id", parent_run_id)]:
            if value is not None:
                clauses.append(f"json_extract(record, '$.{field}') = ?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT id FROM runs" + where + " ORDER BY rowid DESC LIMIT ? OFFSET ?",
                                                [*values, limit, offset])]

    def active_ids(self) -> list[str]:
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT id FROM runs WHERE json_extract(record, '$.state') "
                                                "IN ('starting', 'running', 'cancelling')")]

    def events(self, run_id: str) -> list[dict]:
        with self.connect() as db:
            return [{"sequence": row[0], "time": row[1], "state": row[2]} for row in db.execute(
                "SELECT sequence,time,state FROM events WHERE run_id=? ORDER BY sequence", (run_id,))]
