"""Raw posting store (SQLite): one row per (source, posting id) plus snapshot presence."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS postings (
  source TEXT, posting_id TEXT, first_seen TEXT, last_seen TEXT, payload TEXT,
  PRIMARY KEY (source, posting_id));
CREATE TABLE IF NOT EXISTS presence (
  source TEXT, posting_id TEXT, snapshot TEXT, PRIMARY KEY (source, posting_id, snapshot));
CREATE TABLE IF NOT EXISTS snapshots (
  source TEXT, snapshot TEXT, started_at TEXT, finished_at TEXT, n_postings INT, notes TEXT,
  PRIMARY KEY (source, snapshot));
"""


class PostingStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # shared across crawler threads; every write goes through self.lock
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.executescript(SCHEMA)
        self.lock = threading.RLock()

    def upsert(self, source: str, posting_id: str, snapshot: str, payload: dict[str, Any]) -> bool:
        """Insert or refresh a posting; returns True when it is new."""
        with self.lock:
            return self._upsert(source, posting_id, snapshot, payload)

    def _upsert(self, source: str, posting_id: str, snapshot: str, payload: dict[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False)
        cur = self.con.execute(
            "SELECT 1 FROM postings WHERE source=? AND posting_id=?", (source, posting_id)
        )
        new = cur.fetchone() is None
        if new:
            self.con.execute(
                "INSERT INTO postings VALUES (?,?,?,?,?)",
                (source, posting_id, snapshot, snapshot, body),
            )
        else:
            self.con.execute(
                "UPDATE postings SET last_seen=?, payload=? WHERE source=? AND posting_id=?",
                (snapshot, body, source, posting_id),
            )
        self.con.execute(
            "INSERT OR IGNORE INTO presence VALUES (?,?,?)", (source, posting_id, snapshot)
        )
        return new

    def start_snapshot(self, source: str, snapshot: str, started_at: str) -> None:
        self.con.execute(
            "INSERT OR IGNORE INTO snapshots VALUES (?,?,?,NULL,0,'')",
            (source, snapshot, started_at),
        )
        self.con.commit()

    def finish_snapshot(
        self, source: str, snapshot: str, finished_at: str, n: int, notes: str = ""
    ) -> None:
        self.con.execute(
            "UPDATE snapshots SET finished_at=?, n_postings=?, notes=?"
            " WHERE source=? AND snapshot=?",
            (finished_at, n, notes, source, snapshot),
        )
        self.con.commit()

    def commit(self) -> None:
        with self.lock:
            self.con.commit()

    def iter_payloads(self, source: str | None = None) -> Iterator[tuple[str, str, dict[str, Any]]]:
        q = "SELECT source, posting_id, payload FROM postings"
        args: tuple[str, ...] = ()
        if source:
            q += " WHERE source=?"
            args = (source,)
        for s, pid, body in self.con.execute(q, args).fetchall():
            yield s, pid, json.loads(body)

    def iter_rows(self) -> Iterator[tuple[str, str, dict[str, Any], str, str]]:
        """(source, posting_id, payload, first_seen, last_seen) for every posting."""
        q = "SELECT source, posting_id, payload, first_seen, last_seen FROM postings"
        for s, pid, body, first, last in self.con.execute(q).fetchall():
            yield s, pid, json.loads(body), first, last

    def counts(self) -> dict[str, int]:
        return dict(self.con.execute("SELECT source, COUNT(*) FROM postings GROUP BY source"))
