"""Saved Reports library (SQLite). Every read and write is filtered by owner, so a
user can only ever see or delete their own reports. Deletion is soft: rows get a
`deleted_at` timestamp and stay in the database.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_DDL = """
CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id    TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    deleted_at  TEXT
);
CREATE INDEX IF NOT EXISTS reports_owner ON reports (owner_id, deleted_at);
"""


@dataclass(frozen=True)
class Report:
    id: int
    owner_id: str
    thread_id: str
    title: str
    body: str
    created_at: str

    def summary(self) -> dict:
        return {"id": self.id, "title": self.title, "created_at": self.created_at[:16].replace("T", " ")}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ReportStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(_DDL)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def save(self, owner_id: str, thread_id: str, title: str, body: str) -> Report:
        created = _now()
        with closing(self._connect()) as con, con:
            cur = con.execute(
                "INSERT INTO reports (owner_id, thread_id, title, body, created_at) VALUES (?, ?, ?, ?, ?)",
                (owner_id, thread_id, title, body, created),
            )
        return Report(cur.lastrowid, owner_id, thread_id, title, body, created)

    def find(
        self,
        owner_id: str,
        *,
        thread_id: str | None = None,
        mentions: str | None = None,
        ids: Iterable[int] | None = None,
    ) -> list[Report]:
        """The owner's active reports matching ALL given filters."""
        where, params = ["owner_id = ?", "deleted_at IS NULL"], [owner_id]
        if thread_id is not None:
            where.append("thread_id = ?")
            params.append(thread_id)
        if mentions:
            like = "%" + mentions.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            where.append("(title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\')")
            params += [like, like]
        if ids is not None:
            ids = [int(i) for i in ids]
            if not ids:
                return []
            where.append(f"id IN ({', '.join('?' * len(ids))})")
            params += ids
        sql = f"SELECT id, owner_id, thread_id, title, body, created_at FROM reports WHERE {' AND '.join(where)} ORDER BY id"
        with closing(self._connect()) as con:
            return [Report(**dict(row)) for row in con.execute(sql, params)]

    def soft_delete(self, owner_id: str, ids: Iterable[int]) -> list[int]:
        """Mark the owner's reports as deleted. Returns the ids actually deleted."""
        targets = [r.id for r in self.find(owner_id, ids=ids)]
        if targets:
            with closing(self._connect()) as con, con:
                con.execute(
                    f"UPDATE reports SET deleted_at = ? WHERE owner_id = ? AND id IN ({', '.join('?' * len(targets))})",
                    [_now(), owner_id, *targets],
                )
        return targets
