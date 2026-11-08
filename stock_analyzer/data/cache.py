"""Shared SQLite cache, safe for many threads and processes.

* WAL journal mode: readers never block the writer and vice versa.
* A fresh connection per operation (sqlite3 connections must not be shared across threads),
  with a busy timeout so concurrent writers queue instead of failing.
* Upserts are single atomic statements, so a half-written entry can never be read.
* Values are opaque strings; the service decides what to store.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from stock_analyzer.exceptions import CacheError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheEntry:
    key: str
    value: str
    stored_at: float  # epoch seconds

    def age_seconds(self, now: float) -> float:
        return max(0.0, now - self.stored_at)


class SQLiteCache:
    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_s: float = 30.0,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._timeout = busy_timeout_s
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS cache ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL, stored_at REAL NOT NULL)"
                )
        except (sqlite3.Error, OSError) as exc:
            raise CacheError(f"cannot initialise cache at {self.path}: {exc}") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=self._timeout)) as conn:
            conn.execute("PRAGMA synchronous=NORMAL")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def get(self, key: str) -> CacheEntry | None:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value, stored_at FROM cache WHERE key = ?", (key,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise CacheError(f"cache read failed: {exc}") from exc
        return CacheEntry(key, row[0], row[1]) if row else None

    def set(self, key: str, value: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO cache(key, value, stored_at) VALUES(?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "value = excluded.value, stored_at = excluded.stored_at",
                    (key, value, self._clock()),
                )
        except sqlite3.Error as exc:
            raise CacheError(f"cache write failed: {exc}") from exc

    def delete(self, key: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        except sqlite3.Error as exc:
            raise CacheError(f"cache delete failed: {exc}") from exc

    def purge_older_than(self, max_age_seconds: float) -> int:
        """Delete entries older than ``max_age_seconds``; return how many were removed."""
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM cache WHERE stored_at < ?", (self._clock() - max_age_seconds,)
                )
                return cur.rowcount
        except sqlite3.Error as exc:
            raise CacheError(f"cache purge failed: {exc}") from exc

    def is_fresh(self, entry: CacheEntry, ttl_seconds: float) -> bool:
        return entry.age_seconds(self._clock()) < ttl_seconds
