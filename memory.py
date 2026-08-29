"""
JARVIS — Packet F: Memory (SQLite)
==================================
Persistent storage for preferences, frequently-used apps, research interests,
recurring tasks, the assistant's name, etc. Pure standard library — no install.

Two stores, one tiny schema:
  * prefs(key, value)        -> single-value settings   (get / set / delete)
  * items(category, item)    -> categorized lists with a usage count
                                (remember / recall / forget)

Why a count on items? The goal asks for "frequently-used apps" and "recurring
tasks", so remember() increments a count when the same item is stored again,
and recall() returns items most-used-first. That makes "what apps do I use
most" fall out for free without complicating the schema.

Usage (the flat interface other modules import):
    from memory import get, set, remember, recall

    set("assistant_name", "Jarvis")
    get("assistant_name")                  -> "Jarvis"
    get("missing", default="?")            -> "?"

    remember("apps", "chrome")
    remember("apps", "chrome")             # used again -> count goes up
    remember("apps", "vscode")
    recall("apps")                         -> ["chrome", "vscode"]   (most-used first)

Values in prefs are JSON-encoded, so set/get round-trips real types
(str, int, float, bool, list, dict), not just strings.
"""

import os
import json
import sqlite3
import threading
from typing import Any, Optional

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_memory.db")


class Memory:
    """SQLite-backed store. Safe to call from multiple threads.

    Pass ":memory:" for an ephemeral DB (used in tests). For persistence,
    pass a file path (the module-level functions default to jarvis_memory.db).
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        # check_same_thread=False + a lock: Jarvis runs TTS/listen on separate
        # threads, and any of them may touch memory. One connection guarded by
        # a lock is simplest and keeps :memory: alive for the object's lifetime.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS prefs (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,           -- JSON-encoded
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS items (
                    category   TEXT NOT NULL,
                    item       TEXT NOT NULL,
                    count      INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (category, item)
                );
                """
            )

    # ---- key/value preferences -------------------------------------------
    def set(self, key: str, value: Any) -> None:
        """Store (or overwrite) a single value. Value may be any JSON type."""
        payload = json.dumps(value)
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO prefs (key, value, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(key) DO UPDATE SET
                       value=excluded.value, updated_at=datetime('now')""",
                (key, payload),
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Return the stored value (decoded) or `default` if absent."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM prefs WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]  # tolerate a non-JSON legacy value

    def delete(self, key: str) -> bool:
        """Remove a preference. Returns True if something was deleted."""
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM prefs WHERE key = ?", (key,))
        return cur.rowcount > 0

    # ---- categorized lists -----------------------------------------------
    def remember(self, category: str, item: str) -> None:
        """Add `item` to `category`. Re-adding the same item bumps its count
        (this is how 'frequently used' is tracked)."""
        item = str(item)
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO items (category, item)
                   VALUES (?, ?)
                   ON CONFLICT(category, item) DO UPDATE SET
                       count = count + 1, updated_at = datetime('now')""",
                (category, item),
            )

    def recall(self, category: str, limit: Optional[int] = None,
               with_counts: bool = False):
        """Return items in `category`, most-used first (ties: most recent).

        with_counts=False -> ["chrome", "vscode"]
        with_counts=True  -> [("chrome", 4), ("vscode", 1)]
        """
        sql = ("SELECT item, count FROM items WHERE category = ? "
               "ORDER BY count DESC, updated_at DESC")
        params: tuple = (category,)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        if with_counts:
            return [(r["item"], r["count"]) for r in rows]
        return [r["item"] for r in rows]

    def forget(self, category: str, item: Optional[str] = None) -> int:
        """Remove one item, or the whole category if item is None.
        Returns number of rows removed."""
        with self._lock, self._conn:
            if item is None:
                cur = self._conn.execute(
                    "DELETE FROM items WHERE category = ?", (category,))
            else:
                cur = self._conn.execute(
                    "DELETE FROM items WHERE category = ? AND item = ?",
                    (category, str(item)))
        return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ===========================================================================
# Module-level flat interface (the one the contract specifies).
# Backed by a single shared Memory on the default DB file.
# ===========================================================================
_default: Optional[Memory] = None
_default_lock = threading.Lock()


def _store() -> Memory:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Memory(DEFAULT_DB)
    return _default


def set(key: str, value: Any) -> None:           # noqa: A001 (intentional name)
    _store().set(key, value)


def get(key: str, default: Any = None) -> Any:
    return _store().get(key, default)


def delete(key: str) -> bool:
    return _store().delete(key)


def remember(category: str, item: str) -> None:
    _store().remember(category, item)


def recall(category: str, limit: Optional[int] = None, with_counts: bool = False):
    return _store().recall(category, limit=limit, with_counts=with_counts)


def forget(category: str, item: Optional[str] = None) -> int:
    return _store().forget(category, item)


if __name__ == "__main__":
    # Tiny smoke demo against a throwaway in-memory store.
    m = Memory(":memory:")
    m.set("assistant_name", "Jarvis")
    m.set("brightness_pref", 70)
    m.remember("apps", "chrome"); m.remember("apps", "chrome"); m.remember("apps", "vscode")
    print("name        :", m.get("assistant_name"))
    print("brightness  :", m.get("brightness_pref"), type(m.get("brightness_pref")).__name__)
    print("missing     :", m.get("nope", default="(default)"))
    print("apps        :", m.recall("apps"))
    print("apps+counts :", m.recall("apps", with_counts=True))
    m.close()
