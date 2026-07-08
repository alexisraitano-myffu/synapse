"""
Database access, backed by the Rust core (SYN-110).

The core (`synapse_core`) owns BOTH the schema and the only SQLite library in
the process. That single-library rule is not a style choice: two SQLite
builds in one process (e.g. apsw + the core's bundled SQLite) do not see each
other's POSIX locks — same-process advisory locks don't conflict — so their
transactions interleave and corrupt the database file. Every SQL statement
therefore flows through the core's gateway; apsw and the pip sqlite-vec
extension are gone (vec0 is compiled into the core and available to raw SQL
here too).

`Connection`/`Cursor` keep the small apsw surface the codebase always used:
`execute(sql, params)`, `fetchone/fetchall/description`, iteration,
`last_insert_rowid`, `close`, and `with conn:` transactions (savepoints when
nested, exactly like apsw).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import synapse_core

from config import DB_PATH, EMBEDDING_DIM  # noqa: F401 — EMBEDDING_DIM re-exported


class ExecutionCompleteError(Exception):
    """Raised by `Cursor.description` when there is no result set (the apsw
    behavior `cursor_to_dicts`/`first_row` were built around)."""


class Cursor:
    """apsw-shaped cursor over an eagerly-fetched result set."""

    def __init__(self, columns: list[str] | None, rows: list):
        self._columns = columns
        self._rows = rows
        self._next = 0

    @property
    def description(self):
        if self._columns is None:
            raise ExecutionCompleteError("statement produced no result set")
        return [(name, None) for name in self._columns]

    def fetchone(self):
        if self._next < len(self._rows):
            row = self._rows[self._next]
            self._next += 1
            return tuple(row)
        return None

    def fetchall(self):
        rows = [tuple(r) for r in self._rows[self._next:]]
        self._next = len(self._rows)
        return rows

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row


class Connection:
    """apsw-shaped adapter over one core SQL connection.

    `with conn:` == transaction (BEGIN/COMMIT/ROLLBACK), nesting handled with
    savepoints like apsw. Outside a block every statement autocommits.
    """

    def __init__(self, db_path):
        self._conn = synapse_core.connect(str(db_path))
        self._txn_depth = 0

    def execute(self, sql: str, params=()) -> Cursor:
        columns, rows = self._conn.execute(sql, list(params))
        return Cursor(columns, rows)

    def last_insert_rowid(self) -> int:
        return self._conn.last_insert_rowid()

    def insert_fact(self, *, entity_id, predicate, value, confidence,
                    source_inbox_id=None, persistence_value=3,
                    provenance_capture_id=None, category=None) -> str:
        """Fact write via the core (dedup-reinforce + SYN-37 supersede,
        `routing::insert_fact`) executed on THIS connection: an open
        `with conn:` transaction wraps it. T5: the Python implementation
        (facts_store) is gone."""
        import json

        return self._conn.insert_fact(
            entity_id, predicate, json.dumps(value), confidence,
            json.dumps(source_inbox_id), persistence_value,
            provenance_capture_id, json.dumps(category))

    # SYN-19/68 decay (core decay.rs), executed on THIS connection — the
    # caller's open transaction wraps the writes. `now` = 'YYYY-MM-DD HH:MM:SS'
    # or None (system clock).
    def apply_decay(self, tau_days=None, now=None) -> int:
        return self._conn.apply_decay(tau_days, now)

    def apply_entity_decay(self, tau_days=None, now=None) -> int:
        return self._conn.apply_entity_decay(tau_days, now)

    def reactivate_notes(self, note_ids, factor=1.0, now=None) -> int:
        return self._conn.reactivate_notes([str(n) for n in note_ids], factor, now)

    def reactivate_notes_for_entities(self, entity_names, now=None) -> int:
        return self._conn.reactivate_notes_for_entities(list(entity_names), now)

    def gather_week(self, now=None, days=7) -> str:
        """SYN-23 — the digest's structured week as a JSON string (pure SQL in
        the core, `digest.rs::gather_week`, on THIS connection)."""
        return self._conn.gather_week(now, days)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        if self._txn_depth == 0:
            self.execute("BEGIN")
        else:
            self.execute(f"SAVEPOINT sp_{self._txn_depth}")
        self._txn_depth += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self._txn_depth -= 1
        depth = self._txn_depth
        if exc_type is None:
            self.execute("COMMIT" if depth == 0 else f"RELEASE sp_{depth}")
        elif depth == 0:
            self.execute("ROLLBACK")
        else:
            self.execute(f"ROLLBACK TO sp_{depth}")
            self.execute(f"RELEASE sp_{depth}")
        return False


def get_connection() -> Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return Connection(DB_PATH)


def cursor_to_dicts(cursor: Cursor) -> list[dict]:
    try:
        cols = [d[0] for d in cursor.description]
    except ExecutionCompleteError:
        return []
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def first_row(cursor: Cursor) -> dict | None:
    """Return the first row as a dict, or None if the result set is empty."""
    try:
        cols = [d[0] for d in cursor.description]
    except ExecutionCompleteError:
        return None
    row = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


def init_db() -> None:
    """Create/migrate the schema, now owned by the Rust core (SYN-110).

    The DDL lives in synapse-core (`crates/synapse-core/src/schema.rs`), the
    exact port of the idempotent CREATE/ALTER sequence that used to live here.
    Opening the core store runs it; this wrapper keeps the historical call
    sites (MCP startup, Dream Cycle, tests) unchanged. Do NOT add DDL here —
    schema changes go into the core.
    """
    from core_store import get_brain, get_store

    get_store()
    # Warm the Brain here too: opening it runs (idempotent) schema writes,
    # which must never happen lazily inside a caller's open transaction
    # (SQLITE_BUSY). After this, get_brain() is a cache hit everywhere.
    get_brain()
