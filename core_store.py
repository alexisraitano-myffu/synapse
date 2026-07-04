"""
Access to the Rust core's storage substrate (SYN-110 / T1).

The compiled core (`synapse_core`, the synapse-core PyO3 wheel) owns the
SQLite schema and every vector read/write: the vec0 KNN over atomic_notes and
the entity/resource embedding columns + similarity scans. Python keeps its own
apsw connections for all non-vector SQL against the same database file.

One `Storage` handle per database path, cached for the process: opening runs
the idempotent schema init/migration, and the handle serializes its internal
connection, so per-request re-opens would only add overhead. Keyed by path
(not a singleton) because the test suite points `db.DB_PATH` at a fresh
temporary database per test.
"""

from synapse_core import Storage

_stores: dict[str, Storage] = {}


def get_store() -> Storage:
    """The core storage handle for the current database path."""
    import db  # late import: db.DB_PATH is monkeypatched by the test fixtures

    path = str(db.DB_PATH)
    store = _stores.get(path)
    if store is None:
        store = Storage(path)
        _stores[path] = store
    return store
