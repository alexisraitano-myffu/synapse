import apsw
import sqlite_vec
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, EMBEDDING_DIM


def get_connection() -> apsw.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(DB_PATH))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    return conn


def cursor_to_dicts(cursor: apsw.Cursor) -> list[dict]:
    try:
        cols = [d[0] for d in cursor.description]
    except apsw.ExecutionCompleteError:
        return []
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def first_row(cursor: apsw.Cursor) -> dict | None:
    """Return the first row as a dict, or None if the result set is empty."""
    try:
        cols = [d[0] for d in cursor.description]
    except apsw.ExecutionCompleteError:
        return None
    row = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


def init_db() -> None:
    conn = get_connection()
    try:
        for stmt in [
            """CREATE TABLE IF NOT EXISTS inbox (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                content      TEXT NOT NULL,
                source       TEXT NOT NULL DEFAULT 'manual',
                raw_file     BLOB,
                created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS atomic_notes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT,
                content    TEXT NOT NULL,
                source_ids TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            # Parallel vector index — rowid mirrors atomic_notes.id
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS atomic_notes_vec
                USING vec0(embedding float[{EMBEDDING_DIM}])""",
            """CREATE TABLE IF NOT EXISTS knowledge_graph (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_a   TEXT NOT NULL,
                relation   TEXT NOT NULL,
                entity_b   TEXT NOT NULL,
                context    TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            # ── Phase A+ — Entity graph ────────────────────────────────────────
            """CREATE TABLE IF NOT EXISTS entities (
                id                TEXT PRIMARY KEY,
                type              TEXT,
                canonical_name    TEXT NOT NULL,
                aliases           TEXT DEFAULT '[]',
                attributes        TEXT DEFAULT '{}',
                mention_count     INTEGER DEFAULT 1,
                last_mentioned    DATE,
                persistence_value INTEGER DEFAULT 3,
                summary           TEXT,
                embedding         BLOB,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS facts (
                id                TEXT PRIMARY KEY,
                entity_id         TEXT REFERENCES entities(id),
                predicate         TEXT NOT NULL,
                value             TEXT NOT NULL,
                confidence        REAL DEFAULT 0.5,
                source_inbox_id   TEXT,
                persistence_value INTEGER DEFAULT 3,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_confirmed    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS relations (
                id          TEXT PRIMARY KEY,
                entity_from TEXT REFERENCES entities(id),
                predicate   TEXT NOT NULL,
                entity_to   TEXT REFERENCES entities(id),
                confidence  REAL DEFAULT 0.5,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS resources (
                id                TEXT PRIMARY KEY,
                type              TEXT,
                source            TEXT,
                title             TEXT,
                summary           TEXT,
                tags              TEXT DEFAULT '[]',
                entities_mentioned TEXT DEFAULT '[]',
                embedding         BLOB,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS pending_facts (
                id                  TEXT PRIMARY KEY,
                fact_data           TEXT NOT NULL,
                validation_strategy TEXT DEFAULT 'passive',
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS review_queue (
                id               TEXT PRIMARY KEY,
                fact_data        TEXT NOT NULL,
                suggested_entity TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS intentions (
                id         TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                ttl_hours  INTEGER DEFAULT 48,
                resolved   BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
        ]:
            conn.execute(stmt)

        # Migrations for columns added after initial schema
        for migration in [
            "ALTER TABLE inbox ADD COLUMN processed_at TIMESTAMP",
            "ALTER TABLE inbox ADD COLUMN raw_file BLOB",
        ]:
            try:
                conn.execute(migration)
            except apsw.SQLError:
                pass  # column already present
    finally:
        conn.close()
