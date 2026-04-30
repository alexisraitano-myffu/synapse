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
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def init_db() -> None:
    conn = get_connection()
    try:
        for stmt in [
            """CREATE TABLE IF NOT EXISTS inbox (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                content      TEXT NOT NULL,
                source       TEXT NOT NULL DEFAULT 'manual',
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
        ]:
            conn.execute(stmt)

        # Migration: add processed_at if DB was created before this column existed
        try:
            conn.execute("ALTER TABLE inbox ADD COLUMN processed_at TIMESTAMP")
        except apsw.SQLError:
            pass  # column already present
    finally:
        conn.close()
