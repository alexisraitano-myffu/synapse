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
            # Append-only log of user validations (survives a rebuild, replicates).
            """CREATE TABLE IF NOT EXISTS validation_events (
                id               TEXT PRIMARY KEY,
                fact_id          TEXT,
                entity_canonical TEXT,
                predicate        TEXT,
                value            TEXT,
                confirmed        INTEGER NOT NULL,
                correction       TEXT,
                device_id        TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            # One row per Dream Cycle run (stats for the app's "last/next cycle").
            """CREATE TABLE IF NOT EXISTS cycle_runs (
                id                TEXT PRIMARY KEY,
                started_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                finished_at       TIMESTAMP,
                notes_processed   INTEGER DEFAULT 0,
                entities_total    INTEGER DEFAULT 0,
                pending_total     INTEGER DEFAULT 0,
                status            TEXT DEFAULT 'running',
                trigger           TEXT DEFAULT 'manual',
                error             TEXT
            )""",
            # ── SYN-41 — Projects as aggregate entities ─────────────────────────
            # A project is an entity (type='project') that receives a timeline of
            # entries plus a versioned synthesis. Captures stay in inbox (the
            # immutable source of truth); projections (entries, summaries, facts,
            # atomic_notes, entities, relations) carry a provenance_capture_id
            # back to the inbox row so we never lose lineage.
            """CREATE TABLE IF NOT EXISTS project_entries (
                id                TEXT PRIMARY KEY,
                project_id        TEXT NOT NULL REFERENCES entities(id),
                capture_id        INTEGER NOT NULL REFERENCES inbox(id),
                content           TEXT NOT NULL,
                kind              TEXT NOT NULL DEFAULT 'note',
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS project_state_versions (
                id                TEXT PRIMARY KEY,
                project_id        TEXT NOT NULL REFERENCES entities(id),
                summary_md        TEXT NOT NULL,
                entry_count       INTEGER NOT NULL,
                trigger           TEXT NOT NULL,  -- 'passive' | 'mcp' | 'manual'
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS project_state (
                project_id          TEXT PRIMARY KEY REFERENCES entities(id),
                current_version_id  TEXT REFERENCES project_state_versions(id),
                updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entry_count_at_sync INTEGER DEFAULT 0
            )""",
            # ── SYN-39 — Entity merge proposals ─────────────────────────────────
            # Detected duplicates surface as pending proposals the user accepts,
            # rejects, or postpones. Acceptance reroutes facts/relations/notes
            # from the absorbed entity to the canonical one; the absorbed entity
            # stays in DB with merged_into_id pointing to the survivor (soft
            # link, no DELETE, so a future unmerge stays possible).
            """CREATE TABLE IF NOT EXISTS entity_merge_proposals (
                id                    TEXT PRIMARY KEY,
                candidate_entity_id   TEXT NOT NULL REFERENCES entities(id),
                existing_entity_id    TEXT NOT NULL REFERENCES entities(id),
                similarity_score      REAL NOT NULL,
                similarity_reason     TEXT,
                evidence_capture_id   INTEGER REFERENCES inbox(id),
                status                TEXT NOT NULL DEFAULT 'pending',
                created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at           TIMESTAMP,
                resolved_canonical_id TEXT REFERENCES entities(id)
            )""",
            # SYN-58: the entity-type vocabulary is no longer a closed enum baked
            # into the classifier prompt. `active_entity_types` is the live vocab
            # (built-in + user-validated); the prompt reads it at runtime. When the
            # classifier finds no fitting type it raises an `entity_type_proposals`
            # row (the candidate entity is created in status='pending') instead of
            # forcing a wrong type — the user validates to extend the vocab.
            """CREATE TABLE IF NOT EXISTS active_entity_types (
                type        TEXT PRIMARY KEY,
                source      TEXT NOT NULL,            -- 'builtin' | 'user'
                added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS entity_type_proposals (
                id                  TEXT PRIMARY KEY,
                proposed_type       TEXT NOT NULL,
                reason              TEXT,
                evidence_capture_id INTEGER REFERENCES inbox(id),
                candidate_entity_id TEXT REFERENCES entities(id),
                status              TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | rejected
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at         TIMESTAMP
            )""",
        ]:
            conn.execute(stmt)

        # SYN-58: seed the live vocabulary with the six built-in types. Idempotent
        # (INSERT OR IGNORE) so re-running init_db never disturbs user-added types.
        for builtin in ("person", "place", "project", "concept", "organization", "animal"):
            conn.execute(
                "INSERT OR IGNORE INTO active_entity_types (type, source) VALUES (?, 'builtin')",
                (builtin,),
            )

        # Migration: add processed_at if DB was created before this column existed
        try:
            conn.execute("ALTER TABLE inbox ADD COLUMN processed_at TIMESTAMP")
        except apsw.SQLError:
            pass  # column already present

        # Migration: episodic-memory columns on atomic_notes (spec §3.1 / §7).
        # entities_mentioned links a note to graph entities; memory_strength is
        # the Ebbinghaus retention score (decay logic is Phase C — defaults to 1.0).
        for col, ddl in [
            ("summary", "ALTER TABLE atomic_notes ADD COLUMN summary TEXT"),
            ("entities_mentioned",
             "ALTER TABLE atomic_notes ADD COLUMN entities_mentioned TEXT DEFAULT '[]'"),
            ("memory_strength",
             "ALTER TABLE atomic_notes ADD COLUMN memory_strength REAL DEFAULT 1.0"),
        ]:
            try:
                conn.execute(ddl)
            except apsw.SQLError:
                pass  # column already present

        # Migration: sync columns on inbox (client_id enables idempotent capture).
        for ddl in [
            "ALTER TABLE inbox ADD COLUMN client_id TEXT",
            "ALTER TABLE inbox ADD COLUMN device_id TEXT",
            "ALTER TABLE inbox ADD COLUMN captured_at TIMESTAMP",
            "ALTER TABLE inbox ADD COLUMN status TEXT DEFAULT 'queued'",
        ]:
            try:
                conn.execute(ddl)
            except apsw.SQLError:
                pass  # column already present

        # Idempotency: at most one inbox row per client-generated capture id.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_inbox_client_id "
            "ON inbox(client_id) WHERE client_id IS NOT NULL"
        )

        # Migration: SYN-41 — provenance back-link to the immutable inbox row.
        # Existing rows keep NULL (we only commit to the invariant going forward).
        for ddl in [
            "ALTER TABLE entities     ADD COLUMN provenance_capture_id INTEGER REFERENCES inbox(id)",
            "ALTER TABLE facts        ADD COLUMN provenance_capture_id INTEGER REFERENCES inbox(id)",
            "ALTER TABLE atomic_notes ADD COLUMN provenance_capture_id INTEGER REFERENCES inbox(id)",
            "ALTER TABLE relations    ADD COLUMN provenance_capture_id INTEGER REFERENCES inbox(id)",
        ]:
            try:
                conn.execute(ddl)
            except apsw.SQLError:
                pass  # column already present

        # Timeline access: fetch a project's entries in reverse-chrono order.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_entries_project "
            "ON project_entries(project_id, created_at DESC)"
        )

        # Migration: SYN-44 — distinguish append (incremental) from refinement
        # (from-scratch rebuild) on project_state_versions. Both stay
        # trigger='passive' or 'mcp' or 'manual' — kind is orthogonal.
        try:
            conn.execute(
                "ALTER TABLE project_state_versions "
                "ADD COLUMN kind TEXT NOT NULL DEFAULT 'append'"
            )
        except apsw.SQLError:
            pass  # column already present

        # Migration: SYN-39 — soft-link a merged entity to its absorber. Queries
        # filter on `merged_into_id IS NULL` to hide the absorbed row, but the
        # data stays so we can rebuild lineage and (eventually) unmerge.
        for ddl in [
            "ALTER TABLE entities ADD COLUMN merged_into_id TEXT REFERENCES entities(id)",
            "ALTER TABLE entities ADD COLUMN merged_at TIMESTAMP",
        ]:
            try:
                conn.execute(ddl)
            except apsw.SQLError:
                pass  # column already present

        # Migration: SYN-58 — entity lifecycle status. 'active' is the default and
        # the only value the read views surface; 'pending' = awaiting a type-vocab
        # decision, 'archived' = rejected. Existing rows backfill to 'active'.
        try:
            conn.execute(
                "ALTER TABLE entities ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
            )
        except apsw.SQLError:
            pass  # column already present

        # Quick lookups for the merge-proposals queue.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_merge_proposals_status "
            "ON entity_merge_proposals(status, created_at DESC)"
        )
        # Quick lookups for the type-proposals queue (SYN-58).
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_type_proposals_status "
            "ON entity_type_proposals(status, created_at DESC)"
        )
    finally:
        conn.close()
