"""
Non-regression tests for the unified Dream Cycle structure (Chantier A).

These run OFFLINE (no ANTHROPIC_API_KEY): they cover the package layout, the
episodic-memory schema migration, and the episodic-note writer — none of which
call the Claude API. The full classify→route pipeline (which does call the API)
lives in test_dream_cycle.py and is skipped without a key.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ── Package layout (guards the shadowing bug we fixed) ───────────────────────

def test_dream_cycle_package_not_shadowed():
    """`import dream_cycle` must resolve to the package, exporting the cycle."""
    import dream_cycle
    assert dream_cycle.__file__.endswith("dream_cycle/__init__.py")
    from dream_cycle import run_dream_cycle, main
    assert callable(run_dream_cycle) and callable(main)


# ── Schema migration (spec §3.1 episodic atomic_notes) ───────────────────────

def test_atomic_notes_has_episodic_columns(isolated_db):
    from db import get_connection
    conn = get_connection()
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(atomic_notes)")}
    finally:
        conn.close()
    assert {"summary", "entities_mentioned", "memory_strength"} <= cols


# ── Episodic note writer ─────────────────────────────────────────────────────

def test_write_episodic_note_creates_vectorized_note(isolated_db):
    from dream_cycle.cycle import write_episodic_note
    from db import get_connection

    classified = {
        "input_type": "episodic",
        "summary": "Dîné avec Marie à Lyon",
        "entities": [{"canonical_name": "Marie"}, {"canonical_name": "Lyon"}],
    }
    entry = {"id": 1, "content": "Hier soir, super dîner avec Marie à Lyon."}

    conn = get_connection()
    try:
        with conn:
            write_episodic_note(classified, entry, conn)

        row = conn.execute(
            "SELECT id, summary, entities_mentioned, memory_strength FROM atomic_notes"
        ).fetchone()
        assert row is not None, "episodic note was not written"
        note_id, summary, entities_mentioned, memory_strength = row

        assert summary == "Dîné avec Marie à Lyon"
        assert set(json.loads(entities_mentioned)) == {"Marie", "Lyon"}
        assert memory_strength == 1.0

        vec = conn.execute(
            "SELECT embedding FROM atomic_notes_vec WHERE rowid = ?", (note_id,)
        ).fetchone()
        assert vec is not None, "episodic note was not vectorized"
    finally:
        conn.close()


def test_episodic_note_is_searchable(isolated_db):
    from dream_cycle.cycle import write_episodic_note
    from db import get_connection

    classified = {
        "input_type": "episodic",
        "summary": "Dîné avec Marie à Lyon",
        "entities": [{"canonical_name": "Marie"}],
    }
    entry = {"id": 1, "content": "Hier soir, super dîner avec Marie à Lyon."}

    conn = get_connection()
    try:
        with conn:
            write_episodic_note(classified, entry, conn)
    finally:
        conn.close()

    import mcp_server.server as server
    search = getattr(server.search_memory, "fn", server.search_memory)
    results = json.loads(search("repas avec Marie", limit=3))

    assert results, "expected the episodic note to be searchable"
    assert results[0]["search_type"] == "vector"


# ── Entity creation (decoupled from fact confidence) ─────────────────────────
# step4_route is pure DB logic — testable offline by hand-building `resolved`.

def _route(resolved: dict, source_id: int = 1):
    from dream_cycle.cycle import step4_route
    from db import get_connection
    conn = get_connection()
    try:
        with conn:
            step4_route(resolved, source_id, conn)
    finally:
        conn.close()


def _entities() -> list[dict]:
    from db import get_connection, cursor_to_dicts
    conn = get_connection()
    try:
        return cursor_to_dicts(conn.execute("SELECT * FROM entities"))
    finally:
        conn.close()


def test_entity_created_on_mention_even_without_high_confidence(isolated_db):
    """A first-mention fact (conf 0.75 < 0.85) still creates the entity node;
    the fact itself lands in pending."""
    resolved = {
        "resolved_entities": [{
            "canonical_name": "Marie", "type": "person", "aliases": ["maman"],
            "summary": "La mère de l'utilisateur", "attributes": {"role": "mère"},
            "facts": [{"predicate": "has_birthday", "value": "2026-05-15", "persistence_value": 5}],
            "existing_entity": None,
        }],
        "relations": [],
    }
    _route(resolved)

    ents = _entities()
    assert len(ents) == 1
    marie = ents[0]
    assert marie["canonical_name"] == "Marie"
    assert marie["summary"] == "La mère de l'utilisateur"
    assert json.loads(marie["attributes"])["role"] == "mère"
    assert marie["persistence_value"] == 5

    # the fact is not confirmed yet — it should be pending, not in facts
    from db import get_connection
    conn = get_connection()
    try:
        facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM pending_facts").fetchone()[0]
    finally:
        conn.close()
    assert facts == 0 and pending == 1


def test_noise_entity_is_not_created(isolated_db):
    """An entity whose only fact is pure noise (persistence 1) and that is not in
    a relation must be skipped — anti-pollution garde-fou."""
    resolved = {
        "resolved_entities": [{
            "canonical_name": "Truc entendu en passant", "type": "concept",
            "aliases": [], "summary": None, "attributes": {},
            "facts": [{"predicate": "mentioned", "value": "x", "persistence_value": 1}],
            "existing_entity": None,
        }],
        "relations": [],
    }
    _route(resolved)
    assert _entities() == []


def test_relation_only_entity_is_created(isolated_db):
    """Entities that appear only in a relation (no standalone fact) are created so
    the relation has both endpoints."""
    resolved = {
        "resolved_entities": [
            {"canonical_name": "Marie", "type": "person", "aliases": [],
             "summary": None, "attributes": {}, "facts": [], "existing_entity": None},
            {"canonical_name": "Alexis", "type": "person", "aliases": [],
             "summary": None, "attributes": {}, "facts": [], "existing_entity": None},
        ],
        "relations": [{"from": "Marie", "predicate": "mere_de", "to": "Alexis"}],
    }
    _route(resolved)

    assert len(_entities()) == 2
    from db import get_connection
    conn = get_connection()
    try:
        rel_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    finally:
        conn.close()
    assert rel_count == 1


def test_nameless_entity_skipped(isolated_db):
    resolved = {
        "resolved_entities": [{
            "canonical_name": "  ", "type": "concept", "aliases": [],
            "summary": None, "attributes": {},
            "facts": [{"predicate": "p", "value": "v", "persistence_value": 5}],
            "existing_entity": None,
        }],
        "relations": [],
    }
    _route(resolved)
    assert _entities() == []
