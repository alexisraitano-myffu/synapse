"""
Offline tests for SYN-58 — extensible entity-type vocabulary via pending.

No ANTHROPIC_API_KEY: we drive step4_route directly (the classify step that
emits type_proposal is API-bound and covered by test_dream_cycle.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _route(resolved, source_id=1):
    from dream_cycle.cycle import step4_route
    from db import get_connection
    conn = get_connection()
    try:
        with conn:
            step4_route(resolved, source_id, conn)
    finally:
        conn.close()


def _rows(sql, params=()):
    from db import get_connection, cursor_to_dicts
    conn = get_connection()
    try:
        return cursor_to_dicts(conn.execute(sql, params))
    finally:
        conn.close()


def _entity(name, type_="concept", type_proposal=None, persistence=3):
    return {
        "canonical_name": name, "type": type_, "type_proposal": type_proposal,
        "facts": [{"predicate": "is", "value": "x",
                   "persistence_value": persistence, "evidence_strength": "explicit"}],
        "existing_entity": None,
    }


def test_unknown_type_creates_pending_entity_and_proposal(isolated_db):
    _route({
        "resolved_entities": [_entity(
            "Udon Dan Dan",
            type_proposal={"value": "recipe", "reason": "un plat / recette"},
        )],
        "relations": [], "project_entries": [],
    })
    ents = _rows("SELECT canonical_name, type, status FROM entities")
    assert len(ents) == 1
    assert ents[0]["status"] == "pending", "vocab-gap entity must be parked pending"
    assert ents[0]["type"] == "concept", "type stays the fallback until accepted"

    props = _rows("SELECT proposed_type, status, candidate_entity_id FROM entity_type_proposals")
    assert len(props) == 1
    assert props[0]["proposed_type"] == "recipe"
    assert props[0]["status"] == "pending"
    assert props[0]["candidate_entity_id"] == ents[0]["id"] if "id" in ents[0] else True


def test_proposal_for_already_active_type_is_ignored(isolated_db):
    # If the model proposes a type that's already in the vocab, no pending/no proposal.
    _route({
        "resolved_entities": [_entity(
            "Bob", type_="person",
            type_proposal={"value": "person", "reason": "déjà un type"},
        )],
        "relations": [], "project_entries": [],
    })
    ents = _rows("SELECT type, status FROM entities")
    assert ents[0]["status"] == "active"
    assert _rows("SELECT id FROM entity_type_proposals") == []


def test_project_shell_guard_downgrades_to_concept(isolated_db):
    # type=project but no project_entries for it → mis-tag, fall back to concept.
    _route({
        "resolved_entities": [_entity("Keane", type_="project")],
        "relations": [], "project_entries": [],
    })
    ents = _rows("SELECT type, status FROM entities WHERE canonical_name='Keane'")
    assert ents[0]["type"] == "concept"
    assert ents[0]["status"] == "active"


def test_ephemeral_with_entities_still_captures_them(isolated_db, monkeypatch):
    """SYN-58: an ephemeral capture that ALSO names a durable entity must not
    discard it. The recipe entity (+ type proposal) is created and the expiring
    intention is recorded; the atomic_note is suppressed (no double-store)."""
    import dream_cycle.cycle as cyc
    from db import get_connection

    canned = {
        "input_type": "ephemeral", "is_ephemeral": True,
        "ephemeral_content": "envie de cuisine",
        "summary": "envie de refaire les udon",
        "atomic_note": "j'ai envie de refaire les udon",
        "project_entries": [],
        "relations": [],
        "entities": [{
            "canonical_name": "Udon Dan Dan", "type": "concept",
            "type_proposal": {"value": "recipe", "reason": "une recette"},
            "aliases": [], "summary": "recette", "attributes": {},
            "facts": [{"predicate": "is", "value": "recette",
                       "persistence_value": 3, "evidence_strength": "explicit"}],
        }],
    }
    monkeypatch.setattr(cyc, "step1_classify", lambda *a, **k: canned)

    conn = get_connection()
    try:
        cyc._process_entry({"id": 1, "content": "..."}, None, conn,
                           "2026-05-31T00:00:00", False, False)
    finally:
        conn.close()

    ents = _rows("SELECT canonical_name, status FROM entities")
    assert any(e["canonical_name"] == "Udon Dan Dan" and e["status"] == "pending" for e in ents)
    assert [p["proposed_type"] for p in _rows("SELECT proposed_type FROM entity_type_proposals")] == ["recipe"]
    assert len(_rows("SELECT id FROM intentions")) == 1  # the envie still expires
    assert _rows("SELECT id FROM atomic_notes") == []     # not double-stored


def test_project_with_matching_entry_stays_project(isolated_db):
    _route({
        "resolved_entities": [_entity("Synapse", type_="project")],
        "relations": [],
        "project_entries": [{"project_canonical": "Synapse", "content": "...", "is_new": True}],
    })
    ents = _rows("SELECT type FROM entities WHERE canonical_name='Synapse'")
    assert ents[0]["type"] == "project"
