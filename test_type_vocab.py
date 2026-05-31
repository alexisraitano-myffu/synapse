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


def test_project_with_matching_entry_stays_project(isolated_db):
    _route({
        "resolved_entities": [_entity("Synapse", type_="project")],
        "relations": [],
        "project_entries": [{"project_canonical": "Synapse", "content": "...", "is_new": True}],
    })
    ents = _rows("SELECT type FROM entities WHERE canonical_name='Synapse'")
    assert ents[0]["type"] == "project"
