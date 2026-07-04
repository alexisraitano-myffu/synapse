"""
Offline tests for the SYN-61 embedding fallback in entity merge proposals.

No ANTHROPIC_API_KEY needed — embeddings are local. SYN-111: the merge scan
runs inside the core's routing; we drive `_process_entry` with a classified
that creates the new entity (a persistence≥2 fact passes the creation gate)
and assert on the proposals it raises against pre-inserted candidates.
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _insert(conn, name, type_="person", summary="", with_embedding=False):
    """Insert an entity; optionally vectorize it (candidates need an embedding,
    the freshly-created entity does not — it's embedded on the fly)."""
    from embeddings import embed_text
    from entity_search import entity_embedding_text

    eid = str(uuid.uuid4())
    row = {"canonical_name": name, "type": type_,
           "aliases": "[]", "attributes": "{}", "summary": summary}
    vec = embed_text(entity_embedding_text(row)) if with_embedding else None
    conn.execute(
        "INSERT INTO entities (id, type, canonical_name, aliases, attributes, "
        "summary, embedding) VALUES (?,?,?,?,?,?,?)",
        (eid, type_, name, "[]", "{}", summary, vec),
    )
    return eid



def _route_new_entity(name, type_, summary, source_id=2):
    """Create `name` through the core routing (which then runs the merge scan
    against the committed candidates)."""
    from datetime import datetime, timezone
    from db import get_connection
    from dream_cycle import cycle
    classified = {
        "input_type": "fact",
        "entities": [{
            "canonical_name": name, "type": type_, "aliases": [],
            "summary": summary, "attributes": {},
            "facts": [{"predicate": "is", "value": "x",
                       "persistence_value": 3, "evidence_strength": "explicit"}],
        }],
        "relations": [], "project_entries": [],
    }
    conn = get_connection()
    try:
        cycle._process_entry({"id": source_id, "content": "capture de test"},
                             None, conn, datetime.now(timezone.utc).isoformat(),
                             False, False, classified=classified)
    finally:
        conn.close()

def _proposals(conn):
    from db import cursor_to_dicts
    return cursor_to_dicts(conn.execute(
        "SELECT candidate_entity_id, existing_entity_id, similarity_score, "
        "similarity_reason FROM entity_merge_proposals"
    ))


def test_embedding_fallback_proposes_merge_without_substring(isolated_db, monkeypatch):
    """'Marie Dupont' ↔ 'M. Dupont' share no usable substring token, so only the
    embedding fallback can catch them. Near-identical summaries push the cosine
    well past threshold."""
    monkeypatch.setenv("SYNAPSE_MERGE_EMBEDDING_THRESHOLD", "0.7")
    from db import get_connection

    summary = "Amie proche de l'utilisateur, habite à Lyon, travaille dans la finance"
    conn = get_connection()
    try:
        with conn:
            _insert(conn, "Marie Dupont", "person", summary, with_embedding=True)
    finally:
        conn.close()
    _route_new_entity("M. Dupont", "person", summary)
    conn = get_connection()
    try:
        props = _proposals(conn)
        new_id = conn.execute(
            "SELECT id FROM entities WHERE canonical_name='M. Dupont'").fetchone()[0]
    finally:
        conn.close()

    assert len(props) == 1, "embedding fallback should raise exactly one proposal"
    assert props[0]["candidate_entity_id"] == new_id
    assert props[0]["similarity_reason"].startswith("embedding_"), \
        "reason must flag this as an embedding match, not substring"
    assert props[0]["similarity_score"] >= 0.7


def test_substring_wins_over_embedding(isolated_db):
    """When the substring heuristic matches, it fires and the embedding fallback
    is not consulted (reason stays name_substring)."""
    from db import get_connection

    conn = get_connection()
    try:
        with conn:
            _insert(conn, "Martin Bari", "person", "Collègue", with_embedding=True)
    finally:
        conn.close()
    _route_new_entity("Martin", "person", "Collègue")
    conn = get_connection()
    try:
        props = _proposals(conn)
    finally:
        conn.close()

    assert len(props) == 1
    assert props[0]["similarity_reason"] == "name_substring"


def test_embedding_fallback_respects_type_filter(isolated_db, monkeypatch):
    """A semantically similar entity of a *different* type is never proposed."""
    monkeypatch.setenv("SYNAPSE_MERGE_EMBEDDING_THRESHOLD", "0.5")
    from db import get_connection

    summary = "Concept lié à l'escalade et à la grimpe de bloc"
    conn = get_connection()
    try:
        with conn:
            _insert(conn, "Bloc", "concept", summary, with_embedding=True)
    finally:
        conn.close()
    _route_new_entity("Quelqu'un", "person", summary)
    conn = get_connection()
    try:
        props = _proposals(conn)
    finally:
        conn.close()

    assert props == [], "cross-type pair must not be proposed"


def test_high_threshold_blocks_unrelated(isolated_db, monkeypatch):
    """A near-1.0 threshold means unrelated entities raise no proposal."""
    monkeypatch.setenv("SYNAPSE_MERGE_EMBEDDING_THRESHOLD", "0.99")
    from db import get_connection

    conn = get_connection()
    try:
        with conn:
            _insert(conn, "Banque centrale", "concept",
                    "Politique monétaire et taux directeurs", with_embedding=True)
    finally:
        conn.close()
    _route_new_entity("Vélo de route", "concept", "Sport d'endurance en plein air")
    conn = get_connection()
    try:
        props = _proposals(conn)
    finally:
        conn.close()

    assert props == []


def test_embedding_proposal_is_deduped(isolated_db, monkeypatch):
    """Running the fallback twice on the same pair doesn't double-propose."""
    monkeypatch.setenv("SYNAPSE_MERGE_EMBEDDING_THRESHOLD", "0.7")
    from db import get_connection

    summary = "Laboratoire de recherche en intelligence artificielle"
    conn = get_connection()
    try:
        with conn:
            _insert(conn, "OpenAI", "organization", summary, with_embedding=True)
    finally:
        conn.close()
    # Two captures naming the same new entity: the first creates it and raises
    # the proposal; the second resolves to the existing row (no re-scan) — the
    # pair must not be proposed twice either way.
    _route_new_entity("Open AI", "organization", summary, source_id=2)
    _route_new_entity("Open AI", "organization", summary, source_id=3)
    conn = get_connection()
    try:
        props = _proposals(conn)
    finally:
        conn.close()

    assert len(props) == 1, "the same pair must not be proposed twice"
