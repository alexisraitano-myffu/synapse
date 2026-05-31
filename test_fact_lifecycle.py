"""
Offline tests for SYN-37 — last-writes-wins supersede in facts_store.insert_fact.
(SYN-59 manual archive/obsolete endpoints + view filters are tested in test_api.py.)
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _entity(conn, name="Lucas"):
    eid = str(uuid.uuid4())
    conn.execute("INSERT INTO entities (id, type, canonical_name) VALUES (?,?,?)",
                 (eid, "person", name))
    return eid


def _facts(conn, entity_id):
    from db import cursor_to_dicts
    return {f["id"]: f for f in cursor_to_dicts(conn.execute(
        "SELECT id, predicate, value, confidence, obsoleted_at, obsoleted_by, archived_at "
        "FROM facts WHERE entity_id=?", (entity_id,)))}


def test_single_valued_predicate_supersedes(isolated_db):
    from db import get_connection
    from facts_store import insert_fact
    conn = get_connection()
    try:
        with conn:
            eid = _entity(conn)
            f1 = insert_fact(conn, entity_id=eid, predicate="works_at",
                             value="Stripe", confidence=0.9)
            f2 = insert_fact(conn, entity_id=eid, predicate="works_at",
                             value="OpenAI", confidence=0.9)
        rows = _facts(conn, eid)
    finally:
        conn.close()
    assert rows[f1]["obsoleted_at"] is not None, "old single-valued fact must be obsoleted"
    assert rows[f1]["obsoleted_by"] == f2, "obsoleted_by points at the replacing fact"
    assert rows[f2]["obsoleted_at"] is None, "new fact stays active"


def test_multi_valued_predicate_appends(isolated_db):
    from db import get_connection
    from facts_store import insert_fact
    conn = get_connection()
    try:
        with conn:
            eid = _entity(conn)
            f1 = insert_fact(conn, entity_id=eid, predicate="likes", value="escalade", confidence=0.9)
            f2 = insert_fact(conn, entity_id=eid, predicate="likes", value="vélo", confidence=0.9)
        rows = _facts(conn, eid)
    finally:
        conn.close()
    assert rows[f1]["obsoleted_at"] is None and rows[f2]["obsoleted_at"] is None


def test_lower_confidence_does_not_supersede(isolated_db):
    """A less-confident contradiction coexists (suspicious, not yet resolved)."""
    from db import get_connection
    from facts_store import insert_fact
    conn = get_connection()
    try:
        with conn:
            eid = _entity(conn)
            f1 = insert_fact(conn, entity_id=eid, predicate="works_at", value="Stripe", confidence=0.9)
            f2 = insert_fact(conn, entity_id=eid, predicate="works_at", value="OpenAI", confidence=0.6)
        rows = _facts(conn, eid)
    finally:
        conn.close()
    assert rows[f1]["obsoleted_at"] is None, "higher-confidence fact must not be superseded"
    assert rows[f2]["obsoleted_at"] is None


def test_manual_obsolete_is_preserved(isolated_db):
    """A manually-obsoleted fact (obsoleted_by NULL) is not re-touched by a later
    insert — its earlier obsoleted_at and NULL obsoleted_by stay intact."""
    from db import get_connection
    from facts_store import insert_fact
    conn = get_connection()
    try:
        with conn:
            eid = _entity(conn)
            f1 = insert_fact(conn, entity_id=eid, predicate="works_at", value="Mistral", confidence=0.9)
            conn.execute("UPDATE facts SET obsoleted_at=CURRENT_TIMESTAMP WHERE id=?", (f1,))
            f2 = insert_fact(conn, entity_id=eid, predicate="works_at", value="OpenAI", confidence=0.9)
        rows = _facts(conn, eid)
    finally:
        conn.close()
    assert rows[f1]["obsoleted_by"] is None, "manual obsolescence must stay unlinked"
    assert rows[f2]["obsoleted_at"] is None, "the new fact is active"
