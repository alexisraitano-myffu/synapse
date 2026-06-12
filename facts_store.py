"""
Shared fact insertion + the single-valued predicate registry (SYN-37).

Every path that writes a row into `facts` goes through `insert_fact`, so the
last-writes-wins supersede rule stays consistent across step4 routing, pending
promotion, manual validation, and the reclassify endpoint — no drifting copies.
"""

import uuid

from db import cursor_to_dicts

# SYN-37: predicates that hold at most one *current* value per entity — a new
# fact obsoletes the previous one (last-writes-wins). Everything else is
# multi-valued (plain append). Extend this set as dogfood surfaces new ones.
SINGLE_VALUED_PREDICATES = {
    "works_at", "current_workplace", "employer",
    "lives_in", "current_city", "lives", "address",
    "has_birthday", "birthday", "born_on", "date_of_birth",
    "phone", "phone_number", "email",
    "age", "job_title", "current_role", "role",
}


def is_single_valued(predicate: str) -> bool:
    return (predicate or "").strip().lower() in SINGLE_VALUED_PREDICATES


def insert_fact(conn, *, entity_id, predicate, value, confidence,
                source_inbox_id=None, persistence_value=3,
                provenance_capture_id=None, fact_id=None, category=None) -> str:
    """Insert a fact, applying SYN-37 last-writes-wins for single-valued predicates.

    For a single-valued predicate, any *active* (not obsoleted, not archived) fact
    with the same entity+predicate is marked `obsoleted_at=now, obsoleted_by=new`
    — but only when the new fact is at least as confident, else both coexist and
    the user sees the unresolved conflict. A manually-obsoleted fact is left
    untouched (idempotent: its earlier `obsoleted_at` is preserved). The caller
    owns the transaction. Returns the new fact id.
    """
    fact_id = fact_id or str(uuid.uuid4())
    if is_single_valued(predicate):
        for ex in cursor_to_dicts(conn.execute(
            "SELECT id, confidence FROM facts "
            "WHERE entity_id = ? AND predicate = ? "
            "AND obsoleted_at IS NULL AND archived_at IS NULL",
            (entity_id, predicate),
        )):
            if confidence >= (ex["confidence"] or 0.0):
                conn.execute(
                    "UPDATE facts SET obsoleted_at = CURRENT_TIMESTAMP, obsoleted_by = ? "
                    "WHERE id = ?",
                    (fact_id, ex["id"]),
                )
    conn.execute(
        "INSERT INTO facts "
        "(id, entity_id, predicate, value, confidence, source_inbox_id, "
        " persistence_value, provenance_capture_id, category) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (fact_id, entity_id, predicate, value, confidence,
         source_inbox_id, persistence_value, provenance_capture_id, category),
    )
    return fact_id
