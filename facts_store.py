"""
Shared fact insertion (SYN-37 last-writes-wins + dedup-reinforce).

T5 (SYN-114): the implementation lives in the Rust core (`routing.rs::insert_fact`,
also used by the routing path — one implementation, zero drift). This module is
the host-side shim kept for its call sites (validation, reclassify endpoint) and
the historical signature; the write runs on the CALLER's connection so an open
`with conn:` transaction wraps it. The single-valued predicate registry lives in
the core too (`routing.rs::SINGLE_VALUED_PREDICATES`).
"""


def insert_fact(conn, *, entity_id, predicate, value, confidence,
                source_inbox_id=None, persistence_value=3,
                provenance_capture_id=None, category=None) -> str:
    """Insert a fact through the core. Returns the fact id (the existing one
    when an identical active fact was reinforced instead of duplicated)."""
    return conn.insert_fact(
        entity_id=entity_id, predicate=predicate, value=value,
        confidence=confidence, source_inbox_id=source_inbox_id,
        persistence_value=persistence_value,
        provenance_capture_id=provenance_capture_id, category=category)
