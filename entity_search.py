"""
Shared semantic search over the entity graph (SYN-60).

Entity embeddings live as raw float32 BLOBs in `entities.embedding`. Entity ids
are UUID strings, so they can't share the int-rowid `vec0` table used for
`atomic_notes`; similarity is an exact linear scan (sub-millisecond at personal
scale). Since SYN-110 the scan runs inside the Rust core (`synapse_core`,
`Storage.search_entities` / `search_resources`) with the exact same candidate
filters and scoring; this module keeps the historical signatures so every
caller — MCP `search_memory`, merge V2 (SYN-61), semantic suggestions (SYN-62)
— stays on one shared implementation.

Vectors are L2-normalized at embed time, so the L2 distance lives in [0, 2] and
is monotonic with cosine similarity, keeping the `score = 1 - distance/2`
mapping valid (related entities land ~0.9, unrelated ~1.4 with this model).
"""

import json
import struct

from core_store import get_store
from embeddings import embed_text


def entity_embedding_text(entity: dict) -> str:
    """Build the composite text embedded for an entity (SYN-60).

    Used by both the Dream Cycle's `step6_vectorize` and the backfill script, so
    a backfilled vector lives in the same space as a freshly-cycled one. We embed
    identity + context (name, type, aliases, attributes, summary) but *not* facts
    — facts are too volatile and would trigger constant re-embeds.

    Accepts a raw `entities` row: `aliases`/`attributes` may be JSON strings (as
    stored) or already-decoded list/dict.
    """
    aliases = entity.get("aliases") or []
    if isinstance(aliases, str):
        try:
            aliases = json.loads(aliases)
        except (ValueError, TypeError):
            aliases = []
    attributes = entity.get("attributes") or {}
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except (ValueError, TypeError):
            attributes = {}
    return (
        f"Nom: {entity['canonical_name']}\n"
        f"Type: {entity.get('type', '')}\n"
        f"Aliases: {', '.join(aliases)}\n"
        f"Attributs: {json.dumps(attributes, ensure_ascii=False)}\n"
        f"Résumé: {entity.get('summary', '') or ''}"
    )


def deserialize_vec(blob: bytes) -> tuple[float, ...]:
    """Decode a sqlite-vec serialized float32 blob into a tuple of floats."""
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def _score_from_distance(dist: float) -> float:
    """Map an L2 distance on unit vectors ([0, 2]) to a [0, 1] similarity."""
    return round(max(0.0, 1 - dist / 2), 4)


def search_entities_by_vector(
    conn,
    query_vec: bytes,
    *,
    limit: int = 10,
    min_score: float = 0.0,
    type_filter: str | None = None,
    exclude_ids: set[str] | list[str] | None = None,
) -> list[dict]:
    """Top-K entities most similar to `query_vec` (a serialized float32 blob).

    Soft-merged entities (`merged_into_id` set) are always excluded — they are
    tombstones pointing at a survivor and must not surface as results.

    Args:
        conn: an open apsw connection.
        query_vec: the query embedding, as returned by `embed_text`.
        limit: max results to return.
        min_score: drop results scoring below this (0..1).
        type_filter: if set, only consider entities of this `type`.
        exclude_ids: entity ids to skip (e.g. the query entity itself).

    Returns dicts `{id, canonical_name, type, summary, score}`, score-descending.
    """
    # The candidate filters (vectorized, not soft-merged, status='active',
    # not user-archived, optional type, stale-dim skip) and the scoring live
    # in the core — one implementation for desktop and mobile. `conn` is kept
    # in the signature for call-site compatibility; the core has its own.
    hits = get_store().search_entities(
        bytes(query_vec),
        limit=limit,
        min_score=min_score,
        type_filter=type_filter,
        exclude_ids=list(exclude_ids or ()),
    )
    return [
        {
            "id": eid,
            "canonical_name": name,
            "type": etype,
            "summary": summary,
            "score": score,
        }
        for eid, name, etype, summary, score in hits
    ]


def search_entities_by_text(conn, text: str, **kwargs) -> list[dict]:
    """Convenience wrapper: embed `text` locally, then run the vector search."""
    return search_entities_by_vector(conn, embed_text(text), **kwargs)


def search_resources_by_vector(conn, query_vec: bytes, *, limit: int = 10) -> list[dict]:
    """SYN-21: top-K stored resources by cosine on their embedded summary
    (resources, like entities, use UUID ids → manual scan). Returns dicts
    `{id, title, url, summary, score}`, score-descending."""
    hits = get_store().search_resources(bytes(query_vec), limit=limit)
    return [
        {"id": rid, "title": title, "url": url, "summary": summary, "score": score}
        for rid, title, url, summary, score in hits
    ]
