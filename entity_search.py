"""
Shared semantic search over the entity graph (SYN-60).

Entity embeddings live as raw float32 BLOBs in `entities.embedding`. Entity ids
are UUID strings, so they can't share the int-rowid `vec0` table used for
`atomic_notes` — instead the Dream Cycle's `step6_vectorize` writes the vector
straight into the BLOB column. At personal scale (hundreds to a few thousand
entities) a linear cosine scan in Python is sub-millisecond, so we skip the ANN
index entirely; `vec0` only earns its keep past ~100k vectors.

This module is the single implementation every caller shares — MCP
`search_memory`, merge V2 (SYN-61), and semantic suggestions (SYN-62) — so the
scoring stays consistent and we don't grow three drifting copies of the scan.

Vectors are L2-normalized at embed time, so the L2 distance lives in [0, 2] and
is monotonic with cosine similarity, keeping the `score = 1 - distance/2`
mapping valid (related entities land ~0.9, unrelated ~1.4 with this model).
"""

import json
import struct

from db import cursor_to_dicts
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
    q = deserialize_vec(query_vec)
    exclude = set(exclude_ids or ())

    # status='active' (SYN-58): pending (awaiting type validation) and archived
    # (rejected) entities must never surface as search hits or merge candidates.
    # archived_at IS NULL (SYN-59): user-archived entities are hidden too.
    sql = (
        "SELECT id, canonical_name, type, summary, embedding FROM entities "
        "WHERE embedding IS NOT NULL AND merged_into_id IS NULL "
        "AND status = 'active' AND archived_at IS NULL"
    )
    params: list = []
    if type_filter:
        sql += " AND type = ?"
        params.append(type_filter)

    scored: list[tuple[float, dict]] = []
    for row in cursor_to_dicts(conn.execute(sql, params)):
        if row["id"] in exclude:
            continue
        v = deserialize_vec(row["embedding"])
        if len(v) != len(q):
            continue  # stale dim (model changed before a re-embed) — skip
        dist = sum((a - b) ** 2 for a, b in zip(q, v)) ** 0.5
        score = _score_from_distance(dist)
        if score < min_score:
            continue
        scored.append((score, row))

    scored.sort(key=lambda x: -x[0])
    return [
        {
            "id": row["id"],
            "canonical_name": row["canonical_name"],
            "type": row.get("type"),
            "summary": row.get("summary") or "",
            "score": score,
        }
        for score, row in scored[:limit]
    ]


def search_entities_by_text(conn, text: str, **kwargs) -> list[dict]:
    """Convenience wrapper: embed `text` locally, then run the vector search."""
    return search_entities_by_vector(conn, embed_text(text), **kwargs)
