"""
Offline tests for entity vectorization + shared semantic search (SYN-60).

No ANTHROPIC_API_KEY needed — embeddings are local (fastembed) and we exercise
the storage/search path directly, not the classify step.
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _insert_entity(conn, name, type_="concept", summary="", aliases=None,
                   attributes=None, embed=True, merged_into=None):
    """Insert an entity, embedding it the same way the cycle does (unless embed=False)."""
    from embeddings import embed_text
    from entity_search import entity_embedding_text

    eid = str(uuid.uuid4())
    row = {
        "canonical_name": name,
        "type": type_,
        "summary": summary,
        "aliases": json.dumps(aliases or []),
        "attributes": json.dumps(attributes or {}),
    }
    vec = embed_text(entity_embedding_text(row)) if embed else None
    conn.execute(
        "INSERT INTO entities (id, type, canonical_name, aliases, attributes, "
        "summary, embedding, merged_into_id) VALUES (?,?,?,?,?,?,?,?)",
        (eid, type_, name, row["aliases"], row["attributes"], summary, vec, merged_into),
    )
    return eid


# ── entity_embedding_text ────────────────────────────────────────────────────

def test_entity_embedding_text_handles_json_string_fields():
    from entity_search import entity_embedding_text

    text = entity_embedding_text({
        "canonical_name": "Escalade",
        "type": "concept",
        "aliases": json.dumps(["grimpe", "climbing"]),       # stored as JSON string
        "attributes": json.dumps({"intensité": "haute"}),
        "summary": "Sport de grimpe",
    })
    assert "Nom: Escalade" in text
    assert "grimpe, climbing" in text
    assert "intensité" in text
    assert "Sport de grimpe" in text


def test_entity_embedding_text_tolerates_none_summary():
    from entity_search import entity_embedding_text
    text = entity_embedding_text({"canonical_name": "X", "summary": None})
    assert text.rstrip().endswith("Résumé:")  # no crash on None summary


# ── Shared cosine search ─────────────────────────────────────────────────────

def test_search_finds_semantically_close_entity(isolated_db):
    from db import get_connection
    from embeddings import embed_text
    from entity_search import search_entities_by_vector

    conn = get_connection()
    try:
        with conn:
            _insert_entity(conn, "Escalade", summary="Grimper des parois et des blocs")
            _insert_entity(conn, "Bouldering", summary="Grimpe de bloc sans corde")
            _insert_entity(conn, "Politique monétaire",
                           summary="Taux directeurs de la banque centrale")

        results = search_entities_by_vector(
            conn, embed_text("escalade et grimpe de bloc"), limit=3
        )
    finally:
        conn.close()

    assert results, "expected at least one entity match"
    top_names = [r["canonical_name"] for r in results[:2]]
    assert "Escalade" in top_names or "Bouldering" in top_names
    # The unrelated finance entity must not outrank the climbing ones.
    assert results[0]["canonical_name"] != "Politique monétaire"
    # Scores are descending and in [0, 1].
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_search_respects_type_filter_and_exclude_ids(isolated_db):
    from db import get_connection
    from embeddings import embed_text
    from entity_search import search_entities_by_vector

    conn = get_connection()
    try:
        with conn:
            climbing = _insert_entity(conn, "Escalade", type_="concept",
                                      summary="Grimpe")
            _insert_entity(conn, "Marie", type_="person", summary="Grimpe aussi")

        q = embed_text("grimpe")
        # type_filter keeps only persons → the concept is gone.
        persons = search_entities_by_vector(conn, q, type_filter="person")
        assert all(r["type"] == "person" for r in persons)
        assert all(r["canonical_name"] != "Escalade" for r in persons)

        # exclude_ids drops the query entity itself.
        others = search_entities_by_vector(conn, q, exclude_ids={climbing})
        assert all(r["id"] != climbing for r in others)
    finally:
        conn.close()


def test_search_excludes_soft_merged_entities(isolated_db):
    from db import get_connection
    from embeddings import embed_text
    from entity_search import search_entities_by_vector

    conn = get_connection()
    try:
        with conn:
            survivor = _insert_entity(conn, "OpenAI", summary="Labo IA")
            _insert_entity(conn, "Open AI", summary="Labo IA", merged_into=survivor)

        results = search_entities_by_vector(conn, embed_text("OpenAI labo IA"), limit=10)
    finally:
        conn.close()

    assert all(r["canonical_name"] != "Open AI" for r in results), \
        "soft-merged tombstone must not surface in search"


def test_search_min_score_filters_weak_matches(isolated_db):
    from db import get_connection
    from embeddings import embed_text
    from entity_search import search_entities_by_vector

    conn = get_connection()
    try:
        with conn:
            _insert_entity(conn, "Escalade", summary="Grimpe de paroi")
        # A wildly unrelated query with a high floor should return nothing.
        results = search_entities_by_vector(
            conn, embed_text("dérivés financiers et fiscalité"), min_score=0.95
        )
    finally:
        conn.close()
    assert results == []


# ── Cycle vectorization path (step6) ─────────────────────────────────────────

def test_step6_vectorize_fills_embedding_and_reacts_to_summary(isolated_db):
    from db import get_connection
    from dream_cycle.cycle import step6_vectorize

    conn = get_connection()
    try:
        with conn:
            eid = _insert_entity(conn, "Schopenhauer", type_="person",
                                 summary="", embed=False)

        # Cycle embeds the touched entity (client is ignored — local embeddings).
        assert step6_vectorize([eid], conn, client=None) == 1
        before = conn.execute(
            "SELECT embedding FROM entities WHERE id=?", (eid,)
        ).fetchone()[0]
        assert before is not None, "step6 must fill entities.embedding"

        # Summary changes → re-embedding yields a different vector.
        with conn:
            conn.execute("UPDATE entities SET summary=? WHERE id=?",
                         ("Philosophe pessimiste allemand", eid))
        step6_vectorize([eid], conn, client=None)
        after = conn.execute(
            "SELECT embedding FROM entities WHERE id=?", (eid,)
        ).fetchone()[0]
    finally:
        conn.close()

    assert after != before, "embedding should change when the summary changes"


# ── Backfill ─────────────────────────────────────────────────────────────────

def test_backfill_fills_missing_and_is_idempotent(isolated_db):
    from db import get_connection
    from scripts.backfill_entity_embeddings import backfill

    conn = get_connection()
    try:
        with conn:
            _insert_entity(conn, "Sans embedding 1", embed=False, summary="a")
            _insert_entity(conn, "Sans embedding 2", embed=False, summary="b")
            _insert_entity(conn, "Avec embedding", embed=True, summary="c")
    finally:
        conn.close()

    # First run embeds the two NULL rows only.
    assert backfill() == 2
    # Second run is a no-op (idempotent).
    assert backfill() == 0

    conn = get_connection()
    try:
        missing = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE embedding IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    assert missing == 0
