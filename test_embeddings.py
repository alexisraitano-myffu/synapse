"""
Non-regression tests for the local embedding layer (Chantier B).

These run FULLY OFFLINE — no ANTHROPIC_API_KEY required — because embeddings
now come from a local fastembed model. They guard:
  - vector shape / normalization / determinism
  - the backward-compatible `embed_text(text, client=None)` signature
  - semantic ranking quality (related closer than unrelated)
  - search_memory's vector path working without an API key, and its text fallback
"""

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from config import EMBEDDING_DIM
from embeddings import embed_text


def _deserialize(blob: bytes) -> list[float]:
    """Decode a sqlite-vec serialized float32 blob back to a Python list."""
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


# ── Vector shape & properties ───────────────────────────────────────────────

def test_embed_text_dimension_matches_config():
    """The model output must match EMBEDDING_DIM, or the vec0 table breaks."""
    vec = _deserialize(embed_text("un texte de test"))
    assert len(vec) == EMBEDDING_DIM


def test_embed_text_is_l2_normalized():
    """Downstream score = 1 - distance/2 relies on unit-norm vectors."""
    vec = _deserialize(embed_text("vecteur normalisé attendu"))
    magnitude = sum(x * x for x in vec) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-3)


def test_embed_text_is_deterministic():
    """Same text must always produce the same bytes (reproducible index)."""
    assert embed_text("phrase identique") == embed_text("phrase identique")


def test_embed_text_ignores_client_argument():
    """`client` is kept only for backward compat and must not affect output."""
    assert embed_text("rétrocompatibilité") == embed_text("rétrocompatibilité", client=object())


def test_embed_text_works_without_api_key(monkeypatch):
    """Embedding is local — it must succeed even with no API key in the env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    vec = _deserialize(embed_text("aucune clé api nécessaire"))
    assert len(vec) == EMBEDDING_DIM


# ── Semantic quality ────────────────────────────────────────────────────────

def test_related_text_is_closer_than_unrelated():
    """A paraphrase must be nearer than an off-topic sentence (French)."""
    base = _deserialize(embed_text("Le chat dort sur le canapé"))
    related = _deserialize(embed_text("Un félin se repose sur le sofa"))
    unrelated = _deserialize(embed_text("La politique monétaire de la banque centrale"))

    assert _cos(base, related) > _cos(base, unrelated)


# ── search_memory end-to-end ─────────────────────────────────────────────────

def _insert_note(conn, title: str, content: str, with_vector: bool = True) -> int:
    conn.execute("INSERT INTO atomic_notes (title, content) VALUES (?, ?)", (title, content))
    note_id = conn.last_insert_rowid()
    if with_vector:
        conn.execute(
            "INSERT OR REPLACE INTO atomic_notes_vec(rowid, embedding) VALUES (?, ?)",
            (note_id, embed_text(f"{title}\n{content}")),
        )
    return note_id


def _search_fn():
    """Return the underlying search_memory callable (unwrapped from FastMCP)."""
    import mcp_server.server as server
    return getattr(server.search_memory, "fn", server.search_memory)


def test_search_memory_vector_ranks_relevant_first(isolated_db, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from db import get_connection
    conn = get_connection()
    try:
        with conn:
            _insert_note(conn, "Choix base de données",
                         "On a retenu sqlite-vec comme moteur vectoriel local.")
            _insert_note(conn, "Anniversaire de maman",
                         "Aujourd hui c est l anniversaire de ma mère, le 15 mai.")
            _insert_note(conn, "Recette",
                         "Pour la tarte aux pommes, préchauffer le four à 180 degrés.")
    finally:
        conn.close()

    results = json.loads(_search_fn()("quelle base de données vectorielle", limit=3))

    assert results, "expected vector results"
    assert results[0]["search_type"] == "vector"
    assert results[0]["title"] == "Choix base de données"


def test_search_memory_text_fallback_when_no_vectors(isolated_db, monkeypatch):
    """With no vectors indexed, search must fall back to LIKE keyword search."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from db import get_connection
    conn = get_connection()
    try:
        with conn:
            _insert_note(conn, "Note brute", "contient le token ZQXWV unique", with_vector=False)
    finally:
        conn.close()

    results = json.loads(_search_fn()("ZQXWV", limit=5))

    assert results, "expected a text-fallback hit"
    assert results[0]["search_type"] == "text"
    assert "ZQXWV" in results[0]["content"]
