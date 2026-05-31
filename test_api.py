"""
Offline tests for the HTTP API (FastAPI TestClient). No ANTHROPIC_API_KEY:
capture, feed, graph, entity, pending, validate, changes and auth all work
without the Claude API. /dream-cycle/run is not tested here (it calls Claude).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def client(isolated_db, monkeypatch):
    monkeypatch.delenv("SYNAPSE_API_TOKEN", raising=False)  # auth off by default
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)


def _conn():
    from db import get_connection
    return get_connection()


# ── Health ───────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Capture (idempotent) ─────────────────────────────────────────────────────

def test_capture_is_idempotent(client):
    first = client.post("/capture", json={"id": "uuid-1", "content": "première note"})
    assert first.status_code == 200
    assert first.json()["status"] == "queued"

    dup = client.post("/capture", json={"id": "uuid-1", "content": "doublon réseau"})
    assert dup.json()["status"] == "duplicate"

    feed = client.get("/feed").json()
    assert sum(1 for x in feed if x["client_id"] == "uuid-1") == 1


def test_feed_reports_status(client):
    client.post("/capture", json={"id": "u2", "content": "à traiter", "device_id": "air"})
    feed = client.get("/feed").json()
    row = next(x for x in feed if x["client_id"] == "u2")
    assert row["status"] == "queued"


def test_feed_reports_failed_and_legacy_processed(client):
    conn = _conn()
    try:
        with conn:
            # an entry the cycle marked failed
            conn.execute("INSERT INTO inbox (content, source, client_id, status, processed_at) "
                         "VALUES ('mauvaise note','test','f1','failed','2026-05-27T00:00:00')")
            # a legacy entry: processed_at set but status still default 'queued'
            conn.execute("INSERT INTO inbox (content, source, client_id, processed_at) "
                         "VALUES ('vieille note','test','l1','2026-05-27T00:00:00')")
    finally:
        conn.close()
    feed = {x["client_id"]: x for x in client.get("/feed").json()}
    assert feed["f1"]["status"] == "failed"
    assert feed["l1"]["status"] == "processed"


# ── Graph ────────────────────────────────────────────────────────────────────

def _seed_graph():
    conn = _conn()
    try:
        with conn:
            conn.execute("INSERT INTO entities (id, type, canonical_name, mention_count, persistence_value) "
                         "VALUES ('e1','person','Marie',5,5)")
            conn.execute("INSERT INTO entities (id, type, canonical_name, mention_count, persistence_value) "
                         "VALUES ('e2','person','Alexis',2,4)")
            conn.execute("INSERT INTO relations (id, entity_from, predicate, entity_to, confidence) "
                         "VALUES ('r1','e1','mere_de','e2',0.9)")
    finally:
        conn.close()


def test_graph_full(client):
    _seed_graph()
    g = client.get("/graph").json()
    assert len(g["nodes"]) == 2
    assert len(g["edges"]) == 1
    marie = next(n for n in g["nodes"] if n["label"] == "Marie")
    assert marie["mention_count"] == 5
    assert "memory_strength" in marie  # frozen field present even if null


def test_graph_ego(client):
    _seed_graph()
    g = client.get("/graph", params={"entity": "Marie", "mode": "ego"}).json()
    labels = {n["label"] for n in g["nodes"]}
    assert "Marie" in labels and "Alexis" in labels


def test_entity_detail(client):
    _seed_graph()
    g = client.get("/graph").json()
    marie_id = next(n["id"] for n in g["nodes"] if n["label"] == "Marie")
    d = client.get(f"/entity/{marie_id}").json()
    assert d["canonical_name"] == "Marie"
    assert any(r["entity_to"] == "Alexis" for r in d["relations"])


# ── Semantic suggestions (SYN-62) ─────────────────────────────────────────────

def _seed_similar():
    """Insert vectorized entities so /similar has something to score."""
    import uuid as _uuid
    from embeddings import embed_text
    from entity_search import entity_embedding_text
    ids = {}
    conn = _conn()
    try:
        with conn:
            for name, type_, summary in [
                ("Escalade", "concept", "Grimper des parois et des blocs en falaise"),
                ("Bouldering", "concept", "Grimpe de bloc sans corde, en salle ou en falaise"),
                ("Politique monétaire", "concept", "Taux directeurs de la banque centrale"),
                ("Marie", "person", "Une amie qui fait de la grimpe"),
            ]:
                eid = str(_uuid.uuid4())
                row = {"canonical_name": name, "type": type_,
                       "aliases": "[]", "attributes": "{}", "summary": summary}
                vec = embed_text(entity_embedding_text(row))
                conn.execute(
                    "INSERT INTO entities (id, type, canonical_name, aliases, "
                    "attributes, summary, embedding) VALUES (?,?,?,?,?,?,?)",
                    (eid, type_, name, "[]", "{}", summary, vec),
                )
                ids[name] = eid
    finally:
        conn.close()
    return ids


def test_entity_similar_returns_semantic_neighbours(client):
    ids = _seed_similar()
    r = client.get(f"/entity/{ids['Escalade']}/similar", params={"min_score": 0.3})
    assert r.status_code == 200
    body = r.json()
    assert body["entity_id"] == ids["Escalade"]
    names = [s["canonical_name"] for s in body["similar"]]
    # The entity itself is never in its own suggestions.
    assert "Escalade" not in names
    # Climbing-adjacent entity should rank above the finance one.
    assert "Bouldering" in names
    assert names[0] == "Bouldering"
    # Scores are descending and carry the expected fields.
    scores = [s["similarity_score"] for s in body["similar"]]
    assert scores == sorted(scores, reverse=True)
    assert all({"entity_id", "canonical_name", "type", "similarity_score"} <= s.keys()
               for s in body["similar"])


def test_entity_similar_same_type_filter(client):
    ids = _seed_similar()
    r = client.get(f"/entity/{ids['Escalade']}/similar",
                   params={"min_score": 0.1, "same_type": True})
    types = {s["type"] for s in r.json()["similar"]}
    assert types <= {"concept"}, "same_type=true must keep only concepts"
    assert all(s["canonical_name"] != "Marie" for s in r.json()["similar"])


def test_entity_similar_404_unknown(client):
    assert client.get("/entity/does-not-exist/similar").status_code == 404


# ── Pending + validate (event-sourced) ───────────────────────────────────────

def _seed_pending():
    conn = _conn()
    try:
        with conn:
            conn.execute("INSERT INTO inbox (content, source) VALUES ('Marie bosse à l hôpital','test')")
            inbox_id = conn.last_insert_rowid()
            fact_data = json.dumps({
                "entity_canonical": "Marie", "predicate": "works_at",
                "value": "Hôpital", "confidence": 0.6,
                "persistence_value": 4, "source_inbox_id": inbox_id,
            })
            conn.execute("INSERT INTO pending_facts (id, fact_data, validation_strategy) "
                         "VALUES ('p1', ?, 'passive')", (fact_data,))
    finally:
        conn.close()


def test_pending_shows_question_and_source(client):
    _seed_pending()
    items = client.get("/pending").json()
    assert len(items) == 1
    it = items[0]
    assert it["entity"] == "Marie"
    assert "Marie" in it["question"]
    assert "hôpital" in it["source_text"].lower()


def test_validate_confirm_consolidates_and_logs_event(client):
    _seed_pending()
    r = client.post("/pending/p1/validate", json={"confirmed": True, "device_id": "air"})
    assert r.json()["status"] == "confirmed"

    conn = _conn()
    try:
        facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM pending_facts").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM validation_events WHERE confirmed=1").fetchone()[0]
    finally:
        conn.close()
    assert facts == 1 and pending == 0 and events == 1


def test_validate_reject_logs_event_and_discards(client):
    _seed_pending()
    r = client.post("/pending/p1/validate", json={"confirmed": False})
    assert r.json()["status"] == "rejected"
    conn = _conn()
    try:
        facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM validation_events WHERE confirmed=0").fetchone()[0]
    finally:
        conn.close()
    assert facts == 0 and events == 1


def test_validate_unknown_fact_404(client):
    assert client.post("/pending/nope/validate", json={"confirmed": True}).status_code == 404


# ── Changes (replication snapshot) ───────────────────────────────────────────

def test_changes_returns_derived_state(client):
    _seed_graph()
    body = client.get("/changes").json()
    assert {"entities", "facts", "relations", "atomic_notes", "cursor"} <= body.keys()
    assert len(body["entities"]) == 2


# ── Auth ─────────────────────────────────────────────────────────────────────

def test_auth_enforced_when_token_set(client, monkeypatch):
    monkeypatch.setenv("SYNAPSE_API_TOKEN", "secret")
    # protected route without header → 401
    assert client.get("/graph").status_code == 401
    # with correct header → ok
    ok = client.get("/graph", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    # health stays open
    assert client.get("/health").status_code == 200
