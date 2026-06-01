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


def test_graph_default_is_entities_only(client):
    """SYN-68 — the legacy shape is preserved: no notes, no cluster pass."""
    _seed_graph()
    g = client.get("/graph").json()
    assert {n["kind"] for n in g["nodes"]} == {"entity"}
    assert all(n["community_id"] is None for n in g["nodes"])


def test_graph_map_adds_notes_and_clusters(client):
    """SYN-68 — include_notes adds atomic_notes as a 2nd node kind with mention
    edges; cluster tags every node with a community_id."""
    _seed_graph()
    conn = _conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO atomic_notes (id, content, summary, entities_mentioned) "
                "VALUES (1, 'une pensée sur Marie', 'pensée', '[\"Marie\"]')")
    finally:
        conn.close()
    g = client.get("/graph", params={"include_notes": "true", "cluster": "true"}).json()
    assert {n["kind"] for n in g["nodes"]} == {"entity", "atomic_note"}
    note = next(n for n in g["nodes"] if n["kind"] == "atomic_note")
    assert note["id"] == "n:1"
    assert any(e["from"] == "n:1" and e["label"] == "mentions" for e in g["edges"])
    assert all(n["community_id"] is not None for n in g["nodes"])


def test_graph_layout_is_stable_and_incremental(client):
    """SYN-69 — positions persist (same map on reopen), and adding a node does
    not move the nodes already placed."""
    _seed_graph()
    first = client.get("/graph", params={"layout": "true"}).json()
    pos1 = {n["id"]: (n["x"], n["y"]) for n in first["nodes"]}
    assert all("x" in n and "y" in n for n in first["nodes"])

    # reopen → identical positions (read from node_positions, no re-layout)
    second = client.get("/graph", params={"layout": "true"}).json()
    pos2 = {n["id"]: (n["x"], n["y"]) for n in second["nodes"]}
    assert pos2 == pos1

    # add a new entity, reopen → existing positions untouched, newcomer placed
    conn = _conn()
    try:
        with conn:
            conn.execute("INSERT INTO entities (id, type, canonical_name, mention_count) "
                         "VALUES ('e3','person','Karim',1)")
    finally:
        conn.close()
    third = client.get("/graph", params={"layout": "true"}).json()
    pos3 = {n["id"]: (n["x"], n["y"]) for n in third["nodes"]}
    assert pos3["e1"] == pos1["e1"] and pos3["e2"] == pos1["e2"]  # not disturbed
    assert "e3" in pos3  # newcomer got a position


def test_graph_anti_hairball_filters(client):
    """SYN-71 — the five filters compose and a hard node cap always applies."""
    _seed_graph()  # e1 Marie + e2 Alexis (ms default 1.0) with a relation
    conn = _conn()
    try:
        with conn:
            # a stale, isolated entity: low liveness, no relation
            conn.execute("INSERT INTO entities (id, type, canonical_name, memory_strength, "
                         "last_mentioned) VALUES ('e_stale','concept','Vieux',0.01,'2020-01-01')")
            conn.execute("INSERT INTO atomic_notes (id, content, summary, entities_mentioned) "
                         "VALUES (1,'pensée','p','[\"Marie\"]')")
    finally:
        conn.close()

    # node_types=entities → notes excluded even with include_notes
    g = client.get("/graph", params={"include_notes": "true", "node_types": "entities"}).json()
    assert {n["kind"] for n in g["nodes"]} == {"entity"}

    # memory_strength_min drops the stale entity
    labels = {n["label"] for n in client.get(
        "/graph", params={"memory_strength_min": "0.5"}).json()["nodes"]}
    assert "Vieux" not in labels and "Marie" in labels

    # since keeps only recently-active nodes
    labels = {n["label"] for n in client.get(
        "/graph", params={"since": "2026-01-01"}).json()["nodes"]}
    assert "Vieux" not in labels

    # include_isolated=false drops the disconnected entity
    labels = {n["label"] for n in client.get(
        "/graph", params={"include_isolated": "false"}).json()["nodes"]}
    assert "Vieux" not in labels and {"Marie", "Alexis"} <= labels

    # top_pct_per_cluster keeps ≥1 per community
    g = client.get("/graph", params={"cluster": "true", "top_pct_per_cluster": "0.5"}).json()
    assert 1 <= len(g["nodes"]) < 3

    # max_nodes is a hard ceiling
    assert len(client.get("/graph", params={"max_nodes": "1"}).json()["nodes"]) == 1


def test_graph_clusters_section(client, monkeypatch):
    """SYN-70 — clusters=true adds labelled regions with a hull. Force the
    fallback (factory → None) so the test stays offline and deterministic
    regardless of whether an Anthropic key is configured."""
    import api.app as appmod
    monkeypatch.setattr(appmod, "_anthropic_client_factory", lambda: None)
    _seed_graph()
    g = client.get("/graph", params={"clusters": "true"}).json()
    assert "clusters" in g and g["clusters"]
    c = g["clusters"][0]
    assert {"community_id", "label", "size", "hull"} <= set(c)
    assert c["label"] == f"Cluster {c['community_id']}"  # forced fallback
    assert isinstance(c["hull"], list)
    # nodes carry positions (clusters imply layout)
    assert all("x" in n and "y" in n for n in g["nodes"])


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


# ── Entity-type proposals (SYN-58) ────────────────────────────────────────────

def _seed_type_proposal():
    """A pending entity + its type proposal, as the cycle would create them."""
    import uuid as _uuid
    conn = _conn()
    try:
        with conn:
            conn.execute("INSERT INTO inbox (content, source) VALUES ('recette udon','test')")
            cid = conn.last_insert_rowid()
            eid = str(_uuid.uuid4())
            conn.execute(
                "INSERT INTO entities (id, type, canonical_name, status) VALUES (?,?,?,?)",
                (eid, "concept", "Udon Dan Dan", "pending"),
            )
            pid = str(_uuid.uuid4())
            conn.execute(
                "INSERT INTO entity_type_proposals "
                "(id, proposed_type, reason, evidence_capture_id, candidate_entity_id) "
                "VALUES (?,?,?,?,?)",
                (pid, "recipe", "un plat", cid, eid),
            )
    finally:
        conn.close()
    return pid, eid


def _entity_row(eid):
    from db import first_row
    conn = _conn()
    try:
        return first_row(conn.execute("SELECT type, status FROM entities WHERE id=?", (eid,)))
    finally:
        conn.close()


def _active_types():
    conn = _conn()
    try:
        return {r[0] for r in conn.execute("SELECT type FROM active_entity_types")}
    finally:
        conn.close()


def test_type_proposals_list_shows_candidate_and_evidence(client):
    pid, eid = _seed_type_proposal()
    rows = client.get("/entity-type-proposals").json()
    row = next(p for p in rows if p["id"] == pid)
    assert row["proposed_type"] == "recipe"
    assert row["candidate_name"] == "Udon Dan Dan"
    assert "udon" in row["evidence_content"].lower()


def test_type_proposal_accept_extends_vocab_and_activates_entity(client):
    pid, eid = _seed_type_proposal()
    assert "recipe" not in _active_types()
    r = client.post(f"/entity-type-proposals/{pid}/accept", json={})
    assert r.status_code == 200 and r.json()["type"] == "recipe"
    assert "recipe" in _active_types()
    row = _entity_row(eid)
    assert row["type"] == "recipe" and row["status"] == "active"
    # terminal: a second accept is rejected
    assert client.post(f"/entity-type-proposals/{pid}/accept", json={}).status_code == 400


def test_type_proposal_accept_honours_rename(client):
    pid, eid = _seed_type_proposal()
    r = client.post(f"/entity-type-proposals/{pid}/accept", json={"type": "plat"})
    assert r.json()["type"] == "plat"
    assert "plat" in _active_types() and "recipe" not in _active_types()
    assert _entity_row(eid)["type"] == "plat"


def test_type_proposal_reject_archives_entity(client):
    pid, eid = _seed_type_proposal()
    assert client.post(f"/entity-type-proposals/{pid}/reject").status_code == 200
    assert _entity_row(eid)["status"] == "archived"
    # archived entity is absent from the default graph view
    g = client.get("/graph").json()
    assert all(n["id"] != eid for n in g["nodes"])


def test_pending_entity_hidden_from_graph(client):
    _pid, eid = _seed_type_proposal()
    g = client.get("/graph").json()
    assert all(n["id"] != eid for n in g["nodes"]), "pending entity must not leak into graph"


# ── Lifecycle: archive / obsolete (SYN-59) ────────────────────────────────────

def _seed_entity_with_fact(name="Michel", predicate="works_at", value="Mistral"):
    import uuid as _uuid
    from facts_store import insert_fact
    conn = _conn()
    try:
        eid = str(_uuid.uuid4())
        with conn:
            conn.execute("INSERT INTO entities (id, type, canonical_name) VALUES (?,?,?)",
                         (eid, "person", name))
            fid = insert_fact(conn, entity_id=eid, predicate=predicate,
                              value=value, confidence=0.95)
    finally:
        conn.close()
    return eid, fid


def _fact_ids(client, eid, **params):
    return [f["id"] for f in client.get(f"/entity/{eid}", params=params).json()["facts"]]


def test_fact_obsolete_then_restore(client):
    eid, fid = _seed_entity_with_fact()
    assert fid in _fact_ids(client, eid)                      # visible by default
    assert client.post(f"/fact/{fid}/obsolete").status_code == 200
    assert fid not in _fact_ids(client, eid)                  # hidden
    assert fid in _fact_ids(client, eid, include="obsolete")  # opt back in
    assert client.post(f"/fact/{fid}/restore").status_code == 200
    assert fid in _fact_ids(client, eid)                      # back in default view


def test_fact_archive_then_unarchive(client):
    eid, fid = _seed_entity_with_fact()
    client.post(f"/fact/{fid}/archive")
    assert fid not in _fact_ids(client, eid)
    assert fid in _fact_ids(client, eid, include="archived")
    client.post(f"/fact/{fid}/unarchive")
    assert fid in _fact_ids(client, eid)


def test_entity_archive_hides_from_graph(client):
    eid, _fid = _seed_entity_with_fact()
    assert any(n["id"] == eid for n in client.get("/graph").json()["nodes"])
    assert client.post(f"/entity/{eid}/archive").status_code == 200
    assert all(n["id"] != eid for n in client.get("/graph").json()["nodes"])
    assert any(n["id"] == eid
               for n in client.get("/graph", params={"include_archived": True}).json()["nodes"])
    client.post(f"/entity/{eid}/unarchive")
    assert any(n["id"] == eid for n in client.get("/graph").json()["nodes"])


def test_lifecycle_404_on_unknown(client):
    assert client.post("/fact/does-not-exist/obsolete").status_code == 404
    assert client.post("/entity/does-not-exist/archive").status_code == 404


def test_supersede_visible_through_entity_endpoint(client):
    """SYN-37 end-to-end through the API: a second works_at hides the first."""
    eid, fid1 = _seed_entity_with_fact(value="Stripe")
    from facts_store import insert_fact
    conn = _conn()
    try:
        with conn:
            fid2 = insert_fact(conn, entity_id=eid, predicate="works_at",
                               value="OpenAI", confidence=0.95)
    finally:
        conn.close()
    active = _fact_ids(client, eid)
    assert fid2 in active and fid1 not in active           # only the latest is active
    assert fid1 in _fact_ids(client, eid, include="obsolete")


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
