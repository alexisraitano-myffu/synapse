"""
Offline tests for the P2P sync transport (SYN-112 T3 phase 3): /sync/*
endpoints, owner-lock run-guard, peer pull (HTTP stubbed — the "peer" is a
second real core Storage in a temp dir), cursor advance and the
double-routed-rows dedup pass. No network, no Claude API.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def client(isolated_db, monkeypatch):
    monkeypatch.delenv("SYNAPSE_API_TOKEN", raising=False)
    monkeypatch.delenv("SYNAPSE_SYNC_PEERS", raising=False)
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)


def _conn():
    from db import get_connection
    return get_connection()


# ── /sync/changes + /sync/status ─────────────────────────────────────────────

def test_sync_changes_exposes_protocol_v1(client):
    client.post("/capture", json={"id": "cap-1", "content": "note à répliquer"})
    page = client.get("/sync/changes", params={"since": 0, "limit": 10000}).json()
    assert page["protocol"] == 1
    assert page["next"] > 0
    row = next(r for r in page["rows"] if r["t"] == "inbox" and r["pk"] == "cap-1")
    assert row["cols"]["content"]["v"] == "note à répliquer"
    assert "hlc" in row["cols"]["content"]


def test_sync_status_shape(client):
    status = client.get("/sync/status").json()
    assert status["device_id"]
    assert status["journal_seq"] >= 0
    assert status["owner"] is None          # fresh install: nobody owns the cycle
    assert status["is_owner"] is False
    assert status["cursors"] == {}


# ── Owner-lock + run-guard ───────────────────────────────────────────────────

def test_owner_implicit_claim_then_guard_blocks_foreign_device(client):
    from api.sync_peers import ensure_cycle_owner
    from core_store import get_store

    # First run on a fresh install: self-claim, then pass.
    ensure_cycle_owner()
    me = get_store().sync_device_id()
    owner = client.get("/sync/owner").json()
    assert owner["owner"]["device_id"] == me
    assert owner["is_owner"] is True
    ensure_cycle_owner()  # still owner → still passes

    # Hand the lock to another device: the guard must now refuse, and the
    # cycle endpoint must 409 before doing any work.
    claimed = client.put("/sync/owner", json={"device_id": "other-mac"}).json()
    assert claimed["owner"]["device_id"] == "other-mac"
    assert claimed["owner"]["epoch"] == 2
    with pytest.raises(Exception) as exc:
        ensure_cycle_owner()
    assert "409" in str(getattr(exc.value, "status_code", "")) or \
        getattr(exc.value, "status_code", None) == 409
    r = client.post("/dream-cycle/run")
    assert r.status_code == 409
    assert "other-mac" in r.json()["detail"]

    # Claiming it back (epoch 3) reopens the cycle.
    client.put("/sync/owner", json={})
    ensure_cycle_owner()


# ── Peer pull (HTTP stubbed, real second Storage) ────────────────────────────

class _Resp:
    def __init__(self, text):
        self.text = text
        self.status_code = 200

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass


@pytest.fixture
def peer(tmp_path_factory):
    """A second real core database standing in for the other Mac."""
    import synapse_core
    peer_dir = tmp_path_factory.mktemp("peer-home")
    store = synapse_core.Storage(str(peer_dir / "synapse.db"))
    gate = synapse_core.connect(str(peer_dir / "synapse.db"))
    return store, gate


@pytest.fixture
def stubbed_http(peer, monkeypatch):
    """Route sync_peers' HTTP calls to the peer Storage, no sockets."""
    store, _ = peer

    class _Requests:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            if url.endswith("/sync/status"):
                return _Resp(json.dumps({"device_id": store.sync_device_id()}))
            if url.endswith("/sync/changes"):
                return _Resp(store.sync_changes_since(
                    int(params["since"]), int(params["limit"])))
            raise AssertionError(f"unexpected URL {url}")

    from api import sync_peers
    monkeypatch.setattr(sync_peers, "requests", _Requests)
    return _Requests


def test_pull_from_peer_bootstraps_and_is_idempotent(client, peer, stubbed_http):
    store, gate = peer
    gate.execute(
        "INSERT INTO inbox (id, content, source) VALUES ('pc-1', 'depuis le pair', 'test')", [])
    gate.execute(
        "INSERT INTO atomic_notes (id, content, kind) VALUES ('pn-1', 'note du pair', 'note')", [])
    gate.execute(
        "INSERT INTO entities (id, canonical_name, type) VALUES ('pe-1', 'Pixel', 'concept')", [])

    from api.sync_peers import pull_from_peer
    report = pull_from_peer("http://peer.test:8000")
    assert report["peer_device"] == store.sync_device_id()
    assert report["rows_created"] >= 3
    assert report["cursor"] > 0

    conn = _conn()
    try:
        assert conn.execute("SELECT content FROM inbox WHERE id='pc-1'").fetchone()[0] \
            == "depuis le pair"
        assert conn.execute("SELECT count(*) FROM atomic_notes WHERE id='pn-1'").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM entities WHERE id='pe-1'").fetchone()[0] == 1
        # Cursor persisted per peer device.
        saved = conn.execute("SELECT v FROM sync_meta WHERE k = ?",
                             (f"cursor:{store.sync_device_id()}",)).fetchone()[0]
        assert int(saved) == report["cursor"]
    finally:
        conn.close()

    # Second pull: cursor did its job, nothing new lands.
    again = pull_from_peer("http://peer.test:8000")
    assert again["rows_created"] == 0
    assert again["rows_deleted"] == 0

    # And the peer's deletes replicate as tombstones on the next pull.
    gate.execute("DELETE FROM inbox WHERE id='pc-1'", [])
    third = pull_from_peer("http://peer.test:8000")
    assert third["rows_deleted"] == 1
    conn = _conn()
    try:
        assert conn.execute("SELECT count(*) FROM inbox WHERE id='pc-1'").fetchone()[0] == 0
    finally:
        conn.close()


def test_pull_skips_self(client, monkeypatch):
    from api import sync_peers
    from core_store import get_store

    class _Requests:
        @staticmethod
        def get(url, params=None, headers=None, timeout=None):
            return _Resp(json.dumps({"device_id": get_store().sync_device_id()}))

    monkeypatch.setattr(sync_peers, "requests", _Requests)
    report = sync_peers.pull_from_peer("http://loop.test:8000")
    assert report["skipped"] == "self"


# ── Dedup of double-routed derived rows ──────────────────────────────────────

def test_dedup_collapses_twin_derived_rows(client):
    conn = _conn()
    try:
        with conn:
            conn.execute("INSERT INTO inbox (id, content) VALUES ('cap-x', 'source')")
            conn.execute("INSERT INTO entities (id, canonical_name) VALUES ('ent-1', 'Alexis')")
            # Twin notes: same capture, same content — two devices routed it.
            conn.execute(
                "INSERT INTO atomic_notes (id, content, kind, provenance_capture_id) "
                "VALUES ('n-bbb', 'même note', 'note', 'cap-x')")
            conn.execute(
                "INSERT INTO atomic_notes (id, content, kind, provenance_capture_id) "
                "VALUES ('n-aaa', 'même note', 'note', 'cap-x')")
            # A legitimately different note on the same capture must survive.
            conn.execute(
                "INSERT INTO atomic_notes (id, content, kind, provenance_capture_id) "
                "VALUES ('n-ccc', 'autre contenu', 'note', 'cap-x')")
            # Twin facts.
            conn.execute(
                "INSERT INTO facts (id, entity_id, predicate, value, provenance_capture_id) "
                "VALUES ('f-bbb', 'ent-1', 'aime', 'le café', 'cap-x')")
            conn.execute(
                "INSERT INTO facts (id, entity_id, predicate, value, provenance_capture_id) "
                "VALUES ('f-aaa', 'ent-1', 'aime', 'le café', 'cap-x')")
    finally:
        conn.close()

    from api.sync_peers import dedup_after_pull
    removed = dedup_after_pull()
    assert removed == {"atomic_notes": 1, "facts": 1}

    conn = _conn()
    try:
        notes = [r[0] for r in conn.execute(
            "SELECT id FROM atomic_notes ORDER BY id").fetchall()]
        assert notes == ["n-aaa", "n-ccc"]  # smallest uuid of the twins + the distinct one
        facts = [r[0] for r in conn.execute("SELECT id FROM facts ORDER BY id").fetchall()]
        assert facts == ["f-aaa"]
        # The collapse journals tombstones → it replicates to peers.
        tomb = conn.execute(
            "SELECT count(*) FROM sync_log WHERE col = '-' AND pk IN ('n-bbb', 'f-bbb')"
        ).fetchone()[0]
        assert tomb == 2
    finally:
        conn.close()

    # Idempotent.
    assert dedup_after_pull() == {}


# ── Push (SYN-113: the phone sends its pages, it can't be pulled from) ───────

def test_sync_push_applies_a_peer_changeset(client, peer):
    store, gate = peer
    gate.execute(
        "INSERT INTO inbox (id, content, source) VALUES ('push-1', 'depuis le tel', 'ios')", [])
    gate.execute(
        "INSERT INTO atomic_notes (id, content, kind) VALUES ('push-n1', 'note du tel', 'note')", [])
    page = store.sync_changes_since(0, 10000)

    r = client.post("/sync/push", content=page,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 200
    report = r.json()
    assert report["rows_created"] >= 2
    assert "reembedded" in report and "deduped" in report

    conn = _conn()
    try:
        assert conn.execute(
            "SELECT content FROM inbox WHERE id='push-1'").fetchone()[0] == "depuis le tel"
        assert conn.execute(
            "SELECT count(*) FROM atomic_notes WHERE id='push-n1'").fetchone()[0] == 1
    finally:
        conn.close()

    # Re-pushing the same page is an echo — nothing changes.
    again = client.post("/sync/push", content=page,
                        headers={"Content-Type": "application/json"}).json()
    assert again["rows_created"] == 0
    assert again["rows_updated"] == 0
    assert again["rows_deleted"] == 0


def test_sync_push_rejects_garbage(client):
    r = client.post("/sync/push", content="pas du json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
