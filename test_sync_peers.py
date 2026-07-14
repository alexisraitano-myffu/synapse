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
    # SYN-129: the guard's detail is structured so clients render a human
    # message (code + owner identity + epoch).
    detail = r.json()["detail"]
    assert detail["code"] == "not_cycle_owner"
    assert detail["owner_device_id"] == "other-mac"
    assert detail["epoch"] == 2
    assert "other-mac" in detail["message"]

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


# ── Espace + registre d'appareils (SYN-127) ──────────────────────────────────

def test_register_self_device_seeds_then_only_refreshes(client):
    from api.sync_peers import register_self_device
    from core_store import get_store
    me = get_store().sync_device_id()

    register_self_device()
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT name, platform, last_seen FROM devices WHERE device_id = ?",
            (me,)).fetchone()
        assert row is not None and row[0] and row[1]
        # Un rename utilisateur survit aux boots suivants.
        with conn:
            conn.execute("UPDATE devices SET name = 'Mon Mac' WHERE device_id = ?", (me,))
    finally:
        conn.close()

    register_self_device()
    conn = _conn()
    try:
        name = conn.execute(
            "SELECT name FROM devices WHERE device_id = ?", (me,)).fetchone()[0]
        assert name == "Mon Mac"
    finally:
        conn.close()


def test_ensure_space_is_owner_only(client):
    from api.sync_peers import claim_owner, ensure_space, get_space
    from core_store import get_store

    # Pas d'owner → pas de création (une réplique fraîche ne fonde jamais).
    ensure_space()
    conn = _conn()
    try:
        assert get_space(conn) is None
    finally:
        conn.close()

    # Owner = moi → fondation, puis idempotent.
    claim_owner(get_store().sync_device_id())
    ensure_space()
    ensure_space()
    conn = _conn()
    try:
        space = get_space(conn)
        assert space and space["name"] == "Ma mémoire" and space["space_id"]
    finally:
        conn.close()

    # Owner = un autre device → un non-owner ne fonde rien.
    conn = _conn()
    try:
        with conn:
            conn.execute("DELETE FROM space")
            conn.execute("UPDATE sync_owner SET device_id = 'other-device'")
    finally:
        conn.close()
    ensure_space()
    conn = _conn()
    try:
        assert get_space(conn) is None
    finally:
        conn.close()


def test_space_and_devices_endpoints(client):
    from api.sync_peers import claim_owner, ensure_space, register_self_device
    from core_store import get_store
    me = get_store().sync_device_id()

    register_self_device()
    claim_owner(me)
    ensure_space()

    body = client.get("/space").json()
    assert body["space"]["name"] == "Ma mémoire"
    assert body["device_id"] == me and body["owner_device_id"] == me

    renamed = client.patch("/space", json={"name": "Mémoire d'Alexis"}).json()
    assert renamed["space"]["name"] == "Mémoire d'Alexis"
    assert client.patch("/space", json={"name": "  "}).status_code == 422

    devices = client.get("/devices").json()["devices"]
    assert len(devices) == 1
    assert devices[0]["is_self"] and devices[0]["is_owner"] and not devices[0]["revoked"]

    # Garde-fous de révocation : soi-même et l'owner sont intouchables.
    assert client.patch(f"/device/{me}", json={"revoked": True}).status_code == 409
    conn = _conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO devices (device_id, name, platform) "
                "VALUES ('peer-1', 'Pixel', 'android')")
    finally:
        conn.close()

    out = client.patch("/device/peer-1", json={"name": "Pixel d'Alexis",
                                               "revoked": True}).json()
    assert out["name"] == "Pixel d'Alexis" and out["revoked"]

    # Un pair révoqué est sauté par la boucle de pull.
    from api.sync_peers import device_revoked
    conn = _conn()
    try:
        assert device_revoked(conn, "peer-1") is True
    finally:
        conn.close()

    restored = client.patch("/device/peer-1", json={"revoked": False}).json()
    assert not restored["revoked"]
    assert client.patch("/device/inconnu", json={"name": "x"}).status_code == 404


# ── Appairage (SYN-128) ──────────────────────────────────────────────────────

def test_pairing_end_to_end_transfers_secrets(client, monkeypatch):
    """Member offers a QR → joiner scans (real core crypto) → member approves
    with key opt-in → joiner opens the sealed payload and gets space_id, token
    and the key. No token needed on the joiner endpoints."""
    import base64
    import json
    from synapse_core import pairing_accept, pairing_offer_addrs, pairing_open

    from api.sync_peers import claim_owner, ensure_space
    from core_store import get_store

    # Member founds a space + has a key; the request carries a bearer token.
    monkeypatch.setenv("SYNAPSE_API_TOKEN", "member-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    claim_owner(get_store().sync_device_id())
    ensure_space()
    auth = {"Authorization": "Bearer member-token"}

    # 1. Member starts the offer (auth required).
    assert client.post("/pair/offer").status_code == 401
    qr = client.post("/pair/offer", headers=auth).json()["qr"]
    assert pairing_offer_addrs(qr) is not None

    # 2. Joiner scans locally (core), submits accept_pub — NO auth.
    accept_pub, joiner_key = pairing_accept(qr)
    req = client.post("/pair/request", json={
        "accept_pub_b64": base64.b64encode(accept_pub).decode(),
        "name": "Pixel d'Alexis", "platform": "android"}).json()
    request_id = req["request_id"]

    # 3. Member sees the pending request and approves with the key.
    pend = client.get("/pair/pending", headers=auth).json()["requests"]
    assert any(p["request_id"] == request_id and p["name"] == "Pixel d'Alexis" for p in pend)
    assert client.post("/pair/approve", headers=auth,
                       json={"request_id": request_id, "include_key": True}).status_code == 200

    # 4. Joiner polls, opens the sealed payload with its channel key.
    res = client.get(f"/pair/result/{request_id}").json()
    assert res["status"] == "approved"
    offer_pub = base64.b64decode(_offer_pub_from_qr(qr))
    opened = pairing_open(joiner_key, offer_pub, accept_pub, res["sealed"])
    payload = json.loads(opened)
    assert payload["token"] == "member-token"
    assert payload["space_id"]
    assert payload["anthropic_key"] == "sk-ant-secret"

    # 5. One-shot: a second poll no longer returns the secret.
    assert client.get(f"/pair/result/{request_id}").json()["status"] == "expired"


def test_pairing_denied_and_key_optout(client, monkeypatch):
    import base64
    import json
    from synapse_core import pairing_accept, pairing_open

    from api.sync_peers import claim_owner, ensure_space
    from core_store import get_store

    monkeypatch.setenv("SYNAPSE_API_TOKEN", "member-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    claim_owner(get_store().sync_device_id())
    ensure_space()
    auth = {"Authorization": "Bearer member-token"}

    qr = client.post("/pair/offer", headers=auth).json()["qr"]
    accept_pub, joiner_key = pairing_accept(qr)

    # Opt-OUT of the key.
    rid = client.post("/pair/request", json={
        "accept_pub_b64": base64.b64encode(accept_pub).decode(),
        "name": "Mac", "platform": "darwin"}).json()["request_id"]
    client.post("/pair/approve", headers=auth,
                json={"request_id": rid, "include_key": False})
    res = client.get(f"/pair/result/{rid}").json()
    offer_pub = base64.b64decode(_offer_pub_from_qr(qr))
    payload = json.loads(pairing_open(joiner_key, offer_pub, accept_pub, res["sealed"]))
    assert "anthropic_key" not in payload

    # A denied request tells the joiner nothing sealed.
    rid2 = client.post("/pair/request", json={
        "accept_pub_b64": base64.b64encode(accept_pub).decode(),
        "name": "X", "platform": "y"}).json()["request_id"]
    client.post("/pair/deny", headers=auth, json={"request_id": rid2})
    assert client.get(f"/pair/result/{rid2}").json()["status"] == "denied"


def _offer_pub_from_qr(qr: str) -> str:
    # QR wire form: "v|offer_pub_b64|secret_b64|addrs"
    return qr.split("|")[1]
