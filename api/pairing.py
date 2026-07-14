"""Device pairing sessions (SYN-128) — the MEMBER side.

The member (a device already in the space, holding the data + token + optional
key) shows a QR and, after the user approves, hands a joining device the
secrets it needs to join the mesh. The cryptography lives in the Rust core
(`synapse_core.PairingSession` / `pairing_*`, see `synapse-core/src/pairing.rs`):
this module is only the in-memory session state + the transport.

Security model. The joiner endpoints (`/pair/request`, `/pair/result`) are
UNauthenticated on purpose — a fresh device has no bearer token yet. That is
safe because everything those endpoints return is AEAD-sealed under a channel
key derived from the QR secret: an attacker who never saw the QR cannot derive
the key, so the sealed payload is useless to them. The member endpoints
(`/pair/offer`, `/pair/pending`, `/pair/approve`, `/pair/deny`) require the
bearer token: they drive the member's own device.

State is process-local and ephemeral (one active offer per member, short TTL):
pairing is a one-shot in-person action, nothing here is persisted.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid

from synapse_core import PairingSession, pairing_seal

from config_store import get_anthropic_key
from db import first_row, get_connection

# TTL for an offer and for an unclaimed request. Pairing is done face to face
# in under a minute; anything older is stale and dropped.
_OFFER_TTL = 120.0
_REQUEST_TTL = 120.0

_lock = threading.Lock()
# One active offer per process (the member shows one QR at a time).
_offer: dict | None = None
# request_id -> pending/approved/denied request dict.
_requests: dict[str, dict] = {}


def _now() -> float:
    return time.monotonic()


def _prune(now: float) -> None:
    """Drop the offer + requests that have outlived their TTL. Caller holds
    the lock."""
    global _offer
    if _offer is not None and now - _offer["created"] > _OFFER_TTL:
        _offer = None
    stale = [rid for rid, r in _requests.items() if now - r["created"] > _REQUEST_TTL]
    for rid in stale:
        _requests.pop(rid, None)


def _local_addrs() -> list[str]:
    """Reachable base URLs for THIS backend, for the QR. The primary LAN IP
    (via a dummy UDP connect, no traffic sent) + the API port."""
    port = os.environ.get("SYNAPSE_API_PORT", "8000")
    ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        ip = None
    addrs = []
    if ip:
        addrs.append(f"http://{ip}:{port}")
    return addrs


def start_offer() -> dict:
    """Begin showing a QR (member side). Replaces any prior offer. Returns
    `{qr}` — render `qr` as a QR code for the joiner to scan."""
    global _offer
    session, qr = PairingSession.offer(_local_addrs())
    with _lock:
        _prune(_now())
        _offer = {"session": session, "qr": qr, "offer_pub": session.offer_pub(),
                  "created": _now()}
        # A fresh offer invalidates requests aimed at the previous one.
        _requests.clear()
    return {"qr": qr}


def submit_request(accept_pub: bytes, name: str, platform: str) -> dict:
    """Joiner side (unauthenticated): submit the scanner's public key + who we
    are. Returns `{request_id}`. The member must still approve."""
    with _lock:
        _prune(_now())
        if _offer is None:
            raise _PairingError(409, "no active pairing offer")
        channel_key = _offer["session"].channel_key(accept_pub)
        request_id = str(uuid.uuid4())
        _requests[request_id] = {
            "name": (name or "").strip()[:80] or "Nouvel appareil",
            "platform": (platform or "").strip()[:32] or "unknown",
            "accept_pub": accept_pub,
            "channel_key": channel_key,
            "status": "pending",
            "sealed": None,
            "created": _now(),
        }
        return {"request_id": request_id}


def list_pending() -> list[dict]:
    """Member side: the requests awaiting the user's approval."""
    with _lock:
        _prune(_now())
        return [
            {"request_id": rid, "name": r["name"], "platform": r["platform"]}
            for rid, r in _requests.items()
            if r["status"] == "pending"
        ]


def approve(request_id: str, include_key: bool) -> dict:
    """Member side: the user approved. Seal the payload (space_id, name, the
    member's sync token, peer URLs, and the API key IFF opted in) under the
    request's channel key. The joiner fetches it via `/pair/result`."""
    with _lock:
        _prune(_now())
        req = _requests.get(request_id)
        if req is None:
            raise _PairingError(404, "unknown or expired request")
        if req["status"] != "pending":
            raise _PairingError(409, f"request already {req['status']}")
        if _offer is None:
            raise _PairingError(409, "offer expired — restart pairing")
        payload = _build_payload(include_key)
        sealed = pairing_seal(
            req["channel_key"], _offer["offer_pub"], req["accept_pub"], payload
        )
        req["status"] = "approved"
        req["sealed"] = sealed
    return {"status": "approved"}


def deny(request_id: str) -> dict:
    with _lock:
        req = _requests.get(request_id)
        if req is None:
            raise _PairingError(404, "unknown or expired request")
        req["status"] = "denied"
        req["sealed"] = None
    return {"status": "denied"}


def poll_result(request_id: str) -> dict:
    """Joiner side (unauthenticated): poll for the outcome. On approval,
    returns the sealed payload ONCE, then consumes the request so the
    single-use secret can't be re-fetched."""
    with _lock:
        _prune(_now())
        req = _requests.get(request_id)
        if req is None:
            return {"status": "expired"}
        if req["status"] == "approved":
            sealed = req["sealed"]
            _requests.pop(request_id, None)  # one-shot delivery
            return {"status": "approved", "sealed": sealed}
        if req["status"] == "denied":
            _requests.pop(request_id, None)
            return {"status": "denied"}
        return {"status": "pending"}


def _build_payload(include_key: bool) -> bytes:
    """The secrets the joiner needs, as compact JSON bytes. Never logged."""
    import json

    from api.sync_peers import known_peers

    conn = get_connection()
    try:
        row = first_row(conn.execute(
            "SELECT space_id, name FROM space WHERE id = 'space'"))
    finally:
        conn.close()
    space_id = row["space_id"] if row else None
    space_name = (row["name"] if row else None) or "Ma mémoire"
    peers = [p["url"] for p in known_peers()] + _local_addrs()
    payload = {
        "space_id": space_id,
        "space_name": space_name,
        "token": os.environ.get("SYNAPSE_API_TOKEN", ""),
        "peers": sorted(set(peers)),
    }
    if include_key:
        key = get_anthropic_key()
        if key:
            payload["anthropic_key"] = key
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class _PairingError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)
