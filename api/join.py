"""SYN-137 — joiner-side orchestration of the code pairing (Mac↔Mac).

The joiner is a VIRGIN install (no captures, no entities — enforced): it runs
the SPAKE2 handshake against the member found on the LAN (mDNS, or an explicit
URL), waits for the human approval there, opens the sealed payload and adopts
the space: mesh token stored locally, our own founding rows dropped
unjournaled, then a bootstrap pull from the member. Runs in a background
thread; the app polls `GET /pair/join-status`.

Never log the code, the handshake material or the opened payload.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import time

import requests

from synapse_core import CodePairing, pairing_code_confirm_mac, pairing_open

from db import get_connection

# How long the joiner waits for the human approval on the member (matches the
# member's own offer/request TTL).
_APPROVAL_TTL = 120.0

_lock = threading.Lock()
_state: dict = {"status": "idle"}


class _JoinError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


def status() -> dict:
    with _lock:
        return dict(_state)


def _set(**kw) -> None:
    with _lock:
        _state.clear()
        _state.update(kw)


def is_virgin() -> bool:
    """v1 guard: joining is only offered to a device that never captured
    anything — adopting a space from a lived-in install (merge) is a later
    product stage."""
    conn = get_connection()
    try:
        n_inbox = conn.execute("SELECT COUNT(*) FROM inbox").fetchone()[0]
        n_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        return n_inbox == 0 and n_entities == 0
    finally:
        conn.close()


def start_join(code: str, url: str | None) -> dict:
    code = (code or "").strip().replace(" ", "")
    if len(code) != 6 or not code.isdigit():
        raise _JoinError(422, "code must be 6 digits")
    if not is_virgin():
        raise _JoinError(409, "device_not_virgin")
    with _lock:
        if _state.get("status") in ("searching", "waiting_approval", "applying"):
            raise _JoinError(409, "join already in progress")
        _state.clear()
        _state.update({"status": "searching"})
    threading.Thread(target=_run, args=(code, url), daemon=True).start()
    return {"status": "searching"}


def _candidates(url: str | None) -> list[str]:
    if url:
        return [url.rstrip("/")]
    from api.sync_peers import known_peers

    return [p["url"] for p in known_peers()]


def _run(code: str, url: str | None) -> None:
    try:
        cands = _candidates(url)
        if not cands:
            _set(status="not_found")
            return
        for base in cands:
            if _try_member(code, base) != "next":
                return  # terminal state already set
        _set(status="not_found")
    except Exception:  # noqa: BLE001 — never leak handshake material in a trace
        _set(status="failed")


def _try_member(code: str, base: str) -> str:
    """One full attempt against one candidate member. Returns "next" to try
    the following candidate; any other value means a terminal state was set."""
    session = CodePairing(code)
    msg_j = bytes(session.msg())
    try:
        r = requests.post(
            f"{base}/pair/request-code",
            json={
                "msg": base64.b64encode(msg_j).decode("ascii"),
                "name": socket.gethostname().split(".", 1)[0] or "Nouvel appareil",
                "platform": "desktop",
            },
            timeout=5,
        )
    except requests.RequestException:
        return "next"
    if r.status_code != 200:
        return "next"  # no code shown there (or attempts burned)
    data = r.json()
    request_id = data["request_id"]
    msg_m = base64.b64decode(data["msg"])
    key = bytes(session.finish(msg_m))
    mac = bytes(pairing_code_confirm_mac(key, msg_m, msg_j))
    try:
        c = requests.post(
            f"{base}/pair/confirm-code",
            json={"request_id": request_id,
                  "mac": base64.b64encode(mac).decode("ascii")},
            timeout=5,
        )
    except requests.RequestException:
        return "next"
    if c.status_code != 200:
        # This member shows a DIFFERENT code — maybe another candidate matches.
        return "next"

    _set(status="waiting_approval", member_url=base)
    deadline = time.monotonic() + _APPROVAL_TTL
    while time.monotonic() < deadline:
        try:
            res = requests.get(f"{base}/pair/result/{request_id}", timeout=5).json()
        except requests.RequestException:
            time.sleep(2)
            continue
        st = res.get("status")
        if st == "approved":
            _set(status="applying")
            payload = json.loads(bytes(pairing_open(key, msg_m, msg_j, res["sealed"])))
            _apply(payload, base)
            _set(status="done", space_name=payload.get("space_name"))
            return "done"
        if st in ("denied", "expired"):
            _set(status=st)
            return st
        time.sleep(2)
    _set(status="expired")
    return "expired"


def _apply(payload: dict, base: str) -> None:
    """Adopt the space: mesh token (local, never replicated), opt-in Anthropic
    key, founding rows dropped, bootstrap pull, then our device row joins the
    replicated registry."""
    from api import sync_peers
    from config_store import set_anthropic_key

    token = payload.get("token") or ""
    if token:
        sync_peers.set_mesh_token(token)
    key = payload.get("anthropic_key")
    if key:
        set_anthropic_key(key)
    _adopt_reset()
    sync_peers.pull_from_peer(base)
    sync_peers.register_self_device()


def _adopt_reset() -> None:
    """Drop OUR founding rows (virgin install) WITHOUT journaling — under the
    `applying` flag the sync triggers stay silent — and purge their journal
    entries, so none of our newer-HLC space/owner rows can win the LWW merge
    at the bootstrap pull or ship back to the mesh as changes."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("UPDATE sync_meta SET v = 1 WHERE k = 'applying'")
            conn.execute("DELETE FROM space")
            conn.execute("DELETE FROM sync_owner")
            conn.execute("UPDATE sync_meta SET v = 0 WHERE k = 'applying'")
            conn.execute("DELETE FROM sync_log WHERE tbl IN ('space', 'sync_owner')")
    finally:
        conn.close()
