"""
SYN-112 (T3, phase 3) — Mac↔Mac transport on top of the core sync engine.

Pull-based mesh: each backend periodically pulls `/sync/changes` from its
peers (mDNS-discovered and/or `SYNAPSE_SYNC_PEERS`), merges through the
core's per-column LWW (`Storage.sync_apply`), re-embeds the notes the merge
touched, dedups double-routed derived rows, and advances a per-peer cursor.

Trust model: same as the rest of the API — LAN/Tailscale peers sharing ONE
`SYNAPSE_API_TOKEN` (each side presents its own token when pulling; without
a token, auth is dev-disabled everywhere).

Owner-lock: `sync_owner` is a REPLICATED singleton row (it travels with the
sync itself) naming the one device allowed to run the Dream Cycle — the
whole derived layer stays single-writer, which is what keeps the LWW merge
trivially correct. `ensure_cycle_owner()` is the run-guard: implicit claim
on first run (single-device install), 409 when another device holds it.
The dedup pass is the safety net for the exceptional double-route (lock
transferred mid-window): identical derived rows created on two devices from
the same capture collapse onto the smallest uuid, and the deletions
propagate as tombstones.
"""

import json
import logging
import os
import threading
import time

import requests
from fastapi import HTTPException

from core_store import get_store
from db import first_row, get_connection

log = logging.getLogger(__name__)

SYNC_PAGE_LIMIT = 5000


def _headers() -> dict:
    token = os.environ.get("SYNAPSE_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def sync_interval() -> int:
    """Seconds between automatic pull rounds; 0 disables the loop."""
    try:
        return int(os.environ.get("SYNAPSE_SYNC_INTERVAL", "600"))
    except ValueError:
        return 600


# ── Per-peer cursors (local state, deliberately NOT replicated) ──────────────

def get_cursor(conn, peer_device: str) -> int:
    row = conn.execute(
        "SELECT v FROM sync_meta WHERE k = ?", (f"cursor:{peer_device}",)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def set_cursor(conn, peer_device: str, seq: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO sync_meta (k, v) VALUES (?, ?)",
        (f"cursor:{peer_device}", int(seq)),
    )


def all_cursors(conn) -> dict:
    rows = conn.execute(
        "SELECT k, v FROM sync_meta WHERE k LIKE 'cursor:%'"
    ).fetchall()
    return {k.split(":", 1)[1]: int(v) for k, v in rows}


# ── Owner-lock (replicated) + run-guard ──────────────────────────────────────

def get_owner(conn) -> dict | None:
    row = first_row(conn.execute(
        "SELECT device_id, epoch, claimed_at FROM sync_owner WHERE id = 'owner'"
    ))
    return dict(row) if row else None


def claim_owner(device_id: str) -> dict:
    """Claim the cycle for `device_id`. INSERT OR REPLACE journals the whole
    row at one HLC → concurrent claims are settled row-atomically by LWW."""
    conn = get_connection()
    try:
        with conn:
            current = get_owner(conn)
            epoch = (current["epoch"] if current else 0) + 1
            conn.execute(
                "INSERT OR REPLACE INTO sync_owner (id, device_id, epoch, claimed_at) "
                "VALUES ('owner', ?, ?, CURRENT_TIMESTAMP)",
                (device_id, epoch),
            )
        return get_owner(conn)
    finally:
        conn.close()


def ensure_cycle_owner() -> None:
    """Run-guard for the Dream Cycle. No owner yet = single-device install →
    implicit claim (the production Mac self-elects on its first cycle)."""
    me = get_store().sync_device_id()
    conn = get_connection()
    try:
        owner = get_owner(conn)
    finally:
        conn.close()
    if owner is None:
        claim_owner(me)
        return
    if owner["device_id"] != me:
        raise HTTPException(
            status_code=409,
            detail=f"dream cycle owned by device {owner['device_id']} "
                   f"(epoch {owner['epoch']}) — transfer it with PUT /sync/owner",
        )


# ── Pulling from a peer ──────────────────────────────────────────────────────

def pull_from_peer(base_url: str, timeout: int = 30) -> dict:
    """Pull + merge everything new from one peer. Idempotent and resumable:
    the cursor advances after each applied page."""
    base = base_url.rstrip("/")
    store = get_store()
    me = store.sync_device_id()

    status = requests.get(
        f"{base}/sync/status", headers=_headers(), timeout=timeout
    ).json()
    peer_device = status.get("device_id")
    if not peer_device:
        return {"url": base, "error": "peer exposes no sync device_id"}
    if peer_device == me:
        return {"url": base, "peer_device": peer_device, "skipped": "self"}

    conn = get_connection()
    try:
        cursor = get_cursor(conn, peer_device)
    finally:
        conn.close()

    agg = {"rows_created": 0, "rows_updated": 0, "rows_deleted": 0,
           "skipped": 0, "conflicts": 0}
    notes: set[str] = set()
    pages = 0
    while True:
        r = requests.get(
            f"{base}/sync/changes",
            params={"since": cursor, "limit": SYNC_PAGE_LIMIT},
            headers=_headers(), timeout=timeout,
        )
        r.raise_for_status()
        # r.text goes to the core verbatim — the protocol check lives there.
        report = json.loads(store.sync_apply(r.text))
        payload = r.json()
        for k in agg:
            agg[k] += report.get(k, 0)
        notes.update(report.get("notes_changed", []))
        cursor = payload["next"]
        pages += 1
        conn = get_connection()
        try:
            with conn:
                set_cursor(conn, peer_device, cursor)
        finally:
            conn.close()
        if not payload.get("has_more"):
            break

    reembedded = reembed_notes(sorted(notes))
    deduped = dedup_after_pull()
    return {"url": base, "peer_device": peer_device, "pages": pages,
            "cursor": cursor, "reembedded": reembedded, "deduped": deduped,
            **agg}


def apply_pushed(changes_json: str) -> dict:
    """SYN-113: merge one changeset PUSHED by a peer (a phone is not
    reachable over HTTP, so unlike the Mac↔Mac pull mesh it must send its
    pages). Same post-merge machinery as a pull: re-embed + twin dedup. The
    payload goes to the core verbatim — the protocol check lives there."""
    store = get_store()
    report = json.loads(store.sync_apply(changes_json))
    reembedded = reembed_notes(sorted(report.get("notes_changed", [])))
    deduped = dedup_after_pull()
    return {"reembedded": reembedded, "deduped": deduped,
            **{k: v for k, v in report.items() if k != "notes_changed"}}


def reembed_notes(note_ids) -> int:
    """The vec0 index is derived and never on the wire: recompute locally for
    every note the merge created or changed (same text shape as routing:
    title + content). A tombstoned note was already dropped by the merge."""
    if not note_ids:
        return 0
    from embeddings import embed_text, embed_text_chunks

    store = get_store()
    done = 0
    conn = get_connection()
    try:
        for nid in note_ids:
            row = first_row(conn.execute(
                "SELECT title, content FROM atomic_notes WHERE id = ?", (nid,)
            ))
            if not row:
                continue
            content = (row["content"] or "").strip()
            title = row["title"] or content[:60]
            try:
                store.upsert_note_vectors(nid, embed_text_chunks(f"{title}\n{content}"))
                done += 1
            except Exception as exc:  # noqa: BLE001 — model may be absent
                log.warning("sync: re-embed failed for note %s: %s", nid, exc)
    finally:
        conn.close()
    return done


# ── Dedup of double-routed derived rows (post-merge safety net) ──────────────

# (table, guard column that must be non-null, natural-identity columns)
_DEDUP_RULES = [
    ("atomic_notes", "provenance_capture_id",
     ["provenance_capture_id", "content", "kind"]),
    ("facts", "provenance_capture_id",
     ["entity_id", "predicate", "value", "provenance_capture_id"]),
    ("relations", "provenance_capture_id",
     ["entity_from", "predicate", "entity_to", "provenance_capture_id"]),
    ("project_entries", "capture_id",
     ["project_id", "capture_id", "content", "kind"]),
]


def dedup_after_pull() -> dict:
    """Collapse rows that two devices derived from the same capture (they
    carry different uuids but the same natural identity) onto the smallest
    uuid. Deletions journal as tombstones → the collapse replicates.
    Entities are NOT deduped here: the existing merge-proposal machinery
    (embedding similarity) already handles same-name entities gracefully."""
    removed: dict[str, int] = {}
    doomed_notes: list[str] = []
    conn = get_connection()
    try:
        with conn:
            for table, guard, keys in _DEDUP_RULES:
                key_expr = ", ".join(keys)
                doomed = [r[0] for r in conn.execute(
                    f"SELECT id FROM {table} WHERE {guard} IS NOT NULL "
                    f"AND id NOT IN (SELECT min(id) FROM {table} "
                    f"               WHERE {guard} IS NOT NULL GROUP BY {key_expr})"
                ).fetchall()]
                if not doomed:
                    continue
                ph = ",".join("?" * len(doomed))
                conn.execute(f"DELETE FROM {table} WHERE id IN ({ph})", doomed)
                removed[table] = len(doomed)
                if table == "atomic_notes":
                    doomed_notes = doomed
    finally:
        conn.close()
    # Vector cleanup outside the transaction (core writes must never nest
    # inside a host transaction — T1 rule).
    store = get_store()
    for nid in doomed_notes:
        try:
            store.delete_note_vector(nid)
        except Exception:  # noqa: BLE001
            pass
    return removed


# ── Peer assembly + periodic loop ────────────────────────────────────────────

def known_peers() -> list[dict]:
    """Static peers (SYNAPSE_SYNC_PEERS=url,url) + mDNS-discovered ones."""
    peers: list[dict] = []
    static = os.environ.get("SYNAPSE_SYNC_PEERS", "")
    for u in (s.strip() for s in static.split(",")):
        if u:
            peers.append({"url": u.rstrip("/"), "source": "static"})
    try:
        from api.discovery import discovered_peers
        for p in discovered_peers():
            peers.append({**p, "source": "mdns"})
    except Exception:  # noqa: BLE001 — zeroconf optional/disabled
        pass
    seen: set[str] = set()
    return [p for p in peers if not (p["url"] in seen or seen.add(p["url"]))]


def pull_all() -> list[dict]:
    reports = []
    for peer in known_peers():
        try:
            reports.append(pull_from_peer(peer["url"]))
        except Exception as exc:  # noqa: BLE001 — peer down is normal life
            reports.append({"url": peer["url"], "error": f"{type(exc).__name__}: {exc}"})
    return reports


def _sync_loop() -> None:
    interval = sync_interval()
    time.sleep(min(30, interval))  # let the mDNS browser populate first
    while True:
        try:
            reports = pull_all()
            for rep in reports:
                if rep.get("rows_created") or rep.get("rows_updated") \
                        or rep.get("rows_deleted"):
                    log.info("sync: pulled from %s: %s", rep.get("url"), rep)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval)


def start_sync_thread() -> None:
    if sync_interval() <= 0:
        return
    threading.Thread(target=_sync_loop, daemon=True, name="peer-sync").start()
