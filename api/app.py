"""
Synapse HTTP API (FastAPI) — capture, retrieval, validation, sync.

Runs on the "brain" Mac. Auth = bearer token (`SYNAPSE_API_TOKEN`); if that env
var is unset, auth is DISABLED (dev mode) with a warning. Designed for LAN /
Tailscale, no cloud. Response shapes are frozen on the target spec (fields like
`memory_strength` are present even when not yet populated).

Run:  python -m api   (uvicorn on 0.0.0.0:8000)
"""

import contextlib
import io
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config_store
from config import BASE_DIR
from db import cursor_to_dicts, first_row, get_connection, init_db
from embeddings import embed_text
from entity_search import entity_embedding_text, search_entities_by_vector

init_db()


# ── Auto-trigger (event-driven, debounced) ───────────────────────────────────
# After captures arrive, run the cycle once the inbox has been quiet for the
# debounce window (a synced batch → one cycle). Also acts as a catch-up safety
# net (queued entries from a previous session run shortly after startup).
# Disabled unless SYNAPSE_AUTO_CYCLE is truthy. The lock prevents overlap.

_last_capture_ts = 0.0


def _auto_cycle_enabled() -> bool:
    return os.environ.get("SYNAPSE_AUTO_CYCLE", "").lower() in ("1", "true", "yes", "on")


def _debounce_seconds() -> int:
    try:
        return int(os.environ.get("SYNAPSE_CYCLE_DEBOUNCE_SECONDS", "120"))
    except ValueError:
        return 120


def _scheduler_loop() -> None:
    while True:
        time.sleep(30)
        try:
            if not _auto_cycle_enabled():
                continue
            conn = get_connection()
            try:
                queued = conn.execute(
                    "SELECT COUNT(*) FROM inbox WHERE processed_at IS NULL"
                ).fetchone()[0]
            finally:
                conn.close()
            if queued == 0 or time.time() - _last_capture_ts < _debounce_seconds():
                continue
            try:
                dream_cycle_run(trigger="auto")  # guarded by the lock; errors land in cycle_runs
            except Exception:
                pass  # retry next tick
        except Exception:
            pass


@contextlib.asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_scheduler_loop, daemon=True).start()
    # Loopback-only builds (bundled tester binary) set SYNAPSE_DISABLE_MDNS=1
    # to skip zeroconf — mobile clients on the LAN won't reach this instance.
    azc = None
    if not os.environ.get("SYNAPSE_DISABLE_MDNS"):
        from api.discovery import start_advertising, stop_advertising
        azc = await start_advertising()
    try:
        yield
    finally:
        if azc is not None:
            from api.discovery import stop_advertising
            await stop_advertising(azc)


app = FastAPI(title="Synapse API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Auth ─────────────────────────────────────────────────────────────────────

def require_auth(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("SYNAPSE_API_TOKEN")
    if not token:
        return  # dev mode — auth disabled
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# ── Cycle lock (single-instance guard) ───────────────────────────────────────

_LOCK_PATH = BASE_DIR / "cycle.lock"
_LOCK_STALE_SECONDS = 1800


@contextlib.contextmanager
def cycle_lock():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if _LOCK_PATH.exists() and time.time() - _LOCK_PATH.stat().st_mtime > _LOCK_STALE_SECONDS:
        _LOCK_PATH.unlink(missing_ok=True)  # recover a stale lock
    try:
        fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise HTTPException(status_code=409, detail="a dream cycle is already running")
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        _LOCK_PATH.unlink(missing_ok=True)


# ── Models ───────────────────────────────────────────────────────────────────

class CaptureIn(BaseModel):
    id: str                       # client-generated UUID (idempotency key)
    content: str
    device_id: str | None = None
    captured_at: str | None = None
    type: str = "text"
    source: str = "manual"


class ValidateIn(BaseModel):
    confirmed: bool
    correction: str | None = None
    device_id: str | None = None


EntityType = Literal["person", "place", "project", "concept", "organization", "animal"]


class EntityUpdate(BaseModel):
    type: EntityType


class AnthropicKeyIn(BaseModel):
    key: str


# SYN-45 — bodies for project-entry correction endpoints
class ProjectEntryMoveIn(BaseModel):
    project_id: str


class ProjectEntryFactIn(BaseModel):
    predicate: str
    value: str
    entity_id: str
    persistence_value: int = 3


# SYN-39 — merge proposal acceptance body
class MergeAcceptIn(BaseModel):
    canonical_id: str  # which of the two entities survives as canonical


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    conn = get_connection()
    try:
        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM pending_facts").fetchone()[0]
        queued = conn.execute(
            "SELECT COUNT(*) FROM inbox WHERE processed_at IS NULL"
        ).fetchone()[0]
        return {"status": "ok", "entities": entities, "pending": pending, "inbox_queued": queued}
    finally:
        conn.close()


@app.get("/config", dependencies=[Depends(require_auth)])
def get_config():
    """Status only — never echoes the key back. Used by the wizard to know
    whether to prompt for one."""
    return {"anthropic_key_set": config_store.has_anthropic_key()}


@app.put("/config/anthropic-key", dependencies=[Depends(require_auth)])
def put_anthropic_key(body: AnthropicKeyIn):
    """Stores the key in ~/.synapse/config.json (0600). Lets the desktop app
    push the key without touching .env files."""
    key = body.key.strip()
    if not key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="invalid key format (expected sk-...)")
    config_store.set_anthropic_key(key)
    return {"status": "ok"}


@app.post("/capture", dependencies=[Depends(require_auth)])
def capture(item: CaptureIn):
    """Idempotent on the client-generated id — re-POSTing the same id is a no-op."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO inbox "
                "(content, source, client_id, device_id, captured_at, status) "
                "VALUES (?,?,?,?,?, 'queued')",
                (item.content, item.source, item.id, item.device_id, item.captured_at),
            )
        row = first_row(conn.execute(
            "SELECT id, created_at FROM inbox WHERE client_id=?", (item.id,)
        ))
        # changes() tells us whether the INSERT actually added a row
        created = conn.execute("SELECT changes()").fetchone()[0] == 1
        if created:
            global _last_capture_ts
            _last_capture_ts = time.time()  # arm the debounced auto-trigger
        return {
            "id": row["id"] if row else None,
            "client_id": item.id,
            "status": "queued" if created else "duplicate",
        }
    finally:
        conn.close()


@app.get("/feed", dependencies=[Depends(require_auth)])
def feed(limit: int = 30):
    limit = min(max(1, limit), 100)
    conn = get_connection()
    try:
        rows = cursor_to_dicts(conn.execute(
            "SELECT id, client_id, content, source, created_at, captured_at, processed_at, status "
            "FROM inbox ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ))
        for r in rows:
            status = r.get("status") or "queued"
            # legacy rows processed before the status column existed
            if status == "queued" and r.get("processed_at"):
                status = "processed"
            r["status"] = status
        return rows
    finally:
        conn.close()


def _ego_filter(entities: list[dict], relations: list[dict], focus: str):
    """Keep the focus entity and its direct neighbours."""
    focus_ids = {
        e["id"] for e in entities
        if e["id"] == focus or (e.get("label", "").lower() == focus.lower())
    }
    if not focus_ids:
        return [], []
    keep = set(focus_ids)
    kept_rels = []
    for r in relations:
        if r["from"] in focus_ids or r["to"] in focus_ids:
            keep.add(r["from"]); keep.add(r["to"]); kept_rels.append(r)
    return [e for e in entities if e["id"] in keep], kept_rels


@app.get("/graph", dependencies=[Depends(require_auth)])
def graph(entity: str | None = None, mode: str = "full"):
    """Nodes = entities (size ~ mention_count), edges = relations."""
    conn = get_connection()
    try:
        nodes = []
        # SYN-39: hide soft-merged rows; their data already lives on the canonical one.
        for e in cursor_to_dicts(conn.execute(
            "SELECT e.id, e.canonical_name, e.type, e.mention_count, e.persistence_value, "
            "       e.summary, e.last_mentioned, "
            "       (SELECT COUNT(*) FROM facts f WHERE f.entity_id = e.id) AS facts_count "
            "FROM entities e WHERE e.merged_into_id IS NULL"
        )):
            nodes.append({
                "id": e["id"],
                "label": e["canonical_name"],
                "type": e.get("type"),
                "mention_count": e.get("mention_count", 1),
                "persistence_value": e.get("persistence_value", 3),
                "summary": e.get("summary"),
                "last_mentioned": e.get("last_mentioned"),
                "facts_count": e.get("facts_count", 0),
                "memory_strength": None,  # reserved (Phase C)
            })
        edges = [
            {"from": r["entity_from"], "to": r["entity_to"],
             "label": r["predicate"], "confidence": r["confidence"]}
            for r in cursor_to_dicts(conn.execute(
                "SELECT entity_from, entity_to, predicate, confidence FROM relations"
            ))
        ]
        if entity and mode == "ego":
            nodes, edges = _ego_filter(nodes, edges, entity)
        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()


@app.get("/entity/{entity_id}", dependencies=[Depends(require_auth)])
def entity_detail(entity_id: str):
    import json as _json
    conn = get_connection()
    try:
        e = first_row(conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)))
        if not e:
            raise HTTPException(status_code=404, detail="entity not found")
        # SYN-39: an absorbed entity redirects to its canonical so clients that
        # held the old id don't 404 silently — they get pointed at the survivor.
        if e.get("merged_into_id"):
            raise HTTPException(
                status_code=410,
                detail={"reason": "merged", "merged_into_id": e["merged_into_id"]},
            )
        # SYN-54: surface provenance_capture_id on every projection so the client
        # can render a "source" chip pointing back to the immutable inbox row.
        facts = cursor_to_dicts(conn.execute(
            "SELECT predicate, value, confidence, persistence_value, created_at, "
            "       provenance_capture_id "
            "FROM facts WHERE entity_id=? ORDER BY confidence DESC", (entity_id,),
        ))
        relations = cursor_to_dicts(conn.execute(
            "SELECT r.predicate, e.canonical_name AS entity_to, r.confidence, "
            "       r.provenance_capture_id "
            "FROM relations r JOIN entities e ON e.id=r.entity_to WHERE r.entity_from=?",
            (entity_id,),
        ))
        try:
            aliases = _json.loads(e.get("aliases", "[]"))
        except (ValueError, TypeError):
            aliases = []
        return {
            "id": e["id"], "canonical_name": e["canonical_name"], "type": e.get("type"),
            "aliases": aliases, "summary": e.get("summary"),
            "mention_count": e.get("mention_count", 1),
            "persistence_value": e.get("persistence_value", 3),
            "last_mentioned": e.get("last_mentioned"),
            "facts_count": len(facts),
            "facts": facts, "relations": relations,
            "provenance_capture_id": e.get("provenance_capture_id"),
        }
    finally:
        conn.close()


@app.get("/entity/{entity_id}/similar", dependencies=[Depends(require_auth)])
def entity_similar(
    entity_id: str,
    limit: int = 5,
    min_score: float = 0.7,
    same_type: bool = False,
):
    """SYN-62: soft semantic neighbours of an entity — links the user never
    stated explicitly ('Escalade' ↔ 'Bouldering', 'Schopenhauer' ↔ 'Nietzsche').

    Suggestion only, never a materialized relation: each call recomputes against
    the current graph (no cache — fastembed is local and the scan is sub-ms), so
    results evolve as new entities/summaries appear. `same_type=true` restricts
    to the entity's own type. Soft-merged entities are excluded by the search.
    """
    conn = get_connection()
    try:
        e = first_row(conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)))
        if not e:
            raise HTTPException(status_code=404, detail="entity not found")
        if e.get("merged_into_id"):
            raise HTTPException(
                status_code=410,
                detail={"reason": "merged", "merged_into_id": e["merged_into_id"]},
            )
        # Reuse the stored vector when present (step6 fills it); fall back to an
        # on-the-fly embed for an entity not yet vectorized.
        query_vec = e.get("embedding") or embed_text(entity_embedding_text(e))
        matches = search_entities_by_vector(
            conn, query_vec,
            limit=max(1, min(limit, 50)),
            min_score=min_score,
            type_filter=e["type"] if same_type else None,
            exclude_ids={entity_id},
        )
        return {
            "entity_id": entity_id,
            "similar": [
                {
                    "entity_id": m["id"],
                    "canonical_name": m["canonical_name"],
                    "type": m["type"],
                    "summary": m["summary"],
                    "similarity_score": m["score"],
                }
                for m in matches
            ],
        }
    finally:
        conn.close()


@app.get("/projects", dependencies=[Depends(require_auth)])
def projects_list():
    """List entities of type=project with synthesis preview (SYN-53).

    One row per project, joined to its current_state (if any) and counted
    against project_entries. Single trip — avoids the N+1 /entity/{id} +
    /project/{id}/state walk a client would otherwise do.
    """
    conn = get_connection()
    try:
        rows = cursor_to_dicts(conn.execute(
            "SELECT e.id, e.canonical_name, e.mention_count, e.persistence_value, "
            "       e.last_mentioned, e.summary, "
            "       psv.summary_md AS current_summary_md, "
            "       psv.kind        AS current_kind, "
            "       psv.created_at  AS current_synthesized_at, "
            "       (SELECT COUNT(*) FROM project_entries pe WHERE pe.project_id = e.id) "
            "         AS entries_total "
            "FROM entities e "
            "LEFT JOIN project_state ps ON ps.project_id = e.id "
            "LEFT JOIN project_state_versions psv ON psv.id = ps.current_version_id "
            "WHERE e.type = 'project' AND e.merged_into_id IS NULL "
            "ORDER BY COALESCE(e.last_mentioned, e.created_at) DESC"
        ))
        return rows
    finally:
        conn.close()


@app.get("/atomic-notes", dependencies=[Depends(require_auth)])
def atomic_notes_list(
    limit: int = 50,
    q: str | None = None,
    entity: str | None = None,
):
    """List atomic_notes for the Notes view (SYN-52).

    Filters are AND-combined and best-effort:
    - q: substring match on title or content (case-insensitive)
    - entity: matches any note whose entities_mentioned JSON array contains
      the canonical name (LIKE on the serialized list — cheap, no JSON1)
    """
    limit = min(max(1, limit), 200)
    conn = get_connection()
    try:
        clauses = []
        params: list = []
        if q:
            clauses.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)")
            needle = f"%{q.lower()}%"
            params.extend([needle, needle])
        if entity:
            clauses.append("entities_mentioned LIKE ?")
            params.append(f'%"{entity}"%')
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = cursor_to_dicts(conn.execute(
            f"SELECT id, title, content, summary, entities_mentioned, memory_strength, "
            f"       provenance_capture_id, created_at, updated_at "
            f"FROM atomic_notes {where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        ))
        import json as _json
        for r in rows:
            try:
                r["entities_mentioned"] = _json.loads(r.get("entities_mentioned") or "[]")
            except (ValueError, TypeError):
                r["entities_mentioned"] = []
        return rows
    finally:
        conn.close()


@app.get("/merge-proposals", dependencies=[Depends(require_auth)])
def merge_proposals_list(status: str = "pending"):
    """List entity merge proposals filtered by status (SYN-39).

    Joins the two side entities + their fact previews so the client can render
    a side-by-side card without follow-up requests.
    """
    if status not in {"pending", "accepted", "rejected"}:
        raise HTTPException(status_code=400, detail="invalid status filter")
    conn = get_connection()
    try:
        rows = cursor_to_dicts(conn.execute(
            "SELECT p.id, p.candidate_entity_id, p.existing_entity_id, "
            "       p.similarity_score, p.similarity_reason, p.evidence_capture_id, "
            "       p.status, p.created_at, p.resolved_at, p.resolved_canonical_id, "
            "       ec.canonical_name AS candidate_name, ec.type AS candidate_type, "
            "       ec.mention_count   AS candidate_mention_count, "
            "       ee.canonical_name AS existing_name,  ee.type AS existing_type, "
            "       ee.mention_count   AS existing_mention_count "
            "FROM entity_merge_proposals p "
            "JOIN entities ec ON ec.id = p.candidate_entity_id "
            "JOIN entities ee ON ee.id = p.existing_entity_id "
            "WHERE p.status = ? "
            "ORDER BY p.created_at DESC",
            (status,),
        ))
        for r in rows:
            for side, eid in (("candidate", r["candidate_entity_id"]),
                              ("existing",  r["existing_entity_id"])):
                facts = cursor_to_dicts(conn.execute(
                    "SELECT predicate, value, confidence FROM facts "
                    "WHERE entity_id = ? ORDER BY confidence DESC LIMIT 5",
                    (eid,),
                ))
                r[f"{side}_facts"] = facts
        return rows
    finally:
        conn.close()


def _reroute_to_canonical(conn, absorbed_id: str, canonical_id: str) -> None:
    """Move every projection that points at `absorbed_id` over to `canonical_id`.

    Touches facts.entity_id, relations.entity_from / entity_to, and the JSON
    `entities_mentioned` array on atomic_notes (canonical_name swap, since the
    list stores names not ids — see SYN-42). Caller owns the transaction.
    """
    import json as _json
    absorbed = first_row(conn.execute(
        "SELECT canonical_name FROM entities WHERE id = ?", (absorbed_id,)
    ))
    canonical = first_row(conn.execute(
        "SELECT canonical_name FROM entities WHERE id = ?", (canonical_id,)
    ))
    if not absorbed or not canonical:
        raise HTTPException(status_code=404, detail="entity not found during reroute")

    conn.execute("UPDATE facts SET entity_id = ? WHERE entity_id = ?",
                 (canonical_id, absorbed_id))
    conn.execute("UPDATE relations SET entity_from = ? WHERE entity_from = ?",
                 (canonical_id, absorbed_id))
    conn.execute("UPDATE relations SET entity_to = ? WHERE entity_to = ?",
                 (canonical_id, absorbed_id))

    # atomic_notes.entities_mentioned stores canonical names, not ids — swap by name.
    notes = cursor_to_dicts(conn.execute(
        "SELECT id, entities_mentioned FROM atomic_notes "
        "WHERE entities_mentioned LIKE ?",
        (f'%"{absorbed["canonical_name"]}"%',),
    ))
    for n in notes:
        try:
            arr = _json.loads(n.get("entities_mentioned") or "[]")
        except (ValueError, TypeError):
            continue
        if absorbed["canonical_name"] not in arr:
            continue
        new_arr = [canonical["canonical_name"] if x == absorbed["canonical_name"] else x
                   for x in arr]
        # Dedup while keeping order
        seen = set()
        new_arr = [x for x in new_arr if not (x in seen or seen.add(x))]
        conn.execute(
            "UPDATE atomic_notes SET entities_mentioned = ? WHERE id = ?",
            (_json.dumps(new_arr, ensure_ascii=False), n["id"]),
        )


@app.post("/merge-proposals/{proposal_id}/accept", dependencies=[Depends(require_auth)])
def merge_proposal_accept(proposal_id: str, body: MergeAcceptIn):
    """Accept a merge proposal: reroute everything to the chosen canonical and
    soft-mark the absorbed entity (no DELETE, lineage preserved).
    """
    conn = get_connection()
    try:
        p = first_row(conn.execute(
            "SELECT id, candidate_entity_id, existing_entity_id, status "
            "FROM entity_merge_proposals WHERE id = ?",
            (proposal_id,),
        ))
        if not p:
            raise HTTPException(status_code=404, detail="proposal not found")
        if p["status"] != "pending":
            raise HTTPException(status_code=400, detail=f"proposal already {p['status']}")

        sides = {p["candidate_entity_id"], p["existing_entity_id"]}
        if body.canonical_id not in sides:
            raise HTTPException(status_code=400,
                                detail="canonical_id must be one of the proposal sides")
        absorbed_id = (sides - {body.canonical_id}).pop()

        canonical = first_row(conn.execute(
            "SELECT id, canonical_name, aliases FROM entities WHERE id = ?",
            (body.canonical_id,),
        ))
        absorbed = first_row(conn.execute(
            "SELECT id, canonical_name FROM entities WHERE id = ?",
            (absorbed_id,),
        ))
        if not canonical or not absorbed:
            raise HTTPException(status_code=404, detail="entity not found")

        import json as _json
        try:
            aliases = _json.loads(canonical.get("aliases") or "[]")
        except (ValueError, TypeError):
            aliases = []
        if absorbed["canonical_name"] not in aliases:
            aliases.append(absorbed["canonical_name"])

        with conn:
            _reroute_to_canonical(conn, absorbed_id, body.canonical_id)
            conn.execute(
                "UPDATE entities SET aliases = ? WHERE id = ?",
                (_json.dumps(aliases, ensure_ascii=False), body.canonical_id),
            )
            conn.execute(
                "UPDATE entities SET merged_into_id = ?, merged_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (body.canonical_id, absorbed_id),
            )
            conn.execute(
                "UPDATE entity_merge_proposals SET status = 'accepted', "
                "resolved_at = CURRENT_TIMESTAMP, resolved_canonical_id = ? "
                "WHERE id = ?",
                (body.canonical_id, proposal_id),
            )
        return {"status": "accepted", "proposal_id": proposal_id,
                "canonical_id": body.canonical_id, "absorbed_id": absorbed_id}
    finally:
        conn.close()


@app.post("/merge-proposals/{proposal_id}/reject", dependencies=[Depends(require_auth)])
def merge_proposal_reject(proposal_id: str):
    """Mark the proposal rejected (status terminal, won't be re-proposed)."""
    conn = get_connection()
    try:
        p = first_row(conn.execute(
            "SELECT id, status FROM entity_merge_proposals WHERE id = ?",
            (proposal_id,),
        ))
        if not p:
            raise HTTPException(status_code=404, detail="proposal not found")
        if p["status"] != "pending":
            raise HTTPException(status_code=400, detail=f"proposal already {p['status']}")
        with conn:
            conn.execute(
                "UPDATE entity_merge_proposals SET status = 'rejected', "
                "resolved_at = CURRENT_TIMESTAMP WHERE id = ?",
                (proposal_id,),
            )
        return {"status": "rejected", "proposal_id": proposal_id}
    finally:
        conn.close()


@app.get("/capture/{capture_id}", dependencies=[Depends(require_auth)])
def capture_detail(capture_id: int):
    """Return the raw inbox row by id. Powers the 'source' chip — anything
    elsewhere with a provenance_capture_id can resolve to a full capture
    payload here without leaking SQL details to the client."""
    conn = get_connection()
    try:
        row = first_row(conn.execute(
            "SELECT id, content, source, device_id, captured_at, created_at, "
            "       processed_at, status, client_id "
            "FROM inbox WHERE id = ?", (capture_id,)
        ))
        if not row:
            raise HTTPException(status_code=404, detail="capture not found")
        return row
    finally:
        conn.close()


@app.post("/project-entries/{entry_id}/move", dependencies=[Depends(require_auth)])
def move_project_entry(entry_id: str, body: ProjectEntryMoveIn):
    """Reassign a project_entry to a different project.

    SYN-45: the immutable capture (inbox row) is never touched — we only
    reassign the projection. Caller is expected to trigger a refinement
    later if the synthesis on either project needs updating.
    """
    conn = get_connection()
    try:
        entry = first_row(conn.execute(
            "SELECT id, project_id FROM project_entries WHERE id = ?", (entry_id,)
        ))
        if not entry:
            raise HTTPException(status_code=404, detail="project_entry not found")
        target = first_row(conn.execute(
            "SELECT id FROM entities WHERE id = ? AND type = 'project'",
            (body.project_id,),
        ))
        if not target:
            raise HTTPException(status_code=404, detail="target project not found")
        with conn:
            conn.execute(
                "UPDATE project_entries SET project_id = ? WHERE id = ?",
                (body.project_id, entry_id),
            )
        return {"status": "moved", "entry_id": entry_id,
                "from_project_id": entry["project_id"],
                "to_project_id": body.project_id}
    finally:
        conn.close()


@app.post("/project-entries/{entry_id}/attach-to-project", dependencies=[Depends(require_auth)])
def attach_entry_to_project(entry_id: str, body: ProjectEntryMoveIn):
    """Add a parallel rattachement to another project (additive, not move).

    SYN-55 + SYN-57: a capture can belong to N projects (the schema already
    allows multiple project_entries per capture_id, and the classifier emits
    project_entries in batch). This manual endpoint mirrors that capability
    for the UI: it takes an existing entry as a template and INSERTs a new
    row pointing to the target project, keeping the source row intact. The
    immutable inbox row is never touched.
    """
    import uuid as _uuid
    conn = get_connection()
    try:
        src = first_row(conn.execute(
            "SELECT id, project_id, capture_id, content, kind FROM project_entries WHERE id = ?",
            (entry_id,),
        ))
        if not src:
            raise HTTPException(status_code=404, detail="project_entry not found")
        if src["project_id"] == body.project_id:
            raise HTTPException(status_code=400, detail="already attached to this project")
        target = first_row(conn.execute(
            "SELECT id FROM entities WHERE id = ? AND type = 'project'",
            (body.project_id,),
        ))
        if not target:
            raise HTTPException(status_code=404, detail="target project not found")
        # Prevent duplicate parallel rattachements on the same (project, capture).
        dup = first_row(conn.execute(
            "SELECT id FROM project_entries WHERE project_id = ? AND capture_id = ?",
            (body.project_id, src["capture_id"]),
        ))
        if dup:
            raise HTTPException(status_code=409, detail="capture already attached to that project")
        new_id = str(_uuid.uuid4())
        with conn:
            conn.execute(
                "INSERT INTO project_entries (id, project_id, capture_id, content, kind) "
                "VALUES (?, ?, ?, ?, ?)",
                (new_id, body.project_id, src["capture_id"], src["content"], src["kind"]),
            )
        return {"status": "attached", "new_entry_id": new_id,
                "project_id": body.project_id, "capture_id": src["capture_id"]}
    finally:
        conn.close()


@app.post("/project-entries/{entry_id}/detach", dependencies=[Depends(require_auth)])
def detach_project_entry(entry_id: str):
    """Remove the project rattachement; the capture in inbox is preserved.

    SYN-45: the projection is destroyed but the source-of-truth capture stays
    in inbox. The user can re-route later from another correction endpoint,
    or capture again.
    """
    conn = get_connection()
    try:
        entry = first_row(conn.execute(
            "SELECT id, project_id, capture_id FROM project_entries WHERE id = ?",
            (entry_id,),
        ))
        if not entry:
            raise HTTPException(status_code=404, detail="project_entry not found")
        with conn:
            conn.execute("DELETE FROM project_entries WHERE id = ?", (entry_id,))
        return {"status": "detached", "entry_id": entry_id,
                "former_project_id": entry["project_id"],
                "capture_id": entry["capture_id"]}
    finally:
        conn.close()


@app.post("/project-entries/{entry_id}/reclassify-as-fact", dependencies=[Depends(require_auth)])
def reclassify_entry_as_fact(entry_id: str, body: ProjectEntryFactIn):
    """Turn a project_entry into an explicit fact on a target entity.

    SYN-45: the entry is removed (the projection), the capture stays in inbox,
    and a new fact is created with confidence=1.0 (the user vouches for it).
    Provenance points back to the capture that originally spawned the entry.
    """
    import uuid as _uuid
    conn = get_connection()
    try:
        entry = first_row(conn.execute(
            "SELECT id, capture_id FROM project_entries WHERE id = ?", (entry_id,)
        ))
        if not entry:
            raise HTTPException(status_code=404, detail="project_entry not found")
        ent = first_row(conn.execute(
            "SELECT id FROM entities WHERE id = ?", (body.entity_id,)
        ))
        if not ent:
            raise HTTPException(status_code=404, detail="target entity not found")
        with conn:
            fact_id = str(_uuid.uuid4())
            conn.execute(
                "INSERT INTO facts "
                "(id, entity_id, predicate, value, confidence, source_inbox_id, "
                " persistence_value, provenance_capture_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    fact_id, body.entity_id, body.predicate, body.value,
                    1.0, str(entry["capture_id"]),
                    body.persistence_value, entry["capture_id"],
                ),
            )
            conn.execute("DELETE FROM project_entries WHERE id = ?", (entry_id,))
        return {"status": "reclassified", "fact_id": fact_id,
                "entity_id": body.entity_id,
                "capture_id": entry["capture_id"]}
    finally:
        conn.close()


@app.get("/project/{project_id}/state", dependencies=[Depends(require_auth)])
def project_state(project_id: str):
    """Live synthesis of a project entity (SYN-43).

    Returns the current summary_md + metadata, plus a small recent-entries slice
    so the client can show the timeline without a second round-trip.
    """
    conn = get_connection()
    try:
        ent = first_row(conn.execute(
            "SELECT id, canonical_name, type FROM entities "
            "WHERE id = ? AND type = 'project'", (project_id,)
        ))
        if not ent:
            raise HTTPException(status_code=404, detail="project not found")
        state = first_row(conn.execute(
            "SELECT psv.id AS version_id, psv.summary_md, psv.entry_count, "
            "       psv.trigger, psv.kind, psv.created_at, ps.updated_at "
            "FROM project_state ps "
            "JOIN project_state_versions psv ON psv.id = ps.current_version_id "
            "WHERE ps.project_id = ?", (project_id,)
        ))
        entries = cursor_to_dicts(conn.execute(
            "SELECT id, content, kind, capture_id, created_at "
            "FROM project_entries WHERE project_id = ? "
            "ORDER BY created_at DESC LIMIT 50", (project_id,)
        ))
        total_entries = conn.execute(
            "SELECT COUNT(*) FROM project_entries WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
        return {
            "project_id": ent["id"],
            "canonical_name": ent["canonical_name"],
            "current_state": state,            # may be None if no entry yet
            "entries_recent": entries,
            "entries_total": total_entries,
        }
    finally:
        conn.close()


@app.patch("/entity/{entity_id}", dependencies=[Depends(require_auth)])
def update_entity(entity_id: str, body: EntityUpdate):
    """Update an entity's type. Closed enum (see EntityType)."""
    conn = get_connection()
    try:
        e = first_row(conn.execute("SELECT id FROM entities WHERE id=?", (entity_id,)))
        if not e:
            raise HTTPException(status_code=404, detail="entity not found")
        with conn:
            conn.execute("UPDATE entities SET type=? WHERE id=?", (body.type, entity_id))
        return {"id": entity_id, "type": body.type}
    finally:
        conn.close()


@app.get("/pending", dependencies=[Depends(require_auth)])
def pending():
    """Pending facts as validatable cards: question + source quote + confidence."""
    import json as _json
    conn = get_connection()
    try:
        out = []
        for item in cursor_to_dicts(conn.execute(
            "SELECT id, fact_data, created_at FROM pending_facts ORDER BY created_at DESC"
        )):
            try:
                fd = _json.loads(item["fact_data"])
            except (ValueError, TypeError):
                continue
            source_text = None
            src_id = fd.get("source_inbox_id")
            if src_id is not None:
                src = first_row(conn.execute(
                    "SELECT content FROM inbox WHERE id=?", (src_id,)
                ))
                source_text = src["content"] if src else None
            entity = fd.get("entity_canonical", "")
            out.append({
                "id": item["id"],
                "entity": entity,
                "predicate": fd.get("predicate"),
                "value": fd.get("value"),
                "confidence": fd.get("confidence"),
                "question": f"{entity} — {fd.get('predicate')} : {fd.get('value')} ?",
                "source_text": source_text,
                "created_at": item["created_at"],
            })
        return out
    finally:
        conn.close()


@app.post("/pending/{fact_id}/validate", dependencies=[Depends(require_auth)])
def validate(fact_id: str, body: ValidateIn):
    from dream_cycle.validation import record_and_apply_validation
    conn = get_connection()
    try:
        with conn:
            result = record_and_apply_validation(
                conn, fact_id, body.confirmed, body.correction, body.device_id
            )
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    finally:
        conn.close()


@app.post("/dream-cycle/run", dependencies=[Depends(require_auth)])
def dream_cycle_run(trigger: str = "manual"):
    """Run the cycle now (manual/testing). Guarded by a single-instance lock."""
    run_id = str(uuid.uuid4())
    with cycle_lock():
        conn = get_connection()
        try:
            processed_before = conn.execute(
                "SELECT COUNT(*) FROM inbox WHERE processed_at IS NOT NULL"
            ).fetchone()[0]
            with conn:
                conn.execute(
                    "INSERT INTO cycle_runs (id, trigger, status) VALUES (?,?, 'running')",
                    (run_id, trigger),
                )
        finally:
            conn.close()

        from dream_cycle import run_dream_cycle
        buf = io.StringIO()
        now = datetime.now(timezone.utc).isoformat()
        try:
            with contextlib.redirect_stdout(buf):
                run_dream_cycle()
        except EnvironmentError as e:
            _finish_run(run_id, status="error", error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # noqa: BLE001
            _finish_run(run_id, status="error", error=f"{type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

        conn = get_connection()
        try:
            processed_after = conn.execute(
                "SELECT COUNT(*) FROM inbox WHERE processed_at IS NOT NULL"
            ).fetchone()[0]
            entities_total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            pending_total = conn.execute("SELECT COUNT(*) FROM pending_facts").fetchone()[0]
        finally:
            conn.close()

        _finish_run(
            run_id, status="ok",
            notes_processed=processed_after - processed_before,
            entities_total=entities_total, pending_total=pending_total, finished_at=now,
        )
        return _last_run()


def _finish_run(run_id, status, error=None, notes_processed=0,
                entities_total=0, pending_total=0, finished_at=None):
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE cycle_runs SET status=?, error=?, notes_processed=?, "
                "entities_total=?, pending_total=?, finished_at=? WHERE id=?",
                (status, error, notes_processed, entities_total, pending_total,
                 finished_at or datetime.now(timezone.utc).isoformat(), run_id),
            )
    finally:
        conn.close()


def _last_run():
    conn = get_connection()
    try:
        return first_row(conn.execute(
            "SELECT * FROM cycle_runs ORDER BY started_at DESC LIMIT 1"
        ))
    finally:
        conn.close()


@app.get("/dream-cycle/last", dependencies=[Depends(require_auth)])
def dream_cycle_last():
    return _last_run() or {"status": "never_run"}


@app.get("/changes", dependencies=[Depends(require_auth)])
def changes(since: str | None = None):
    """
    Pull replication: returns the derived state for the read replicas.

    Entities are returned in full (they mutate); notes/facts can be filtered by
    `since` (created_at). The graph is small, so a full pull is fine for v1.
    `cursor` is the timestamp to pass as `since` next time.
    """
    conn = get_connection()
    try:
        entities = cursor_to_dicts(conn.execute("SELECT * FROM entities"))
        relations = cursor_to_dicts(conn.execute("SELECT * FROM relations"))
        if since:
            facts = cursor_to_dicts(conn.execute(
                "SELECT * FROM facts WHERE created_at > ?", (since,)))
            notes = cursor_to_dicts(conn.execute(
                "SELECT * FROM atomic_notes WHERE created_at > ?", (since,)))
        else:
            facts = cursor_to_dicts(conn.execute("SELECT * FROM facts"))
            notes = cursor_to_dicts(conn.execute("SELECT * FROM atomic_notes"))
        return {
            "entities": entities, "facts": facts,
            "relations": relations, "atomic_notes": notes,
            "cursor": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        conn.close()
