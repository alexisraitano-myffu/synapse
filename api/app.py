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
import json
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
from config import BASE_DIR, CLAUDE_MODEL
from db import cursor_to_dicts, first_row, get_connection, init_db
from embeddings import embed_text
from entity_search import entity_embedding_text, search_entities_by_vector
from facts_store import insert_fact

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
                # SYN-89: user fact edits flag summaries stale — they warrant a
                # (cheap) cycle run even with an empty inbox.
                stale = conn.execute(
                    "SELECT COUNT(*) FROM entities "
                    "WHERE summary_stale = 1 AND merged_into_id IS NULL"
                ).fetchone()[0]
            finally:
                conn.close()
            if queued == 0 and stale == 0:
                continue
            if queued > 0 and time.time() - _last_capture_ts < _debounce_seconds():
                continue
            try:
                dream_cycle_run(trigger="auto")  # guarded by the lock; errors land in cycle_runs
            except Exception:
                pass  # retry next tick
        except Exception:
            pass


def _recover_interrupted_runs() -> None:
    """SYN-77 — a run left 'running' with no live cycle was killed mid-cycle
    (process death / machine shutdown). Surface it as an error instead of
    showing a phantom 'running' forever; its unprocessed entries stay queued
    and the auto-cycle catch-up picks them up. A fresh lock file means a cycle
    may genuinely be running in another process — leave it alone."""
    if _LOCK_PATH.exists() and time.time() - _LOCK_PATH.stat().st_mtime <= _LOCK_STALE_SECONDS:
        return
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE cycle_runs SET status='error', "
                "error='interrupted (process died mid-run)', finished_at=? "
                "WHERE status='running'",
                (datetime.now(timezone.utc).isoformat(),),
            )
    finally:
        conn.close()


@contextlib.asynccontextmanager
async def lifespan(_app):
    _recover_interrupted_runs()
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
    # SYN-82 — user edits on the fiche: both optional, at least one required.
    type: EntityType | None = None
    canonical_name: str | None = None


class FactUpdate(BaseModel):
    # SYN-82 — user correction of a fact (predicate and/or value).
    predicate: str | None = None
    value: str | None = None


class RelationCreate(BaseModel):
    # SYN-84 — manual relation between two EXISTING entities. Optional client id
    # (offline action log) so replica and master agree on the row identity.
    id: str | None = None
    entity_from: str
    predicate: str
    entity_to: str


class RelationUpdate(BaseModel):
    # SYN-84 — user correction of a relation's predicate.
    predicate: str


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


class TypeProposalAcceptIn(BaseModel):
    type: str | None = None  # optional rename of the proposed type before adding it


class NotePromoteIn(BaseModel):
    canonical_name: str | None = None  # optional project name (else derived from the note)


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
        return {"status": "ok", "entities": entities, "pending": pending,
                "inbox_queued": queued, "instance_id": config_store.get_instance_id()}
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
            "SELECT id, client_id, content, source, created_at, captured_at, processed_at, "
            "status, error "
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


@app.post("/inbox/{entry_id}/requeue", dependencies=[Depends(require_auth)])
def inbox_requeue(entry_id: int):
    """Put a failed entry back in the queue (SYN-77 — user-driven retry)."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE inbox SET status='queued', processed_at=NULL, error=NULL "
                "WHERE id=? AND status='failed'",
                (entry_id,),
            )
            requeued = conn.execute("SELECT changes()").fetchone()[0] == 1
        if not requeued:
            raise HTTPException(status_code=404, detail="no failed inbox entry with this id")
        return {"id": entry_id, "status": "queued"}
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


def _anthropic_client_factory():
    """Build an Anthropic client from the configured key, or None if unset — the
    cluster labeller (SYN-70) then falls back to generic labels. Reuses the same
    key path and model as the Dream Cycle (BYOK)."""
    import anthropic

    from config_store import get_anthropic_key
    key = get_anthropic_key()
    return anthropic.Anthropic(api_key=key) if key else None


def _assign_communities(nodes: list[dict], edges: list[dict]) -> None:
    """Tag every node dict with a `community_id` via Louvain community detection
    (SYN-66/68). Best-effort: if networkx is unavailable the nodes keep
    community_id=None rather than failing the request. Deterministic (fixed seed)
    so the same graph yields the same colouring across calls."""
    if not nodes:
        return
    try:
        import networkx as nx
    except ImportError:
        return
    ids = {n["id"] for n in nodes}
    g = nx.Graph()
    g.add_nodes_from(ids)
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a in ids and b in ids and a != b:
            w = float(e.get("confidence") or 1.0)
            if g.has_edge(a, b):
                g[a][b]["weight"] += w
            else:
                g.add_edge(a, b, weight=w)
    try:
        communities = nx.community.louvain_communities(g, weight="weight", seed=42)
    except Exception:
        return  # never let clustering break the endpoint
    cid = {nid: i for i, comm in enumerate(communities) for nid in comm}
    for n in nodes:
        n["community_id"] = cid.get(n["id"])


# ── Anti-hairball filters (SYN-71) ────────────────────────────────────────────

def _prune(nodes: list[dict], edges: list[dict], keep: set) -> tuple[list, list]:
    """Keep only `keep` nodes and the edges whose both endpoints survive."""
    return ([n for n in nodes if n["id"] in keep],
            [e for e in edges if e["from"] in keep and e["to"] in keep])


def _set_degree(nodes: list[dict], edges: list[dict]) -> None:
    """(Re)compute node degree from the current edge set — drives node size on
    the map (SYN-64: size ~ memory_strength × degree)."""
    deg: dict = {}
    for e in edges:
        deg[e["from"]] = deg.get(e["from"], 0) + 1
        deg[e["to"]] = deg.get(e["to"], 0) + 1
    for n in nodes:
        n["degree"] = deg.get(n["id"], 0)


def _node_score(n: dict) -> float:
    """Composite salience for ranking: liveness × connectivity. A node kept alive
    by recent mentions and well connected outranks a stale or isolated one."""
    return (n.get("memory_strength") or 0.0) * (n.get("degree", 0) + 1)


def _top_pct_per_cluster(nodes: list[dict], edges: list[dict], pct: float) -> tuple[list, list]:
    """Within each community keep the top `pct` nodes by salience (always ≥1), so
    a dense cluster is summarised rather than dumped wholesale."""
    import math as _math
    by_comm: dict = {}
    for n in nodes:
        by_comm.setdefault(n.get("community_id"), []).append(n)
    keep: set = set()
    for members in by_comm.values():
        members.sort(key=_node_score, reverse=True)
        k = max(1, _math.ceil(len(members) * pct))
        keep.update(n["id"] for n in members[:k])
    return _prune(nodes, edges, keep)


@app.get("/graph", dependencies=[Depends(require_auth)])
def graph(entity: str | None = None, mode: str = "full", include_archived: bool = False,
          include_notes: bool = False, cluster: bool = False,
          layout: bool = False, relayout: bool = False,
          node_types: str = "both", memory_strength_min: float | None = None,
          since: str | None = None, top_pct_per_cluster: float | None = None,
          include_isolated: bool = True, max_nodes: int = 1000,
          clusters: bool = False, semantic_layout: bool = True):
    """Nodes = entities (size ~ mention_count), edges = relations.
    `include_archived=true` also returns user-archived entities (SYN-59).

    Living-map options (SYN-66/68/69), all default off so the legacy entity-list /
    ego consumers keep the original shape:
    - `include_notes=true` adds atomic_notes as a second node kind (id `n:<id>`,
      kind `atomic_note`) plus `mentions` edges note→entity (resolved from each
      note's entities_mentioned).
    - `cluster=true` runs community detection and tags every node with a
      `community_id` (entities and notes share the colouring).
    - `layout=true` attaches persisted `x`/`y` map positions (ForceAtlas2);
      missing nodes are placed incrementally near their cluster. `relayout=true`
      forces a full recompute of the whole map. layout implies clustering (the
      community id drives incremental placement).

    Anti-hairball filters (SYN-71), composable — the UI tightens them by default
    and reveals more on demand:
    - `node_types`: `entities` | `atomic_notes` | `both` (default both).
    - `memory_strength_min`: drop nodes below this liveness.
    - `since`: ISO date — keep only nodes active since then (by last_mentioned).
    - `top_pct_per_cluster`: keep the top fraction (e.g. 0.2) per community.
    - `include_isolated`: keep degree-0 nodes (default true).
    - `max_nodes`: hard ceiling (default 1000) — the densest-by-salience survive,
      so the endpoint never returns an unbounded hairball.

    `clusters=true` (SYN-70) adds a top-level `clusters: [{community_id, label,
    size, hull}]` — a short Haiku label per community (cached) and a convex hull
    around its node positions. Implies clustering + layout."""
    import json
    conn = get_connection()
    try:
        nodes = []
        # SYN-39: hide soft-merged rows; their data already lives on the canonical one.
        # SYN-58: hide non-active rows (pending type-validation / archived).
        # SYN-59: hide user-archived entities unless include_archived; facts_count
        # counts only active facts (not archived / obsolete).
        archived_clause = "" if include_archived else " AND e.archived_at IS NULL"
        for e in cursor_to_dicts(conn.execute(
            "SELECT e.id, e.canonical_name, e.type, e.mention_count, e.persistence_value, "
            "       e.summary, e.last_mentioned, e.archived_at, e.memory_strength, "
            "       (SELECT COUNT(*) FROM facts f WHERE f.entity_id = e.id "
            "          AND f.archived_at IS NULL AND f.obsoleted_at IS NULL) AS facts_count "
            "FROM entities e WHERE e.merged_into_id IS NULL AND e.status = 'active'"
            + archived_clause
        )):
            nodes.append({
                "id": e["id"],
                "kind": "entity",
                "label": e["canonical_name"],
                "type": e.get("type"),
                "mention_count": e.get("mention_count", 1),
                "persistence_value": e.get("persistence_value", 3),
                "summary": e.get("summary"),
                "last_mentioned": e.get("last_mentioned"),
                "facts_count": e.get("facts_count", 0),
                "memory_strength": e.get("memory_strength"),  # SYN-68 (was reserved)
                "archived_at": e.get("archived_at"),  # SYN-59
                "community_id": None,
            })
        edges = [
            {"from": r["entity_from"], "to": r["entity_to"],
             "label": r["predicate"], "confidence": r["confidence"]}
            for r in cursor_to_dicts(conn.execute(
                "SELECT entity_from, entity_to, predicate, confidence FROM relations"
            ))
        ]

        if include_notes:
            # entities_mentioned stores canonical names, not ids → resolve to the
            # entity nodes already in the graph so a note links to real targets.
            name_to_id = {n["label"].lower(): n["id"] for n in nodes}
            present = {n["id"] for n in nodes}
            for n in cursor_to_dicts(conn.execute(
                "SELECT id, title, summary, content, memory_strength, "
                "       last_reactivated_at, created_at, entities_mentioned "
                # Digests are global weekly summaries (mention many entities) — they'd
                # hairball the map; archived notes are hidden everywhere else too.
                "FROM atomic_notes WHERE archived_at IS NULL AND kind != 'digest'"
            )):
                nid = f"n:{n['id']}"
                preview = n.get("title") or n.get("summary") or (n.get("content") or "")
                nodes.append({
                    "id": nid,
                    "kind": "atomic_note",
                    "label": (preview or "")[:60],
                    "type": "atomic_note",
                    "summary": n.get("summary"),
                    "last_mentioned": n.get("last_reactivated_at") or n.get("created_at"),
                    "memory_strength": n.get("memory_strength"),
                    "community_id": None,
                })
                try:
                    mentioned = json.loads(n.get("entities_mentioned") or "[]")
                except (ValueError, TypeError):
                    mentioned = []
                for name in mentioned:
                    eid = name_to_id.get(str(name).lower())
                    if eid in present:
                        edges.append({"from": nid, "to": eid,
                                      "label": "mentions", "confidence": 1.0})

        # ── Anti-hairball filters (SYN-71) ───────────────────────────────────
        # Cheap value filters first (kind / liveness / recency), then prune edges.
        if node_types in ("entities", "entity"):
            nodes = [n for n in nodes if n["kind"] == "entity"]
        elif node_types in ("atomic_notes", "atomic_note", "notes"):
            nodes = [n for n in nodes if n["kind"] == "atomic_note"]
        if memory_strength_min is not None:
            nodes = [n for n in nodes if (n.get("memory_strength") or 0.0) >= memory_strength_min]
        if since:
            nodes = [n for n in nodes if (n.get("last_mentioned") or "") >= since]
        nodes, edges = _prune(nodes, edges, {n["id"] for n in nodes})

        if entity and mode == "ego":
            nodes, edges = _ego_filter(nodes, edges, entity)

        _set_degree(nodes, edges)
        if not include_isolated:
            nodes, edges = _prune(nodes, edges, {n["id"] for n in nodes if n["degree"] > 0})

        # top_pct needs community_id; layout/clusters need it too.
        if cluster or layout or clusters or top_pct_per_cluster is not None:
            _assign_communities(nodes, edges)
        if top_pct_per_cluster is not None:
            nodes, edges = _top_pct_per_cluster(nodes, edges, top_pct_per_cluster)
        if max_nodes and len(nodes) > max_nodes:
            keep = {n["id"] for n in sorted(nodes, key=_node_score, reverse=True)[:max_nodes]}
            nodes, edges = _prune(nodes, edges, keep)
        _set_degree(nodes, edges)  # final degree after structural pruning

        if layout or clusters:  # clusters need positions for their hulls
            from graph_layout import ensure_positions
            positions = ensure_positions(conn, nodes, edges, full=relayout, semantic=semantic_layout)
            for n in nodes:
                xy = positions.get(n["id"])
                if xy:
                    n["x"], n["y"] = xy["x"], xy["y"]

        result = {"nodes": nodes, "edges": edges}
        if clusters:
            from graph_clusters import build_clusters
            result["clusters"] = build_clusters(
                conn, nodes, client_factory=_anthropic_client_factory, model=CLAUDE_MODEL)
        return result
    finally:
        conn.close()


@app.get("/entity/{entity_id}", dependencies=[Depends(require_auth)])
def entity_detail(entity_id: str, include: str | None = None):
    """`include` (CSV) opts archived/obsolete facts back into the response —
    e.g. `?include=archived,obsolete` for the history/archive sections."""
    import json as _json
    inc = {p.strip() for p in (include or "").split(",") if p.strip()}
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
        # SYN-59: hide archived/obsolete facts by default; opt back in via ?include.
        fact_filter = ""
        if "obsolete" not in inc:
            fact_filter += " AND obsoleted_at IS NULL"
        if "archived" not in inc:
            fact_filter += " AND archived_at IS NULL"
        facts = cursor_to_dicts(conn.execute(
            "SELECT id, predicate, value, confidence, persistence_value, created_at, "
            "       provenance_capture_id, archived_at, obsoleted_at, obsoleted_by, category "
            "FROM facts WHERE entity_id=?" + fact_filter + " ORDER BY confidence DESC",
            (entity_id,),
        ))
        relations = cursor_to_dicts(conn.execute(
            "SELECT r.id, r.predicate, e.canonical_name AS entity_to, r.confidence, "
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
            "status": e.get("status"),               # SYN-58
            "archived_at": e.get("archived_at"),     # SYN-59
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
            "  AND e.status = 'active' "       # SYN-58: hide pending/archived-status
            "  AND e.archived_at IS NULL "      # SYN-59: hide user-archived projects
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
    kind: str | None = None,
):
    """List atomic_notes for the Notes view (SYN-52).

    Filters are AND-combined and best-effort:
    - q: substring match on title or content (case-insensitive)
    - entity: matches any note whose entities_mentioned JSON array contains
      the canonical name (LIKE on the serialized list — cheap, no JSON1)
    - kind: note | task | event (SYN-85)
    User-archived notes are hidden (SYN-85 "rendre obsolète").
    """
    limit = min(max(1, limit), 200)
    conn = get_connection()
    try:
        clauses = ["archived_at IS NULL"]
        params: list = []
        if q:
            clauses.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)")
            needle = f"%{q.lower()}%"
            params.extend([needle, needle])
        if entity:
            clauses.append("entities_mentioned LIKE ?")
            params.append(f'%"{entity}"%')
        if kind:
            if kind not in ("note", "task", "event"):
                raise HTTPException(status_code=400, detail="invalid kind filter")
            clauses.append("kind = ?")
            params.append(kind)
        where = "WHERE " + " AND ".join(clauses)
        params.append(limit)
        rows = cursor_to_dicts(conn.execute(
            f"SELECT id, title, content, summary, entities_mentioned, memory_strength, "
            f"       provenance_capture_id, created_at, updated_at, "
            f"       kind, event_date, event_recurring, archived_at "
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


@app.get("/atomic-note/{note_id}", dependencies=[Depends(require_auth)])
def atomic_note_detail(note_id: int):
    """A single atomic_note for the map / notes detail (SYN-64).

    Same shape as the list rows, plus the source capture's content when the
    provenance link resolves — so the client can show where the note came from.
    """
    conn = get_connection()
    try:
        rows = cursor_to_dicts(conn.execute(
            "SELECT id, title, content, summary, entities_mentioned, memory_strength, "
            "       provenance_capture_id, created_at, updated_at, "
            "       kind, event_date, event_recurring, archived_at "
            "FROM atomic_notes WHERE id = ?",
            (note_id,),
        ))
        if not rows:
            raise HTTPException(status_code=404, detail="note not found")
        r = rows[0]
        import json as _json
        try:
            r["entities_mentioned"] = _json.loads(r.get("entities_mentioned") or "[]")
        except (ValueError, TypeError):
            r["entities_mentioned"] = []
        if r.get("provenance_capture_id"):
            cap = cursor_to_dicts(conn.execute(
                "SELECT content FROM inbox WHERE id = ?", (r["provenance_capture_id"],)))
            r["provenance_content"] = cap[0]["content"] if cap else None
        return r
    finally:
        conn.close()


@app.post("/atomic-note/{note_id}/archive", dependencies=[Depends(require_auth)])
def atomic_note_archive(note_id: int):
    """SYN-85 — user gesture « rendre obsolète » : hide a note (task done /
    event passé / pensée périmée) without deleting it (reversible)."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE atomic_notes SET archived_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), note_id),
            )
            changed = conn.execute("SELECT changes()").fetchone()[0] == 1
        if not changed:
            raise HTTPException(status_code=404, detail="note not found")
        return {"id": note_id, "archived": True}
    finally:
        conn.close()


@app.post("/atomic-note/{note_id}/unarchive", dependencies=[Depends(require_auth)])
def atomic_note_unarchive(note_id: int):
    conn = get_connection()
    try:
        with conn:
            conn.execute("UPDATE atomic_notes SET archived_at=NULL WHERE id=?", (note_id,))
            changed = conn.execute("SELECT changes()").fetchone()[0] == 1
        if not changed:
            raise HTTPException(status_code=404, detail="note not found")
        return {"id": note_id, "archived": False}
    finally:
        conn.close()


@app.post("/atomic-note/{note_id}/promote-to-project", dependencies=[Depends(require_auth)])
def atomic_note_promote_to_project(note_id: int, body: NotePromoteIn):
    """Point 1 (D) — promote a mis-classified task/note into a project: create (or
    reuse) the project entity, seed it with a project_entry built from the note,
    then archive the note (reversible). Best handled upstream by the classifier
    (SYN — projet vs tâche), but this rescues a task after the fact."""
    from dream_cycle.cycle import _persist_project_entry
    conn = get_connection()
    try:
        note = first_row(conn.execute(
            "SELECT id, title, content, provenance_capture_id FROM atomic_notes WHERE id = ?",
            (note_id,)))
        if not note:
            raise HTTPException(status_code=404, detail="note not found")
        capture_id = note.get("provenance_capture_id")
        if not capture_id:
            raise HTTPException(
                status_code=400,
                detail="note sans capture source — promotion impossible")
        canonical = (body.canonical_name or note.get("title") or note["content"]).strip()[:80]
        if not canonical:
            raise HTTPException(status_code=400, detail="nom de projet vide")
        client = _anthropic_client_factory()
        with conn:
            project_id, entry_id = _persist_project_entry(
                project_canonical=canonical,
                content=note["content"],
                capture_id=capture_id,
                conn=conn,
                is_new_project=True,
                client=client,
            )
            conn.execute(
                "UPDATE atomic_notes SET archived_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), note_id))
        return {"status": "promoted", "note_id": note_id, "project_id": project_id,
                "entry_id": entry_id, "project_canonical": canonical}
    finally:
        conn.close()


@app.post("/atomic-note/{note_id}/reinforce", dependencies=[Depends(require_auth)])
def atomic_note_reinforce(note_id: int):
    """SYN-23 (digest) — user gesture 👍 « garder ça » on a fading note: full
    reactivation. Moves last_reactivated_at to now so Ebbinghaus springs the
    memory_strength back up; sets it to 1.0 immediately for the UI."""
    now = datetime.now(timezone.utc)
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE atomic_notes SET last_reactivated_at=?, memory_strength=1.0 WHERE id=?",
                (now.strftime("%Y-%m-%d %H:%M:%S"), note_id),
            )
            changed = conn.execute("SELECT changes()").fetchone()[0] == 1
        if not changed:
            raise HTTPException(status_code=404, detail="note not found")
        return {"id": note_id, "memory_strength": 1.0}
    finally:
        conn.close()


@app.post("/atomic-note/{note_id}/date", dependencies=[Depends(require_auth)])
def atomic_note_set_date(note_id: int, event_date: str | None = None, recurring: bool = False):
    """SYN-23 (dated tasks) — set (or clear) a note's date. Lets a task carry an
    `event_date` without becoming an event, so it surfaces in « À venir » like an
    event. `event_date=null` clears it. Absolute date (YYYY-MM-DD) expected."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE atomic_notes SET event_date=?, event_recurring=? WHERE id=?",
                (event_date or None, 1 if (event_date and recurring) else 0, note_id),
            )
            changed = conn.execute("SELECT changes()").fetchone()[0] == 1
        if not changed:
            raise HTTPException(status_code=404, detail="note not found")
        return {"id": note_id, "event_date": event_date or None, "event_recurring": bool(event_date and recurring)}
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


# ── Entity-type proposals (SYN-58) ────────────────────────────────────────────

@app.get("/entity-type-proposals", dependencies=[Depends(require_auth)])
def type_proposals_list(status: str = "pending"):
    """List entity-type proposals filtered by status (SYN-58).

    Joins the candidate entity + the evidence capture so the client can render
    the card ("create type `recipe` for entity X?") without follow-up requests.
    """
    if status not in {"pending", "accepted", "rejected"}:
        raise HTTPException(status_code=400, detail="invalid status filter")
    conn = get_connection()
    try:
        return cursor_to_dicts(conn.execute(
            "SELECT p.id, p.proposed_type, p.reason, p.status, p.created_at, "
            "       p.resolved_at, p.candidate_entity_id, p.evidence_capture_id, "
            "       e.canonical_name AS candidate_name, e.type AS candidate_type, "
            "       e.summary        AS candidate_summary, e.status AS candidate_status, "
            "       i.content        AS evidence_content "
            "FROM entity_type_proposals p "
            "LEFT JOIN entities e ON e.id = p.candidate_entity_id "
            "LEFT JOIN inbox i    ON i.id = p.evidence_capture_id "
            "WHERE p.status = ? "
            "ORDER BY p.created_at DESC",
            (status,),
        ))
    finally:
        conn.close()


@app.post("/entity-type-proposals/{proposal_id}/accept", dependencies=[Depends(require_auth)])
def type_proposal_accept(proposal_id: str, body: TypeProposalAcceptIn):
    """Accept a type proposal: extend the vocab, promote the candidate entity.

    The user may rename the proposed type via `body.type`. Adds it to
    `active_entity_types` (source='user'), sets the candidate entity's type and
    flips it to status='active', then marks the proposal accepted. Idempotent on
    the vocab insert so accepting twice can't duplicate the type.
    """
    conn = get_connection()
    try:
        p = first_row(conn.execute(
            "SELECT id, status, proposed_type, candidate_entity_id "
            "FROM entity_type_proposals WHERE id = ?", (proposal_id,),
        ))
        if not p:
            raise HTTPException(status_code=404, detail="proposal not found")
        if p["status"] != "pending":
            raise HTTPException(status_code=400, detail=f"proposal already {p['status']}")
        new_type = (body.type or p["proposed_type"] or "").strip()
        if not new_type:
            raise HTTPException(status_code=400, detail="empty type")
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO active_entity_types (type, source) VALUES (?, 'user')",
                (new_type,),
            )
            if p["candidate_entity_id"]:
                conn.execute(
                    "UPDATE entities SET type = ?, status = 'active' WHERE id = ?",
                    (new_type, p["candidate_entity_id"]),
                )
            conn.execute(
                "UPDATE entity_type_proposals SET status = 'accepted', "
                "resolved_at = CURRENT_TIMESTAMP WHERE id = ?", (proposal_id,),
            )
        return {"status": "accepted", "proposal_id": proposal_id, "type": new_type,
                "entity_id": p["candidate_entity_id"]}
    finally:
        conn.close()


@app.post("/entity-type-proposals/{proposal_id}/reject", dependencies=[Depends(require_auth)])
def type_proposal_reject(proposal_id: str):
    """Reject a type proposal: archive the candidate entity (drops out of the
    default views) and mark the proposal rejected."""
    conn = get_connection()
    try:
        p = first_row(conn.execute(
            "SELECT id, status, candidate_entity_id "
            "FROM entity_type_proposals WHERE id = ?", (proposal_id,),
        ))
        if not p:
            raise HTTPException(status_code=404, detail="proposal not found")
        if p["status"] != "pending":
            raise HTTPException(status_code=400, detail=f"proposal already {p['status']}")
        with conn:
            if p["candidate_entity_id"]:
                conn.execute(
                    "UPDATE entities SET status = 'archived' WHERE id = ?",
                    (p["candidate_entity_id"],),
                )
            conn.execute(
                "UPDATE entity_type_proposals SET status = 'rejected', "
                "resolved_at = CURRENT_TIMESTAMP WHERE id = ?", (proposal_id,),
            )
        return {"status": "rejected", "proposal_id": proposal_id}
    finally:
        conn.close()


# ── Lifecycle: archive / obsolete (SYN-59) ────────────────────────────────────

def _set_timestamp(table: str, row_id: str, columns: dict, label: str):
    """Set/clear lifecycle timestamp columns on one entity/fact row. `columns`
    maps column→'now'|None. 404 if the row doesn't exist."""
    conn = get_connection()
    try:
        if not first_row(conn.execute(f"SELECT id FROM {table} WHERE id = ?", (row_id,))):
            raise HTTPException(status_code=404, detail=f"{table[:-1]} not found")
        sets = ", ".join(
            f"{c} = CURRENT_TIMESTAMP" if v == "now" else f"{c} = NULL"
            for c, v in columns.items()
        )
        with conn:
            conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", (row_id,))
            if table == "facts":
                # SYN-89: a fact lifecycle change invalidates the derived summary.
                conn.execute(
                    "UPDATE entities SET summary_stale = 1 "
                    "WHERE id = (SELECT entity_id FROM facts WHERE id = ?)",
                    (row_id,),
                )
        return {"status": label, "id": row_id}
    finally:
        conn.close()


@app.post("/entity/{entity_id}/archive", dependencies=[Depends(require_auth)])
def entity_archive(entity_id: str):
    return _set_timestamp("entities", entity_id, {"archived_at": "now"}, "archived")


@app.post("/entity/{entity_id}/unarchive", dependencies=[Depends(require_auth)])
def entity_unarchive(entity_id: str):
    return _set_timestamp("entities", entity_id, {"archived_at": None}, "unarchived")


@app.post("/fact/{fact_id}/archive", dependencies=[Depends(require_auth)])
def fact_archive(fact_id: str):
    return _set_timestamp("facts", fact_id, {"archived_at": "now"}, "archived")


@app.post("/fact/{fact_id}/unarchive", dependencies=[Depends(require_auth)])
def fact_unarchive(fact_id: str):
    return _set_timestamp("facts", fact_id, {"archived_at": None}, "unarchived")


@app.post("/fact/{fact_id}/obsolete", dependencies=[Depends(require_auth)])
def fact_obsolete(fact_id: str):
    """Manual obsolescence (no replacement): obsoleted_by stays NULL — that's
    SYN-37's job when a newer fact supersedes."""
    return _set_timestamp("facts", fact_id, {"obsoleted_at": "now"}, "obsoleted")


@app.post("/fact/{fact_id}/restore", dependencies=[Depends(require_auth)])
def fact_restore(fact_id: str):
    """Resurrect an obsolete fact — clears both obsoleted_at and obsoleted_by."""
    return _set_timestamp("facts", fact_id,
                          {"obsoleted_at": None, "obsoleted_by": None}, "restored")


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


@app.get("/capture/{capture_id}/generated", dependencies=[Depends(require_auth)])
def capture_generated(capture_id: int):
    """SYN-92 — the reverse provenance index: what the Dream Cycle produced from
    one capture. Every derived table carries provenance_capture_id (the inbox row
    that first created the entity / fact / relation / note), so this is a uniform
    fan-out. Powers the app's "ce qui en est sorti" panel under a journal line.

    Entities are listed only when this capture *created* them (a capture that merely
    re-mentions an existing entity won't appear) — which is exactly "ce qui a été créé".
    Facts/relations carry resolved entity names so the client can render readable lines."""
    conn = get_connection()
    try:
        entities = cursor_to_dicts(conn.execute(
            "SELECT id, canonical_name, type FROM entities "
            "WHERE provenance_capture_id = ? AND merged_into_id IS NULL "
            "ORDER BY created_at", (capture_id,)
        ))
        facts = cursor_to_dicts(conn.execute(
            "SELECT f.id, f.predicate, f.value, f.entity_id, f.confidence, f.category, "
            "       e.canonical_name AS entity_name, f.archived_at, f.obsoleted_at "
            "FROM facts f LEFT JOIN entities e ON e.id = f.entity_id "
            "WHERE f.provenance_capture_id = ? ORDER BY f.created_at", (capture_id,)
        ))
        relations = cursor_to_dicts(conn.execute(
            "SELECT r.id, r.entity_from, r.predicate, r.entity_to, r.confidence, "
            "       ef.canonical_name AS entity_from_name, et.canonical_name AS entity_to_name "
            "FROM relations r "
            "LEFT JOIN entities ef ON ef.id = r.entity_from "
            "LEFT JOIN entities et ON et.id = r.entity_to "
            "WHERE r.provenance_capture_id = ? ORDER BY r.created_at", (capture_id,)
        ))
        notes = cursor_to_dicts(conn.execute(
            "SELECT id, title, content, summary, kind, archived_at "
            "FROM atomic_notes WHERE provenance_capture_id = ? ORDER BY created_at",
            (capture_id,)
        ))
        return {"entities": entities, "facts": facts,
                "relations": relations, "notes": notes}
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
            fact_id = insert_fact(
                conn, entity_id=body.entity_id,
                predicate=body.predicate, value=body.value, confidence=1.0,
                source_inbox_id=str(entry["capture_id"]),
                persistence_value=body.persistence_value,
                provenance_capture_id=entry["capture_id"],
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
    """User edit of the fiche (SYN-82): type (closed enum) and/or rename.

    A rename keeps the old canonical_name as an alias so the resolver still
    matches future mentions of the old name."""
    new_name = body.canonical_name.strip() if body.canonical_name else None
    if body.type is None and not new_name:
        raise HTTPException(status_code=400, detail="nothing to update")
    conn = get_connection()
    try:
        e = first_row(conn.execute(
            "SELECT id, canonical_name, aliases FROM entities WHERE id=?", (entity_id,)
        ))
        if not e:
            raise HTTPException(status_code=404, detail="entity not found")
        with conn:
            if body.type is not None:
                conn.execute("UPDATE entities SET type=? WHERE id=?", (body.type, entity_id))
            if new_name and new_name != e["canonical_name"]:
                try:
                    aliases = json.loads(e["aliases"] or "[]")
                except (ValueError, TypeError):
                    aliases = []
                if e["canonical_name"] and e["canonical_name"] not in aliases:
                    aliases.append(e["canonical_name"])
                aliases = [a for a in aliases if a.lower() != new_name.lower()]
                conn.execute(
                    "UPDATE entities SET canonical_name=?, aliases=? WHERE id=?",
                    (new_name, json.dumps(aliases, ensure_ascii=False), entity_id),
                )
        return {"id": entity_id, "type": body.type,
                "canonical_name": new_name or e["canonical_name"]}
    finally:
        conn.close()


@app.post("/relation", dependencies=[Depends(require_auth)])
def create_relation(body: RelationCreate):
    """SYN-84 — user-created relation (the cycle only extracts from new notes;
    a fact edit never regenerates relations, so wrong/missing ones are fixed here).
    Both entities must already exist; user origin → confidence 1.0."""
    predicate = body.predicate.strip()
    if not predicate:
        raise HTTPException(status_code=400, detail="predicate required")
    conn = get_connection()
    try:
        for eid in (body.entity_from, body.entity_to):
            if not first_row(conn.execute(
                    "SELECT id FROM entities WHERE id=? AND merged_into_id IS NULL", (eid,))):
                raise HTTPException(status_code=404, detail=f"entity not found: {eid}")
        rel_id = body.id or str(uuid.uuid4())
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO relations (id, entity_from, predicate, entity_to, confidence) "
                "VALUES (?,?,?,?,1.0)",
                (rel_id, body.entity_from, predicate, body.entity_to),
            )
        return {"id": rel_id, "entity_from": body.entity_from,
                "predicate": predicate, "entity_to": body.entity_to, "confidence": 1.0}
    finally:
        conn.close()


@app.patch("/relation/{relation_id}", dependencies=[Depends(require_auth)])
def update_relation(relation_id: str, body: RelationUpdate):
    """SYN-84 — user correction of a relation's predicate (authoritative → 1.0)."""
    predicate = body.predicate.strip()
    if not predicate:
        raise HTTPException(status_code=400, detail="predicate required")
    conn = get_connection()
    try:
        if not first_row(conn.execute("SELECT id FROM relations WHERE id=?", (relation_id,))):
            raise HTTPException(status_code=404, detail="relation not found")
        with conn:
            conn.execute(
                "UPDATE relations SET predicate=?, confidence=1.0 WHERE id=?",
                (predicate, relation_id),
            )
        return {"id": relation_id, "predicate": predicate, "confidence": 1.0}
    finally:
        conn.close()


@app.delete("/relation/{relation_id}", dependencies=[Depends(require_auth)])
def delete_relation(relation_id: str):
    """SYN-84 — remove a wrongly-extracted relation (provenance stays in the capture)."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM relations WHERE id=?", (relation_id,))
            deleted = conn.execute("SELECT changes()").fetchone()[0] == 1
        if not deleted:
            raise HTTPException(status_code=404, detail="relation not found")
        return {"id": relation_id, "deleted": True}
    finally:
        conn.close()


@app.patch("/fact/{fact_id}", dependencies=[Depends(require_auth)])
def update_fact(fact_id: str, body: FactUpdate):
    """User correction of a fact (SYN-82) — authoritative: confidence → 1.0."""
    predicate = body.predicate.strip() if body.predicate else None
    value = body.value.strip() if body.value else None
    if not predicate and not value:
        raise HTTPException(status_code=400, detail="nothing to update")
    conn = get_connection()
    try:
        f = first_row(conn.execute("SELECT id FROM facts WHERE id=?", (fact_id,)))
        if not f:
            raise HTTPException(status_code=404, detail="fact not found")
        sets, params = ["confidence=1.0", "last_confirmed=?"], [
            datetime.now(timezone.utc).isoformat()]
        if predicate:
            sets.append("predicate=?"); params.append(predicate)
        if value:
            sets.append("value=?"); params.append(value)
        params.append(fact_id)
        with conn:
            conn.execute(f"UPDATE facts SET {', '.join(sets)} WHERE id=?", params)
            # SYN-89: a user correction invalidates the derived summary.
            conn.execute(
                "UPDATE entities SET summary_stale = 1 "
                "WHERE id = (SELECT entity_id FROM facts WHERE id = ?)",
                (fact_id,),
            )
        return {"id": fact_id, "predicate": predicate, "value": value, "confidence": 1.0}
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


# ── SYN-23 — Weekly digest ───────────────────────────────────────────────────────

@app.post("/digest/run", dependencies=[Depends(require_auth)])
def digest_run(days: int = 7, dry_run: bool = False):
    """Generate the weekly digest now (manual / testing — production is the
    weekly launchd job). Returns the digest markdown + the note id."""
    from dream_cycle.digest import generate_weekly_digest
    try:
        return generate_weekly_digest(days=days, dry_run=dry_run)
    except EnvironmentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/digest/latest", dependencies=[Depends(require_auth)])
def digest_latest():
    """The most recent stored digest (kind='digest' atomic_note)."""
    conn = get_connection()
    try:
        row = first_row(conn.execute(
            "SELECT id, title, content, summary, created_at FROM atomic_notes "
            "WHERE kind = 'digest' AND archived_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1"
        ))
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Aucun digest généré pour l'instant.")
    return row


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
        import base64
        entities = cursor_to_dicts(conn.execute("SELECT * FROM entities"))
        for e in entities:
            # SYN-91: ship the embedding (raw float32 BLOB) as base64 so the replica can compute
            # « entités liées » (cosine) offline. JSON can't hold bytes; the raw BLOB is dropped.
            emb = e.pop("embedding", None)
            e["embedding_b64"] = base64.b64encode(emb).decode("ascii") if emb else None
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
            "instance_id": config_store.get_instance_id(),
        }
    finally:
        conn.close()
