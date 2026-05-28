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

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import BASE_DIR
from db import cursor_to_dicts, first_row, get_connection, init_db

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
    from api.discovery import start_advertising, stop_advertising
    threading.Thread(target=_scheduler_loop, daemon=True).start()
    azc = await start_advertising()
    try:
        yield
    finally:
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
        for e in cursor_to_dicts(conn.execute(
            "SELECT id, canonical_name, type, mention_count, persistence_value, summary FROM entities"
        )):
            nodes.append({
                "id": e["id"],
                "label": e["canonical_name"],
                "type": e.get("type"),
                "mention_count": e.get("mention_count", 1),
                "persistence_value": e.get("persistence_value", 3),
                "summary": e.get("summary"),
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
        facts = cursor_to_dicts(conn.execute(
            "SELECT predicate, value, confidence, persistence_value, created_at "
            "FROM facts WHERE entity_id=? ORDER BY confidence DESC", (entity_id,),
        ))
        relations = cursor_to_dicts(conn.execute(
            "SELECT r.predicate, e.canonical_name AS entity_to, r.confidence "
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
            "facts": facts, "relations": relations,
        }
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
