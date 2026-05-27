import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from db import get_connection, init_db

app = FastAPI(title="Synapse Visualizer")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

STOP_WORDS = {
    "a", "an", "the", "is", "in", "of", "and", "or", "for",
    "to", "with", "on", "at", "from", "by", "its", "this", "that",
}


def _group(title: str) -> int:
    words = [
        w for w in re.findall(r"\b[a-z]{3,}\b", (title or "").lower())
        if w not in STOP_WORDS
    ]
    key = words[0] if words else "default"
    return abs(hash(key)) % 8


def _strip_markdown(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/nodes")
async def get_nodes():
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id, title, content, source_ids, created_at FROM atomic_notes ORDER BY id"
        )
        cols = [d[0] for d in cur.description]
        notes = [dict(zip(cols, row)) for row in cur.fetchall()]

        # Connection count from knowledge_graph
        kg_cur = conn.execute(
            "SELECT entity_a, entity_b FROM knowledge_graph"
        )
        # Count how many times each note_id appears as a source
        connection_count: dict[int, int] = {n["id"]: 0 for n in notes}

        # Also count via source_ids overlap
        note_sources: dict[int, set] = {}
        for note in notes:
            try:
                note_sources[note["id"]] = set(json.loads(note["source_ids"] or "[]"))
            except (ValueError, TypeError):
                note_sources[note["id"]] = set()

        for i, ids_a in enumerate(note_sources.values()):
            note_id_a = list(note_sources.keys())[i]
            for note_id_b, ids_b in note_sources.items():
                if note_id_a != note_id_b and ids_a & ids_b:
                    connection_count[note_id_a] = connection_count.get(note_id_a, 0) + 1

        result = []
        for note in notes:
            content = note["content"] or ""
            plain = _strip_markdown(content)
            summary = plain[:140] + ("…" if len(plain) > 140 else "")
            result.append({
                "id": note["id"],
                "title": note["title"] or "(untitled)",
                "summary": summary,
                "created_at": note["created_at"],
                "group": _group(note["title"] or ""),
                "size": max(1, len(json.loads(note["source_ids"] or "[]"))),
                "connections": connection_count.get(note["id"], 0),
            })
        return result
    finally:
        conn.close()


def _build_edges(conn) -> list[dict]:
    """Compute vector-similarity edges. Shared by /api/edges and /api/stats."""
    first_note = conn.execute("SELECT id FROM atomic_notes LIMIT 1").fetchone()
    if not first_note:
        return []
    first_vec = conn.execute(
        "SELECT embedding FROM atomic_notes_vec WHERE rowid = ?", (first_note[0],)
    ).fetchone()
    if not first_vec:
        return []

    note_ids = [r[0] for r in conn.execute("SELECT id FROM atomic_notes ORDER BY id").fetchall()]

    seen: set[tuple] = set()
    edges = []
    # L2 distance on unit vectors, range [0,2]. With the local fastembed model,
    # related notes land ~0.9 and unrelated ~1.4, so 1.1 separates them well.
    # Tune up for a denser graph, down for a sparser one.
    DISTANCE_THRESHOLD = 1.1
    K = 4  # k-1 real neighbours + self

    for nid in note_ids:
        vec_row = conn.execute(
            "SELECT embedding FROM atomic_notes_vec WHERE rowid = ?", (nid,)
        ).fetchone()
        if not vec_row:
            continue

        cur = conn.execute(
            "SELECT rowid, distance FROM atomic_notes_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_row[0], K),
        )
        cols = [d[0] for d in cur.description]
        for nb in [dict(zip(cols, r)) for r in cur.fetchall()]:
            nb_id, dist = nb["rowid"], nb["distance"]
            if nb_id == nid or dist > DISTANCE_THRESHOLD:
                continue
            pair = (min(nid, nb_id), max(nid, nb_id))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({"source": nid, "target": nb_id, "weight": round(1 - dist / 2, 3)})

    return edges


@app.get("/api/edges")
async def get_edges():
    conn = get_connection()
    try:
        return _build_edges(conn)
    finally:
        conn.close()


@app.get("/api/note/{note_id}")
async def get_note(note_id: int):
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id, title, content, source_ids, created_at, updated_at "
            "FROM atomic_notes WHERE id = ?",
            (note_id,),
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Note not found")
        note = dict(zip(cols, row))
        try:
            note["source_ids"] = json.loads(note["source_ids"] or "[]")
        except (ValueError, TypeError):
            note["source_ids"] = []
        return note
    finally:
        conn.close()


@app.get("/api/stats")
async def get_stats():
    conn = get_connection()
    try:
        notes_total = conn.execute("SELECT COUNT(*) FROM atomic_notes").fetchone()[0]
        # Count vectorized notes via point lookups (vec0 doesn't support COUNT(*))
        all_ids = [r[0] for r in conn.execute("SELECT id FROM atomic_notes").fetchall()]
        notes_vectorized = sum(
            1 for nid in all_ids
            if conn.execute("SELECT embedding FROM atomic_notes_vec WHERE rowid = ?", (nid,)).fetchone()
        )
        inbox_pending = conn.execute(
            "SELECT COUNT(*) FROM inbox WHERE processed_at IS NULL"
        ).fetchone()[0]
        connections = len(_build_edges(conn))
        return {
            "notes_total": notes_total,
            "notes_vectorized": notes_vectorized,
            "connections": connections,
            "inbox_pending": inbox_pending,
        }
    finally:
        conn.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
