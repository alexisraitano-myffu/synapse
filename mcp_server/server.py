import contextlib
import io
import json
import struct
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

from db import get_connection, cursor_to_dicts, first_row, init_db
from embeddings import embed_text

# ── Startup ───────────────────────────────────────────────────────────────────

init_db()


def _format_result(row: dict, search_type: str) -> dict:
    result = {
        "id": row["id"],
        "title": row.get("title") or "(untitled)",
        "content": row["content"],
        "created_at": row.get("created_at"),
        "search_type": search_type,
    }
    if "distance" in row:
        result["score"] = round(max(0.0, 1 - float(row["distance"]) / 2), 4)
    raw_sources = row.get("source_ids")
    if raw_sources:
        try:
            result["sources"] = json.loads(raw_sources)
        except (ValueError, TypeError):
            pass
    return result


def _deserialize_vec(blob: bytes) -> tuple[float, ...]:
    """Decode a sqlite-vec serialized float32 blob into floats."""
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def _search_entities(query_vec: bytes, limit: int, conn) -> list[dict]:
    """
    Semantic search over the entity graph.

    Entity embeddings live as raw BLOBs (entity ids are UUIDs, so they can't
    share the int-rowid vec0 table), so we score them with a manual L2 distance
    on the unit vectors — fine for a personal-scale graph.
    """
    q = _deserialize_vec(query_vec)
    scored: list[tuple[float, dict]] = []
    for row in cursor_to_dicts(conn.execute(
        "SELECT id, canonical_name, type, summary, embedding FROM entities "
        "WHERE embedding IS NOT NULL"
    )):
        v = _deserialize_vec(row["embedding"])
        if len(v) != len(q):
            continue
        dist = sum((a - b) ** 2 for a, b in zip(q, v)) ** 0.5
        scored.append((dist, row))

    scored.sort(key=lambda x: x[0])
    results = []
    for dist, row in scored[:limit]:
        results.append({
            "id": row["id"],
            "title": row["canonical_name"],
            "content": row.get("summary") or "",
            "type": row.get("type"),
            "score": round(max(0.0, 1 - dist / 2), 4),
            "search_type": "entity",
        })
    return results


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "Synapse",
    instructions="Your personal semantic memory — store, search and recall knowledge.",
)


@mcp.tool()
def add_to_inbox(content: str, source: str = "manual") -> str:
    """
    Add a raw piece of information to the Synapse inbox.

    The Dream Cycle will later clean, deduplicate and vectorise this entry.
    Use this for quick captures: thoughts, meeting snippets, web clippings.

    Args:
        content: The raw text to store.
        source:  Origin hint (e.g. 'voice', 'chrome', 'meeting', 'manual').

    Returns:
        JSON with the new inbox id and confirmation status.
    """
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO inbox (content, source) VALUES (?, ?)",
                (content, source),
            )
        row_id = conn.last_insert_rowid()
        return json.dumps({"id": row_id, "status": "added", "source": source})
    finally:
        conn.close()


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """
    Search the knowledge base using hybrid search (vector + keyword fallback).

    Step 1 — Vector search (fully local, no API key needed): embeds the query
    locally and ranks both episodic notes (atomic_notes) and graph entities by
    similarity, merged into one score-sorted list.
    Step 2 — Text fallback: if the vector path yields nothing (empty index),
    falls back to LIKE keyword search across atomic_notes and inbox.

    Args:
        query: Natural-language question or keyword phrase.
        limit: Maximum number of results (default 5, max 20).

    Returns:
        JSON array of results with title, content, score, search_type
        ('vector' for notes, 'entity' for graph entities, 'text' for fallback).
    """
    limit = min(max(1, limit), 20)

    # ── Step 1: vector search over notes + entities (local, no API key required)
    try:
        query_vec = embed_text(query)
        conn = get_connection()
        try:
            cur = conn.execute(
                """
                SELECT n.id, n.title, n.content, n.source_ids, n.created_at, v.distance
                FROM   atomic_notes_vec v
                JOIN   atomic_notes n ON n.id = v.rowid
                WHERE  v.embedding MATCH ?
                AND    k = ?
                ORDER  BY v.distance
                """,
                (query_vec, limit),
            )
            note_results = [_format_result(r, "vector") for r in cursor_to_dicts(cur)]
            entity_results = _search_entities(query_vec, limit, conn)
        finally:
            conn.close()

        merged = note_results + entity_results
        merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        if merged:
            return json.dumps(merged[:limit], ensure_ascii=False, default=str)
    except Exception:
        pass  # fall through to text search

    # ── Step 2: text fallback
    pattern = f"%{query}%"
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT id, title, content, source_ids, created_at, 'note' AS type
            FROM   atomic_notes
            WHERE  content LIKE ? OR title LIKE ?
            UNION ALL
            SELECT id, NULL, content, NULL, created_at, 'inbox' AS type
            FROM   inbox
            WHERE  content LIKE ?
            ORDER  BY created_at DESC
            LIMIT  ?
            """,
            (pattern, pattern, pattern, limit),
        )
        results = [_format_result(r, "text") for r in cursor_to_dicts(cur)]
        return json.dumps(results, ensure_ascii=False, default=str)
    finally:
        conn.close()


@mcp.tool()
def list_recent(limit: int = 10) -> str:
    """
    List the most recent raw entries in the inbox, not yet processed.

    Useful to review what has been captured recently before the Dream Cycle runs.

    Args:
        limit: Number of entries to return (default 10, max 50).

    Returns:
        JSON array of inbox entries, newest first.
    """
    limit = min(max(1, limit), 50)

    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id, content, source, created_at FROM inbox ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return json.dumps(cursor_to_dicts(cur), ensure_ascii=False, default=str)
    finally:
        conn.close()


@mcp.tool()
def run_dream_cycle() -> str:
    """
    Trigger the Dream Cycle directly (handy for testing — normally cron-driven).

    Processes all unprocessed inbox entries: Claude classifies each one and
    routes it to the entity graph (facts), atomic_notes (episodic memory) or
    intentions (ephemeral), scoring confidence and vectorizing as it goes.

    Requires ANTHROPIC_API_KEY for the classification/extraction step.
    Add it to the MCP server config in claude_desktop_config.json:
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }

    Returns:
        JSON with status and the full cycle output log.
    """
    from dream_cycle import run_dream_cycle as _run_cycle

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _run_cycle()
        return json.dumps({"status": "success", "output": buf.getvalue()})
    except EnvironmentError as e:
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })


# ── Phase A+ tools ────────────────────────────────────────────────────────────

@mcp.tool()
def get_entity(name: str) -> str:
    """
    Search an entity by canonical name or alias.

    Returns the entity's facts (predicates + values + confidence) and
    outgoing relations. Useful to inspect what Synapse knows about a person,
    place, or concept.

    Args:
        name: Canonical name or any known alias.

    Returns:
        JSON with the entity record, its facts, and its relations.
    """
    conn = get_connection()
    try:
        entity = first_row(conn.execute(
            "SELECT * FROM entities WHERE LOWER(canonical_name)=LOWER(?)", (name,)
        ))

        # Alias fallback
        if not entity:
            for candidate in cursor_to_dicts(conn.execute("SELECT * FROM entities")):
                try:
                    aliases = json.loads(candidate.get("aliases", "[]"))
                except (ValueError, TypeError):
                    aliases = []
                if name.lower() in [a.lower() for a in aliases]:
                    entity = candidate
                    break

        if not entity:
            return json.dumps({"found": False, "name": name})

        entity_id = entity["id"]

        facts = cursor_to_dicts(conn.execute(
            "SELECT predicate, value, confidence, persistence_value, created_at "
            "FROM facts WHERE entity_id=? ORDER BY confidence DESC",
            (entity_id,),
        ))

        relations = cursor_to_dicts(conn.execute(
            """SELECT r.predicate, e.canonical_name AS entity_to, r.confidence
               FROM relations r
               JOIN entities e ON e.id = r.entity_to
               WHERE r.entity_from=?""",
            (entity_id,),
        ))

        return json.dumps(
            {
                "found": True,
                "id": entity_id,
                "canonical_name": entity["canonical_name"],
                "type": entity.get("type"),
                "aliases": json.loads(entity.get("aliases", "[]")),
                "mention_count": entity.get("mention_count", 0),
                "summary": entity.get("summary"),
                "facts": facts,
                "relations": relations,
            },
            ensure_ascii=False,
            default=str,
        )
    finally:
        conn.close()


@mcp.tool()
def list_pending() -> str:
    """
    List facts waiting for validation (pending_facts table).

    These are facts extracted by the Dream Cycle with confidence between
    0.5 and 0.85 — plausible but not yet confirmed. Use validate_fact()
    to accept or reject individual items.

    Returns:
        JSON array of pending facts with parsed fact_data.
    """
    conn = get_connection()
    try:
        result = []
        for item in cursor_to_dicts(conn.execute(
            "SELECT id, fact_data, validation_strategy, created_at "
            "FROM pending_facts ORDER BY created_at DESC"
        )):
            try:
                item["fact_data"] = json.loads(item["fact_data"])
            except (ValueError, TypeError):
                pass
            result.append(item)
        return json.dumps(result, ensure_ascii=False, default=str)
    finally:
        conn.close()


@mcp.tool()
def validate_fact(fact_id: str, confirmed: bool, correction: str = None) -> str:
    """
    Validate or reject a pending fact.

    If confirmed=True, the fact is consolidated into entities/facts with
    confidence 0.95 (user-confirmed). If correction is provided, its value
    overrides the extracted one. If confirmed=False, the fact is discarded.

    Args:
        fact_id:    ID from list_pending().
        confirmed:  True to accept, False to discard.
        correction: Optional corrected value (overrides extracted value).

    Returns:
        JSON with status and details of what was done.
    """
    from dream_cycle.validation import record_and_apply_validation

    conn = get_connection()
    try:
        with conn:
            result = record_and_apply_validation(conn, fact_id, confirmed, correction)
        return json.dumps(result, ensure_ascii=False, default=str)
    finally:
        conn.close()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
