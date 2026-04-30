import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from mcp.server.fastmcp import FastMCP

from db import get_connection, cursor_to_dicts, init_db
from embeddings import embed_text

# ── Startup ───────────────────────────────────────────────────────────────────

init_db()


def _get_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


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

    Step 1 — Vector search: embeds the query via Claude Haiku, finds the closest
    atomic notes using cosine similarity in sqlite-vec.
    Step 2 — Text fallback: if no vector results (empty index or no API key),
    falls back to LIKE keyword search across atomic_notes and inbox.

    Args:
        query: Natural-language question or keyword phrase.
        limit: Maximum number of results (default 5, max 20).

    Returns:
        JSON array of results with title, content, date, score, sources, search_type.
    """
    limit = min(max(1, limit), 20)

    # ── Step 1: vector search
    client = _get_client()
    if client:
        try:
            query_vec = embed_text(query, client)
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
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                results = [_format_result(dict(zip(cols, r)), "vector") for r in rows]
            finally:
                conn.close()

            if results:
                return json.dumps(results, ensure_ascii=False, default=str)
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
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        results = [_format_result(dict(zip(cols, r)), "text") for r in rows]
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
    Trigger the Dream Cycle directly from Claude Desktop.

    Processes all unprocessed inbox entries through 3 phases:
    1. Filtering — Claude Haiku extracts key facts and removes noise
    2. Synthesis — cleaned notes are written to atomic_notes
    3. Vectorization — embeddings generated and stored in sqlite-vec

    Requires ANTHROPIC_API_KEY to be set in the environment.
    Add it to the MCP server config in claude_desktop_config.json:
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }

    Returns:
        JSON with status and the full cycle output log.
    """
    from dream_cycle.cycle import run_cycle

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_cycle()
        return json.dumps({"status": "success", "output": buf.getvalue()})
    except EnvironmentError as e:
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
