import contextlib
import io
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic
from mcp.server.fastmcp import FastMCP

from db import get_connection, cursor_to_dicts, first_row, init_db
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
                results = [_format_result(r, "vector") for r in cursor_to_dicts(cur)]
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
    conn = get_connection()
    try:
        pending = first_row(conn.execute(
            "SELECT id, fact_data FROM pending_facts WHERE id=?", (fact_id,)
        ))
        if not pending:
            return json.dumps({"status": "error", "message": f"fact_id '{fact_id}' not found"})
        try:
            fact_data = json.loads(pending["fact_data"])
        except (ValueError, TypeError):
            return json.dumps({"status": "error", "message": "invalid fact_data JSON"})

        with conn:
            if not confirmed:
                conn.execute("DELETE FROM pending_facts WHERE id=?", (fact_id,))
                return json.dumps({"status": "rejected", "fact_id": fact_id})

            if correction:
                fact_data["value"] = correction

            entity_name = fact_data.get("entity_canonical", "unknown")
            row = conn.execute(
                "SELECT id FROM entities WHERE LOWER(canonical_name)=LOWER(?)", (entity_name,)
            ).fetchone()
            if row:
                entity_id = row[0]
            else:
                entity_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO entities (id, canonical_name) VALUES (?,?)",
                    (entity_id, entity_name),
                )

            conn.execute(
                "INSERT INTO facts "
                "(id, entity_id, predicate, value, confidence, source_inbox_id, persistence_value) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()), entity_id,
                    fact_data.get("predicate"), fact_data.get("value"),
                    0.95,
                    fact_data.get("source_inbox_id"),
                    fact_data.get("persistence_value", 3),
                ),
            )
            conn.execute("DELETE FROM pending_facts WHERE id=?", (fact_id,))

        return json.dumps(
            {
                "status": "confirmed",
                "fact_id": fact_id,
                "entity": entity_name,
                "predicate": fact_data.get("predicate"),
                "value": fact_data.get("value"),
            }
        )
    finally:
        conn.close()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
