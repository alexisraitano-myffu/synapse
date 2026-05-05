# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Synapse is a local-first personal semantic memory system exposed as an MCP server. It captures raw notes into an inbox, processes them via Claude Haiku into atomic knowledge units (Dream Cycle), and makes them searchable via vector similarity. The primary interface is MCP tools consumed by Claude Desktop or Claude Code.

## Commands

**Setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run MCP server** (Claude Desktop/Code integration):
```bash
source .venv/bin/activate
python mcp_server/server.py
```

**Run Dream Cycle** (process inbox → atomic notes):
```bash
source .venv/bin/activate
python run_cycle.py
```

**Run web visualizer** (knowledge graph at http://127.0.0.1:8080):
```bash
source .venv/bin/activate
python visualizer/app.py
```

**Environment variable:** `ANTHROPIC_API_KEY` is required for Dream Cycle and vector search. `SYNAPSE_HOME` overrides the default DB location (`~/.synapse/synapse.db`).

## Architecture

### Data Flow

```
Capture → inbox table → Dream Cycle → atomic_notes table + atomic_notes_vec (sqlite-vec)
```

The Dream Cycle (`dream_cycle/cycle.py`) is a 3-phase pipeline:
1. **Filter/Extract** — Claude Haiku reads raw inbox entries and extracts distinct atomic notes as JSON (uses prompt caching)
2. **Synthesize** — Writes notes to `atomic_notes`, marks inbox entries `processed_at`
3. **Vectorize** — Claude Haiku extracts 12 semantic concepts per note; concepts are hash-projected into a 384-dim vector and stored in `atomic_notes_vec` (sqlite-vec `vec0` virtual table)

### MCP Tools (mcp_server/server.py)

Four tools exposed to Claude:
- `add_to_inbox(content, source)` — raw capture
- `search_memory(query, limit)` — hybrid vector + text-fallback search
- `list_recent(limit)` — show unprocessed inbox entries
- `run_dream_cycle()` — trigger processing pipeline

### Embedding Strategy

**No PyTorch, no external embedding server.** Embeddings use Claude Haiku to extract semantic concepts, then deterministically project them to a 384-dim vector via hash projection (`embeddings.py`). This means embeddings are reproducible and cheap (prompt caching), but are concept-based rather than subword-based.

Search is hybrid: vector k-NN via sqlite-vec first, falling back to `LIKE %query%` if vectors are unavailable or API key is missing.

### Database

SQLite at `~/.synapse/synapse.db`. Key tables:
- `inbox` — raw captures with `processed_at` NULL until Dream Cycle runs
- `atomic_notes` — processed notes with `source_ids` (JSON array of contributing inbox IDs)
- `atomic_notes_vec` — vec0 virtual table; rowid mirrors `atomic_notes.id`
- `knowledge_graph` — explicit entity relations (reserved for Phase D)

Connection and schema initialization live in `db/__init__.py`. The sqlite-vec extension is loaded at connection time via `apsw`.

### Config (config.py)

```python
BASE_DIR = Path(os.getenv("SYNAPSE_HOME", Path.home() / ".synapse"))
DB_PATH = BASE_DIR / "synapse.db"
EMBEDDING_DIM = 384
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
```

### Visualizer (visualizer/)

FastAPI REST API (`app.py`) serving `/api/nodes`, `/api/edges`, `/api/stats`, `/api/note/{id}`, backed by the same SQLite DB. Static frontend uses D3.js force-directed graph (`static/graph.js`).
