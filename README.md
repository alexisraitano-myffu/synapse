# Synapse

**A local-first personal semantic memory system.** Capture raw notes; an AI cleans, links and structures them into a queryable knowledge graph + episodic memory — all on your own machine. **No cloud: your data never leaves your devices.**

Synapse runs on a Mac (the "brain"), exposes its memory to AI agents over **MCP** (Claude Desktop / Claude Code) and to apps over a small **HTTP API**. Processing is done by a nightly/event-driven "Dream Cycle" using Claude Haiku; embeddings are **fully local** (no API call).

> Philosophy: *capture passively, process actively.* Open source (Apache-2.0). The optional mobile app is a separate project.

## How it works

```
Capture → inbox → Dream Cycle ─┬─ fact      → entities / facts / relations  (semantic memory)
                               ├─ episodic  → atomic_notes (vectorized)      (episodic memory)
                               └─ ephemeral → intentions (48h TTL)
                                        ↓
                         MCP tools  ·  HTTP API  ·  web visualizer
```

- **Entities** are created on mention; **facts** are confidence-scored and either consolidated, queued for validation, or sent to a review queue.
- **Embeddings**: local `fastembed` (ONNX, multilingual, 384-d) → SQLite + `sqlite-vec`. No PyTorch, no embedding server, no network.
- Full design + diagrams: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # used only by the Dream Cycle (reasoning), not for embeddings
```

**Process the inbox** (the Dream Cycle):
```bash
python -m dream_cycle               # --dry-run / --verbose available
```

**HTTP API** (backend for apps; FastAPI on `0.0.0.0:8000`):
```bash
python -m api                       # auth via SYNAPSE_API_TOKEN; SYNAPSE_AUTO_CYCLE=1 for debounced auto-runs
```

**MCP server** (Claude Desktop / Claude Code, over stdio):
```bash
python mcp_server/server.py
```

**Web visualizer** (D3 knowledge graph at http://127.0.0.1:8080):
```bash
python visualizer/app.py
```

## HTTP API

Bearer auth (`SYNAPSE_API_TOKEN`; disabled if unset = dev). Contract: [`openapi.json`](openapi.json).

`GET /health` · `POST /capture` (idempotent on a client UUID) · `GET /feed` · `GET /graph` (entity graph + opt-in **living-map** layers: atomic-note nodes, Louvain clusters, ForceAtlas2 positions, labelled regions, anti-hairball filters) · `GET /entity/{id}` · `GET /pending` · `POST /pending/{id}/validate` · `POST /dream-cycle/run` · `GET /dream-cycle/last` · `GET /changes` (pull-replication).

Designed for LAN / private mesh (Tailscale). Captures carry a client UUID + device id (idempotent, offline-safe); validations are recorded as append-only events; derived state is rebuildable → multi-device replication without a multi-master database.

## MCP tools

`add_to_inbox` · `search_memory` (local vector over notes + entities, text fallback) · `list_recent` · `run_dream_cycle` · `get_entity` · `list_pending` · `validate_fact`.

## Configuration

`config.py`: `SYNAPSE_HOME` (DB location, default `~/.synapse`), `EMBEDDING_MODEL` (local), `CLAUDE_MODEL` (Dream Cycle reasoning). The API also reads `SYNAPSE_API_TOKEN`, `SYNAPSE_API_PORT`, `SYNAPSE_AUTO_CYCLE`, `SYNAPSE_CYCLE_DEBOUNCE_SECONDS`.

## Tests

```bash
pytest                              # offline suites run without an API key
```
`test_embeddings.py`, `test_cycle.py`, `test_api.py` are fully offline; `test_dream_cycle.py` exercises the live classify→route pipeline and is skipped unless `ANTHROPIC_API_KEY` is set.

## License

Apache-2.0. See [LICENSE](LICENSE).
