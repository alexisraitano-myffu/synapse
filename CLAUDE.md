# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Synapse is a local-first personal semantic memory system exposed as an MCP server. It captures raw notes into an inbox, then a single "Dream Cycle" processes them with Claude Haiku into a structured memory (entity graph + episodic notes), searchable via local vector similarity. The primary interface is MCP tools consumed by Claude Desktop or Claude Code.

## Commands

**Setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run MCP server** (Claude Desktop/Code integration; communicates over stdio):
```bash
python mcp_server/server.py
```

**Run the Dream Cycle** (process the inbox; normally cron-driven, e.g. `0 3 * * *`):
```bash
python -m dream_cycle              # or: python run_cycle.py  (identical)
python -m dream_cycle --dry-run --verbose   # preview without DB writes, per-step logs
```

**Run tests:**
```bash
pytest                             # whole suite
pytest test_embeddings.py          # offline NR tests (no API key needed)
pytest test_cycle.py::test_episodic_note_is_searchable   # a single test
```
`test_embeddings.py`, `test_cycle.py` and `test_api.py` run fully offline (local embeddings, FastAPI TestClient). `test_dream_cycle.py` hits the live Claude API for the classify→route pipeline and is **skipped** unless `ANTHROPIC_API_KEY` is set. Shared fixture `isolated_db` lives in `conftest.py`.

**Re-embed** after changing the embedding model:
```bash
python reembed.py
```

**Run the HTTP API** (backend for the mobile/desktop apps; FastAPI on `0.0.0.0:8000`):
```bash
python -m api                      # env: SYNAPSE_API_TOKEN (bearer auth), SYNAPSE_API_PORT,
                                   # SYNAPSE_AUTO_CYCLE=1 (debounced auto-run after captures),
                                   # SYNAPSE_CYCLE_DEBOUNCE_SECONDS (default 120)
```

**Run web visualizer** (knowledge graph at http://127.0.0.1:8080):
```bash
python visualizer/app.py
```

**Environment:** `ANTHROPIC_API_KEY` is required for the Dream Cycle's classification step (NOT for embeddings or search — those are local). `SYNAPSE_HOME` overrides the default DB location (`~/.synapse/synapse.db`); tests set it to a temp dir for isolation. The Dream Cycle and MCP server load `.env` via `python-dotenv`.

## Architecture

### Data flow

```
Capture → inbox → Dream Cycle ─┬─ fact      → entities / facts / relations
                               ├─ episodic  → atomic_notes (+ atomic_notes_vec)
                               ├─ ephemeral → intentions (48h TTL)
                               └─ resource  → (routed like fact for now; fetch+summary TODO)
```

`import dream_cycle` resolves to the **package** `dream_cycle/`; the pipeline lives in `dream_cycle/cycle.py` and is exported as `run_dream_cycle` (also `python -m dream_cycle`). There is one cycle — the earlier two-implementation split has been merged.

### The Dream Cycle (`dream_cycle/cycle.py`)

Operates per inbox entry, with French prompts. Classifies each entry, then routes by `input_type`:

- **fact** → the 6-step graph pipeline below.
- **episodic** → `write_episodic_note`: stores raw content + summary + `entities_mentioned` in `atomic_notes` with `memory_strength=1.0`, and vectorizes it into `atomic_notes_vec`.
- **ephemeral** → `intentions` (48h TTL), skips the graph.
- **resource** → currently falls through to the fact pipeline (fetch+summary into `resources` is a future step, per spec).

The 6 steps for facts:
1. **Classify** — Haiku tags `input_type` and extracts entities, facts (snake_case predicates + `persistence_value` 1–5), relations, and a one-line summary.
2. **Resolve** — matches entities to existing rows (canonical name or alias); resolves relative dates to absolute via `dateparser` (date-like predicates only).
3. **Score** (`compute_confidence`) — explicit-statement (+0.5) + context (+0.3) + mention-count (≤+0.2) + persistence bonus → [0,1].
4. **Route** — **entity nodes are created on mention** (decoupled from fact confidence) as long as they pass the anti-pollution garde-fou `MIN_ENTITY_PERSISTENCE` (≥2, i.e. not pure noise) OR appear in a relation OR already exist. **Facts** are still confidence-gated: > 0.85 → `facts`; 0.5–0.85 → `pending_facts`; < 0.5 → `review_queue`. So a fresh entity exists in the graph while its facts await corroboration/validation. Relations are written when both endpoints exist (now reliably, since entities are created eagerly). `_upsert_entity` fills `summary`/`attributes`/`persistence_value`.
5. **Behavioral validation** — a pending fact corroborated by a new mention in the same run is promoted into `facts`.

Per-entry resilience: each entry is processed in isolation. An `anthropic.APIError` (no/invalid key, network) **aborts the whole run** and leaves entries queued for a retry; a content error on one entry marks just that entry `status='failed'` (`processed_at` set so it isn't retried) and the run continues. The API can auto-run the cycle debounced after captures (`SYNAPSE_AUTO_CYCLE`).
6. **Vectorize** — embeds touched entities into the `entities.embedding` BLOB column.

`memory_strength` (Ebbinghaus decay / graceful forgetting) is in the schema but not yet computed — that's the planned Phase C.

### MCP tools (`mcp_server/server.py`)

- `add_to_inbox(content, source)` — raw capture
- `search_memory(query, limit)` — local vector search over **both** `atomic_notes` (episodic) and `entities` (graph), merged and score-sorted; falls back to `LIKE` keyword search if the vector path yields nothing
- `list_recent(limit)` — recent inbox entries
- `run_dream_cycle()` — triggers the unified cycle (kept for testing; production is cron-driven)
- `get_entity(name)` — entity by canonical name or alias, with its facts and relations
- `list_pending()` — facts awaiting validation (`pending_facts`)
- `validate_fact(fact_id, confirmed, correction)` — confirm (→ `facts` at confidence 0.95) or reject a pending fact. Shares logic with the HTTP API via `dream_cycle/validation.py::record_and_apply_validation` (records an append-only `validation_events` row, then applies).

### HTTP API (`api/app.py`)

FastAPI app for the mobile/desktop clients (run `python -m api`, port 8000). Bearer auth via `SYNAPSE_API_TOKEN` (auth **disabled** if unset — dev mode). Endpoints: `GET /health`, `POST /capture` (**idempotent on client UUID** — `INSERT OR IGNORE` on `inbox.client_id`), `GET /feed`, `GET /graph` (`?mode=ego&entity=`), `GET /entity/{id}`, `GET /pending` (question + source quote), `POST /pending/{id}/validate`, `POST /dream-cycle/run` (single-instance **file lock** + writes `cycle_runs`), `GET /dream-cycle/last`, `GET /changes` (pull-replication snapshot of derived state). Per-request apsw connections (sync endpoints). Sync model: captures carry `id`/`device_id`/`captured_at`, validations are append-only events → state rebuildable, multi-device replication possible (see `docs/ARCHITECTURE.md`).

### Embedding strategy

**Fully local, no PyTorch, no API call.** `embeddings.py` uses **fastembed** (ONNX runtime) with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~50 languages incl. French — set via `EMBEDDING_MODEL` in `config.py`). The model loads lazily as a process-level singleton (~220 MB, downloaded once, then offline). `embed_text(text, client=None)` returns an L2-normalized serialized vector; the `client` arg is ignored (kept for backward compat with the old API-based signature). Run `python reembed.py` after changing `EMBEDDING_MODEL` to regenerate existing vectors.

Vectors are normalized so the sqlite-vec `vec0` **L2 distance** stays in [0, 2] and is monotonic with cosine — keeping the `score = 1 - distance/2` mapping valid. With this model, related notes land ~0.9 and unrelated ~1.4 (the visualizer edge threshold is 1.1).

Search is hybrid: vector k-NN via sqlite-vec first (no API key needed — embeddings are local), falling back to `LIKE %query%` across `atomic_notes` and `inbox` only if the vector path errors or returns nothing.

### Database

SQLite at `~/.synapse/synapse.db`, opened via `apsw` (stdlib `sqlite3` on macOS can't load extensions). The sqlite-vec extension is loaded at connection time. Schema and connection helpers (`get_connection`, `cursor_to_dicts`, `first_row`, `init_db`) live in `db/__init__.py`; `init_db()` is idempotent (`CREATE TABLE IF NOT EXISTS` + best-effort `ALTER TABLE` migrations wrapped in try/except) and is called at MCP startup and at the top of the Dream Cycle.

Tables:
- `inbox` — raw captures; `processed_at` NULL until the Dream Cycle consumes them. Sync columns: `client_id` (UNIQUE partial index → idempotent capture), `device_id`, `captured_at`, `status`
- `validation_events` — append-only log of user validate/reject decisions (durable, replicable)
- `cycle_runs` — one row per Dream Cycle run (stats for `GET /dream-cycle/last`)
- `atomic_notes` / `atomic_notes_vec` — episodic memory; vec0 rowid mirrors `atomic_notes.id`. Columns include `summary`, `entities_mentioned` (JSON), `memory_strength` (added by migration)
- `entities`, `facts`, `relations`, `resources` — entity graph (entity/fact IDs are UUID strings); entity embeddings live in `entities.embedding` as raw BLOBs (UUID ids can't use the int-rowid vec0 table, so entity search does manual cosine)
- `pending_facts`, `review_queue`, `intentions` — routing buckets
- `knowledge_graph` — legacy explicit relations table; unused by current code

vec0 virtual tables don't support `COUNT(*)`; count by point-looking-up each rowid (see `visualizer/app.py::get_stats`).

### Config (`config.py`)

```python
BASE_DIR = Path(os.getenv("SYNAPSE_HOME", Path.home() / ".synapse"))
DB_PATH = BASE_DIR / "synapse.db"
EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # local fastembed
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Dream Cycle reasoning only
```

### Visualizer (`visualizer/`)

FastAPI app (`app.py`) serving `/api/nodes`, `/api/edges`, `/api/stats`, `/api/note/{id}`, backed by the same SQLite DB. It reads the `atomic_notes` (episodic) world. Edges are computed live from vector similarity (k-NN per note, L2 distance threshold 1.1). Static frontend is a D3.js force-directed graph (`static/graph.js`). Note: it does not yet render the entity graph — wiring that to `/api/nodes` is a natural next step.

## Clients

The HTTP API has known clients beyond MCP:
- A **mobile app** (Android + iOS, Kotlin Multiplatform + Compose Multiplatform) lives in a separate **private/proprietary** repo `synapse-app` and talks to this backend over the LAN (`POST /capture`, `GET /feed`, `GET /changes`, `POST /pending/{id}/validate`). The frozen contract it codes against is the generated `openapi.json` in this repo. Keep that file up to date when endpoints change.
- (Future) a desktop app and a managed sync relay are part of the wider product but live outside this repo.

The roadmap (Phase C — memory_strength decay, coreference window, resource fetch, weekly digest, etc.) is tracked in an **internal task tracker outside this repo**. Don't reference internal tooling URLs from this file (public repo).
