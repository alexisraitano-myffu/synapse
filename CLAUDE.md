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

**Run the decay job** (SYN-19/68 — recompute `memory_strength` via Ebbinghaus for `atomic_notes` and `entities`; also runs at the end of every Dream Cycle, but a nightly cron covers empty-inbox days):
```bash
python -m dream_cycle.decay        # env: SYNAPSE_DECAY_TAU_DAYS (default 30)
```

**Run the weekly digest** (SYN-23 — condense the past week + the week ahead into one durable `atomic_note` of `kind='digest'`; retrospective new entities/facts/notes/trends + forward-looking dated events & open tasks, rendered by Haiku, idempotent per ISO week):
```bash
python -m dream_cycle.digest                 # generate + store this week's digest
python -m dream_cycle.digest --dry-run --verbose   # preview the markdown without writing
```
On-demand from a client: `POST /digest/run` (`?dry_run=true` to preview); `GET /digest/latest` returns the last stored digest. Production trigger = a weekly LaunchAgent (Monday 08h), machine-specific — see the launchd note below. The API backend also self-heals it: `_ensure_weekly_digest` (in the scheduler loop) generates the current ISO week's digest if it's missing, so a scheduled fire missed while the Mac slept is recovered within the hour once it's awake.

**Run the HTTP API** (backend for the mobile/desktop apps; FastAPI on `0.0.0.0:8000`):
```bash
python -m api                      # env: SYNAPSE_API_TOKEN (bearer auth), SYNAPSE_API_PORT,
                                   # SYNAPSE_AUTO_CYCLE=1 (debounced auto-run after captures),
                                   # SYNAPSE_CYCLE_DEBOUNCE_SECONDS (default 120)
```

**Production on this Mac — launchd (since 2026-06-12).** The API runs as a user LaunchAgent
(`~/Library/LaunchAgents/fr.myffu.synapse.backend.plist`, NOT in this repo — machine-specific):
`RunAtLoad` + `KeepAlive` (auto-start at login, auto-restart on crash), `WorkingDirectory` = this
repo (so `.env` provides `ANTHROPIC_API_KEY` + `SYNAPSE_AUTO_CYCLE=1`), program = `.venv/bin/python -m api`.
This replaced ad-hoc manual runs — a machine reboot used to silently kill processing (the
original dogfood incident: a cycle died mid-run and nothing processed for a day).
```bash
tail -f ~/.synapse/api.log                                        # logs (stdout+stderr)
launchctl kickstart -k gui/$(id -u)/fr.myffu.synapse.backend      # restart (ALWAYS after a backend code change)
launchctl bootout gui/$(id -u)/fr.myffu.synapse.backend           # stop/disable
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/fr.myffu.synapse.backend.plist  # (re)enable
curl -s http://localhost:8000/health                              # liveness + counters
```

**Weekly digest LaunchAgent (SYN-23).** Like the backend agent, the schedule lives in a
machine-specific plist outside this repo (`~/Library/LaunchAgents/fr.myffu.synapse.digest.plist`):
`StartCalendarInterval` Monday (`Weekday 1`) 08:00, `WorkingDirectory` = this repo (so `.env`
provides `ANTHROPIC_API_KEY`), program = `.venv/bin/python -m dream_cycle.digest`, logs to
`~/.synapse/digest.log`. It writes one `kind='digest'` note per ISO week (re-running overwrites it).
Belt-and-braces: the API backend self-heals a missed fire (`_ensure_weekly_digest`) — if the Mac
was asleep at 08h Monday, the running backend regenerates the week's digest on its next hourly
check. Both triggers target the same current-week label, so there's never a duplicate.
```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/fr.myffu.synapse.digest.plist   # enable
launchctl kickstart -k gui/$(id -u)/fr.myffu.synapse.digest                              # run now
launchctl bootout   gui/$(id -u)/fr.myffu.synapse.digest                                 # disable
```

**Run web visualizer** (knowledge graph at http://127.0.0.1:8080):
```bash
python visualizer/app.py
```

**Seed synthetic demo data for the living map** (SYN-64 dogfood — ~95 entities in 10 communities + 50 notes, varied `memory_strength`; idempotent, test data only, writes into `SYNAPSE_HOME`):
```bash
python -m scripts.seed_demo_map            # (re)seed · --clean removes all synthetic rows
```

**Environment:** `ANTHROPIC_API_KEY` is required for the Dream Cycle's classification step (NOT for embeddings or search — those are local). `SYNAPSE_HOME` overrides the default DB location (`~/.synapse/synapse.db`); tests set it to a temp dir for isolation. The Dream Cycle and MCP server load `.env` via `python-dotenv`.

## Architecture

### Data flow

```
Capture → inbox → Dream Cycle ─┬─ fact      → entities / facts / relations
                               ├─ episodic  → atomic_notes (+ atomic_notes_vec)
                               ├─ ephemeral → intentions (48h TTL)   ← NON-exclusive: durable
                               │                                       entities are still extracted
                               └─ any URL   → fetch + Haiku summary → resources (searchable, SYN-21)
```

Routing is **non-exclusive**: one capture can produce entities + atomic_note + project_entries + an intention + a resource at once (`_process_entry`). URL-driven resource fetch runs for any capture, even a pure intention.

`import dream_cycle` resolves to the **package** `dream_cycle/`; the pipeline lives in `dream_cycle/cycle.py` and is exported as `run_dream_cycle` (also `python -m dream_cycle`). There is one cycle — the earlier two-implementation split has been merged.

### The Dream Cycle (`dream_cycle/cycle.py`)

Operates per inbox entry, with French prompts. Classifies each entry, then routes by `input_type`:

- **fact** → the 6-step graph pipeline below.
- **episodic** → `write_episodic_note`: stores raw content + summary + `entities_mentioned` in `atomic_notes` with `memory_strength=1.0`, and vectorizes it into `atomic_notes_vec`.
- **ephemeral** → `intentions` (48h TTL). Non-exclusive: durable entities in the same capture are still routed (SYN-58).
- **resource** → any URL in the capture is fetched (`httpx`) + extracted (stdlib `html.parser`, no trafilatura dep) + summarised (Haiku) + stored in `resources`, searchable via its embedded summary (`dream_cycle/resources.py`, SYN-21).

The 6 steps for facts:
1. **Classify** — Haiku tags `input_type` and extracts entities, facts (snake_case predicates + `persistence_value` 1–5), relations, summary. **Entity type vocab is dynamic** (SYN-58): the prompt reads `active_entity_types` at runtime (uncached block); an entity that fits no active type carries `type_proposal{value,reason}` instead of being mis-typed. Garde-fou: `type=project` only with a matching `project_entries` item.
2. **Resolve** — matches entities to existing rows (canonical name or alias); resolves relative dates to absolute via `dateparser` (date-like predicates only).
3. **Score** (`compute_confidence`) — evidence base (`explicit` 0.92 · `hedged` 0.65 · `implicit` 0.40) + existing/mention/persistence bonuses → [0,1]; `hedged` clamped to 0.84.
4. **Route** — **entity nodes are created on mention** (decoupled from fact confidence) if they pass `MIN_ENTITY_PERSISTENCE` (≥2) OR appear in a relation OR already exist. A vocab-gap entity is created `status='pending'` + an `entity_type_proposals` row. **Facts** are confidence-gated: > 0.85 → `facts`; 0.5–0.85 → `pending_facts`; < 0.5 → `review_queue`. Newly-created entities are scanned for duplicates → `entity_merge_proposals` (substring SYN-39, then embedding fallback SYN-61). All fact writes go through **`facts_store.insert_fact`**, which applies SYN-37 last-writes-wins: a single-valued predicate (`works_at`, `lives_in`, …) obsoletes the prior active fact (`obsoleted_at`/`obsoleted_by`) when the new one is ≥ as confident.
5. **Behavioral validation** — a pending fact corroborated by a new mention in the same run is promoted into `facts`.
6. **Vectorize** — embeds touched entities into `entities.embedding` (BLOB). Then **decay** (SYN-19/68): `apply_decay` + `apply_entity_decay` recompute `memory_strength` for all `atomic_notes` and `entities`.

Per-entry resilience: each entry is processed in isolation. An `anthropic.APIError` (no/invalid key, network) **aborts the whole run** and leaves entries queued for a retry; a content error on one entry marks just that entry `status='failed'` and the run continues. The API can auto-run the cycle debounced after captures (`SYNAPSE_AUTO_CYCLE`).

**Memory strength / graceful forgetting (SYN-19/68, `dream_cycle/decay.py`)**: `memory_strength = exp(-Δdays/τ)` recomputed cadence-independently (τ via `SYNAPSE_DECAY_TAU_DAYS`, default 30) for **both** `atomic_notes` (`apply_decay`, anchor `last_reactivated_at`) **and** `entities` (`apply_entity_decay`, anchor `last_mentioned`, SYN-68). Reactivation: a mention in a new capture is a strong bump; a `search_memory` hit is a light one. Runs at the end of each cycle + standalone `python -m dream_cycle.decay` (nightly cron for empty-inbox days).

**Living-map graph (SYN-66, `graph_layout.py` + `graph_clusters.py`)**: `GET /graph` assembles a projection (no new source of truth) — entities ∪ atomic_notes as nodes, relations + mentions as edges — then Louvain clustering (networkx), ForceAtlas2 layout persisted/incremental in `node_positions`, and batched+cached Haiku cluster labels (`cluster_labels`, keyed by a signature of the cluster's defining entities) + pure-Python convex hulls. A community must hold ≥`MIN_CLUSTER_SIZE` (3) nodes to become a region — smaller ones aren't forced into a zone; the frontend (SYN-64) floats them as orphans. On a full recompute, `semantic_edges` (SYN-64) adds embedding-kNN soft springs (top-4 cosine ≥ 0.80, weight `0.45×score`, **layout-only — never returned**) so vector-similar entities drift together; `semantic_layout=false` disables. **New dep: `networkx>=3.2`** (pure-Python; packages into the PyInstaller .dmg, unlike igraph/leidenalg). Visual mapping: size = `memory_strength`×`degree`, colour = `community_id`, saturation = `memory_strength`, position = `node_positions`. See `docs/ARCHITECTURE.md` §5.

**Lifecycle (SYN-37/59)**: `facts` and `entities` carry `archived_at` (user "filed away") and facts also `obsoleted_at`/`obsoleted_by` ("no longer true" — auto by SYN-37 supersede or manual). Read views hide them by default; `?include=archived,obsolete` (entity facts) and `?include_archived=true` (graph) opt them back in.

**Shared modules**: `entity_search.py` (entity/resource cosine search + composite-text helper, used by MCP search, merge fallback, `/similar`), `facts_store.py` (single source of fact writes + supersede), `dream_cycle/decay.py`, `dream_cycle/resources.py`, `graph_layout.py` (ForceAtlas2 + `node_positions`, SYN-69), `graph_clusters.py` (Haiku labels + hulls, SYN-70).

### Update 2026-06-12 — dogfood batch (SYN-77 → SYN-89)

- **Inbox diagnosability (SYN-77/78)**: `inbox.error` stores the per-entry failure reason (exposed on `/feed`); `POST /inbox/{id}/requeue` retries a failed entry; API startup marks orphan `running` cycle_runs as `error` (process died mid-run, guarded by `cycle.lock` freshness). Cycle fixes: `_intention_text()` coerces object/list `ephemeral_content`; classify `max_tokens` 1536→4096 + explicit `stop_reason` check.
- **Fiche edits (SYN-82/84)**: `PATCH /entity/{id}` also renames (old canonical_name kept as **alias** so the resolver still matches); `PATCH /fact/{id}` = user correction → `confidence 1.0` + `last_confirmed`; relation CRUD `POST/PATCH/DELETE /relation` (optional client id; `/entity` exposes `relations[].id`). User edits are **source of truth**.
- **Note kinds (SYN-85)**: `atomic_notes.kind` ∈ `note|task|event` + `event_date` (absolute, classifier-resolved), `event_recurring` (yearly), `archived_at` (user « rendre obsolète », `POST /atomic-note/{id}/archive|unarchive`). Tasks = retrievable backlog, **no due date/checkbox** — decay forgets them. Durable (task/event) notes **bypass the ephemeral gates** (pure-intention fast exit + SYN-58 anti-double-store), a project-routed note always **mentions its project**, and an entity anchoring a durable note passes the noise garde-fou (`anchors_durable_note`).
- **Fact categories (SYN-88)**: `facts.category` ∈ `identity|dates|work|places|relations|preferences|health|other`, assigned by the classifier, propagated through every write path via `insert_fact`. Clients group facts into collapsible sections.
- **Entity re-summary (SYN-89)**: the summary is **purely derived** (never user-edited). `entities.summary_stale` is set by every fact write (`insert_fact`) and fact-edit/lifecycle endpoints; `step_resummarize` rebuilds summaries from the **active** facts + relations (Haiku, `_RESUMMARY_SYSTEM`) for touched ∪ stale entities, then they're re-vectorized. Hard rule: summaries are **timeless** (absolute dates only — never « la semaine prochaine »); same rule in the extraction prompt. The cycle and the auto-scheduler also run on an empty inbox when stale summaries exist.
- **Alias-aware promotion (SYN-87)**: both pending-fact promotion paths (step5 + `validation.py`) resolve entities through `_find_existing_entity` (aliases included) — canonical-only lookup used to spawn duplicate shells.

### Update 2026-06-17 — weekly digest, reinforce, dated tasks, offline embeddings

- **Weekly digest (SYN-23)**: new module `dream_cycle/digest.py` — a weekly job (separate from the cycle) that condenses the past week + the week ahead into one durable `atomic_notes` row with **`kind='digest'`** (idempotent per ISO week, vectorized, `memory_strength=1.0`). `gather_week()` is pure SQL (offline-testable): new entities/facts/notes, *tendances* (most-mentioned entities over the window), forward-looking **dated events AND dated tasks** within 7 days (incl. recurring birthdays), and open undated tasks; `summarize_digest()` renders French markdown via Haiku with the **timeless rule** (absolute dates only); empty weeks are skipped. Run `python -m dream_cycle.digest [--dry-run]`; production = a weekly LaunchAgent (`fr.myffu.synapse.digest`, Sunday 23h, machine-specific plist). Endpoints `POST /digest/run` (`?days`,`?dry_run`) + `GET /digest/latest`.
- **Reinforce + dated tasks (SYN-23)**: `POST /atomic-note/{id}/reinforce` = user 👍 « keep » on a fading note → full reactivation (`last_reactivated_at`=now, `memory_strength`=1.0). `POST /atomic-note/{id}/date?event_date=&recurring=` sets/clears a note's date — a **task may carry an `event_date` (échéance) without becoming an event** (then surfaces under the digest's « à venir »). Classifier rule (d): a dated to-do stays `kind='task'`; rule (e) clarified — **event = an occurrence that HAPPENS vs task = a thing to DO**; `write_typed_note` stores `event_date`/`event_recurring` for `kind in (event, task)`.
- **`/graph` excludes digests (SYN-23)**: nodes now filter `kind != 'digest' AND archived_at IS NULL` (a digest mentions many entities → would hairball the map).
- **Offline « entités liées » (SYN-91)**: `GET /changes` ships each entity's embedding as base64 in `embedding_b64` (null until vectorized) — previously the raw BLOB was dropped. Lets the mobile replica compute the cosine « entités liées » **offline**. Cost ≈ 2 KB/entity in `/changes`; revisit with delta-sync as the base grows.
- **Map layout moved client-side (SYN-64)**: the mobile app now computes the living-map layout itself (a vis-network `forceAtlas2Based` port, `ForceLayout.kt`) — the backend `graph_layout.py` (ForceAtlas2 → `node_positions`) is unchanged but **advisory** for the mobile map.

### MCP tools (`mcp_server/server.py`)

- `add_to_inbox(content, source)` — raw capture
- `search_memory(query, limit)` — local vector search over `atomic_notes` (episodic), `entities` (graph) **and `resources`** (SYN-21), merged and score-sorted; falls back to `LIKE` keyword search if the vector path yields nothing. A hit lightly reactivates the surfaced notes (SYN-19).
- `list_recent(limit)` — recent inbox entries
- `run_dream_cycle()` — triggers the unified cycle (kept for testing; production is cron-driven)
- `get_entity(name)` — entity by canonical name or alias, with its facts and relations
- `list_pending()` — facts awaiting validation (`pending_facts`)
- `validate_fact(fact_id, confirmed, correction)` — confirm (→ `facts` at confidence 0.95) or reject a pending fact. Shares logic with the HTTP API via `dream_cycle/validation.py::record_and_apply_validation` (records an append-only `validation_events` row, then applies).

### HTTP API (`api/app.py`)

FastAPI app for the mobile/desktop clients (run `python -m api`, port 8000), **~38 endpoints**; the frozen contract is `openapi.json` (regenerate via `app.openapi()` when it changes — the app codes against it). Bearer auth via `SYNAPSE_API_TOKEN` (auth **disabled** if unset — dev). Core: `GET /health`, `POST /capture` (**idempotent on client UUID**), `GET /feed`, `GET /graph` (living-map SYN-66: base = entities+relations; opt-in flags `include_notes` adds atomic_notes as `n:<id>` nodes + mention edges, `cluster` → `community_id` (Louvain), `layout`/`relayout` → `x`/`y` (ForceAtlas2, persisted in `node_positions`), `clusters` → `{label, hull}` regions; filters `node_types`/`memory_strength_min`/`since`/`top_pct_per_cluster`/`include_isolated`/`max_nodes`), `GET /entity/{id}` (`?include=archived,obsolete`), `GET /atomic-note/{id}` (single note + `provenance_content`, SYN-64), `GET /pending`, `POST /pending/{id}/validate`, `POST /dream-cycle/run` (file lock + `cycle_runs`), `GET /dream-cycle/last`, `GET /changes`, `GET /atomic-notes`, `GET /projects`, `GET /project/{id}/state`, project-entry ops. **Entity-graph endpoints**: `GET /entity/{id}/similar` (SYN-62), `GET/POST /entity-type-proposals*` (SYN-58), `GET/POST /merge-proposals*` (SYN-39), `POST /entity|fact/{id}/archive|unarchive` + `/fact/{id}/obsolete|restore` (SYN-59). **Digest/note endpoints (SYN-23)**: `POST /digest/run`, `GET /digest/latest`, `POST /atomic-note/{id}/reinforce`, `POST /atomic-note/{id}/date`. **SYN-91**: `/changes` carries `embedding_b64` per entity (offline « entités liées »). Per-request apsw connections. Sync model: captures carry `id`/`device_id`/`captured_at`, validations are append-only events → state rebuildable (see `docs/ARCHITECTURE.md`).

### Embedding strategy

**Fully local, no PyTorch, no API call.** `embeddings.py` uses **fastembed** (ONNX runtime) with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~50 languages incl. French — set via `EMBEDDING_MODEL` in `config.py`). The model loads lazily as a process-level singleton (~220 MB, downloaded once, then offline). `embed_text(text, client=None)` returns an L2-normalized serialized vector; the `client` arg is ignored (kept for backward compat with the old API-based signature). Run `python reembed.py` after changing `EMBEDDING_MODEL` to regenerate existing vectors.

Vectors are normalized so the sqlite-vec `vec0` **L2 distance** stays in [0, 2] and is monotonic with cosine — keeping the `score = 1 - distance/2` mapping valid. With this model, related notes land ~0.9 and unrelated ~1.4 (the visualizer edge threshold is 1.1).

Search is hybrid: vector k-NN via sqlite-vec first (no API key needed — embeddings are local), falling back to `LIKE %query%` across `atomic_notes` and `inbox` only if the vector path errors or returns nothing.

### Database

SQLite at `~/.synapse/synapse.db`, opened via `apsw` (stdlib `sqlite3` on macOS can't load extensions). The sqlite-vec extension is loaded at connection time. Schema and connection helpers (`get_connection`, `cursor_to_dicts`, `first_row`, `init_db`) live in `db/__init__.py`; `init_db()` is idempotent (`CREATE TABLE IF NOT EXISTS` + best-effort `ALTER TABLE` migrations wrapped in try/except) and is called at MCP startup and at the top of the Dream Cycle.

Tables:
- `inbox` — raw captures; `processed_at` NULL until consumed. Sync: `client_id` (UNIQUE partial index → idempotent), `device_id`, `captured_at`, `status`
- `validation_events` — append-only log of validate/reject decisions
- `cycle_runs` — one row per Dream Cycle run (stats for `GET /dream-cycle/last`)
- `atomic_notes` / `atomic_notes_vec` — episodic memory; vec0 rowid mirrors `atomic_notes.id`. Columns: `summary`, `entities_mentioned` (JSON), `memory_strength` + `last_reactivated_at` (SYN-19)
- `entities`, `facts`, `relations`, `resources` — entity graph (UUID ids). `entities.embedding` raw BLOB (manual cosine — UUID ids can't use int-rowid vec0). Lifecycle cols: `entities.status` (active|pending|archived, SYN-58) + `entities.archived_at`, `facts.archived_at`/`obsoleted_at`/`obsoleted_by` (SYN-37/59). `entities.memory_strength` (decay, SYN-68). `resources` now has `url`/`content`/`summary`/`embedding`/`fetched_at` (SYN-21, unique index on `url`)
- `node_positions` (carte: `node_id`,`x`,`y` — ForceAtlas2, SYN-69), `cluster_labels` (carte: `signature`,`label` — cached Haiku labels, SYN-70) — projection caches for the living map, never authoritative
- `active_entity_types` (live type vocab: 6 builtin + user-validated) + `entity_type_proposals` (SYN-58)
- `entity_merge_proposals` (SYN-39) — dedup queue; `merged_into_id`/`merged_at` soft-link on `entities`
- `pending_facts`, `review_queue`, `intentions` — routing buckets
- `project_entries`, `project_state`, `project_state_versions` — project aggregate (SYN-40)
- `knowledge_graph` — legacy, unused

vec0 virtual tables don't support `COUNT(*)`; count by point-looking-up each rowid (see `visualizer/app.py::get_stats`).

### Config (`config.py`)

```python
BASE_DIR = Path(os.getenv("SYNAPSE_HOME", Path.home() / ".synapse"))
DB_PATH = BASE_DIR / "synapse.db"
EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # local fastembed
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Dream Cycle reasoning only
```

**Tunable env vars** (consumed by the cycle): `SYNAPSE_AUTO_CYCLE`, `SYNAPSE_CYCLE_DEBOUNCE_SECONDS` (120), `SYNAPSE_REFINEMENT_THRESHOLD`, `SYNAPSE_MERGE_EMBEDDING_THRESHOLD` (0.85, SYN-61), `SYNAPSE_DECAY_TAU_DAYS` (30, SYN-19). Single-valued predicates list (SYN-37): `facts_store.SINGLE_VALUED_PREDICATES`.

### Visualizer (`visualizer/`)

FastAPI app (`app.py`) serving `/api/nodes`, `/api/edges`, `/api/stats`, `/api/note/{id}`, backed by the same SQLite DB. It reads the `atomic_notes` (episodic) world. Edges are computed live from vector similarity (k-NN per note, L2 distance threshold 1.1). Static frontend is a D3.js force-directed graph (`static/graph.js`). Note: it does not yet render the entity graph — wiring that to `/api/nodes` is a natural next step.

## Clients

The HTTP API has known clients beyond MCP:
- A **mobile app** (Android + iOS, Kotlin Multiplatform + Compose Multiplatform) lives in a separate **private/proprietary** repo `synapse-app` and talks to this backend over the LAN (`POST /capture`, `GET /feed`, `GET /changes`, `POST /pending/{id}/validate`). The frozen contract it codes against is the generated `openapi.json` in this repo. Keep that file up to date when endpoints change.
- (Future) a desktop app and a managed sync relay are part of the wider product but live outside this repo.

The roadmap (Phase C — memory_strength decay, coreference window, resource fetch, weekly digest, etc.) is tracked in an **internal task tracker outside this repo**. Don't reference internal tooling URLs from this file (public repo).

## Local-only engine map

If `docs/engine-map.html` is present in the working tree, it is a personal, gitignored visual map — three tabs (Dream Cycle pipeline · data model · **living-map graph model**, SYN-66) with clickable details for prompts, tunable thresholds, schema. Keep it in sync when you change:
- Tunable constants in `dream_cycle/cycle.py` (e.g. `MIN_ENTITY_PERSISTENCE`, `_EVIDENCE_BASE`, bucket thresholds in `step4_route`).
- Env vars consumed by the cycle (`SYNAPSE_AUTO_CYCLE`, `SYNAPSE_CYCLE_DEBOUNCE_SECONDS`, `SYNAPSE_REFINEMENT_THRESHOLD`).
- Classifier prompt rules (`_SYSTEM_CLASSIFIER`) or sub-routing rules (atomic_note, project_entries, ephemeral).
- Schema changes in `db/__init__.py` (new tables, new columns, new soft-link semantics).

The local skill `engine-map-sync` (in `.claude/skills/`, gitignored) documents exactly which DOM block in the HTML each constant maps to. If the file isn't present, ignore this section.
