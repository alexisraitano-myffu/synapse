# Synapse

**A local-first personal semantic memory system.** Capture raw notes, and an AI cleans, links and structures them into a queryable knowledge graph plus episodic memory, all on your own machines. **No cloud: your data never leaves your devices.**

Synapse is built to run **on any machine you own**. The **brain** — classification, routing, confidence scoring, embeddings, storage, vector search, decay and P2P sync — is a single compiled **Rust core, [`synapse-core`](https://github.com/alexisraitano-myffu/synapse-core)**, written once and shared across every platform. **This repository is the desktop host**: it embeds that core (as a PyO3 wheel, `synapse_core`) inside a cross-platform **Python service** that runs in the background on a **desktop (macOS or Windows)**, exposing the memory to AI agents over **MCP** (Claude Desktop, Claude Code) and to the apps over a small **HTTP API**. A **desktop app** (macOS, Windows) and a **mobile app** (iOS, Android) are the clients that capture and browse over the LAN — the mobile apps embed the very same core via UniFFI. Consolidation runs in a batched, sleep-like "Dream Cycle" using Claude Haiku; embeddings are **fully local** (ONNX, no API call, no PyTorch).

> Philosophy: *capture passively, process actively.* Open source (Apache-2.0). The shared brain lives in **[`synapse-core`](https://github.com/alexisraitano-myffu/synapse-core)** (Rust, Apache-2.0) — one implementation, zero logic divergence between platforms; this repo hosts it on the desktop. The apps live in a separate project and talk to this host only through the documented HTTP API.

---

## Table of contents

- [How it works](#how-it-works)
- [Core ideas](#core-ideas)
- [Where it runs](#where-it-runs)
- [Quick start](#quick-start)
- [HTTP API](#http-api)
- [MCP tools](#mcp-tools)
- [Configuration](#configuration)
- [Tests](#tests)
- [Architecture & docs](#architecture--docs)
- [License](#license)

## How it works

```
Capture -> inbox --(batched "sleep" consolidation)--> Dream Cycle
    |-- fact      -> entities / facts / relations           (semantic memory)
    |-- episodic  -> atomic_notes (note | task | event | digest)  (episodic memory)
    |-- ephemeral -> intentions (48h TTL)
    |-- any URL   -> fetch + summarize -> resources (searchable)
                                |
              MCP tools  .  HTTP API  .  living-map graph
```

Routing is **non-exclusive**: one capture can produce entities, an atomic note, a project entry, an intention and a resource at once.

The pipeline logic — classify, resolve, score, route, behavioral-validate, vectorize, then a decay pass — lives **once in the Rust core** (`synapse-core`); `dream_cycle/cycle.py` is the thin host orchestrator that drives the core and persists what it returns. Each entry is processed in isolation (a content error fails only that entry; an API error aborts the run and re-queues everything).

## Core ideas

- **Two timescales, like sleep consolidation (SYN-93).** Captures buffer during the "day"; consolidation runs in a batched "sleep" pass, either twice daily (`SYNAPSE_CONSOLIDATION_HOURS`, default midnight and noon) or when a size valve trips (`SYNAPSE_CONSOLIDATION_MAX_QUEUED`, default 30). A **working-memory** context (a read-only transcript of the batch plus the last 24h) lets the classifier resolve coreference ("he", "that project", "yesterday") *across* captures. The scheduled pass uses the **Message Batches API** for roughly half the cost, and a startup/wake catch-up recovers a slot missed while the machine slept.

- **Entities on mention, facts on confidence.** An **entity** is created the moment it is mentioned (if it clears `MIN_ENTITY_PERSISTENCE` or appears in a relation). Its **facts** are confidence-scored from an evidence base (explicit 0.92, hedged 0.65, implicit 0.40, plus bonuses) and are then either consolidated (above 0.85), queued for validation (0.5 to 0.85), or sent to a review queue. Each fact carries a thematic **category** so clients can group long lists.

- **Nothing uncertain is dropped silently: the "À valider" queue.** Low-confidence tasks and events (`review_status`), low-confidence relations, project-attach proposals, entity-type proposals and merge proposals are all set aside, hidden from every read surface, and surfaced only for explicit confirm or reject.

- **A relation is the canonical form of a fact about two entities.** "Audric is Alexis's cousin" yields a single traversable relation (visible on *both* fiches via `relations` and `relations_incoming`), not a duplicated fact, with a defensive de-dup in routing. Relations are confidence-gated like tasks. Serendipity (embedding proximity) runs on a separate channel and is untouched by this.

- **Notes have kinds.** Free reflections, retrievable **tasks** (deliberately no due-date or checkbox: memory decay forgets them, a one-tap archive dismisses them, and a task *may* carry an optional `event_date`), dated **events** (absolute dates, yearly recurrence), and weekly **digests**.

- **Graceful forgetting (SYN-19/68).** `memory_strength = exp(-Δdays/τ)` (τ default 30d) is recomputed for both notes and entities; a new mention is a strong reactivation, a search hit a light one.

- **Entity summaries are derived, never stored opinions.** They are regenerated from the *active* facts and relations whenever those change, under a hard *timeless* rule (absolute dates only, never "next week"). User edits (rename keeps the old name as an alias, fact corrections, relation CRUD) are authoritative and flow into the next regeneration.

- **A living map.** `GET /graph` assembles a projection (entities and atomic-notes as nodes, relations and mentions as edges) with Louvain communities, a ForceAtlas2 layout and cached Haiku region labels. It is no new source of truth and is fully rebuildable. Node size is `memory_strength × degree`, colour is the community.

- **Embeddings are local, and computed in the core.** The Rust core embeds text (ONNX, `paraphrase-multilingual-MiniLM-L12-v2`, 384-d, about 50 languages) and stores vectors in SQLite plus `sqlite-vec` — one implementation shared by desktop and mobile, so vectors are byte-identical everywhere. No PyTorch, no embedding server, no network. The model files are **data** (`~/.synapse/models/…`), never compiled in (App Store rule 2.5.2).

- **Built for offline, multi-device.** Captures carry a client UUID and device id (idempotent, offline-safe); validations are append-only events; derived state is rebuildable, so replication works without a multi-master database.

## Where it runs

- **Host (this repo):** a cross-platform **Python** service that embeds the Rust core. It runs as an always-on background service on a desktop, on **macOS** (a user LaunchAgent) or **Windows** (a scheduled task at logon). It binds the LAN so the apps can reach it, and can be exposed off-LAN via a private mesh (Tailscale).
- **Desktop app:** macOS and Windows. The distribution build bundles this engine and installs it for you, so a tester runs a single installer.
- **Mobile app:** iOS and Android, a capture client plus an offline read-replica of the derived state.

The apps live in a separate repository and code against the generated `openapi.json` here.

## Quick start

Run the engine directly from source (any desktop OS with Python):

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...  # used only by the Dream Cycle (reasoning), not for embeddings
```

The brain ships as the **`synapse_core` wheel**, which is **not on PyPI yet** — build it once from the [`synapse-core`](https://github.com/alexisraitano-myffu/synapse-core) repo and install it into this venv (it must be the *only* SQLite library in the process):

```bash
git clone https://github.com/alexisraitano-myffu/synapse-core
cd synapse-core/crates/synapse-core-py && maturin build --release
pip install target/wheels/synapse_core-*.whl   # then cd back to this repo
```

**Run the HTTP API** (backend for the apps; FastAPI on `0.0.0.0:8000`):
```bash
python -m api                        # auth via SYNAPSE_API_TOKEN; SYNAPSE_AUTO_CYCLE=1 for batched auto-consolidation
```

**Process the inbox on demand** (the Dream Cycle):
```bash
python -m dream_cycle                # --dry-run / --verbose available
```

**MCP server** (Claude Desktop, Claude Code, over stdio):
```bash
python mcp_server/server.py
```

**Web visualizer** (D3 knowledge graph at http://127.0.0.1:8080):
```bash
python visualizer/app.py
```

**Weekly digest** (condense the past and coming week into one durable note; the production trigger is a Monday-08h scheduled job, self-healed by the API if a run is missed):
```bash
python -m dream_cycle.digest         # --dry-run to preview
```

**Run it as a background service.** On **macOS**, a user LaunchAgent with `RunAtLoad` and `KeepAlive` pointing at `.venv/bin/python -m api` (working directory set to this repo so `.env` is picked up) gives auto-start at login and auto-restart on crash. On **Windows**, a scheduled task at logon launching `python -m api` does the same. In practice, the desktop app installs and manages this service for you.

## HTTP API

Bearer auth (`SYNAPSE_API_TOKEN`; disabled if unset, for dev). **57 endpoints**; the frozen contract is [`openapi.json`](openapi.json). Highlights:

- **Capture / inbox:** `POST /capture` (idempotent on a client UUID), `GET /feed` (includes the per-entry failure reason), `POST /inbox/{id}/requeue`, `POST /inbox/{id}/reprocess` (replay one capture through the cycle after a prompt fix; keeps entities).
- **Graph / living map:** `GET /graph` with opt-in layers (atomic-note nodes, Louvain clusters, ForceAtlas2 positions, labelled regions, anti-hairball filters).
- **Entities / facts / relations:** `GET /entity/{id}` (returns `relations` and `relations_incoming`), `PATCH /entity/{id}` (rename keeps an alias), `PATCH /fact/{id}`, `POST/PATCH/DELETE /relation`, `GET /entity/{id}/similar`, archive / obsolete / restore.
- **"À valider":** `GET /pending` and `/pending/{id}/validate`, `GET /atomic-notes?review_status=pending` and `/atomic-note/{id}/confirm`, `GET /relations/pending` and `/relation/{id}/confirm`, merge-proposals, entity-type-proposals, project-attach-proposals.
- **Notes / projects:** `GET /atomic-notes`, `POST /atomic-note/{id}/reinforce|date|archive|promote-to-project`, `GET /projects`, `GET /project/{id}/state`.
- **Provenance:** `GET /capture/{id}/generated` (what the cycle produced from a capture).
- **Cycle / digest:** `POST /dream-cycle/run`, `GET /dream-cycle/last`, `POST /digest/run`, `GET /digest/latest`.
- **Replication:** `GET /changes?since=` (derived state plus a per-entity `embedding_b64` for offline "related entities").
- **Config / owner:** `GET/PUT /config`, `PUT /config/anthropic-key`, `GET/PUT /owner`.

## MCP tools

`add_to_inbox`, `search_memory` (local vector over notes, entities and resources, with a text fallback), `list_recent`, `run_dream_cycle`, `get_entity`, `list_pending`, `validate_fact`.

## Configuration

`config.py`: `SYNAPSE_HOME` (DB location, default `~/.synapse`), `EMBEDDING_MODEL` (local), `CLAUDE_MODEL` (Dream Cycle reasoning, Haiku). The cycle also reads `SYNAPSE_API_TOKEN`, `SYNAPSE_API_PORT`, `SYNAPSE_AUTO_CYCLE`, `SYNAPSE_CONSOLIDATION_HOURS` (`"0,12"`), `SYNAPSE_CONSOLIDATION_MAX_QUEUED` (30), `SYNAPSE_REVIEW_CONFIDENCE_THRESHOLD` (0.7), `SYNAPSE_DECAY_TAU_DAYS` (30), `SYNAPSE_MERGE_EMBEDDING_THRESHOLD` (0.85).

**Anthropic client and fuel proxy.** `anthropic_client.py` is the single place that builds the client. A normal `sk-ant-…` key calls Anthropic directly. A beta **fuel token** (`syn-fuel-…`) routes through a Cloudflare Worker proxy so closed-beta testers can borrow the maintainer's credits without holding a key; the real key lives only on the Worker. Override the proxy with `SYNAPSE_FUEL_BASE_URL` (empty to disable). The whole cycle runs on Haiku.

## Tests

```bash
pytest                               # offline suites run without an API key
```
`test_embeddings.py`, `test_cycle.py`, `test_api.py` are fully offline; `test_dream_cycle.py` exercises the live classify-to-route pipeline and is skipped unless `ANTHROPIC_API_KEY` is set.

## Architecture & docs

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md):** the full technical spec, covering deployment topology, the Dream Cycle, the two-timescale model, the "À valider" queue, the explicit graph (facts vs relations), the living map, the sync model, and the tunable levers.
- **[docs/engine-map.html](docs/engine-map.html):** an interactive, clickable map of the engine (Dream Cycle pipeline, data model, living-map model) with the prompts, tunable thresholds and schema behind each node. Open it in a browser: `open docs/engine-map.html`.
- **[`synapse-core`](https://github.com/alexisraitano-myffu/synapse-core):** the shared **Rust brain** this host embeds — embeddings, storage, vector search, routing, decay, summaries, LLM orchestration and CRDT sync. Consumed here as the `synapse_core` PyO3 wheel, and by the mobile apps via UniFFI. One implementation, every platform.

**Clients.** The desktop and mobile apps (Kotlin Multiplatform / Compose Multiplatform) live in a separate private repository and talk to this engine over the LAN, coding against the generated `openapi.json` here. Keep that file up to date when endpoints change.

## License

Apache-2.0. See [LICENSE](LICENSE).
