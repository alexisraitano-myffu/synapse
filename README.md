# Synapse

**A local-first personal semantic memory system.** Capture raw notes; an AI cleans, links and structures them into a queryable knowledge graph + episodic memory — all on your own machine. **No cloud: your data never leaves your devices.**

Synapse runs on a Mac (the "brain"), exposes its memory to AI agents over **MCP** (Claude Desktop / Claude Code) and to apps over a small **HTTP API**. Consolidation is done by a batched, sleep-like "Dream Cycle" using Claude Haiku; embeddings are **fully local** (fastembed/ONNX, no API call, no PyTorch).

> Philosophy: *capture passively, process actively.* Open source (Apache-2.0). The optional mobile app is a separate project.

---

## Table of contents

- [How it works](#how-it-works)
- [Core ideas](#core-ideas)
- [Quick start](#quick-start)
- [HTTP API](#http-api)
- [MCP tools](#mcp-tools)
- [Configuration](#configuration)
- [Tests](#tests)
- [Architecture & docs](#architecture--docs)
- [License](#license)

---

## How it works

```
Capture → inbox ──(batched "sleep" consolidation)──> Dream Cycle ─┬─ fact      → entities / facts / relations   (semantic memory)
                                                                  ├─ episodic  → atomic_notes (note | task | event | digest)  (episodic memory)
                                                                  ├─ ephemeral → intentions (48h TTL)
                                                                  └─ any URL   → fetch + summarize → resources (searchable)
                                                            ↓
                                          MCP tools  ·  HTTP API  ·  living-map graph
```

Routing is **non-exclusive**: one capture can produce entities + an atomic note + a project entry + an intention + a resource at once.

The pipeline (one implementation, `dream_cycle/cycle.py`): **classify → resolve → score → route → behavioral-validate → vectorize**, then a decay pass. Each entry is processed in isolation (a content error fails only that entry; an API error aborts the run and re-queues everything).

## Core ideas

- **Two timescales, like sleep consolidation (SYN-93).** Captures buffer during the "day"; consolidation runs in a batched "sleep" pass — twice daily (`SYNAPSE_CONSOLIDATION_HOURS`, default midnight+noon) or when a size valve trips (`SYNAPSE_CONSOLIDATION_MAX_QUEUED`, default 30). A **working-memory** context (a read-only transcript of the batch + the last 24h) lets the classifier resolve coreference ("he / that project / yesterday") *across* captures. The scheduled pass uses the **Message Batches API (−50%)**; a startup/wake catch-up recovers a slot missed while the Mac slept.

- **Entities on mention, facts on confidence.** An **entity** is created the moment it's mentioned (if it clears `MIN_ENTITY_PERSISTENCE` or appears in a relation). Its **facts** are confidence-scored from an evidence base (explicit 0.92 · hedged 0.65 · implicit 0.40, + bonuses) and either consolidated (>0.85), queued for validation (0.5–0.85), or sent to a review queue. Each fact carries a thematic **category** so clients can group long lists.

- **Nothing uncertain is dropped silently — the "À valider" queue.** Low-confidence tasks/events (`review_status`), low-confidence relations, project-attach proposals, entity-type proposals and merge proposals are all set aside, hidden from every read surface, and surfaced only for explicit confirm/reject.

- **A relation is the canonical form of a fact about two entities.** "Audric is Alexis's cousin" yields a single traversable relation (visible on *both* fiches via `relations` + `relations_incoming`), not a duplicated fact — with a defensive de-dup in routing. Relations are confidence-gated like tasks. Serendipity (embedding proximity) runs on a separate channel and is untouched by this.

- **Notes have kinds.** Free reflections, retrievable **tasks** (deliberately no due-date/checkbox — memory decay forgets them, a one-tap archive dismisses them; a task *may* carry an optional `event_date`), dated **events** (absolute dates, yearly recurrence), and weekly **digests**.

- **Graceful forgetting (SYN-19/68).** `memory_strength = exp(-Δdays/τ)` (τ default 30d) recomputed for both notes and entities; a new mention is a strong reactivation, a search hit a light one.

- **Entity summaries are derived, never stored opinions.** Regenerated from the *active* facts + relations whenever they change, under a hard *timeless* rule (absolute dates only — never "next week"). User edits (rename → old name kept as alias, fact corrections, relation CRUD) are authoritative and flow into the next regeneration.

- **A living map.** `GET /graph` assembles a projection (entities ∪ atomic-notes as nodes, relations + mentions as edges) with Louvain communities, ForceAtlas2 layout, and cached Haiku region labels — no new source of truth, fully rebuildable. Node size = `memory_strength × degree`, colour = community.

- **Embeddings are local.** `fastembed` (ONNX, `paraphrase-multilingual-MiniLM-L12-v2`, 384-d, ~50 languages) → SQLite + `sqlite-vec`. No PyTorch, no embedding server, no network.

- **Built for offline, multi-device.** Captures carry a client UUID + device id (idempotent, offline-safe); validations are append-only events; derived state is rebuildable → replication without a multi-master database.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # used only by the Dream Cycle (reasoning), not for embeddings
```

**Run the HTTP API** (backend for apps; FastAPI on `0.0.0.0:8000`):
```bash
python -m api                       # auth via SYNAPSE_API_TOKEN; SYNAPSE_AUTO_CYCLE=1 for batched auto-consolidation
```

**Process the inbox on demand** (the Dream Cycle):
```bash
python -m dream_cycle               # --dry-run / --verbose available
```

**MCP server** (Claude Desktop / Claude Code, over stdio):
```bash
python mcp_server/server.py
```

**Web visualizer** (D3 knowledge graph at http://127.0.0.1:8080):
```bash
python visualizer/app.py
```

**Weekly digest** (condense the past + coming week into one durable note; production trigger = a Monday-08h LaunchAgent, self-healed by the API if a run is missed):
```bash
python -m dream_cycle.digest        # --dry-run to preview
```

**Run it as a service (macOS).** A user LaunchAgent with `RunAtLoad` + `KeepAlive` pointing at `.venv/bin/python -m api` (working directory = this repo so `.env` is picked up) gives auto-start at login and auto-restart on crash. Designed for LAN / private mesh (Tailscale).

## HTTP API

Bearer auth (`SYNAPSE_API_TOKEN`; disabled if unset = dev). **57 endpoints**; the frozen contract is [`openapi.json`](openapi.json). Highlights:

- **Capture / inbox** — `POST /capture` (idempotent on a client UUID) · `GET /feed` (incl. per-entry failure reason) · `POST /inbox/{id}/requeue` · `POST /inbox/{id}/reprocess` (replay one capture through the cycle after a prompt fix; keeps entities).
- **Graph / living map** — `GET /graph` with opt-in layers: atomic-note nodes, Louvain clusters, ForceAtlas2 positions, labelled regions, anti-hairball filters.
- **Entities / facts / relations** — `GET /entity/{id}` (returns `relations` + `relations_incoming`) · `PATCH /entity/{id}` (rename → alias) · `PATCH /fact/{id}` · `POST/PATCH/DELETE /relation` · `GET /entity/{id}/similar` · archive / obsolete / restore.
- **"À valider"** — `GET /pending` + `/pending/{id}/validate` · `GET /atomic-notes?review_status=pending` + `/atomic-note/{id}/confirm` · `GET /relations/pending` + `/relation/{id}/confirm` · merge-proposals · entity-type-proposals · project-attach-proposals.
- **Notes / projects** — `GET /atomic-notes` · `POST /atomic-note/{id}/reinforce|date|archive|promote-to-project` · `GET /projects` · `GET /project/{id}/state`.
- **Provenance** — `GET /capture/{id}/generated` (what the cycle produced from a capture).
- **Cycle / digest** — `POST /dream-cycle/run` · `GET /dream-cycle/last` · `POST /digest/run` · `GET /digest/latest`.
- **Replication** — `GET /changes?since=` (derived state + per-entity `embedding_b64` for offline "related entities").
- **Config / owner** — `GET/PUT /config` · `PUT /config/anthropic-key` · `GET/PUT /owner`.

## MCP tools

`add_to_inbox` · `search_memory` (local vector over notes + entities + resources, text fallback) · `list_recent` · `run_dream_cycle` · `get_entity` · `list_pending` · `validate_fact`.

## Configuration

`config.py`: `SYNAPSE_HOME` (DB location, default `~/.synapse`), `EMBEDDING_MODEL` (local), `CLAUDE_MODEL` (Dream Cycle reasoning — Haiku). The cycle also reads `SYNAPSE_API_TOKEN`, `SYNAPSE_API_PORT`, `SYNAPSE_AUTO_CYCLE`, `SYNAPSE_CONSOLIDATION_HOURS` (`"0,12"`), `SYNAPSE_CONSOLIDATION_MAX_QUEUED` (30), `SYNAPSE_REVIEW_CONFIDENCE_THRESHOLD` (0.7), `SYNAPSE_DECAY_TAU_DAYS` (30), `SYNAPSE_MERGE_EMBEDDING_THRESHOLD` (0.85).

**Anthropic client & fuel proxy.** `anthropic_client.py` is the single place that builds the client. A normal `sk-ant-…` key calls Anthropic directly. A beta **fuel token** (`syn-fuel-…`) routes through a Cloudflare Worker proxy so closed-beta testers can borrow the maintainer's credits without holding a key; the real key lives only on the Worker. Override the proxy with `SYNAPSE_FUEL_BASE_URL` (empty to disable). The whole cycle runs on Haiku.

## Tests

```bash
pytest                              # offline suites run without an API key
```
`test_embeddings.py`, `test_cycle.py`, `test_api.py` are fully offline; `test_dream_cycle.py` exercises the live classify→route pipeline and is skipped unless `ANTHROPIC_API_KEY` is set.

## Architecture & docs

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the full technical spec: deployment topology, the Dream Cycle, the two-timescale model, the "À valider" queue, the explicit graph (facts vs relations), the living map, the sync model, and the tunable levers.
- **[docs/engine-map.html](docs/engine-map.html)** — an interactive, clickable map of the engine (Dream Cycle pipeline · data model · living-map model) with the prompts, tunable thresholds and schema behind each node. Open it in a browser: `open docs/engine-map.html`.

**Clients.** A mobile app (Android + iOS, Kotlin Multiplatform / Compose Multiplatform) lives in a separate private repo and talks to this backend over the LAN, coding against the generated `openapi.json` here — keep that file up to date when endpoints change.

## License

Apache-2.0. See [LICENSE](LICENSE).
