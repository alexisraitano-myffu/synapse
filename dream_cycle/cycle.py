#!/usr/bin/env python3
"""
Synapse Dream Cycle — unified pipeline.

Per inbox entry, Claude classifies the input and routes it:
  - fact      → entity graph (entities / facts / relations), confidence-scored
  - episodic  → atomic_notes (episodic memory, vectorized for search)
  - ephemeral → intentions (short TTL)
  - resource  → (currently routed like fact; fetch+summary is a future step)

6 steps for facts: classify → resolve → score → route → behavioral-validate → vectorize.

Entity creation is decoupled from fact confidence: an entity NODE is created as
soon as it is mentioned (with an anti-pollution garde-fou — see MIN_ENTITY_PERSISTENCE),
while its FACTS remain confidence-gated (pending until corroborated/validated).
"""

import argparse
import json
import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic

from config import CLAUDE_MODEL
from db import get_connection, cursor_to_dicts, first_row, init_db
from embeddings import embed_text

try:
    import dateparser
    _HAS_DATEPARSER = True
except ImportError:
    _HAS_DATEPARSER = False

_TODAY = date.today().isoformat()


# ── Claude client ──────────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


# ── Step 1 — Classifier ────────────────────────────────────────────────────────

_SYSTEM_CLASSIFIER = """\
Tu es un extracteur de mémoire pour un second cerveau personnel.
Analyse l'entrée et retourne UNIQUEMENT un JSON valide (sans markdown) :
{{
  "input_type": "fact|episodic|ephemeral|resource",
  "entities": [
    {{
      "canonical_name": "string",
      "type": "person|place|project|concept",
      "aliases": ["string"],
      "summary": "string (1 phrase qui décrit cette entité, ou null si rien de notable)",
      "attributes": {{"clé": "valeur"}},
      "facts": [
        {{
          "predicate": "string (snake_case ex: has_birthday, works_at, lives_in)",
          "value": "string",
          "persistence_value": 1
        }}
      ]
    }}
  ],
  "relations": [
    {{
      "from": "canonical_name",
      "predicate": "string",
      "to": "canonical_name"
    }}
  ],
  "summary": "string (résumé en 1 phrase)",
  "is_ephemeral": false,
  "ephemeral_content": null
}}
Règles persistence_value :
5 = permanent (date naissance, lien familial, prénom)
4 = stable modifiable (lieu de travail, adresse)
3 = état actuel (projet en cours)
2 = contextuel (événement ponctuel)
1 = bruit (mention passagère)
Résous les dates relatives vers des dates absolues.
La date d'aujourd'hui est : {today}.\
"""


def step1_classify(entry: dict, client: anthropic.Anthropic, verbose: bool = False) -> dict:
    system = _SYSTEM_CLASSIFIER.format(today=_TODAY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": entry["content"]}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    result = json.loads(raw)
    if verbose:
        u = response.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0)
        print(f"    [classify] type={result.get('input_type')} "
              f"entities={len(result.get('entities', []))} "
              f"tokens={u.input_tokens}/{u.output_tokens}"
              + (f" cache_hit={cache_read}" if cache_read else ""))
    return result


# ── Step 2 — Resolver ─────────────────────────────────────────────────────────

def _resolve_date(value: str) -> str:
    if not _HAS_DATEPARSER:
        return value
    parsed = dateparser.parse(
        value,
        settings={"PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False},
    )
    return parsed.date().isoformat() if parsed else value


def _find_existing_entity(canonical_name: str, aliases: list[str], conn) -> dict | None:
    row = first_row(conn.execute(
        "SELECT * FROM entities WHERE LOWER(canonical_name) = LOWER(?)", (canonical_name,)
    ))
    if row:
        return row

    search_names = {n.lower() for n in [canonical_name] + aliases}
    for entity in cursor_to_dicts(conn.execute("SELECT * FROM entities")):
        try:
            entity_aliases = json.loads(entity.get("aliases", "[]"))
        except (ValueError, TypeError):
            entity_aliases = []
        existing_names = {entity["canonical_name"].lower()} | {a.lower() for a in entity_aliases}
        if search_names & existing_names:
            return entity
    return None


def step2_resolve(classified: dict, conn, verbose: bool = False) -> dict:
    resolved_entities = []
    for entity_data in classified.get("entities", []):
        aliases = entity_data.get("aliases", [])
        existing = _find_existing_entity(entity_data["canonical_name"], aliases, conn)

        resolved_facts = []
        for fact in entity_data.get("facts", []):
            value = fact["value"]
            if any(kw in fact["predicate"] for kw in
                   ("birthday", "birth", "date", "born", "anniversary", "anniversaire")):
                value = _resolve_date(value)
            resolved_facts.append({**fact, "value": value})

        resolved_entities.append({
            **entity_data,
            "facts": resolved_facts,
            "existing_entity": existing,
        })
        if verbose and existing:
            print(f"    [resolve] '{entity_data['canonical_name']}' → existing id={existing['id']}")

    return {**classified, "resolved_entities": resolved_entities}


# ── Step 3 — Confidence ────────────────────────────────────────────────────────

_PERSISTENCE_BONUS = {5: 0.2, 4: 0.15, 3: 0.05, 2: 0.0, 1: -0.1}

# Anti-pollution garde-fou for entity creation: an entity is created on mention
# only if it carries at least this much persistence in one of its facts (i.e. it
# is more than pure noise) — unless it already exists or appears in a relation.
# Tune UP to be stricter (fewer entities), DOWN to capture more. 1 = create for
# everything mentioned (noisiest), 2 = skip pure "mention passagère".
MIN_ENTITY_PERSISTENCE = 2


def compute_confidence(
    fact: dict,
    has_explicit_statement: bool,
    context_supports: bool,
    mention_count: int,
) -> float:
    score = 0.0
    if has_explicit_statement:
        score += 0.5
    if context_supports:
        score += 0.3
    score += min(0.2, mention_count * 0.05)
    score += _PERSISTENCE_BONUS.get(fact.get("persistence_value", 3), 0)
    return min(1.0, max(0.0, score))


# ── Step 4 — Router ────────────────────────────────────────────────────────────

def _entity_persistence(entity_data: dict) -> int:
    """Entity persistence = the strongest persistence among its facts (default 3)."""
    vals = [f.get("persistence_value", 3) for f in entity_data.get("facts", [])]
    return max(vals) if vals else 3


def _upsert_entity(entity_data: dict, conn) -> str:
    """Create or update an entity node, filling summary / attributes / persistence."""
    existing = entity_data.get("existing_entity")
    now = datetime.now(timezone.utc).date().isoformat()
    summary = entity_data.get("summary")
    attributes = entity_data.get("attributes") or {}
    persistence = _entity_persistence(entity_data)

    if existing:
        entity_id = existing["id"]
        try:
            existing_aliases = json.loads(existing.get("aliases", "[]"))
        except (ValueError, TypeError):
            existing_aliases = []
        merged_aliases = json.dumps(list(set(existing_aliases + entity_data.get("aliases", []))))
        try:
            existing_attrs = json.loads(existing.get("attributes", "{}"))
        except (ValueError, TypeError):
            existing_attrs = {}
        merged_attrs = {**existing_attrs, **attributes}  # new keys win
        new_summary = summary or existing.get("summary")  # keep old summary if none provided
        conn.execute(
            "UPDATE entities SET aliases=?, attributes=?, summary=?, "
            "mention_count=mention_count+1, last_mentioned=?, "
            "persistence_value=MAX(persistence_value, ?) WHERE id=?",
            (
                merged_aliases,
                json.dumps(merged_attrs, ensure_ascii=False),
                new_summary,
                now,
                persistence,
                entity_id,
            ),
        )
    else:
        entity_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO entities "
            "(id, type, canonical_name, aliases, attributes, summary, last_mentioned, persistence_value) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                entity_id,
                entity_data.get("type", "concept"),
                entity_data["canonical_name"],
                json.dumps(entity_data.get("aliases", [])),
                json.dumps(attributes, ensure_ascii=False),
                summary,
                now,
                persistence,
            ),
        )
    return entity_id


def step4_route(
    resolved: dict,
    source_inbox_id: int,
    conn,
    dry_run: bool = False,
    verbose: bool = False,
) -> list[str]:
    entity_ids: list[str] = []

    # Names appearing in relations — these entities are created so the relation
    # has both endpoints (an entity may be mentioned only as a relation target).
    relation_names = set()
    for rel in resolved.get("relations", []):
        for key in ("from", "to"):
            if rel.get(key):
                relation_names.add(rel[key].strip().lower())

    for entity_data in resolved.get("resolved_entities", []):
        canonical = (entity_data.get("canonical_name") or "").strip()
        if not canonical:
            continue  # garde-fou: never create a nameless entity

        existing = entity_data.get("existing_entity")
        mention_count = (existing.get("mention_count", 1) + 1) if existing else 1

        scored: list[tuple[dict, float]] = []
        for fact in entity_data.get("facts", []):
            confidence = compute_confidence(
                fact,
                has_explicit_statement=True,
                context_supports=bool(existing),
                mention_count=mention_count,
            )
            scored.append((fact, confidence))

        # ── Entity creation is DECOUPLED from fact confidence ──
        # Create the node as soon as the entity is mentioned, provided it carries
        # a minimal signal (anti-pollution garde-fou):
        #   - already known (re-mention → bump mention_count), OR
        #   - part of a relation, OR
        #   - has a fact with persistence >= MIN_ENTITY_PERSISTENCE (not pure noise).
        # Its facts are still confidence-routed below (a fresh entity's facts
        # typically land in pending until corroborated/validated).
        max_persistence = _entity_persistence(entity_data) if entity_data.get("facts") else 0
        should_create = (
            bool(existing)
            or canonical.lower() in relation_names
            or max_persistence >= MIN_ENTITY_PERSISTENCE
        )

        entity_id: str | None = None
        if should_create and not dry_run:
            entity_id = _upsert_entity(entity_data, conn)
            if entity_id not in entity_ids:
                entity_ids.append(entity_id)
        elif verbose and not should_create:
            print(f"    [route] entity '{canonical}' skipped — noise, no relation")

        for fact, confidence in scored:
            bucket = (
                "entities" if confidence > 0.85
                else "pending" if confidence >= 0.5
                else "review"
            )
            if verbose:
                print(f"    [route] '{fact['predicate']}' conf={confidence:.2f} → {bucket}")

            fact_data = {
                "entity_canonical": entity_data["canonical_name"],
                "predicate": fact["predicate"],
                "value": fact["value"],
                "persistence_value": fact.get("persistence_value", 3),
                "confidence": confidence,
                "source_inbox_id": source_inbox_id,
            }

            if dry_run:
                continue

            if confidence > 0.85:
                if entity_id:
                    conn.execute(
                        "INSERT INTO facts "
                        "(id, entity_id, predicate, value, confidence, source_inbox_id, persistence_value) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (
                            str(uuid.uuid4()), entity_id,
                            fact["predicate"], fact["value"],
                            confidence, str(source_inbox_id),
                            fact.get("persistence_value", 3),
                        ),
                    )
            elif confidence >= 0.5:
                conn.execute(
                    "INSERT INTO pending_facts (id, fact_data, validation_strategy) VALUES (?,?,?)",
                    (str(uuid.uuid4()), json.dumps(fact_data), "passive"),
                )
            else:
                conn.execute(
                    "INSERT INTO review_queue (id, fact_data, suggested_entity) VALUES (?,?,?)",
                    (str(uuid.uuid4()), json.dumps(fact_data), entity_data["canonical_name"]),
                )

    # Relations — only if both entities already exist
    if not dry_run:
        for rel in resolved.get("relations", []):
            from_name, predicate, to_name = (
                rel.get("from"), rel.get("predicate"), rel.get("to")
            )
            if not (from_name and predicate and to_name):
                continue
            from_row = conn.execute(
                "SELECT id FROM entities WHERE LOWER(canonical_name)=LOWER(?)", (from_name,)
            ).fetchone()
            to_row = conn.execute(
                "SELECT id FROM entities WHERE LOWER(canonical_name)=LOWER(?)", (to_name,)
            ).fetchone()
            if from_row and to_row:
                conn.execute(
                    "INSERT INTO relations (id, entity_from, predicate, entity_to) VALUES (?,?,?,?)",
                    (str(uuid.uuid4()), from_row[0], predicate, to_row[0]),
                )
                if verbose:
                    print(f"    [route] relation {from_name} —{predicate}→ {to_name}")

    return entity_ids


# ── Step 5 — Behavioral validation ────────────────────────────────────────────

def step5_validate_pending(
    new_facts: list[dict],
    conn,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    promoted = 0
    pending = conn.execute(
        "SELECT id, fact_data FROM pending_facts"
    ).fetchall()

    for pending_id, fact_data_raw in pending:
        try:
            pf = json.loads(fact_data_raw)
        except (ValueError, TypeError):
            continue

        corroborating = any(
            nf.get("predicate") == pf.get("predicate")
            and nf.get("entity_canonical", "").lower() == pf.get("entity_canonical", "").lower()
            for nf in new_facts
        )
        if not corroborating:
            continue

        new_conf = compute_confidence(
            {"predicate": pf.get("predicate"), "value": pf.get("value"),
             "persistence_value": pf.get("persistence_value", 3)},
            has_explicit_statement=True,
            context_supports=True,
            mention_count=2,
        )
        if new_conf <= 0.85:
            continue

        if verbose:
            print(f"    [validate] promoting '{pf.get('predicate')}' conf={new_conf:.2f}")

        if not dry_run:
            entity_name = pf.get("entity_canonical", "unknown")
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
                    pf.get("predicate"), pf.get("value"),
                    new_conf, pf.get("source_inbox_id"),
                    pf.get("persistence_value", 3),
                ),
            )
            conn.execute("DELETE FROM pending_facts WHERE id=?", (pending_id,))
        promoted += 1

    return promoted


# ── Step 6 — Vectorization ────────────────────────────────────────────────────

def step6_vectorize(
    entity_ids: list[str],
    conn,
    client: anthropic.Anthropic,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    vectorized = 0
    for entity_id in entity_ids:
        entity = first_row(conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)))
        if not entity:
            continue

        try:
            aliases = json.loads(entity.get("aliases", "[]"))
        except (ValueError, TypeError):
            aliases = []
        try:
            attributes = json.loads(entity.get("attributes", "{}"))
        except (ValueError, TypeError):
            attributes = {}

        text = (
            f"Nom: {entity['canonical_name']}\n"
            f"Type: {entity.get('type', '')}\n"
            f"Aliases: {', '.join(aliases)}\n"
            f"Attributs: {json.dumps(attributes, ensure_ascii=False)}\n"
            f"Résumé: {entity.get('summary', '')}"
        )

        if dry_run:
            if verbose:
                print(f"    [vectorize] would embed '{entity['canonical_name']}'")
            continue

        try:
            vec_bytes = embed_text(text, client)
            conn.execute("UPDATE entities SET embedding=? WHERE id=?", (vec_bytes, entity_id))
            vectorized += 1
            if verbose:
                print(f"    [vectorize] embedded '{entity['canonical_name']}'")
        except Exception as exc:
            if verbose:
                print(f"    [vectorize] error for '{entity['canonical_name']}': {exc}")

    return vectorized


# ── Intentions ─────────────────────────────────────────────────────────────────

def handle_intentions(
    resolved: dict,
    conn,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    if dry_run:
        return
    # Clean expired intentions
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    conn.execute(
        "DELETE FROM intentions WHERE created_at < ? AND resolved = 0", (cutoff,)
    )
    if resolved.get("is_ephemeral") or resolved.get("input_type") == "ephemeral":
        content = resolved.get("ephemeral_content") or resolved.get("summary", "")
        if content:
            conn.execute(
                "INSERT INTO intentions (id, content, ttl_hours) VALUES (?,?,?)",
                (str(uuid.uuid4()), content, 48),
            )
            if verbose:
                print(f"    [intention] created: '{content[:70]}'")


# ── Episodic memory ──────────────────────────────────────────────────────────

def write_episodic_note(
    classified: dict,
    entry: dict,
    conn,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Store an episodic entry as a vectorized atomic_note (spec §7, level 2)."""
    content = entry["content"]
    summary = classified.get("summary") or ""
    entities_mentioned = [
        e["canonical_name"]
        for e in classified.get("entities", [])
        if e.get("canonical_name")
    ]
    title = (summary or content)[:60]

    if dry_run:
        if verbose:
            print(f"    [episodic] would write note: {title!r}")
        return

    conn.execute(
        "INSERT INTO atomic_notes (title, content, summary, entities_mentioned, memory_strength) "
        "VALUES (?,?,?,?,?)",
        (title, content, summary, json.dumps(entities_mentioned, ensure_ascii=False), 1.0),
    )
    note_id = conn.last_insert_rowid()

    try:
        vec_bytes = embed_text(f"{title}\n{content}")
        conn.execute(
            "INSERT OR REPLACE INTO atomic_notes_vec(rowid, embedding) VALUES (?, ?)",
            (note_id, vec_bytes),
        )
    except Exception as exc:
        if verbose:
            print(f"    [episodic] vectorize error: {exc}")

    if verbose:
        print(f"    [episodic] note id={note_id}: {title!r}")


# ── Per-entry processing ─────────────────────────────────────────────────────

def _mark(conn, entry_id: int, now: str, status: str, dry_run: bool = False) -> None:
    """Mark an inbox entry done with an outcome status (processed | failed)."""
    if dry_run:
        return
    conn.execute(
        "UPDATE inbox SET processed_at=?, status=? WHERE id=?", (now, status, entry_id)
    )


def _process_entry(entry, client, conn, now, dry_run, verbose) -> tuple[list[str], list[dict]]:
    """Process one inbox entry; mark it processed. Returns (entity_ids, new_facts).

    Raises anthropic.APIError on infrastructure failure (caller aborts the run and
    leaves the entry queued); other exceptions are content errors the caller marks
    as 'failed'.
    """
    classified = step1_classify(entry, client, verbose)

    # Ephemeral → intentions only
    if classified.get("is_ephemeral") or classified.get("input_type") == "ephemeral":
        with conn:
            handle_intentions(classified, conn, dry_run, verbose)
            _mark(conn, entry["id"], now, "processed", dry_run)
        return [], []

    # Episodic → vectorized atomic_note
    if classified.get("input_type") == "episodic":
        with conn:
            write_episodic_note(classified, entry, conn, dry_run, verbose)
            _mark(conn, entry["id"], now, "processed", dry_run)
        return [], []

    # Fact / resource → entity graph
    resolved = step2_resolve(classified, conn, verbose)
    new_facts = [
        {**fact, "entity_canonical": ent["canonical_name"]}
        for ent in resolved.get("resolved_entities", [])
        for fact in ent.get("facts", [])
    ]
    with conn:
        entity_ids = step4_route(resolved, entry["id"], conn, dry_run, verbose)
        handle_intentions(resolved, conn, dry_run, verbose)
        _mark(conn, entry["id"], now, "processed", dry_run)
    return entity_ids, new_facts


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_dream_cycle(dry_run: bool = False, verbose: bool = False) -> None:
    print("═" * 60)
    print("  SYNAPSE  ·  Dream Cycle  A+")
    if dry_run:
        print("  ⚠  DRY RUN — no writes to database")
    print("═" * 60)

    client = _get_client()
    init_db()

    conn = get_connection()
    try:
        entries = cursor_to_dicts(conn.execute(
            "SELECT id, content, source, created_at "
            "FROM inbox WHERE processed_at IS NULL ORDER BY created_at"
        ))

        if not entries:
            print("\n  Inbox empty — nothing to process.")
            print("═" * 60)
            return

        print(f"\n  {len(entries)} unprocessed entr{'y' if len(entries) == 1 else 'ies'} found\n")

        now = datetime.now(timezone.utc).isoformat()
        all_entity_ids: list[str] = []
        all_new_facts: list[dict] = []

        failed = 0
        for entry in entries:
            print(f"▸ inbox id={entry['id']}: {entry['content'][:70]!r}")
            try:
                entity_ids, new_facts = _process_entry(
                    entry, client, conn, now, dry_run, verbose
                )
                all_entity_ids.extend(e for e in entity_ids if e not in all_entity_ids)
                all_new_facts.extend(new_facts)
            except anthropic.APIError as exc:
                # Infrastructure failure (no/invalid key, network, rate limit):
                # abort the run and leave every remaining entry queued for a retry.
                print(f"  ⚠ erreur API ({type(exc).__name__}) — run interrompu, entrées laissées en file")
                raise
            except Exception as exc:  # noqa: BLE001 — content error for THIS entry
                print(f"  ⚠ échec entrée id={entry['id']}: {exc}")
                if not dry_run:
                    with conn:
                        _mark(conn, entry["id"], now, "failed")
                failed += 1
            print()

        # Step 5 — Behavioral validation
        if all_new_facts:
            if verbose:
                print("▸ Step 5 — Behavioral Validation")
            with conn:
                promoted = step5_validate_pending(all_new_facts, conn, dry_run, verbose)
            if promoted:
                print(f"  → {promoted} pending fact(s) promoted")

        # Step 6 — Vectorize touched entities
        if all_entity_ids:
            print("▸ Step 6 — Vectorization")
            with conn:
                vectorized = step6_vectorize(all_entity_ids, conn, client, dry_run, verbose)
            print(f"  → {vectorized} entit{'y' if vectorized == 1 else 'ies'} vectorized")

        print("\n" + "═" * 60)
        summary = f"  Done  ·  {len(all_entity_ids)} entit{'y' if len(all_entity_ids) == 1 else 'ies'} updated"
        if failed:
            summary += f"  ·  {failed} failed"
        print(summary)
        print("═" * 60 + "\n")

    finally:
        conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synapse Dream Cycle")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing to DB",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Detailed per-step logging",
    )
    args = parser.parse_args(argv)
    run_dream_cycle(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
