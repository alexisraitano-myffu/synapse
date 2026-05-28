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
from config_store import get_anthropic_key
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
    key = get_anthropic_key()
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Either export it (export ANTHROPIC_API_KEY=sk-ant-...) "
            "or set it from the desktop app (Settings → Clé Anthropic API)."
        )
    return anthropic.Anthropic(api_key=key)


# ── Step 1 — Classifier ────────────────────────────────────────────────────────

_SYSTEM_CLASSIFIER = """\
Tu es un extracteur de mémoire pour un second cerveau personnel.

Une même capture peut produire PLUSIEURS sorties simultanément (routing non-exclusif).
Une réflexion dense qui mentionne plusieurs projets, des personnes et énonce des faits,
doit produire à la fois project_entries (N items) + atomic_note + entities + facts dans
le même JSON.

Retourne UNIQUEMENT un JSON valide (sans markdown) :
{{
  "input_type": "fact|episodic|ephemeral|resource",
  "atomic_note": "string ou null (réflexion libre / pensée non-factuelle ; on la garde comme nœud à part qui MENTIONNE des entités sans en devenir une)",
  "project_entries": [
    {{
      "project_canonical": "string (nom du projet auquel rattacher ; si 'nouveau projet : X', mets X)",
      "content": "string (l'extrait de la capture pertinent pour CE projet précis)",
      "is_new": true|false
    }}
  ],
  "entities": [
    {{
      "canonical_name": "string",
      "type": "person|place|project|concept|organization|animal",
      "aliases": ["string"],
      "summary": "string (1 phrase qui décrit cette entité, ou null si rien de notable)",
      "attributes": {{"clé": "valeur"}},
      "facts": [
        {{
          "predicate": "string (snake_case ex: has_birthday, works_at, lives_in)",
          "value": "string",
          "persistence_value": 1,
          "evidence_strength": "explicit|hedged|implicit"
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

Règles atomic_note :
- Une "réflexion libre" ("j'ai pensé à…", "j'ai eu une idée sur…", journal personnel) → atomic_note,
  PAS une entité fourre-tout. Les concepts mentionnés deviennent des entities reliées séparément.
- Si la capture est purement factuelle (ex: "anniversaire de maman = 26 mars"), atomic_note = null.

Règles project_entries :
- Si la capture est explicitement liée à un OU PLUSIEURS projets (déclarés ou nommés), produire UNE entrée par projet dans le tableau project_entries.
- Une même capture peut mentionner plusieurs projets ("j'ai avancé Synapse et Atlas aujourd'hui") → 2 items, un pour chaque projet, avec un `content` propre qui reprend uniquement l'extrait pertinent à ce projet.
- "nouveau projet : X" → is_new=true, project_canonical=X (et toujours dans le tableau, même s'il n'y a qu'un seul item).
- La liste des projets existants te sera fournie en contexte ci-dessous — préfère un nom existant à une variante orthographique.
- Si aucun projet identifiable → project_entries = [] (tableau vide).
- Ne jamais émettre deux items pour le même project_canonical dans une même capture — fusionne le contenu dans un seul item.

Règles persistence_value :
5 = permanent (date naissance, lien familial, prénom)
4 = stable modifiable (lieu de travail, adresse)
3 = état actuel (projet en cours)
2 = contextuel (événement ponctuel)
1 = bruit (mention passagère)
Règles evidence_strength (s'applique à la langue de la capture, FR/EN/autre) :
explicit = fait énoncé directement, sans marqueur d'incertitude
hedged   = marqueur d'incertitude épistémique présent (ex FR: "semble", "je crois", "il paraît", "devrait", "peut-être", "probablement" ; EN: "seems", "I think", "apparently", "probably", "might" ; même critère dans toute autre langue)
implicit = fait non énoncé mais déduit du contexte (inférence indirecte, ex: on parle du déménagement de Pierre sans dire où)
Résous les dates relatives vers des dates absolues.
La date d'aujourd'hui est : {today}.\
"""


def _load_active_projects_block(conn) -> str:
    """Builds the context block listing existing project entities for the prompt.

    Returned as a separate (uncached) system text block so changes to the project
    list don't bust the cache of the stable rules above.
    """
    rows = cursor_to_dicts(conn.execute(
        "SELECT canonical_name, summary, aliases FROM entities "
        "WHERE type='project' ORDER BY mention_count DESC, last_mentioned DESC LIMIT 50"
    ))
    if not rows:
        return "[PROJETS EXISTANTS]\n(aucun pour l'instant — toute mention de 'nouveau projet : X' doit créer l'entité)"
    lines = ["[PROJETS EXISTANTS — utilise leur canonical_name exact pour le rattachement]"]
    for r in rows:
        try:
            aliases = json.loads(r.get("aliases") or "[]")
        except (ValueError, TypeError):
            aliases = []
        alias_str = f" (alias: {', '.join(aliases)})" if aliases else ""
        summary = (r.get("summary") or "").strip().replace("\n", " ")[:120]
        lines.append(f"- {r['canonical_name']}{alias_str}{(' — ' + summary) if summary else ''}")
    return "\n".join(lines)


def step1_classify(
    entry: dict,
    client: anthropic.Anthropic,
    verbose: bool = False,
    conn=None,
) -> dict:
    system_stable = _SYSTEM_CLASSIFIER.format(today=_TODAY)
    system_blocks = [
        {"type": "text", "text": system_stable, "cache_control": {"type": "ephemeral"}},
    ]
    if conn is not None:
        projects_block = _load_active_projects_block(conn)
        # NOT cached — varies as projects are created.
        system_blocks.append({"type": "text", "text": projects_block})
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1536,
        system=system_blocks,
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

# Evidence strength → confidence floor. The pivot of routing: Claude decides
# how the fact is asserted in the source text, Python adds marginal bonuses.
#   - explicit  → directly stated, no hedge → tends to land in `facts`
#   - hedged    → modal of uncertainty present → tends to land in `pending`
#   - implicit  → inferred from context, not stated → tends to be rejected
_EVIDENCE_BASE = {"explicit": 0.92, "hedged": 0.65, "implicit": 0.40}

# Anti-pollution garde-fou for entity creation: an entity is created on mention
# only if it carries at least this much persistence in one of its facts (i.e. it
# is more than pure noise) — unless it already exists or appears in a relation.
# Tune UP to be stricter (fewer entities), DOWN to capture more. 1 = create for
# everything mentioned (noisiest), 2 = skip pure "mention passagère".
MIN_ENTITY_PERSISTENCE = 2


def compute_confidence(
    fact: dict,
    evidence_strength: str,
    existing: bool,
    mention_count: int,
) -> float:
    base = _EVIDENCE_BASE.get(evidence_strength, _EVIDENCE_BASE["explicit"])
    bonus = 0.0
    if existing:
        bonus += 0.05
    bonus += min(0.05, mention_count * 0.02)
    bonus += _PERSISTENCE_BONUS.get(fact.get("persistence_value", 3), 0)
    score = base + bonus
    # Invariant: a hedged fact must remain in the pending zone for user validation,
    # regardless of how high its persistence is. Clamp just under the facts threshold.
    if evidence_strength == "hedged":
        score = min(score, 0.84)
    return min(1.0, max(0.0, score))


# ── Step 4 — Router ────────────────────────────────────────────────────────────

def _entity_persistence(entity_data: dict) -> int:
    """Entity persistence = the strongest persistence among its facts (default 3)."""
    vals = [f.get("persistence_value", 3) for f in entity_data.get("facts", [])]
    return max(vals) if vals else 3


def _upsert_entity(entity_data: dict, conn, capture_id: int | None = None) -> str:
    """Create or update an entity node, filling summary / attributes / persistence.

    SYN-41: a newly created entity carries its `provenance_capture_id` back to
    the immutable inbox row that spawned it. UPDATE path leaves provenance alone
    (first-mention provenance is the lineage we care about; subsequent mentions
    don't overwrite history).
    """
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
            "(id, type, canonical_name, aliases, attributes, summary, last_mentioned, "
            " persistence_value, provenance_capture_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                entity_id,
                entity_data.get("type", "concept"),
                entity_data["canonical_name"],
                json.dumps(entity_data.get("aliases", [])),
                json.dumps(attributes, ensure_ascii=False),
                summary,
                now,
                persistence,
                capture_id,
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
                evidence_strength=fact.get("evidence_strength", "explicit"),
                existing=bool(existing),
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
            entity_id = _upsert_entity(entity_data, conn, capture_id=source_inbox_id)
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
                "evidence_strength": fact.get("evidence_strength", "explicit"),
                "confidence": confidence,
                "source_inbox_id": source_inbox_id,
            }

            if dry_run:
                continue

            if confidence > 0.85:
                if entity_id:
                    conn.execute(
                        "INSERT INTO facts "
                        "(id, entity_id, predicate, value, confidence, source_inbox_id, "
                        " persistence_value, provenance_capture_id) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (
                            str(uuid.uuid4()), entity_id,
                            fact["predicate"], fact["value"],
                            confidence, str(source_inbox_id),
                            fact.get("persistence_value", 3),
                            source_inbox_id,
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
                    "INSERT INTO relations "
                    "(id, entity_from, predicate, entity_to, provenance_capture_id) "
                    "VALUES (?,?,?,?,?)",
                    (str(uuid.uuid4()), from_row[0], predicate, to_row[0], source_inbox_id),
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

        # A pending is promoted only when *another* note corroborates it.
        # Excluding the pending's own source prevents self-corroboration (the
        # very fact that created the pending would otherwise auto-promote it).
        corroborator = next(
            (nf for nf in new_facts
             if nf.get("predicate") == pf.get("predicate")
             and nf.get("entity_canonical", "").lower() == pf.get("entity_canonical", "").lower()
             and str(nf.get("source_inbox_id")) != str(pf.get("source_inbox_id"))),
            None,
        )
        if corroborator is None:
            continue

        # Use the corroborator's evidence_strength so two hedged sources stay
        # in pending (the SYN-30 clamp keeps the score below the facts threshold);
        # only an explicit corroboration lifts the doubt.
        new_conf = compute_confidence(
            {"predicate": pf.get("predicate"), "value": pf.get("value"),
             "persistence_value": pf.get("persistence_value", 3)},
            evidence_strength=corroborator.get("evidence_strength", "explicit"),
            existing=True,
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
            # SYN-41: provenance traces back to the original capture that spawned
            # the pending fact (or whichever corroborator promoted it).
            try:
                prov_id = int(pf.get("source_inbox_id")) if pf.get("source_inbox_id") else None
            except (TypeError, ValueError):
                prov_id = None
            if row:
                entity_id = row[0]
            else:
                entity_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO entities (id, canonical_name, provenance_capture_id) VALUES (?,?,?)",
                    (entity_id, entity_name, prov_id),
                )
            conn.execute(
                "INSERT INTO facts "
                "(id, entity_id, predicate, value, confidence, source_inbox_id, "
                " persistence_value, provenance_capture_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()), entity_id,
                    pf.get("predicate"), pf.get("value"),
                    new_conf, pf.get("source_inbox_id"),
                    pf.get("persistence_value", 3),
                    prov_id,
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


# ── SYN-42 — Multi-output routing helpers ────────────────────────────────────

def _persist_atomic_note(
    content: str,
    summary: str,
    entities_mentioned: list[str],
    capture_id: int,
    conn,
    verbose: bool = False,
) -> int | None:
    """Persist a free-form thought as an atomic_note with provenance.

    Differs from the legacy write_episodic_note: doesn't require a full
    `classified` dict, accepts an explicit content (so the multi-output router
    can pass either the raw capture or a Claude-extracted excerpt), and carries
    the provenance_capture_id from SYN-41.
    """
    title = (summary or content)[:60]
    conn.execute(
        "INSERT INTO atomic_notes "
        "(title, content, summary, entities_mentioned, memory_strength, provenance_capture_id) "
        "VALUES (?,?,?,?,?,?)",
        (title, content, summary,
         json.dumps(entities_mentioned, ensure_ascii=False),
         1.0, capture_id),
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
            print(f"    [atomic_note] vectorize error: {exc}")
    if verbose:
        print(f"    [atomic_note] id={note_id}: {title!r}")
    return note_id


_PROJECT_SUMMARY_SYSTEM = """\
Tu maintiens une synthèse vivante d'un projet personnel. La synthèse doit :
- rester en markdown clair (titres ##, listes, max ~500 mots),
- résumer l'état du projet : objectifs, décisions, idées émergentes, blocages, prochaines étapes,
- éviter la redondance (si l'info existe déjà, ne pas la répéter),
- préserver la nuance (idées floues restent floues, contradictions restent visibles).

Retourne UNIQUEMENT le markdown mis à jour, sans préambule.\
"""


_PROJECT_REFINEMENT_SYSTEM = """\
Tu es le "garbage collector" d'une synthèse de projet personnel. On te donne TOUTES
les entrées du projet dans l'ordre chronologique. Reconstruis from-scratch une
synthèse propre :
- déduplique les infos répétées entre entrées,
- résous les contradictions quand c'est possible (la plus récente fait foi sauf si
  une entrée explicite le contraire),
- élague le périmé (anciennes intentions remplacées, idées abandonnées),
- préserve l'historique des décisions importantes (avec dates si pertinent),
- markdown propre, hiérarchie claire, ~500-800 mots max.

Retourne UNIQUEMENT le markdown, sans préambule.\
"""


# After how many *new* entries since the last refinement do we trigger another one ?
# Configurable via env so tests can use a small value.
def _refinement_threshold() -> int:
    try:
        return max(1, int(os.environ.get("SYNAPSE_REFINEMENT_THRESHOLD", "20")))
    except ValueError:
        return 20


def _refine_project_summary(
    project_id: str,
    project_name: str,
    conn,
    client: "anthropic.Anthropic",
    verbose: bool = False,
) -> str | None:
    """Rebuild a project's synthesis from-scratch from all its entries.

    SYN-44: triggered by `_append_project_summary` once a threshold of new
    entries has accumulated since the last refinement. ~5-10× the cost of an
    append, but rare. INSERT version with `kind='refinement'` so the caller
    can know what the latest snapshot represents.
    """
    entries = cursor_to_dicts(conn.execute(
        "SELECT content, created_at FROM project_entries "
        "WHERE project_id = ? ORDER BY created_at ASC LIMIT 200",
        (project_id,),
    ))
    if not entries:
        return None

    timeline = "\n\n".join(
        f"[{e['created_at']}] {e['content']}" for e in entries
    )
    user_msg = (
        f"Projet : {project_name}\n\n"
        f"Toutes les entrées dans l'ordre chronologique :\n---\n{timeline}\n---\n\n"
        f"Reconstruis from-scratch la synthèse du projet."
    )

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=[{"type": "text", "text": _PROJECT_REFINEMENT_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        if verbose:
            print(f"    [refinement] Claude error: {exc}")
        return None

    summary_md = response.content[0].text.strip()
    if summary_md.startswith("```"):
        summary_md = summary_md.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    entry_count = len(entries)
    version_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO project_state_versions "
        "(id, project_id, summary_md, entry_count, trigger, kind) "
        "VALUES (?,?,?,?,'passive','refinement')",
        (version_id, project_id, summary_md, entry_count),
    )
    conn.execute(
        "UPDATE project_state SET current_version_id=?, updated_at=CURRENT_TIMESTAMP, "
        "entry_count_at_sync=? WHERE project_id=?",
        (version_id, entry_count, project_id),
    )

    if verbose:
        u = response.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0)
        print(f"    [refinement] from-scratch v{entry_count} for '{project_name}' "
              f"({len(entries)} entries) tokens={u.input_tokens}/{u.output_tokens}"
              + (f" cache_hit={cache_read}" if cache_read else ""))
    return summary_md


def _append_project_summary(
    project_id: str,
    project_name: str,
    new_entry_content: str,
    new_entry_count: int,
    conn,
    client: "anthropic.Anthropic",
    verbose: bool = False,
) -> str | None:
    """Generate or amend a project's live synthesis after a new entry.

    SYN-43: one Haiku call per project_entry. If a current version exists,
    we ask Claude to amend it (cheap, entropic). The "garbage collector" pass
    (SYN-44 refinement passif) corrects accumulated drift.
    """
    # Look up the current synthesis, if any
    current = first_row(conn.execute(
        "SELECT psv.summary_md FROM project_state ps "
        "JOIN project_state_versions psv ON psv.id = ps.current_version_id "
        "WHERE ps.project_id = ?",
        (project_id,),
    ))
    current_summary = current["summary_md"] if current else None

    if current_summary:
        user_msg = (
            f"Projet : {project_name}\n\n"
            f"Synthèse actuelle :\n---\n{current_summary}\n---\n\n"
            f"Nouvelle entrée à intégrer :\n---\n{new_entry_content}\n---\n\n"
            f"Mets à jour la synthèse pour intégrer la nouvelle entrée."
        )
    else:
        user_msg = (
            f"Projet : {project_name}\n\n"
            f"Première entrée :\n---\n{new_entry_content}\n---\n\n"
            f"Écris la synthèse initiale du projet à partir de cette entrée."
        )

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": _PROJECT_SUMMARY_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        # Don't block the cycle on a synthesis failure — keep the entry, retry later.
        if verbose:
            print(f"    [project_summary] Claude error: {exc}")
        return None

    summary_md = response.content[0].text.strip()
    # Strip optional ```markdown fences
    if summary_md.startswith("```"):
        summary_md = summary_md.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    version_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO project_state_versions "
        "(id, project_id, summary_md, entry_count, trigger, kind) "
        "VALUES (?,?,?,?,'passive','append')",
        (version_id, project_id, summary_md, new_entry_count),
    )
    existing_state = first_row(conn.execute(
        "SELECT project_id FROM project_state WHERE project_id = ?", (project_id,)
    ))
    if existing_state:
        conn.execute(
            "UPDATE project_state SET current_version_id=?, updated_at=CURRENT_TIMESTAMP, "
            "entry_count_at_sync=? WHERE project_id=?",
            (version_id, new_entry_count, project_id),
        )
    else:
        conn.execute(
            "INSERT INTO project_state "
            "(project_id, current_version_id, entry_count_at_sync) VALUES (?,?,?)",
            (project_id, version_id, new_entry_count),
        )

    if verbose:
        u = response.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0)
        print(f"    [project_summary] append v{new_entry_count} for '{project_name}' "
              f"tokens={u.input_tokens}/{u.output_tokens}"
              + (f" cache_hit={cache_read}" if cache_read else ""))

    # SYN-44: trigger from-scratch refinement once enough new entries have
    # accumulated since the last refinement.
    last_refinement = first_row(conn.execute(
        "SELECT MAX(entry_count) AS last_count FROM project_state_versions "
        "WHERE project_id = ? AND kind = 'refinement'", (project_id,)
    ))
    last_count = (last_refinement or {}).get("last_count") or 0
    if new_entry_count - last_count >= _refinement_threshold():
        _refine_project_summary(
            project_id=project_id,
            project_name=project_name,
            conn=conn,
            client=client,
            verbose=verbose,
        )

    return summary_md


def _persist_project_entry(
    project_canonical: str,
    content: str,
    capture_id: int,
    conn,
    is_new_project: bool = False,
    verbose: bool = False,
    client: "anthropic.Anthropic | None" = None,
) -> tuple[str, str]:
    """Find or create the project entity then INSERT into project_entries.

    Returns (project_id, entry_id). Uses canonical_name case-insensitively to
    match the resolver's existing convention.

    SYN-43: if `client` is provided, also amends the project's live synthesis
    after persisting the new entry (one Haiku call).
    """
    canonical = project_canonical.strip()
    row = first_row(conn.execute(
        "SELECT id FROM entities WHERE type='project' AND LOWER(canonical_name) = LOWER(?)",
        (canonical,),
    ))
    if row:
        project_id = row["id"]
        # Bump mention_count so the project shows up in Force-sorted views.
        conn.execute(
            "UPDATE entities SET mention_count = mention_count + 1, last_mentioned = DATE('now') "
            "WHERE id = ?",
            (project_id,),
        )
    else:
        project_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO entities "
            "(id, type, canonical_name, mention_count, last_mentioned, persistence_value, "
            " summary, provenance_capture_id) "
            "VALUES (?, 'project', ?, 1, DATE('now'), 3, ?, ?)",
            (project_id, canonical,
             "Projet créé automatiquement par le Dream Cycle." if is_new_project else None,
             capture_id),
        )
        if verbose:
            print(f"    [project] auto-created '{canonical}' id={project_id}")

    entry_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO project_entries (id, project_id, capture_id, content, kind) "
        "VALUES (?, ?, ?, ?, 'note')",
        (entry_id, project_id, capture_id, content),
    )
    if verbose:
        print(f"    [project] '{canonical}' ← entry id={entry_id}")

    # SYN-43: amend the live synthesis right after the entry lands.
    if client is not None:
        new_count = conn.execute(
            "SELECT COUNT(*) FROM project_entries WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
        _append_project_summary(
            project_id=project_id,
            project_name=canonical,
            new_entry_content=content,
            new_entry_count=new_count,
            conn=conn,
            client=client,
            verbose=verbose,
        )

    return project_id, entry_id


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

    SYN-42 + SYN-57: routing is NON-EXCLUSIVE and N-per-projection. A single
    capture can simultaneously produce N project_entries (one per mentioned
    project), an atomic_note, entities, facts and relations. Each sub-routing
    is gated by what Haiku put in the JSON, not by an exclusive if/elif chain.

    Raises anthropic.APIError on infrastructure failure (caller aborts the run and
    leaves the entry queued); other exceptions are content errors the caller marks
    as 'failed'.
    """
    classified = step1_classify(entry, client, verbose, conn=conn)

    # Ephemeral stays a special exit — it's "this isn't memory, just an
    # intention that expires". No graph, no atomic_note, no project.
    if classified.get("is_ephemeral") or classified.get("input_type") == "ephemeral":
        with conn:
            handle_intentions(classified, conn, dry_run, verbose)
            _mark(conn, entry["id"], now, "processed", dry_run)
        return [], []

    # All other inputs go through the multi-output router below.
    capture_id = entry["id"]
    entity_ids: list[str] = []
    new_facts: list[dict] = []

    if dry_run:
        if verbose:
            outs = []
            for pe in classified.get("project_entries") or []:
                if pe and pe.get("project_canonical"):
                    outs.append(f"project_entry→{pe['project_canonical']}")
            if classified.get("atomic_note"):
                outs.append("atomic_note")
            if classified.get("entities"):
                outs.append(f"{len(classified['entities'])} entities")
            print(f"    [dry] would route: {', '.join(outs) or '(nothing)'}")
        return [], []

    # 1. Graph (entities + facts + relations) — runs whenever Haiku found entities.
    resolved = step2_resolve(classified, conn, verbose) if classified.get("entities") else None
    if resolved:
        new_facts = [
            {**fact, "entity_canonical": ent["canonical_name"], "source_inbox_id": capture_id}
            for ent in resolved.get("resolved_entities", [])
            for fact in ent.get("facts", [])
        ]

    with conn:
        if resolved:
            entity_ids = step4_route(resolved, capture_id, conn, dry_run=False, verbose=verbose)

        # 2. Atomic note — free-form thought that mentions entities without being one.
        atomic = classified.get("atomic_note")
        if atomic and atomic.strip():
            mentioned = [
                e["canonical_name"]
                for e in classified.get("entities", [])
                if e.get("canonical_name")
            ]
            _persist_atomic_note(
                content=atomic.strip(),
                summary=classified.get("summary") or "",
                entities_mentioned=mentioned,
                capture_id=capture_id,
                conn=conn,
                verbose=verbose,
            )
        elif classified.get("input_type") == "episodic" and not atomic:
            # Backward compat: legacy 'episodic' captures with no explicit atomic_note
            # still get persisted as a vectorized note from the raw capture content.
            write_episodic_note(classified, entry, conn, dry_run=False, verbose=verbose)

        # 3. Project entries — N rattachements possibles (SYN-57). Une même
        # capture peut alimenter la timeline de plusieurs projets en parallèle ;
        # chaque projet reçoit son extrait (`content`) et déclenche sa propre
        # mise à jour de synthèse. Dedup par nom canonique au cas où le LLM
        # émettrait des doublons (premier extrait gagne).
        seen_projects: set[str] = set()
        for proj in classified.get("project_entries") or []:
            if not proj or not proj.get("project_canonical"):
                continue
            key = proj["project_canonical"].strip().lower()
            if not key or key in seen_projects:
                continue
            seen_projects.add(key)
            _persist_project_entry(
                project_canonical=proj["project_canonical"],
                content=(proj.get("content") or entry["content"]).strip(),
                capture_id=capture_id,
                conn=conn,
                is_new_project=bool(proj.get("is_new")),
                verbose=verbose,
                client=client,  # SYN-43: triggers live synthesis append per project
            )

        handle_intentions(classified, conn, dry_run=False, verbose=verbose)
        _mark(conn, capture_id, now, "processed", dry_run=False)

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
