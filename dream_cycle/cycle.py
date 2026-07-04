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
import time
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
from core_store import get_brain, get_store
from entity_search import entity_embedding_text
from dream_cycle.decay import apply_decay, apply_entity_decay
from dream_cycle.resources import process_capture_resources

_TODAY = date.today().isoformat()


# ── Claude client ──────────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    # SYN-105: client construction (incl. the fuel-proxy seam) is centralised.
    from anthropic_client import get_client
    return get_client()


# ── Step 1 — Classifier ────────────────────────────────────────────────────────

# ── SYN-111 : le cerveau (classif + routing) vit dans le cœur Rust ───────────
# Le prompt classifieur est une DONNÉE versionnée dans le repo synapse-core
# (prompts/classifier.md), déployée ici et lue à l'exécution par le core.
PROMPTS_DIR = Path(os.getenv("SYNAPSE_PROMPTS_DIR", Path.home() / ".synapse" / "prompts"))

# SYN-93 — working memory. Recent captures are handed to the classifier as a
# read-only context block so coreference ("il / elle / ce projet / hier") resolves
# across a day's captures instead of each entry being classified in a vacuum. The
# block COMMITS NOTHING — only the current capture (the user message) produces
# outputs. Same string for every entry of a run → cached prefix (one write, then hits).
_WM_MAX_CAPTURES = 80
_WM_MAX_CHARS = 8000
_WM_LOOKBACK_HOURS = 24


def _build_day_context(conn, batch_entries, now) -> str | None:
    """Build the working-memory context (SYN-93): captures of the current
    consolidation batch + recently-consolidated captures within the lookback
    window, as a chronological transcript. Returns None when there's nothing to
    resolve against (a lone capture with no recent history)."""
    cutoff = (now - timedelta(hours=_WM_LOOKBACK_HOURS)).isoformat()
    prior = cursor_to_dicts(conn.execute(
        "SELECT content, created_at FROM inbox "
        "WHERE processed_at IS NOT NULL AND created_at >= ? "
        "ORDER BY created_at DESC LIMIT ?",
        (cutoff, _WM_MAX_CAPTURES),
    ))
    prior.reverse()  # back to chronological
    timeline = [(p["content"], p.get("created_at"), "consolidé") for p in prior]
    timeline += [(e["content"], e.get("created_at"), "à consolider") for e in batch_entries]
    if len(timeline) <= 1:
        return None

    lines = [
        "[CONTEXTE — captures récentes, pour RÉSOUDRE LES RÉFÉRENCES "
        "(il, elle, ça, ce projet, « hier »…).",
        "⚠ N'EXTRAIS RIEN de ce bloc : seule la capture COURANTE (le message "
        "utilisateur) doit produire entités/faits/notes. Ce bloc n'est qu'un rappel "
        "du fil pour lever les ambiguïtés.]",
    ]
    used = 0
    for content, created_at, phase in timeline:
        ts = (created_at or "")[:16].replace("T", " ")
        text = " ".join((content or "").split())
        line = f"[{ts} · {phase}] {text}"
        if used + len(line) > _WM_MAX_CHARS:
            lines.append("… (contexte tronqué)")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)

def _classify_params(entry: dict, conn=None, day_context: str | None = None) -> dict:
    """`messages.create` kwargs pour classifier une capture — construits par le
    core (prompt-as-data + blocs vocab/projets/auteur lus dans SA base). Partagé
    par le chemin synchrone et le chemin Batch API. `conn` est conservé pour la
    signature historique ; le core lit toujours ses blocs lui-même."""
    return json.loads(get_brain().build_classify_params(
        entry["content"], day_context, CLAUDE_MODEL, str(PROMPTS_DIR), _TODAY))


def _parse_classify_text(text: str, content_len: int, stop_reason: str | None) -> dict:
    """Parse d'une réponse classifieur — implémenté dans le core (garde
    max_tokens → ValueError, fence strip, JSON). Partagé sync + batch."""
    import synapse_core
    return json.loads(synapse_core.parse_classify_text(text, content_len, stop_reason))


def step1_classify(
    entry: dict,
    client: anthropic.Anthropic,
    verbose: bool = False,
    conn=None,
    day_context: str | None = None,
) -> dict:
    """Classification synchrone via le core (build prompt + HTTP + parse).

    La résolution de la clé (et le seam fuel-proxy, SYN-105) reste côté hôte ;
    le core exécute. Une erreur réseau/HTTP remonte en ConnectionError (la boucle
    interrompt le run, politique anthropic.APIError) ; un contenu invalide en
    ValueError (l'entrée passe en 'failed'). `client`/`conn` gardés pour la
    signature historique."""
    from anthropic_client import is_fuel_token, _fuel_base_url
    from config_store import get_anthropic_key

    key = get_anthropic_key()
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY manquante — exporte-la, mets-la dans .env, ou "
            "règle-la depuis l'app (Réglages → Clé Anthropic API)."
        )
    fuel = is_fuel_token(key)
    raw = get_brain().classify(
        entry["content"], day_context, CLAUDE_MODEL,
        "" if fuel else key, str(PROMPTS_DIR), _TODAY,
        base_url=_fuel_base_url() if fuel else None,
        fuel_token=key if fuel else None,
    )
    result = json.loads(raw)
    if verbose:
        print(f"    [classify] {entry['id']}: input_type={result.get('input_type')}")
    return result



def _batch_classify(
    entries: list[dict], client, conn, day_context: str | None = None,
    verbose: bool = False, poll_seconds: int = 10, timeout_seconds: int = 3600,
) -> dict:
    """SYN-93 — classify a whole batch via the Message Batches API (~-50% on the
    nightly pass; latency is acceptable for a "sleep" consolidation). Submits one
    request per entry, polls until the batch ends, and returns {entry_id: classified}.
    An entry whose request errored or whose JSON won't parse maps to None — the
    caller then classifies just that one synchronously. Raises on infrastructure
    failure (submit/poll) so the caller can fall back to the fully-sync path."""
    from anthropic.types.messages.batch_create_params import Request
    requests = [
        Request(custom_id=f"e{e['id']}", params=_classify_params(e, conn, day_context))
        for e in entries
    ]
    batch = client.messages.batches.create(requests=requests)
    if verbose:
        print(f"    [batch] submitted {len(requests)} classify requests · id={batch.id}")
    waited = 0
    while batch.processing_status != "ended":
        if waited >= timeout_seconds:
            raise TimeoutError(f"batch {batch.id} pas terminé après {timeout_seconds}s")
        time.sleep(poll_seconds)
        waited += poll_seconds
        batch = client.messages.batches.retrieve(batch.id)

    by_custom = {f"e{e['id']}": e for e in entries}
    out: dict = {}
    for res in client.messages.batches.results(batch.id):
        entry = by_custom.get(res.custom_id)
        if entry is None:
            continue
        r = res.result
        if getattr(r, "type", None) != "succeeded":
            out[entry["id"]] = None  # caller retries this one synchronously
            continue
        try:
            msg = r.message
            out[entry["id"]] = _parse_classify_text(
                msg.content[0].text, len(entry["content"]), msg.stop_reason)
        except Exception as exc:  # noqa: BLE001 — content error for this entry
            out[entry["id"]] = None
            if verbose:
                print(f"    [batch] {res.custom_id} parse error: {exc}")
    return out


# ── Step 2 — Resolver ─────────────────────────────────────────────────────────





# ── SYN-89 — Entity re-summary ────────────────────────────────────────────────

_RESUMMARY_SYSTEM = """\
Tu écris le résumé d'une fiche d'entité d'un second cerveau personnel.

Contraintes STRICTES :
- 1 à 2 phrases en français, factuelles, ton neutre.
- INTEMPOREL : le résumé doit rester vrai dans 6 mois. Dates ABSOLUES uniquement
  ("anniversaire le 16 juin"), JAMAIS de relatif qui expire ("la semaine prochaine",
  "bientôt", "récemment", "vient de").
- Fonde-toi UNIQUEMENT sur les faits et relations fournis — n'invente rien, ne
  spécule pas. Les faits sont la source de vérité (ceux validés/édités par
  l'utilisateur font foi).
- Ignore le bruit : ne mentionne pas tout, garde l'essentiel qui identifie l'entité.

Retourne UNIQUEMENT le texte du résumé, sans préambule ni markdown.\
"""


def step_resummarize(
    touched_ids: list[str],
    conn,
    client,
    verbose: bool = False,
) -> list[str]:
    """SYN-89 — regenerate entity summaries from scratch (derived, never edited).

    Targets = entities touched by this run + entities flagged summary_stale
    (user fact edits between cycles). The summary is rebuilt from the ACTIVE
    facts + relations only, so corrections/obsolescence flow in and stale
    layers never pile up. Returns the regenerated entity ids (to re-vectorize).
    """
    stale = [r["id"] for r in cursor_to_dicts(conn.execute(
        "SELECT id FROM entities WHERE summary_stale = 1 AND merged_into_id IS NULL"
    ))]
    targets = list(dict.fromkeys(list(touched_ids) + stale))
    regenerated: list[str] = []
    for eid in targets:
        e = first_row(conn.execute(
            "SELECT id, canonical_name, type FROM entities "
            "WHERE id = ? AND merged_into_id IS NULL", (eid,)
        ))
        if not e:
            continue
        facts = cursor_to_dicts(conn.execute(
            "SELECT predicate, value FROM facts WHERE entity_id = ? "
            "AND obsoleted_at IS NULL AND archived_at IS NULL "
            "ORDER BY confidence DESC LIMIT 30", (eid,)
        ))
        relations = cursor_to_dicts(conn.execute(
            "SELECT r.predicate, x.canonical_name AS target FROM relations r "
            "JOIN entities x ON x.id = r.entity_to "
            "WHERE r.entity_from = ? AND r.review_status != 'pending'", (eid,)
        ))
        if not facts and not relations:
            # Nothing to derive from — keep the extraction summary, clear the flag.
            conn.execute("UPDATE entities SET summary_stale = 0 WHERE id = ?", (eid,))
            continue
        lines = [f"Entité : {e['canonical_name']}" + (f" (type {e['type']})" if e.get("type") else "")]
        if facts:
            lines.append("Faits :")
            lines += [f"- {f['predicate']} : {f['value']}" for f in facts]
        if relations:
            lines.append("Relations :")
            lines += [f"- {r['predicate']} → {r['target']}" for r in relations]
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,
                system=_RESUMMARY_SYSTEM,
                messages=[{"role": "user", "content": "\n".join(lines)}],
            )
            summary = response.content[0].text.strip()
        except anthropic.APIError as exc:
            # Infra failure — stop here; the stale flags survive for the next run.
            print(f"  ⚠ re-résumé interrompu ({type(exc).__name__}) — repris au prochain cycle")
            break
        except Exception as exc:  # noqa: BLE001 — skip this entity only
            if verbose:
                print(f"    [resummary] échec {e['canonical_name']}: {exc}")
            continue
        if summary:
            conn.execute(
                "UPDATE entities SET summary = ?, summary_stale = 0 WHERE id = ?",
                (summary, eid),
            )
            regenerated.append(eid)
            if verbose:
                print(f"    [resummary] {e['canonical_name']}: {summary[:80]!r}")
    return regenerated


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

        text = entity_embedding_text(entity)

        if dry_run:
            if verbose:
                print(f"    [vectorize] would embed '{entity['canonical_name']}'")
            continue

        try:
            vec_bytes = embed_text(text, client)
            get_store().set_entity_embedding(entity_id, vec_bytes)
            vectorized += 1
            if verbose:
                print(f"    [vectorize] embedded '{entity['canonical_name']}'")
        except Exception as exc:
            if verbose:
                print(f"    [vectorize] error for '{entity['canonical_name']}': {exc}")

    return vectorized


# ── Intentions ─────────────────────────────────────────────────────────────────



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
    capture_id: str,
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

def _mark(conn, entry_id: str, now: str, status: str, dry_run: bool = False,
          error: str | None = None) -> None:
    """Mark an inbox entry done with an outcome status (processed | failed)."""
    if dry_run:
        return
    conn.execute(
        "UPDATE inbox SET processed_at=?, status=?, error=? WHERE id=?",
        (now, status, error, entry_id),
    )


def _process_entry(entry, client, conn, now, dry_run, verbose, day_context=None, classified=None) -> tuple[list[str], list[dict]]:
    """Process one inbox entry; mark it processed. Returns (entity_ids, new_facts).

    SYN-111 : le routing déterministe (résolution, confiance, buckets, dédup
    fait⇄relation, gates, intentions, note atomique, project entries, merge/
    attach proposals, réactivation) vit dans le cœur Rust — classified JSON in,
    écritures DB out. Restent côté hôte : la classification Batch API, le fetch
    de ressources (réseau) et les sous-appels LLM (synthèse projet SYN-43),
    exécutés à partir de la work-list du rapport.

    Raises ConnectionError (HTTP/réseau — le run s'interrompt, entrées laissées
    en file) ; toute autre exception est une erreur de contenu que l'appelant
    marque 'failed'.
    """
    if classified is None:
        classified = step1_classify(entry, client, verbose, conn=conn, day_context=day_context)

    # SYN-21: resource fetch is URL-driven and independent of routing — run it
    # for ANY capture, even a pure intention. Network + LLM outside the core.
    if not dry_run:
        process_capture_resources(entry["content"], conn, client,
                                  capture_id=entry["id"], verbose=verbose)

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

    now_dt = datetime.now(timezone.utc)
    report = json.loads(get_brain().route_capture(
        json.dumps({"id": entry["id"], "content": entry["content"]}, ensure_ascii=False),
        json.dumps(classified, ensure_ascii=False),
        now,
        now_dt.date().isoformat(),
        (now_dt - timedelta(hours=48)).isoformat(),
        now_dt.strftime("%Y-%m-%d %H:%M:%S"),
    ))

    # SYN-43: la synthèse vivante des projets touchés (un appel Haiku chacun)
    # reste côté hôte — le core a persisté les entrées et renvoie la work-list.
    if client is not None:
        for s in report["project_syntheses"]:
            _append_project_summary(
                project_id=s["project_id"],
                project_name=s["project_name"],
                new_entry_content=s["entry_content"],
                new_entry_count=s["entry_count"],
                conn=conn,
                client=client,
                verbose=verbose,
            )

    return report["entity_ids"], report["new_facts"]



# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_dream_cycle(dry_run: bool = False, verbose: bool = False, use_batch: bool = False) -> None:
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
            # SYN-89: a user fact edit between cycles flags summaries stale —
            # an empty-inbox run still regenerates them (then exits).
            stale_count = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE summary_stale = 1 AND merged_into_id IS NULL"
            ).fetchone()[0]
            if not stale_count or dry_run:
                print("\n  Inbox empty — nothing to process.")
                print("═" * 60)
                return
            print(f"\n  Inbox empty — {stale_count} résumé(s) à régénérer\n")

        print(f"\n  {len(entries)} unprocessed entr{'y' if len(entries) == 1 else 'ies'} found\n")

        now = datetime.now(timezone.utc).isoformat()
        # SYN-93 — working memory: one context block for the whole batch (coreference).
        day_context = _build_day_context(conn, entries, datetime.now(timezone.utc))

        # SYN-93 — Batch API for the scheduled "sleep" pass (~-50%). Classify the
        # whole batch up front; route each entry with its pre-computed result.
        # Best-effort: any failure (or an empty result for one entry) falls back to
        # classifying synchronously, so the cycle never gets stuck on the batch path.
        classifications: dict | None = None
        if use_batch and not dry_run and len(entries) >= 2:
            try:
                classifications = _batch_classify(entries, client, conn, day_context, verbose)
                ok = sum(1 for v in classifications.values() if v is not None)
                print(f"  [batch] {ok}/{len(entries)} classifié(s) via Batch API\n")
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ batch classify échoué ({type(exc).__name__}: {exc}) — repli synchrone\n")
                classifications = None

        all_entity_ids: list[str] = []
        all_new_facts: list[dict] = []

        failed = 0
        for entry in entries:
            print(f"▸ inbox id={entry['id']}: {entry['content'][:70]!r}")
            try:
                pre = classifications.get(entry["id"]) if classifications else None
                entity_ids, new_facts = _process_entry(
                    entry, client, conn, now, dry_run, verbose,
                    day_context=day_context, classified=pre,
                )
                all_entity_ids.extend(e for e in entity_ids if e not in all_entity_ids)
                all_new_facts.extend(new_facts)
            except (anthropic.APIError, ConnectionError) as exc:
                # Infrastructure failure (no/invalid key, network, rate limit —
                # anthropic SDK on the batch path, the core's HTTP on the sync
                # path): abort the run, leave every remaining entry queued.
                print(f"  ⚠ erreur API ({type(exc).__name__}) — run interrompu, entrées laissées en file")
                raise
            except Exception as exc:  # noqa: BLE001 — content error for THIS entry
                print(f"  ⚠ échec entrée id={entry['id']}: {exc}")
                if not dry_run:
                    with conn:
                        _mark(conn, entry["id"], now, "failed",
                              error=f"{type(exc).__name__}: {exc}"[:500])
                failed += 1
            print()

        # Step 5 — Behavioral validation
        if all_new_facts:
            if verbose:
                print("▸ Step 5 — Behavioral Validation")
            promoted = 0 if dry_run else get_brain().validate_pending(
                json.dumps(all_new_facts, ensure_ascii=False))
            if promoted:
                print(f"  → {promoted} pending fact(s) promoted")

        # SYN-89 — Re-summary: entities touched by this run + stale ones (user edits).
        if not dry_run:
            with conn:
                resummed = step_resummarize(all_entity_ids, conn, client, verbose)
            if resummed:
                print(f"  → {len(resummed)} résumé(s) d'entité régénéré(s)")
            all_entity_ids.extend(e for e in resummed if e not in all_entity_ids)

        # Step 6 — Vectorize touched entities
        if all_entity_ids:
            print("▸ Step 6 — Vectorization")
            # SYN-110: no `with conn:` here — the writes go through the core's
            # own connection; holding an apsw transaction would deadlock them.
            vectorized = step6_vectorize(all_entity_ids, conn, client, dry_run, verbose)
            print(f"  → {vectorized} entit{'y' if vectorized == 1 else 'ies'} vectorized")

        # SYN-19 / SYN-68 — refresh Ebbinghaus memory_strength on notes AND
        # entities (cadence-free recompute; both anchor on a reactivation date).
        if not dry_run:
            with conn:
                decayed = apply_decay(conn)
                decayed_entities = apply_entity_decay(conn)
            if verbose and (decayed or decayed_entities):
                print(f"  → decay refreshed {decayed} note(s), {decayed_entities} entit{'y' if decayed_entities == 1 else 'ies'}")

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
