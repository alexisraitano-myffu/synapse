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



def _llm_args() -> tuple[str, str | None, str | None]:
    """(api_key, base_url, fuel_token) pour les appels LLM du core — la
    résolution de la clé et le seam fuel-proxy (SYN-105) restent côté hôte.
    Lève EnvironmentError sans clé (même message que step1_classify)."""
    from anthropic_client import is_fuel_token, _fuel_base_url
    from config_store import get_anthropic_key

    key = get_anthropic_key()
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY manquante — exporte-la, mets-la dans .env, ou "
            "règle-la depuis l'app (Réglages → Clé Anthropic API)."
        )
    if is_fuel_token(key):
        return "", _fuel_base_url(), key
    return key, None, None


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

# SYN-89 : le prompt du re-résumé est de la donnée (prompts/resummary.md,
# repo synapse-core, déployé dans ~/.synapse/prompts) — lu par le core.


def step_resummarize(
    touched_ids: list[str],
    conn,
    client,
    verbose: bool = False,
) -> list[str]:
    """SYN-89 — regenerate entity summaries from scratch (derived, never edited).

    T5 : la passe vit dans le core (`Brain.resummarize`, prompt = data). Cibles =
    entités touchées par le run + summary_stale ; reconstruit depuis les faits
    ACTIFS + relations non-pending. Une erreur HTTP arrête la passe (les flags
    stale survivent) ; retourne les ids régénérés (à re-vectoriser). Le core
    écrit sur sa propre connexion : ne PAS appeler sous `with conn:`.
    `conn`/`client` gardés pour la signature historique."""
    key, base_url, fuel = _llm_args()
    raw = get_brain().resummarize(
        list(touched_ids), CLAUDE_MODEL, key, str(PROMPTS_DIR), _TODAY,
        base_url=base_url, fuel_token=fuel,
    )
    regenerated = json.loads(raw)
    if verbose:
        for eid in regenerated:
            print(f"    [resummary] regenerated {eid}")
    return regenerated



# ── Step 6 — Vectorization ────────────────────────────────────────────────────

def step6_vectorize(
    entity_ids: list[str],
    conn,
    client=None,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """T5 : l'embed + write des entités touchées vit dans le core
    (`Brain.vectorize_entities` — texte composite + vecteur, échecs par entité
    ignorés comme avant). `conn`/`client` gardés pour la signature historique."""
    if dry_run:
        if verbose:
            print(f"    [vectorize] would embed {len(entity_ids)} entit(ies)")
        return 0
    return get_brain().vectorize_entities(list(entity_ids))


# ── Intentions ─────────────────────────────────────────────────────────────────



def synthesize_project(project_id: str, project_name: str,
                       entry_content: str, entry_count: int,
                       verbose: bool = False) -> str | None:
    """SYN-43/44 — synthèse vivante d'un projet après une nouvelle entrée.

    T5 : append + refinement (seuil SYNAPSE_REFINEMENT_THRESHOLD) vivent dans
    le core (`Brain.synthesize_project`, prompts = data). Un échec LLM ne
    bloque jamais (retourne None) ; sans clé configurée, no-op silencieux
    (l'entrée est déjà persistée, la synthèse rattrapera au prochain append).
    Appeler HORS transaction hôte (le core écrit sur sa propre connexion)."""
    try:
        key, base_url, fuel = _llm_args()
    except EnvironmentError:
        return None
    summary = get_brain().synthesize_project(
        project_id, project_name, entry_content, int(entry_count),
        CLAUDE_MODEL, key, str(PROMPTS_DIR), _TODAY,
        base_url=base_url, fuel_token=fuel,
    )
    if verbose and summary:
        print(f"    [project_summary] v{entry_count} for '{project_name}'")
    return summary


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
            synthesize_project(
                s["project_id"], s["project_name"],
                s["entry_content"], s["entry_count"], verbose=verbose,
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
            # T5 : le core écrit sur sa propre connexion — pas de `with conn:`.
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
