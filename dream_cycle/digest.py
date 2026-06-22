"""
SYN-23 — Weekly digest.

A cron-driven job (Monday 08h via launchd, see CLAUDE.md) that condenses the
past week AND the week ahead into one durable note (`kind="digest"`). The API
backend also self-heals it (`_ensure_weekly_digest`): if the Mac was asleep at
the scheduled fire, the current ISO week's digest is generated on the next
hourly check once the machine is awake — so it never silently goes missing.

- Retrospective: new entities, new facts, new notes, and the entities most
  reactivated over the window ("tendances").
- Forward-looking: dated events in the next 7 days (incl. recurring birthdays)
  + open tasks — the data SYN-85 made available, so the digest doubles as a
  Sunday review.

The structured week is gathered with plain SQL (`gather_week`, offline-testable),
rendered to French markdown by Haiku (`summarize_digest`, timeless rule — absolute
dates only), then stored as an `atomic_note` (kind="digest", memory_strength=1.0,
vectorized so it's searchable). Idempotent per ISO week: re-running overwrites the
week's digest instead of stacking duplicates.

Run: python -m dream_cycle.digest          (+ --dry-run to preview without writing)
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from config import CLAUDE_MODEL
from config_store import get_anthropic_key
from db import cursor_to_dicts, first_row, get_connection, init_db
from embeddings import embed_text

# Bounds so a busy week doesn't blow up the prompt — the digest is a summary,
# not an exhaustive log.
_MAX_ENTITIES = 25
_MAX_FACTS = 25
_MAX_NOTES = 25
_MAX_TRENDS = 8
_MAX_TASKS = 20


def _get_client() -> anthropic.Anthropic:
    key = get_anthropic_key()
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY manquante — requise pour le digest hebdo "
            "(ou règle-la depuis l'app : Réglages → Clé Anthropic API)."
        )
    return anthropic.Anthropic(api_key=key)


def _now(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(tzinfo=None) if now.tzinfo else now


def _week_start(now: datetime) -> date:
    """Monday of the current ISO week (the digest's stable identity)."""
    d = now.date()
    return d - timedelta(days=d.weekday())


# ── 1. Gather the structured week (pure SQL — offline-testable) ──────────────────

def gather_week(conn, *, now: datetime | None = None, days: int = 7) -> dict:
    """Collect the retrospective (past `days`) and forward-looking (next `days`)
    material for the digest. No API call — safe to unit-test offline."""
    now_n = _now(now)
    since = (now_n - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    today = now_n.date()
    horizon = today + timedelta(days=days)

    new_entities = cursor_to_dicts(conn.execute(
        "SELECT canonical_name, type FROM entities "
        "WHERE created_at >= ? AND (status IS NULL OR status = 'active') "
        "AND merged_into_id IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (since, _MAX_ENTITIES),
    ))

    new_facts = cursor_to_dicts(conn.execute(
        "SELECT e.canonical_name AS entity, f.predicate, f.value "
        "FROM facts f JOIN entities e ON e.id = f.entity_id "
        "WHERE f.created_at >= ? AND f.archived_at IS NULL AND f.obsoleted_at IS NULL "
        "ORDER BY f.created_at DESC LIMIT ?",
        (since, _MAX_FACTS),
    ))

    validated_count = first_row(conn.execute(
        "SELECT COUNT(*) AS n FROM validation_events "
        "WHERE confirmed = 1 AND created_at >= ?",
        (since,),
    ))["n"]

    new_notes = cursor_to_dicts(conn.execute(
        "SELECT title, content, kind FROM atomic_notes "
        "WHERE created_at >= ? AND archived_at IS NULL "
        "AND kind IN ('note', 'task', 'event') "
        "ORDER BY created_at DESC LIMIT ?",
        (since, _MAX_NOTES),
    ))

    # Tendances: entities reactivated (mentioned) over the window, busiest first.
    window_start = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    trends = cursor_to_dicts(conn.execute(
        "SELECT canonical_name, type, mention_count FROM entities "
        "WHERE last_mentioned >= ? AND (status IS NULL OR status = 'active') "
        "AND merged_into_id IS NULL "
        "ORDER BY mention_count DESC, last_mentioned DESC LIMIT ?",
        (window_start, _MAX_TRENDS),
    ))

    captures = first_row(conn.execute(
        "SELECT COUNT(*) AS n FROM inbox WHERE created_at >= ?", (since,)
    ))["n"]

    # Forward-looking — dated events AND dated tasks (SYN-23) not archived, within
    # the horizon. Recurring (birthdays) compared on month-day; one-shots on the
    # absolute date. Filtered in Python so the year-boundary case stays correct.
    dated_raw = cursor_to_dicts(conn.execute(
        "SELECT title, content, kind, event_date, event_recurring FROM atomic_notes "
        "WHERE kind IN ('event', 'task') AND archived_at IS NULL AND event_date IS NOT NULL"
    ))
    upcoming_events = []
    for ev in dated_raw:
        occ = _next_occurrence(ev["event_date"], bool(ev["event_recurring"]), today)
        if occ is not None and today <= occ <= horizon:
            upcoming_events.append({
                "title": ev["title"], "content": ev["content"], "kind": ev["kind"],
                "date": occ.strftime("%Y-%m-%d"),
                "recurring": bool(ev["event_recurring"]),
            })
    upcoming_events.sort(key=lambda e: e["date"])

    # Open tasks WITHOUT a date (dated ones already surface under « à venir »).
    open_tasks = cursor_to_dicts(conn.execute(
        "SELECT title, content FROM atomic_notes "
        "WHERE kind = 'task' AND archived_at IS NULL AND event_date IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (_MAX_TASKS,),
    ))

    return {
        "week_start": _week_start(now_n).strftime("%Y-%m-%d"),
        "generated_at": today.strftime("%Y-%m-%d"),
        "days": days,
        "counts": {
            "captures": captures,
            "new_entities": len(new_entities),
            "new_facts": len(new_facts),
            "validated_facts": validated_count,
            "new_notes": len(new_notes),
        },
        "new_entities": new_entities,
        "new_facts": new_facts,
        "new_notes": new_notes,
        "trends": trends,
        "upcoming_events": upcoming_events,
        "open_tasks": open_tasks,
    }


def _next_occurrence(event_date: str, recurring: bool, today: date) -> date | None:
    """Resolve an event's next concrete date. One-shots return their absolute
    date; recurring ones return this year's (or next year's) matching month-day."""
    try:
        d = datetime.strptime(event_date.strip()[:10], "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None
    if not recurring:
        return d
    try:
        this_year = d.replace(year=today.year)
    except ValueError:  # 29 Feb on a non-leap year → treat as 1 Mar
        this_year = date(today.year, 3, 1)
    return this_year if this_year >= today else this_year.replace(year=today.year + 1)


def has_content(week: dict) -> bool:
    """True if the week is worth a digest (something happened or is coming up)."""
    c = week["counts"]
    return any((
        c["captures"], c["new_entities"], c["new_facts"], c["new_notes"],
        week["upcoming_events"], week["open_tasks"],
    ))


# ── 2. Render to markdown (Haiku) ────────────────────────────────────────────────

_DIGEST_SYSTEM = """\
Tu rédiges le DIGEST HEBDOMADAIRE d'une mémoire personnelle (système Synapse).
On te donne, en JSON, la matière de la semaine écoulée et de la semaine à venir.

Produis un markdown FRANÇAIS concis et vivant (~250–400 mots), structuré :

## Cette semaine
Un court paragraphe de synthèse (ce qui ressort), puis des puces pour les
nouvelles entités notables, les faits marquants et les notes/réflexions. Mets en
avant les TENDANCES (entités les plus actives).

## À venir
Les éléments datés des 7 prochains jours (événements ET tâches à échéance, avec
leur date — `upcoming_events`) puis les tâches ouvertes sans date à ne pas
oublier (`open_tasks`). Si rien n'est à venir, écris une ligne sobre.

RÈGLES STRICTES :
- INTEMPOREL : uniquement des dates ABSOLUES (« le 24 juin »), jamais de relatif
  (« la semaine prochaine », « demain »). Le digest sera relu dans des mois.
- Pas d'invention : ne mentionne que ce qui est dans le JSON. Si une section est
  vide, dis-le brièvement plutôt que de meubler.
- Ton sobre, factuel, à la 2e personne (« tu »). Pas de salutations ni de blabla.
- Commence directement par « ## Cette semaine ». N'ajoute pas de titre H1.
"""


def summarize_digest(week: dict, *, client: anthropic.Anthropic | None = None) -> str:
    """Render the gathered week into French markdown via Haiku."""
    client = client or _get_client()
    payload = json.dumps(week, ensure_ascii=False, indent=2)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1400,
        system=_DIGEST_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Matière de la semaine (semaine du {week['week_start']}) :\n\n{payload}",
        }],
    )
    if not resp.content:
        raise RuntimeError("digest: réponse Haiku vide")
    return resp.content[0].text.strip()


# ── 3. Persist as an atomic_note (kind="digest", idempotent per week) ─────────────

def _digest_title(week: dict) -> str:
    return f"Digest — semaine du {week['week_start']}"


def write_digest_note(conn, week: dict, markdown: str) -> int:
    """Store the digest as an atomic_note (kind='digest'), replacing any existing
    digest for the same ISO week. Vectorized so search_memory surfaces it."""
    title = _digest_title(week)
    names = [e["canonical_name"] for e in week["new_entities"]]
    names += [t["canonical_name"] for t in week["trends"] if t["canonical_name"] not in names]
    summary = f"Digest hebdo : {week['counts']['captures']} captures, " \
              f"{week['counts']['new_entities']} entités, {week['counts']['new_facts']} faits."

    with conn:
        # Idempotent: drop the previous digest for this week (note + its vector row).
        stale = [r["id"] for r in cursor_to_dicts(conn.execute(
            "SELECT id FROM atomic_notes WHERE kind = 'digest' AND title = ?", (title,)
        ))]
        for nid in stale:
            conn.execute("DELETE FROM atomic_notes_vec WHERE rowid = ?", (nid,))
            conn.execute("DELETE FROM atomic_notes WHERE id = ?", (nid,))

        conn.execute(
            "INSERT INTO atomic_notes "
            "(title, content, summary, entities_mentioned, memory_strength, kind) "
            "VALUES (?,?,?,?,?, 'digest')",
            (title, markdown, summary, json.dumps(names, ensure_ascii=False), 1.0),
        )
        note_id = conn.last_insert_rowid()
        try:
            vec = embed_text(f"{title}\n{markdown}")
            conn.execute(
                "INSERT OR REPLACE INTO atomic_notes_vec(rowid, embedding) VALUES (?, ?)",
                (note_id, vec),
            )
        except Exception:  # noqa: BLE001 — vectorization is best-effort
            pass
    return note_id


# ── Orchestration ────────────────────────────────────────────────────────────────

def generate_weekly_digest(
    conn=None,
    *,
    now: datetime | None = None,
    days: int = 7,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Gather → render → store one weekly digest. Returns a JSON-serializable
    result (also used by the API). Skips writing on an empty week."""
    init_db()
    owns_conn = conn is None
    conn = conn or get_connection()
    try:
        week = gather_week(conn, now=now, days=days)
        title = _digest_title(week)
        if not has_content(week):
            if verbose:
                print(f"[digest] semaine du {week['week_start']} vide — rien à résumer.")
            return {"note_id": None, "title": title, "week_start": week["week_start"],
                    "markdown": None, "skipped": "empty"}

        markdown = summarize_digest(week)
        if verbose:
            print(f"[digest] {title}\n\n{markdown}\n")
        if dry_run:
            return {"note_id": None, "title": title, "week_start": week["week_start"],
                    "markdown": markdown, "skipped": "dry-run"}

        note_id = write_digest_note(conn, week, markdown)
        if verbose:
            print(f"[digest] écrit atomic_note id={note_id}")
        return {"note_id": note_id, "title": title, "week_start": week["week_start"],
                "markdown": markdown}
    finally:
        if owns_conn:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère le digest hebdomadaire (SYN-23).")
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans écrire en base.")
    parser.add_argument("--days", type=int, default=7, help="Fenêtre en jours (défaut 7).")
    parser.add_argument("--verbose", action="store_true", help="Logs détaillés.")
    args = parser.parse_args()
    result = generate_weekly_digest(dry_run=args.dry_run, days=args.days, verbose=True)
    if result.get("skipped") == "empty":
        print("Rien à résumer cette semaine.")


if __name__ == "__main__":
    main()
