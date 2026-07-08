"""
SYN-23 — Weekly digest.

A cron-driven job (Monday 08h via launchd, see CLAUDE.md) that condenses the
past week AND the week ahead into one durable note (`kind="digest"`). The API
backend also self-heals it (`_ensure_weekly_digest`): if the Mac was asleep at
the scheduled fire, the current ISO week's digest is generated on the next
hourly check once the machine is awake — so it never silently goes missing.

- Retrospective: new entities, new facts, new notes, and the entities most
  reactivated over the window ("tendances").
- Forward-looking: dated events in the next 7 days (incl. recurring birthdays,
  both as event notes and as `has_birthday` facts — SYN-97) + open tasks — the
  data SYN-85 made available, so the digest doubles as a Sunday review.

T5 (SYN-114): the logic lives in the core (`digest.rs`). `gather_week` is pure
SQL on the caller's connection (offline-testable); the French markdown is
rendered by Haiku through the core's HTTP path with the prompt as DATA
(`prompts/digest.md`, timeless rule — absolute dates only); the note write +
vector go through `Brain.write_digest_note` (idempotent per ISO week:
re-running overwrites the week's digest instead of stacking duplicates). This
module keeps the historical signatures and the orchestration only.

Run: python -m dream_cycle.digest          (+ --dry-run to preview without writing)
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import synapse_core

from config import CLAUDE_MODEL
from core_store import get_brain
from db import get_connection, init_db
from dream_cycle.cycle import PROMPTS_DIR, _TODAY, _llm_args


def _now_sql(now: datetime | None) -> str | None:
    """datetime → 'YYYY-MM-DD HH:MM:SS' for the core's injectable clock."""
    if now is None:
        return None
    n = now.replace(tzinfo=None) if now.tzinfo else now
    return n.strftime("%Y-%m-%d %H:%M:%S")


# ── 1. Gather the structured week (pure SQL in the core — offline-testable) ──────

def gather_week(conn, *, now: datetime | None = None, days: int = 7) -> dict:
    """Collect the retrospective (past `days`) and forward-looking (next `days`)
    material for the digest. No API call — runs on THIS connection (core
    `digest.rs::gather_week`), safe to unit-test offline."""
    return json.loads(conn.gather_week(now=_now_sql(now), days=days))


def _next_occurrence(event_date: str, recurring: bool, today: date) -> date | None:
    """Resolve an event's next concrete date (core logic; this shim keeps the
    historical date-object signature for the tests)."""
    iso = synapse_core.next_occurrence(
        event_date or "", bool(recurring), today.strftime("%Y-%m-%d"))
    return date.fromisoformat(iso) if iso else None


def has_content(week: dict) -> bool:
    """True if the week is worth a digest (something happened or is coming up)."""
    c = week["counts"]
    return any((
        c["captures"], c["new_entities"], c["new_facts"], c["new_notes"],
        week["upcoming_events"], week["open_tasks"],
    ))


# ── 2. Render to markdown (Haiku via the core) ────────────────────────────────────

# SYN-23 : le prompt du digest est de la donnée (prompts/digest.md, repo
# synapse-core, déployé dans ~/.synapse/prompts) — lu par le core.

def summarize_digest(week: dict, *, client=None) -> str:
    """Render the gathered week into French markdown via Haiku (core HTTP
    path). `client` kept for the historical signature (ignored)."""
    key, base_url, fuel = _llm_args()
    return get_brain().summarize_digest(
        json.dumps(week, ensure_ascii=False), CLAUDE_MODEL, key,
        str(PROMPTS_DIR), _TODAY, base_url=base_url, fuel_token=fuel,
    )


# ── 3. Persist as an atomic_note (kind="digest", idempotent per week) ─────────────

def _digest_title(week: dict) -> str:
    return f"Digest — semaine du {week['week_start']}"


def write_digest_note(conn, week: dict, markdown: str) -> str:
    """Store the digest as an atomic_note (kind='digest'), replacing any
    existing digest for the same ISO week; vectorized so search_memory
    surfaces it. The core does note + vector on ITS OWN connection — call
    OUTSIDE `with conn:` (`conn` kept for the historical signature)."""
    return get_brain().write_digest_note(
        json.dumps(week, ensure_ascii=False), markdown)


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
