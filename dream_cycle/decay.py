"""
SYN-19 — Ebbinghaus graceful forgetting for atomic_notes.

`memory_strength = exp(-Δdays / TAU)` where Δdays is the time since the note was
last reactivated (created, mentioned in a new capture, or hit in search). The
score is **recomputed from elapsed time**, never decremented in place — so it's
independent of how often the decay job runs (a missed or doubled run can't
corrupt it). Reactivation moves `last_reactivated_at` toward now, so the
strength springs back up.

- Decay job: `apply_decay(conn)` — recompute every note. Cadence-free; called at
  the top of each Dream Cycle and runnable standalone (`python -m dream_cycle.decay`).
- Mention bump (strong): `reactivate_notes_for_entities` — a note whose
  entities_mentioned includes a freshly-captured entity is fully reactivated.
- Search bump (light): `reactivate_notes(..., factor<1)` — a search hit only
  partially rejuvenates the note.
"""

import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import cursor_to_dicts, first_row, get_connection, init_db

TAU_DAYS = float(os.getenv("SYNAPSE_DECAY_TAU_DAYS", "30"))
_SQLITE_FMT = "%Y-%m-%d %H:%M:%S"


def _parse(ts: str) -> datetime:
    """Parse a SQLite timestamp ('YYYY-MM-DD HH:MM:SS' or ISO) as naive UTC."""
    if not ts:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    ts = ts.strip().replace("T", " ")
    # drop any timezone suffix / fractional seconds for a tolerant parse
    ts = ts.split("+")[0].split(".")[0].strip()
    try:
        return datetime.strptime(ts, _SQLITE_FMT)
    except ValueError:
        return datetime.now(timezone.utc).replace(tzinfo=None)


def _now_naive(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(tzinfo=None) if now.tzinfo else now


def apply_decay(conn, *, tau_days: float | None = None, now: datetime | None = None) -> int:
    """Recompute memory_strength for every atomic_note from elapsed time. Returns
    the number of notes updated. Caller owns the transaction."""
    tau = tau_days if tau_days is not None else TAU_DAYS
    now_n = _now_naive(now)
    rows = cursor_to_dicts(conn.execute(
        "SELECT id, created_at, last_reactivated_at FROM atomic_notes"))
    for r in rows:
        base = _parse(r["last_reactivated_at"] or r["created_at"])
        delta_days = max(0.0, (now_n - base).total_seconds() / 86400.0)
        strength = math.exp(-delta_days / tau)
        conn.execute("UPDATE atomic_notes SET memory_strength = ? WHERE id = ?",
                     (strength, r["id"]))
    return len(rows)


def reactivate_notes(conn, note_ids, *, factor: float = 1.0,
                     now: datetime | None = None) -> int:
    """Move `last_reactivated_at` toward now for each note. factor=1.0 → full
    reset (mention); 0<factor<1 → partial 'light' bump (search hit). Returns the
    count touched. Caller owns the transaction."""
    now_n = _now_naive(now)
    touched = 0
    for nid in note_ids:
        row = first_row(conn.execute(
            "SELECT created_at, last_reactivated_at FROM atomic_notes WHERE id = ?", (nid,)))
        if not row:
            continue
        base = _parse(row["last_reactivated_at"] or row["created_at"])
        new = now_n if factor >= 1.0 else base + (now_n - base) * factor
        conn.execute(
            "UPDATE atomic_notes SET last_reactivated_at = ? WHERE id = ?",
            (new.strftime(_SQLITE_FMT), nid))
        touched += 1
    return touched


def reactivate_notes_for_entities(conn, entity_names, *, now: datetime | None = None) -> int:
    """Strong reactivation of every note that mentions any of `entity_names`
    (matched against the entities_mentioned JSON array)."""
    names = [n for n in (entity_names or []) if n]
    if not names:
        return 0
    ids: set[str] = set()
    for name in names:
        for r in conn.execute(
            "SELECT id FROM atomic_notes WHERE entities_mentioned LIKE ?",
            (f'%"{name}"%',),
        ):
            ids.add(r[0])
    return reactivate_notes(conn, ids, factor=1.0, now=now)


def run_decay() -> None:
    init_db()
    conn = get_connection()
    try:
        with conn:
            n = apply_decay(conn)
        print(f"Decay applied to {n} atomic_note(s) (τ={TAU_DAYS}d).")
    finally:
        conn.close()


if __name__ == "__main__":
    run_decay()
