"""
SYN-19 — Ebbinghaus graceful forgetting for atomic_notes (+ entities, SYN-68).

T5 (SYN-114): the implementation lives in the Rust core (`decay.rs` — the same
module the routing path uses for mention reactivation; one implementation, zero
drift). This module is the host-side shim: historical signatures preserved
(`now` as datetime for the tests' fixed clock), writes run on the CALLER's
connection so an open `with conn:` transaction wraps them.

`memory_strength = exp(-Δdays / τ)`, recomputed from elapsed time — cadence-free
(a missed or doubled run can't corrupt it). τ = SYNAPSE_DECAY_TAU_DAYS (30 d).
"""

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_connection, init_db

TAU_DAYS = float(os.getenv("SYNAPSE_DECAY_TAU_DAYS", "30"))
_SQLITE_FMT = "%Y-%m-%d %H:%M:%S"


def _now_sql(now: datetime | None) -> str | None:
    """None → the core uses the system clock; naive/aware → 'YYYY-MM-DD HH:MM:SS'."""
    if now is None:
        return None
    return (now.replace(tzinfo=None) if now.tzinfo else now).strftime(_SQLITE_FMT)


def apply_decay(conn, *, tau_days: float | None = None, now: datetime | None = None) -> int:
    """Recompute memory_strength for every atomic_note from elapsed time. Returns
    the number of notes updated. Caller owns the transaction."""
    return conn.apply_decay(tau_days, _now_sql(now))


def apply_entity_decay(conn, *, tau_days: float | None = None,
                       now: datetime | None = None) -> int:
    """Same cadence-free law for entities, anchored on `last_mentioned` (SYN-68)."""
    return conn.apply_entity_decay(tau_days, _now_sql(now))


def reactivate_notes(conn, note_ids, *, factor: float = 1.0,
                     now: datetime | None = None) -> int:
    """Move `last_reactivated_at` toward now. factor=1.0 → full reset (mention);
    0<factor<1 → partial 'light' bump (search hit). Returns the count touched."""
    return conn.reactivate_notes(note_ids, factor, _now_sql(now))


def reactivate_notes_for_entities(conn, entity_names, *, now: datetime | None = None) -> int:
    """Strong reactivation of every note that mentions any of `entity_names`."""
    names = [n for n in (entity_names or []) if n]
    if not names:
        return 0
    return conn.reactivate_notes_for_entities(names, _now_sql(now))


def run_decay() -> None:
    init_db()
    conn = get_connection()
    try:
        with conn:
            n = apply_decay(conn)
            m = apply_entity_decay(conn)
        print(f"Decay applied to {n} atomic_note(s) and {m} entit{'y' if m == 1 else 'ies'} (τ={TAU_DAYS}d).")
    finally:
        conn.close()


if __name__ == "__main__":
    run_decay()
