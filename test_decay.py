"""
Offline tests for SYN-19 — Ebbinghaus memory_strength decay + reactivation.

Time is injected via the `now` parameter so the math is deterministic.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

NOW = datetime(2026, 5, 31, 12, 0, 0)
FMT = "%Y-%m-%d %H:%M:%S"


def _note(conn, nid, *, created_days_ago=0, reactivated_days_ago=None, mentions='["Marie"]'):
    created = (NOW - timedelta(days=created_days_ago)).strftime(FMT)
    react = (None if reactivated_days_ago is None
             else (NOW - timedelta(days=reactivated_days_ago)).strftime(FMT))
    conn.execute(
        "INSERT INTO atomic_notes (id, content, created_at, last_reactivated_at, "
        "entities_mentioned) VALUES (?,?,?,?,?)",
        (nid, f"note {nid}", created, react, mentions),
    )  # id is INTEGER PRIMARY KEY (vec0 rowid mirror) — use int nids


def _strength(conn, nid):
    return conn.execute(
        "SELECT memory_strength FROM atomic_notes WHERE id=?", (nid,)).fetchone()[0]


def test_old_note_is_forgotten_fresh_note_retained(isolated_db):
    from db import get_connection
    from dream_cycle.decay import apply_decay
    conn = get_connection()
    try:
        with conn:
            _note(conn, 1, created_days_ago=365)               # never reactivated
            _note(conn, 2, created_days_ago=200, reactivated_days_ago=1)
            apply_decay(conn, now=NOW)
        old, fresh = _strength(conn, 1), _strength(conn, 2)
    finally:
        conn.close()
    assert old < 0.05, f"a year-old note should be nearly forgotten, got {old}"
    assert fresh > 0.5, f"a note reactivated yesterday should stay strong, got {fresh}"


def test_mention_fully_reactivates(isolated_db):
    from db import get_connection
    from dream_cycle.decay import apply_decay, reactivate_notes_for_entities
    conn = get_connection()
    try:
        with conn:
            _note(conn, 1, created_days_ago=365)                  # mentions "Marie"
            reactivate_notes_for_entities(conn, ["Marie"], now=NOW)
            apply_decay(conn, now=NOW)
        s = _strength(conn, 1)
    finally:
        conn.close()
    assert s > 0.9, f"a mentioned note springs back to ~1.0, got {s}"


def test_reactivation_only_targets_mentioned_notes(isolated_db):
    from db import get_connection
    from dream_cycle.decay import apply_decay, reactivate_notes_for_entities
    conn = get_connection()
    try:
        with conn:
            _note(conn, 1, created_days_ago=365, mentions='["Marie"]')
            _note(conn, 2, created_days_ago=365, mentions='["Lucas"]')
            reactivate_notes_for_entities(conn, ["Marie"], now=NOW)
            apply_decay(conn, now=NOW)
        marie, other = _strength(conn, 1), _strength(conn, 2)
    finally:
        conn.close()
    assert marie > 0.9 and other < 0.05


def test_entity_decay_old_vs_recent(isolated_db):
    """SYN-68 — entities decay from last_mentioned like notes decay from
    last_reactivated_at."""
    from db import get_connection
    from dream_cycle.decay import apply_entity_decay
    conn = get_connection()
    try:
        with conn:
            old = (NOW - timedelta(days=365)).strftime(FMT)
            recent = (NOW - timedelta(days=1)).strftime(FMT)
            conn.execute("INSERT INTO entities (id, canonical_name, last_mentioned) "
                         "VALUES ('e_old','Old',?)", (old,))
            conn.execute("INSERT INTO entities (id, canonical_name, last_mentioned) "
                         "VALUES ('e_new','New',?)", (recent,))
            apply_entity_decay(conn, now=NOW)
        s_old = conn.execute("SELECT memory_strength FROM entities WHERE id='e_old'").fetchone()[0]
        s_new = conn.execute("SELECT memory_strength FROM entities WHERE id='e_new'").fetchone()[0]
    finally:
        conn.close()
    assert s_old < 0.05, f"a year-stale entity should be nearly forgotten, got {s_old}"
    assert s_new > 0.9, f"an entity mentioned yesterday should stay strong, got {s_new}"


def test_search_hit_is_a_light_bump(isolated_db):
    from db import get_connection
    from dream_cycle.decay import apply_decay, reactivate_notes
    conn = get_connection()
    try:
        with conn:
            _note(conn, 1, created_days_ago=60)
            apply_decay(conn, now=NOW)
        before = _strength(conn, 1)
        with conn:
            reactivate_notes(conn, [1], factor=0.5, now=NOW)     # search hit
            apply_decay(conn, now=NOW)
        after = _strength(conn, 1)
    finally:
        conn.close()
    assert after > before, "a search hit should rejuvenate the note"
    assert after < 0.99, "but a light bump is not a full reset"
