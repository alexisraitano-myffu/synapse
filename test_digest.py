"""
SYN-23 — weekly digest tests.

Offline: `gather_week` bucketing (retrospective window + forward-looking events
/ tasks, incl. recurring-birthday resolution) and idempotent `write_digest_note`.
The Haiku rendering test is skipped unless ANTHROPIC_API_KEY is set.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from db import get_connection
from dream_cycle.digest import (
    _next_occurrence,
    gather_week,
    generate_weekly_digest,
    has_content,
    summarize_digest,
    write_digest_note,
)

_FMT = "%Y-%m-%d %H:%M:%S"


def _add_entity(conn, name, *, created, last_mentioned=None, mentions=1, status="active"):
    conn.execute(
        "INSERT INTO entities (id, type, canonical_name, mention_count, persistence_value, "
        "last_mentioned, created_at, status) VALUES (?,?,?,?,?,?,?,?)",
        (name.lower(), "person", name, mentions, 3, last_mentioned, created, status),
    )
    return name.lower()


def _add_note(conn, content, *, kind="note", created, event_date=None, recurring=0, archived=None):
    conn.execute(
        "INSERT INTO atomic_notes (title, content, summary, entities_mentioned, memory_strength, "
        "kind, event_date, event_recurring, created_at, archived_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (content[:60], content, None, "[]", 1.0, kind, event_date, recurring, created, archived),
    )


def _add_fact(conn, entity_id, predicate, value, *, created, confidence=0.9):
    conn.execute(
        "INSERT INTO facts (id, entity_id, predicate, value, confidence, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (f"f-{entity_id}-{predicate}", entity_id, predicate, value, confidence, created),
    )


def test_gather_week_retrospective_window(isolated_db):
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    recent = (now - timedelta(days=2)).strftime(_FMT)
    old = (now - timedelta(days=30)).strftime(_FMT)
    conn = get_connection()
    try:
        with conn:
            _add_entity(conn, "Maxwell", created=recent, last_mentioned="2026-06-16", mentions=5)
            _add_entity(conn, "Ancien", created=old, last_mentioned="2026-05-10", mentions=2)
            eid = "maxwell"
            conn.execute(
                "INSERT INTO facts (id, entity_id, predicate, value, confidence, created_at) "
                "VALUES (?,?,?,?,?,?)",
                ("f1", eid, "works_at", "OpenAI", 0.9, recent),
            )
            conn.execute(
                "INSERT INTO validation_events (id, confirmed, created_at) VALUES (?,?,?)",
                ("v1", 1, recent),
            )
            _add_note(conn, "Une réflexion récente", created=recent)
        week = gather_week(conn, now=now, days=7)
    finally:
        conn.close()

    names = [e["canonical_name"] for e in week["new_entities"]]
    assert "Maxwell" in names and "Ancien" not in names      # window filters the old one
    assert week["counts"]["new_facts"] == 1
    assert week["counts"]["validated_facts"] == 1
    assert week["counts"]["new_notes"] == 1
    assert any(t["canonical_name"] == "Maxwell" for t in week["trends"])
    assert has_content(week)


def test_gather_week_forward_looking(isolated_db):
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    today = now.date()
    soon = (today + timedelta(days=3)).strftime("%Y-%m-%d")
    far = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    created = now.strftime(_FMT)
    conn = get_connection()
    try:
        with conn:
            _add_note(conn, "Salon Vivatech", kind="event", event_date=soon, created=created)
            _add_note(conn, "Conf lointaine", kind="event", event_date=far, created=created)
            _add_note(conn, "Refondre le design", kind="task", created=created)
            _add_note(conn, "Tâche faite", kind="task", created=created,
                      archived=created)  # archived → excluded
        week = gather_week(conn, now=now, days=7)
    finally:
        conn.close()

    ev_titles = [e["title"] for e in week["upcoming_events"]]
    assert "Salon Vivatech" in ev_titles and "Conf lointaine" not in ev_titles
    assert len(week["open_tasks"]) == 1
    assert week["open_tasks"][0]["title"] == "Refondre le design"


def test_recurring_birthday_resolves_into_window(isolated_db):
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    today = now.date()
    # A birthday two days out, recorded years ago, recurring yearly.
    bday = today.replace(year=1990) + timedelta(days=2)
    conn = get_connection()
    try:
        with conn:
            _add_note(conn, "Anniversaire Cici", kind="event",
                      event_date=bday.strftime("%Y-%m-%d"), recurring=1,
                      created=now.strftime(_FMT))
        week = gather_week(conn, now=now, days=7)
    finally:
        conn.close()
    assert any(e["title"] == "Anniversaire Cici" for e in week["upcoming_events"])


def test_birthday_fact_surfaces_as_upcoming(isolated_db):
    # SYN-97 — a has_birthday fact within 7 days must appear under « à venir ».
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    today = now.date()
    soon = (today.replace(year=1990) + timedelta(days=2)).strftime("%Y-%m-%d")  # 2 days out
    far = (today.replace(year=1985) + timedelta(days=30)).strftime("%Y-%m-%d")  # >7 days
    created = now.strftime(_FMT)
    conn = get_connection()
    try:
        with conn:
            eid = _add_entity(conn, "Benjamin", created=created)
            fid = _add_entity(conn, "Lointain", created=created)
            _add_fact(conn, eid, "has_birthday", soon, created=created)
            _add_fact(conn, fid, "has_birthday", far, created=created)
        week = gather_week(conn, now=now, days=7)
    finally:
        conn.close()
    bdays = [e for e in week["upcoming_events"] if e["kind"] == "birthday"]
    assert any(e["title"] == "Anniversaire de Benjamin" for e in bdays)
    assert all(e["recurring"] for e in bdays)
    assert not any("Lointain" in e["title"] for e in bdays)   # >7 days out excluded


def test_birthday_fact_deduped_against_event_note(isolated_db):
    # The cycle emits BOTH an event note and a has_birthday fact for a birthday;
    # the digest must not list it twice.
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    today = now.date()
    bday = (today.replace(year=1990) + timedelta(days=2)).strftime("%Y-%m-%d")
    created = now.strftime(_FMT)
    conn = get_connection()
    try:
        with conn:
            eid = _add_entity(conn, "Cici", created=created)
            _add_note(conn, "Anniversaire de Cici", kind="event",
                      event_date=bday, recurring=1, created=created)
            _add_fact(conn, eid, "has_birthday", bday, created=created)
        week = gather_week(conn, now=now, days=7)
    finally:
        conn.close()
    cici = [e for e in week["upcoming_events"] if "cici" in (e["title"] or "").lower()]
    assert len(cici) == 1                # only the event note; the fact was deduped
    assert cici[0]["kind"] == "event"


def test_next_occurrence_recurring_rolls_to_next_year():
    today = datetime(2026, 6, 17).date()
    # A birthday already passed this year → next occurrence is next year.
    assert _next_occurrence("1990-01-10", True, today).year == 2027
    # Upcoming this year stays this year.
    assert _next_occurrence("1990-12-25", True, today).year == 2026
    # One-shot returns its absolute date untouched.
    assert _next_occurrence("2026-06-20", False, today) == datetime(2026, 6, 20).date()


def test_has_content_false_on_empty(isolated_db):
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    conn = get_connection()
    try:
        week = gather_week(conn, now=now, days=7)
    finally:
        conn.close()
    assert not has_content(week)


def test_write_digest_note_is_idempotent_per_week(isolated_db):
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    conn = get_connection()
    try:
        with conn:
            _add_entity(conn, "Maxwell", created=now.strftime(_FMT),
                        last_mentioned="2026-06-16", mentions=3)
        week = gather_week(conn, now=now, days=7)
        write_digest_note(conn, week, "## Cette semaine\nPremière version.")
        write_digest_note(conn, week, "## Cette semaine\nVersion mise à jour.")
        rows = conn.execute(
            "SELECT content FROM atomic_notes WHERE kind = 'digest'"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1                               # second write replaced the first
    assert "mise à jour" in rows[0][0]


def test_generate_skips_empty_week(isolated_db):
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    result = generate_weekly_digest(now=now, days=7)
    assert result["note_id"] is None and result["skipped"] == "empty"


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs live Claude API")
def test_summarize_digest_live(isolated_db):
    now = datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)
    conn = get_connection()
    try:
        with conn:
            _add_entity(conn, "Maxwell", created=now.strftime(_FMT),
                        last_mentioned="2026-06-16", mentions=4)
        week = gather_week(conn, now=now, days=7)
    finally:
        conn.close()
    md = summarize_digest(week)
    assert "## Cette semaine" in md
    assert "## À venir" in md
