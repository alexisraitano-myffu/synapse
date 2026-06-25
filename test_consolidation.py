"""
SYN-93 — working-memory context block + batched consolidation trigger.

Offline: `_build_day_context` (transcript assembly for coreference) and
`_should_consolidate` (the scheduler's batch policy). No API calls.
"""

from datetime import datetime, timedelta, timezone

from db import get_connection
from dream_cycle.cycle import _build_day_context
import api.app as appmod


def _ins(conn, content, created, processed=None):
    conn.execute(
        "INSERT INTO inbox (content, source, created_at, processed_at) VALUES (?,?,?,?)",
        (content, "test", created, processed),
    )


# ── working memory ───────────────────────────────────────────────────────────

def test_build_day_context_includes_prior_and_batch(isolated_db):
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    earlier = (now - timedelta(hours=3)).isoformat()
    conn = get_connection()
    try:
        with conn:
            _ins(conn, "Hier j'ai vu Romain", earlier, processed=earlier)  # consolidated
        batch = [{"content": "Il m'a parlé du projet Atlas", "created_at": now.isoformat()}]
        ctx = _build_day_context(conn, batch, now)
    finally:
        conn.close()
    assert ctx is not None
    assert "Romain" in ctx and "Atlas" in ctx          # prior + current batch both present
    assert "N'EXTRAIS RIEN" in ctx                      # the no-extraction guard


def test_build_day_context_drops_old_prior(isolated_db):
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    old = (now - timedelta(hours=48)).isoformat()       # outside the 24h lookback
    conn = get_connection()
    try:
        with conn:
            _ins(conn, "Capture trop vieille", old, processed=old)
        batch = [
            {"content": "Capture A", "created_at": now.isoformat()},
            {"content": "Capture B", "created_at": now.isoformat()},
        ]
        ctx = _build_day_context(conn, batch, now)
    finally:
        conn.close()
    assert ctx is not None
    assert "trop vieille" not in ctx                    # lookback window excludes it
    assert "Capture A" in ctx and "Capture B" in ctx


def test_build_day_context_none_for_lone_capture(isolated_db):
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    conn = get_connection()
    try:
        batch = [{"content": "Une seule capture", "created_at": now.isoformat()}]
        ctx = _build_day_context(conn, batch, now)
    finally:
        conn.close()
    assert ctx is None                                  # nothing to resolve against


# ── batched consolidation policy ─────────────────────────────────────────────

def test_should_consolidate_stale_only():
    assert appmod._should_consolidate(queued=0, stale=2) is True    # cheap resummary run
    assert appmod._should_consolidate(queued=0, stale=0) is False


def test_should_consolidate_size_valve(monkeypatch):
    other_hour = str((datetime.now().hour + 1) % 24)               # ensure NOT the scheduled hour
    monkeypatch.setenv("SYNAPSE_CONSOLIDATION_HOURS", other_hour)
    monkeypatch.setenv("SYNAPSE_CONSOLIDATION_MAX_QUEUED", "5")
    appmod._last_consolidation_slot = None
    assert appmod._should_consolidate(queued=5, stale=0) is True    # valve hit
    assert appmod._should_consolidate(queued=4, stale=0) is False   # below valve, not the hour


def test_should_consolidate_hour_slot_fires_once(monkeypatch):
    now_hour = str(datetime.now().hour)
    monkeypatch.setenv("SYNAPSE_CONSOLIDATION_HOURS", now_hour)
    monkeypatch.setenv("SYNAPSE_CONSOLIDATION_MAX_QUEUED", "999")   # disable size valve
    appmod._last_consolidation_slot = None
    assert appmod._should_consolidate(queued=1, stale=0) is True    # first tick in the slot
    assert appmod._should_consolidate(queued=1, stale=0) is False   # same slot → no refire
