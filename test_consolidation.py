"""
SYN-93 — working-memory context block + batched consolidation trigger.

Offline: `_build_day_context` (transcript assembly for coreference) and
`_should_consolidate` (the scheduler's batch policy). No API calls.
"""

from datetime import datetime, timedelta, timezone

import pytest

from db import get_connection
from dream_cycle.cycle import (
    _batch_classify,
    _build_day_context,
    _classify_params,
    _parse_classify_text,
)
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
    assert appmod._should_consolidate(queued=0, stale=2) == "stale"   # cheap resummary run
    assert appmod._should_consolidate(queued=0, stale=0) == ""


def test_should_consolidate_size_valve(monkeypatch):
    other_hour = str((datetime.now().hour + 1) % 24)               # ensure NOT the scheduled hour
    monkeypatch.setenv("SYNAPSE_CONSOLIDATION_HOURS", other_hour)
    monkeypatch.setenv("SYNAPSE_CONSOLIDATION_MAX_QUEUED", "5")
    appmod._last_consolidation_slot = None
    assert appmod._should_consolidate(queued=5, stale=0) == "valve"  # valve hit (stays sync)
    assert appmod._should_consolidate(queued=4, stale=0) == ""       # below valve, not the hour


def test_should_consolidate_scheduled_fires_once(monkeypatch, tmp_path):
    now_hour = str(datetime.now().hour)
    monkeypatch.setenv("SYNAPSE_CONSOLIDATION_HOURS", now_hour)
    monkeypatch.setenv("SYNAPSE_CONSOLIDATION_MAX_QUEUED", "999")   # disable size valve
    # Isolate the catch-up marker (mtime-based) from any real run on this machine.
    monkeypatch.setattr(appmod, "_CONSOLIDATION_MARKER", tmp_path / "last_consolidation")
    # Never consolidated → today's scheduled slot is due (batch path).
    assert appmod._should_consolidate(queued=1, stale=0) == "scheduled"
    # The scheduler marks the run done → the same slot no longer refires.
    appmod._mark_consolidated()
    assert appmod._should_consolidate(queued=1, stale=0) == ""


# ── Batch API classify (offline, mocked client) ──────────────────────────────

def test_classify_params_shape_with_working_memory(isolated_db):
    # SYN-111: the params are built by the core (prompt-as-data + live DB
    # blocks, always read from the core's own connection).
    params = _classify_params({"id": 1, "content": "Coucou"}, conn=None, day_context="CTX")
    assert params["model"] and params["max_tokens"] == 4096
    assert params["messages"][0]["content"] == "Coucou"
    sysblocks = params["system"]
    # stable rules + working-memory block, both cached; user content carried separately.
    assert sysblocks[0]["cache_control"] == {"type": "ephemeral"}
    assert any(b.get("text") == "CTX" and b.get("cache_control") for b in sysblocks)
    # uncached live blocks (vocab + projects) follow the cached prefix.
    assert any("TYPES D'ENTITÉ ACTIFS" in b.get("text", "") for b in sysblocks)


def test_parse_classify_text_strips_fence_and_guards_truncation():
    assert _parse_classify_text('```json\n{"input_type":"fact"}\n```', 10, "end_turn") == {"input_type": "fact"}
    with pytest.raises(ValueError):
        _parse_classify_text("{}", 9999, "max_tokens")


class _Msg:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [type("C", (), {"text": text})()]
        self.stop_reason = stop_reason


class _Res:
    def __init__(self, custom_id, type_, message=None):
        self.custom_id = custom_id
        self.result = type("R", (), {"type": type_, "message": message})()


class _Batch:
    id = "batch_1"
    processing_status = "ended"


class _Batches:
    def __init__(self, results): self._results = results
    def create(self, requests=None): return _Batch()
    def retrieve(self, bid): return _Batch()
    def results(self, bid): return iter(self._results)


class _Client:
    def __init__(self, results):
        self.messages = type("M", (), {"batches": _Batches(results)})()


def test_batch_classify_maps_success_and_falls_back_on_error():
    entries = [{"id": 1, "content": "A"}, {"id": 2, "content": "B"}]
    client = _Client([
        _Res("e1", "succeeded", _Msg('{"input_type":"fact"}')),
        _Res("e2", "errored", None),                       # → None, caller retries sync
    ])
    out = _batch_classify(entries, client, conn=None, day_context=None)
    assert out[1] == {"input_type": "fact"}
    assert out[2] is None
