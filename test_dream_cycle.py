"""
Integration tests for the Phase A+ Dream Cycle.
Requires ANTHROPIC_API_KEY. Each test runs against an isolated temp SQLite DB.
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point every test at a fresh temp database."""
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path))

    import importlib
    import config as cfg_mod
    import db as db_mod

    new_db_path = tmp_path / "synapse.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", new_db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", new_db_path)

    db_mod.init_db()


def _add_inbox(content: str, source: str = "test") -> int:
    from db import get_connection
    conn = get_connection()
    try:
        import uuid
        cid = str(uuid.uuid4())
        with conn:
            conn.execute(
                "INSERT INTO inbox (id, content, source) VALUES (?,?,?)", (cid, content, source)
            )
        return cid
    finally:
        conn.close()


def _run_cycle(verbose: bool = False) -> None:
    from dream_cycle import run_dream_cycle
    run_dream_cycle(verbose=verbose)


def _get_entity_facts(name: str) -> list[dict]:
    """Return facts for an entity matching name (canonical or alias), or []."""
    from db import get_connection
    conn = get_connection()
    try:
        # Try canonical
        cur = conn.execute(
            "SELECT * FROM entities WHERE LOWER(canonical_name)=LOWER(?)", (name,)
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()

        if not row:
            # Check aliases
            all_cur = conn.execute("SELECT * FROM entities")
            all_cols = [d[0] for d in all_cur.description]
            for all_row in all_cur.fetchall():
                entity = dict(zip(all_cols, all_row))
                try:
                    aliases = json.loads(entity.get("aliases", "[]"))
                except (ValueError, TypeError):
                    aliases = []
                if name.lower() in [a.lower() for a in aliases]:
                    row, cols = all_row, all_cols
                    break

        if not row:
            return []

        entity_id = dict(zip(cols, row))["id"]
        fcur = conn.execute(
            "SELECT predicate, value, confidence FROM facts WHERE entity_id=?",
            (entity_id,),
        )
        fcols = [d[0] for d in fcur.description]
        return [dict(zip(fcols, r)) for r in fcur.fetchall()]
    finally:
        conn.close()


def _get_pending_facts() -> list[dict]:
    from db import get_connection
    conn = get_connection()
    try:
        cur = conn.execute("SELECT id, fact_data FROM pending_facts")
        result = []
        for row in cur.fetchall():
            try:
                result.append({"id": row[0], **json.loads(row[1])})
            except (ValueError, TypeError):
                pass
        return result
    finally:
        conn.close()


def _get_intentions() -> list[dict]:
    from db import get_connection
    conn = get_connection()
    try:
        cur = conn.execute("SELECT id, content, ttl_hours, resolved FROM intentions")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


# ── Test 1 — Date relative ────────────────────────────────────────────────────

def test_birthday_resolution():
    """
    "aujourd'hui c'est l'anniversaire de maman"
    → get_entity("maman") has a has_birthday fact with today's date.
    """
    _add_inbox("aujourd'hui c'est l'anniversaire de maman")
    _run_cycle()

    today = date.today().isoformat()
    facts = _get_entity_facts("maman")

    # Also accept common alias: "mama", "mother", or the entity may be named differently
    if not facts:
        facts = _get_entity_facts("mama") or _get_entity_facts("mother")

    birthday_facts = [f for f in facts if "birthday" in f["predicate"] or "anniversaire" in f["predicate"]]

    # The birthday must have been extracted either to entities or pending_facts
    if not birthday_facts:
        pending = _get_pending_facts()
        birthday_pending = [
            p for p in pending
            if ("birthday" in p.get("predicate", "") or "anniversaire" in p.get("predicate", ""))
            and "maman" in p.get("entity_canonical", "").lower()
        ]
        assert birthday_pending, (
            "Expected a birthday fact for 'maman' in entities or pending_facts. "
            f"facts={facts}, pending={pending}"
        )
        # Date should resolve to today
        assert today in birthday_pending[0].get("value", ""), (
            f"Expected birthday value to contain today ({today}), "
            f"got: {birthday_pending[0].get('value')}"
        )
    else:
        assert today in birthday_facts[0]["value"], (
            f"Expected has_birthday value to contain today ({today}), "
            f"got: {birthday_facts[0]['value']}"
        )


# ── Test 2 — Fragment sans contexte ───────────────────────────────────────────

def test_phone_fragment():
    """
    "Marie 06 12 34 56 78"
    → entity Marie created with a phone fact in entities or pending_facts.
    """
    _add_inbox("Marie 06 12 34 56 78")
    _run_cycle()

    facts = _get_entity_facts("Marie")
    phone_facts = [f for f in facts if "phone" in f["predicate"] or "tel" in f["predicate"]]

    if not phone_facts:
        pending = _get_pending_facts()
        phone_pending = [
            p for p in pending
            if ("phone" in p.get("predicate", "") or "tel" in p.get("predicate", ""))
            and "marie" in p.get("entity_canonical", "").lower()
        ]
        assert phone_pending, (
            "Expected a phone fact for 'Marie' in entities or pending_facts. "
            f"facts={facts}, pending={pending}"
        )
    else:
        assert any("06" in f["value"] or "0612" in f["value"].replace(" ", "") for f in phone_facts), (
            f"Phone fact found but value doesn't look right: {phone_facts}"
        )


# ── Test 3 — Validation comportementale ───────────────────────────────────────

def test_behavioral_validation():
    """
    First mention of "Jean-Pierre médecin" goes to pending (low confidence).
    Second mention corroborates and should promote to entities via step5.
    """
    _add_inbox("Jean-Pierre est médecin, il a un cabinet rue de Rivoli")
    _run_cycle()

    # Check initial state — could be in entities (high persistence) or pending
    facts_after_first = _get_entity_facts("Jean-Pierre")
    pending_after_first = _get_pending_facts()

    # Haiku may encode "médecin" as a French value ("médecin") OR an English
    # predicate (is_doctor=true) — both are valid. Match the concept in either field.
    _DOCTOR = ("medecin", "médecin", "doctor", "docteur")
    def _is_doctor_fact(predicate: str, value: str) -> bool:
        blob = f"{predicate} {value}".lower()
        return any(tok in blob for tok in _DOCTOR)

    medecin_in_entities = any(
        _is_doctor_fact(f["predicate"], f["value"]) for f in facts_after_first
    )
    medecin_in_pending = any(
        "jean-pierre" in p.get("entity_canonical", "").lower()
        and _is_doctor_fact(p.get("predicate", ""), p.get("value", ""))
        for p in pending_after_first
    )

    assert medecin_in_entities or medecin_in_pending, (
        "Expected Jean-Pierre's médecin fact to be in entities or pending_facts after first entry. "
        f"facts={facts_after_first}, pending={pending_after_first}"
    )

    if medecin_in_pending and not medecin_in_entities:
        # Second entry: corroborate
        _add_inbox("rappeler Jean-Pierre le médecin pour le rdv de vendredi")
        _run_cycle()

        facts_after_second = _get_entity_facts("Jean-Pierre")
        medecin_confirmed = any(
            _is_doctor_fact(f["predicate"], f["value"]) for f in facts_after_second
        )
        assert medecin_confirmed, (
            "Expected Jean-Pierre's médecin fact to be promoted to entities after second mention. "
            f"facts={facts_after_second}"
        )


# ── Test 4 — Ephemeral ────────────────────────────────────────────────────────

def test_ephemeral_goes_to_intentions():
    """
    "penser à acheter du lait"
    → in intentions with ttl_hours=48, NOT in entities.
    """
    _add_inbox("penser à acheter du lait")
    _run_cycle()

    intentions = _get_intentions()
    assert intentions, "Expected at least one intention to be created"

    lait_intentions = [
        i for i in intentions
        if "lait" in i["content"].lower() or "acheter" in i["content"].lower()
    ]
    assert lait_intentions, (
        f"Expected an intention about 'lait', got: {intentions}"
    )
    assert lait_intentions[0]["ttl_hours"] == 48

    # Must NOT be in entities
    from db import get_connection
    conn = get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    finally:
        conn.close()
    assert count == 0, f"Expected 0 entities for an ephemeral entry, found {count}"
