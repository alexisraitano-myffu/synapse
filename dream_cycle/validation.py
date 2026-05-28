"""
Shared validation logic for pending facts.

Records the user's decision as an append-only `validation_events` row (so it
survives a rebuild from the inbox and replicates like any event), then applies
it: confirm → consolidate into `facts` at high confidence; reject → discard.

The caller owns the transaction (wrap in `with conn:`).
"""

import json
import uuid

from db import first_row

CONFIRMED_CONFIDENCE = 0.95  # user-confirmed facts are near-certain


def record_and_apply_validation(
    conn,
    fact_id: str,
    confirmed: bool,
    correction: str | None = None,
    device_id: str | None = None,
) -> dict:
    pending = first_row(conn.execute(
        "SELECT id, fact_data FROM pending_facts WHERE id=?", (fact_id,)
    ))
    if not pending:
        return {"status": "error", "message": f"fact_id '{fact_id}' not found"}
    try:
        fact_data = json.loads(pending["fact_data"])
    except (ValueError, TypeError):
        return {"status": "error", "message": "invalid fact_data JSON"}

    # Append-only event — the durable record of the decision.
    conn.execute(
        "INSERT INTO validation_events "
        "(id, fact_id, entity_canonical, predicate, value, confirmed, correction, device_id) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            str(uuid.uuid4()), fact_id,
            fact_data.get("entity_canonical"), fact_data.get("predicate"),
            fact_data.get("value"), 1 if confirmed else 0, correction, device_id,
        ),
    )

    if not confirmed:
        conn.execute("DELETE FROM pending_facts WHERE id=?", (fact_id,))
        return {"status": "rejected", "fact_id": fact_id}

    if correction:
        fact_data["value"] = correction

    entity_name = fact_data.get("entity_canonical", "unknown")
    row = conn.execute(
        "SELECT id FROM entities WHERE LOWER(canonical_name)=LOWER(?)", (entity_name,)
    ).fetchone()
    # SYN-41: provenance traces back to the capture that spawned the pending.
    try:
        prov_id = int(fact_data.get("source_inbox_id")) if fact_data.get("source_inbox_id") else None
    except (TypeError, ValueError):
        prov_id = None

    if row:
        entity_id = row[0]
    else:
        entity_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO entities (id, canonical_name, provenance_capture_id) VALUES (?,?,?)",
            (entity_id, entity_name, prov_id),
        )

    conn.execute(
        "INSERT INTO facts "
        "(id, entity_id, predicate, value, confidence, source_inbox_id, "
        " persistence_value, provenance_capture_id) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            str(uuid.uuid4()), entity_id,
            fact_data.get("predicate"), fact_data.get("value"),
            CONFIRMED_CONFIDENCE, fact_data.get("source_inbox_id"),
            fact_data.get("persistence_value", 3),
            prov_id,
        ),
    )
    conn.execute("DELETE FROM pending_facts WHERE id=?", (fact_id,))

    return {
        "status": "confirmed",
        "fact_id": fact_id,
        "entity": entity_name,
        "predicate": fact_data.get("predicate"),
        "value": fact_data.get("value"),
    }
