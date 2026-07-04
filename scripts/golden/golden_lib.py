"""
Golden-parity helpers (SYN-111 / T2).

Normalizes a Synapse database into a comparable structure so two routing
implementations (current Python vs Rust core) can be diffed on a frozen
corpus. Everything nondeterministic is canonicalized:

- UUID primary keys → stable `U<n>` tokens assigned in sorted-row order,
  applied consistently to foreign keys;
- wall-clock artifacts (timestamps, `last_mentioned`) → presence booleans /
  a fixed token;
- alias lists → sorted (Python's set-union order is arbitrary);
- attributes / fact_data JSON → canonical (sorted keys);
- floats → rounded to 6 decimals;
- note vectors → an existence flag (embedding bytes differ at the last
  decimal between fastembed-python and the core, by design).

The corpus itself lives OUTSIDE the repos (personal captures):
`$SYNAPSE_GOLDEN_DIR`, default `~/.synapse/golden/`.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

GOLDEN_DIR = Path(os.getenv("SYNAPSE_GOLDEN_DIR", Path.home() / ".synapse" / "golden"))


def _canon_json(raw, default):
    try:
        value = json.loads(raw) if isinstance(raw, str) else (raw or default)
    except (ValueError, TypeError):
        value = default
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _round(x):
    return None if x is None else round(float(x), 6)


def normalize_db(db_path) -> dict:
    """Load a Synapse SQLite file and return the normalized routing state."""
    import synapse_core

    conn = synapse_core.connect(str(db_path))

    def rows(sql):
        cols, data = conn.execute(sql)
        return [dict(zip(cols, r)) for r in data]

    out = {}
    ids = {}  # real uuid -> U<n> token, assigned in deterministic order

    def token(uid):
        if uid is None:
            return None
        if uid not in ids:
            ids[uid] = f"U{len(ids) + 1}"
        return ids[uid]

    # Entities first (everything else references them). Sort on natural keys.
    entities = rows("SELECT * FROM entities")
    entities.sort(key=lambda e: (e["canonical_name"], e.get("type") or ""))
    # Natural key per raw uuid: every other table sorts through THIS, never
    # through the raw uuid (random → the whole dump would reorder run-to-run).
    ent_key = {e["id"]: (e["canonical_name"], e.get("type") or "") for e in entities}

    def ekey(uid):
        return ent_key.get(uid, ("", ""))
    out["entities"] = [
        {
            "id": token(e["id"]),
            "canonical_name": e["canonical_name"],
            "type": e.get("type"),
            "aliases": sorted(json.loads(e.get("aliases") or "[]")),
            "attributes": _canon_json(e.get("attributes"), {}),
            "summary": e.get("summary"),
            "mention_count": e.get("mention_count"),
            "last_mentioned": "D" if e.get("last_mentioned") else None,
            "persistence_value": e.get("persistence_value"),
            "status": e.get("status"),
            "merged_into_id": token(e.get("merged_into_id")),
            "archived": bool(e.get("archived_at")),
            "provenance_capture_id": e.get("provenance_capture_id"),
            "summary_stale": e.get("summary_stale"),
            "has_embedding": e.get("embedding") is not None,
        }
        for e in entities
    ]

    facts = rows("SELECT * FROM facts")
    facts.sort(key=lambda f: (ekey(f.get("entity_id")), f["predicate"], f["value"],
                              str(f.get("category"))))
    fact_tokens = {f["id"]: token(f["id"]) for f in facts}
    out["facts"] = [
        {
            "id": fact_tokens[f["id"]],
            "entity_id": token(f.get("entity_id")),
            "predicate": f["predicate"],
            "value": f["value"],
            "confidence": _round(f.get("confidence")),
            "source_inbox_id": f.get("source_inbox_id"),
            "persistence_value": f.get("persistence_value"),
            "category": f.get("category"),
            "provenance_capture_id": f.get("provenance_capture_id"),
            "archived": bool(f.get("archived_at")),
            "obsoleted": bool(f.get("obsoleted_at")),
            "obsoleted_by": fact_tokens.get(f.get("obsoleted_by")),
        }
        for f in facts
    ]

    relations = rows("SELECT * FROM relations")
    relations.sort(key=lambda r: (ekey(r.get("entity_from")), r["predicate"],
                                  ekey(r.get("entity_to"))))
    out["relations"] = [
        {
            "entity_from": token(r.get("entity_from")),
            "predicate": r["predicate"],
            "entity_to": token(r.get("entity_to")),
            "confidence": _round(r.get("confidence")),
            "review_status": r.get("review_status"),
            "provenance_capture_id": r.get("provenance_capture_id"),
        }
        for r in relations
    ]

    notes = rows("SELECT n.*, (SELECT COUNT(*) FROM atomic_notes_vec v WHERE v.rowid = n.id)"
                 " AS has_vec FROM atomic_notes n ORDER BY n.id")
    out["atomic_notes"] = [
        {
            "id": n["id"],
            "title": n.get("title"),
            "content": n["content"],
            "summary": n.get("summary"),
            "entities_mentioned": _canon_json(n.get("entities_mentioned"), []),
            "kind": n.get("kind"),
            "event_date": n.get("event_date"),
            "event_recurring": n.get("event_recurring"),
            "review_status": n.get("review_status"),
            "memory_strength": _round(n.get("memory_strength")),
            "provenance_capture_id": n.get("provenance_capture_id"),
            "archived": bool(n.get("archived_at")),
            "has_vector": bool(n["has_vec"]),
        }
        for n in notes
    ]

    pending = rows("SELECT * FROM pending_facts")
    out["pending_facts"] = sorted(
        [
            {
                "fact_data": _canon_json(p["fact_data"], {}),
                "validation_strategy": p.get("validation_strategy"),
            }
            for p in pending
        ],
        key=lambda p: p["fact_data"],
    )

    review = rows("SELECT * FROM review_queue")
    out["review_queue"] = sorted(
        [
            {
                "fact_data": _canon_json(r["fact_data"], {}),
                "suggested_entity": r.get("suggested_entity"),
            }
            for r in review
        ],
        key=lambda r: (r["fact_data"], str(r["suggested_entity"])),
    )

    intentions = rows("SELECT * FROM intentions")
    out["intentions"] = sorted(
        [
            {"content": i["content"], "ttl_hours": i.get("ttl_hours"),
             "resolved": i.get("resolved")}
            for i in intentions
        ],
        key=lambda i: i["content"],
    )

    entries = rows("SELECT * FROM project_entries")
    out["project_entries"] = sorted(
        [
            {"project_id": token(e.get("project_id")), "capture_id": e.get("capture_id"),
             "content": e["content"], "kind": e.get("kind")}
            for e in entries
        ],
        key=lambda e: (str(e["capture_id"]), e["content"]),
    )

    merges = rows("SELECT * FROM entity_merge_proposals")
    out["entity_merge_proposals"] = sorted(
        [
            {
                "candidate": token(m.get("candidate_entity_id")),
                "existing": token(m.get("existing_entity_id")),
                "score": None if m.get("similarity_score") is None
                         else round(float(m["similarity_score"]), 4),
                "reason": m.get("similarity_reason"),
                "status": m.get("status"),
                "evidence_capture_id": m.get("evidence_capture_id"),
            }
            for m in merges
        ],
        key=lambda m: (str(m["reason"]), str(m["evidence_capture_id"]), m["score"] or 0),
    )

    types = rows("SELECT * FROM entity_type_proposals")
    out["entity_type_proposals"] = sorted(
        [
            {
                "proposed_type": t["proposed_type"],
                "reason": t.get("reason"),
                "candidate": token(t.get("candidate_entity_id")),
                "status": t.get("status"),
                "evidence_capture_id": t.get("evidence_capture_id"),
            }
            for t in types
        ],
        key=lambda t: (t["proposed_type"], str(t["evidence_capture_id"])),
    )

    attach = rows("SELECT * FROM project_attach_proposals")
    out["project_attach_proposals"] = sorted(
        [
            {
                "capture_id": a.get("capture_id"),
                "note_id": a.get("note_id"),
                "project_id": token(a.get("project_id")),
                "content": a["content"],
                "score": None if a.get("similarity_score") is None
                         else round(float(a["similarity_score"]), 4),
                "status": a.get("status"),
            }
            for a in attach
        ],
        key=lambda a: (str(a["capture_id"]), a["content"]),
    )

    inbox = rows("SELECT * FROM inbox ORDER BY id")
    out["inbox"] = [
        {"id": r["id"], "status": r.get("status"), "error": r.get("error"),
         "processed": bool(r.get("processed_at"))}
        for r in inbox
    ]

    out["active_entity_types"] = sorted(
        (r["type"], r["source"]) for r in rows("SELECT * FROM active_entity_types")
    )

    conn.close()
    return out
