"""
SYN-111 golden corpus, step 2: freeze the CURRENT Python routing reference.

Sequentially replays the frozen corpus (recorded classifications) through the
unmodified Python routing (`_process_entry` with `classified=` pre-computed +
`step5_validate_pending`), against a FRESH database, and writes the
normalized end state + per-entry state hashes to
`$SYNAPSE_GOLDEN_DIR/python_reference.json`.

Determinism boundaries, on purpose:
- LLM sub-calls are out of routing scope: `client=None` (skips project
  synthesis) and `process_capture_resources` is stubbed to a no-op (network).
- The embedding-driven decisions (merge fallback ≥0.85, project-attach ≥0.30)
  DO run, with the real local model — that's part of routing.
- `--db-out` keeps the raw replayed SQLite for inspection.

Usage:
    python -m scripts.golden.golden_replay [--db-out PATH]
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# The replay must land in an isolated database: point SYNAPSE_HOME somewhere
# fresh BEFORE importing config/db (they read the env at import time).
_REPLAY_HOME = tempfile.mkdtemp(prefix="synapse-golden-replay-")
os.environ["SYNAPSE_HOME"] = _REPLAY_HOME

# Frozen wall-clock for the replay writes (routing takes `now` as a param).
REPLAY_NOW = "2026-07-04T00:00:00+00:00"


def state_hash(norm: dict) -> str:
    return hashlib.sha256(
        json.dumps(norm, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-out", default=None,
                        help="also copy the replayed SQLite file here")
    args = parser.parse_args()

    from scripts.golden.golden_lib import GOLDEN_DIR, normalize_db

    corpus_path = GOLDEN_DIR / "corpus.json"
    corpus = json.loads(corpus_path.read_text())

    import dream_cycle.cycle as cycle
    from db import get_connection, init_db, DB_PATH

    # Routing scope only: no network, no LLM sub-calls.
    cycle.process_capture_resources = lambda *a, **k: None

    init_db()
    conn = get_connection()
    per_entry = []
    all_new_facts = []
    try:
        for item in corpus["entries"]:
            conn.execute(
                "INSERT INTO inbox (id, content, source, created_at, status) "
                "VALUES (?,?,?,?, 'queued')",
                (item["capture_id"], item["content"], "golden", item["created_at"]),
            )
            entry = {"id": item["capture_id"], "content": item["content"],
                     "source": "golden", "created_at": item["created_at"]}
            entity_ids, new_facts = cycle._process_entry(
                entry, None, conn, REPLAY_NOW, dry_run=False, verbose=False,
                classified=item["classified"],
            )
            all_new_facts.extend(new_facts)
            per_entry.append({
                "capture_id": item["capture_id"],
                "entity_ids_count": len(entity_ids),
                "state_hash": state_hash(normalize_db(DB_PATH)),
            })
            print(f"  routed id={item['capture_id']} "
                  f"({len(entity_ids)} entité(s)) → {per_entry[-1]['state_hash']}")

        promoted = cycle.step5_validate_pending(all_new_facts, conn, False, False)
        print(f"step5: {promoted} pending fact(s) promoted")
    finally:
        conn.close()

    final = normalize_db(DB_PATH)
    out = {
        "replayed_at_now": REPLAY_NOW,
        "corpus_recorded_at": corpus["recorded_at"],
        "per_entry": per_entry,
        "final_state_hash": state_hash(final),
        "final_state": final,
    }
    out_path = GOLDEN_DIR / "python_reference.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nréférence Python figée → {out_path}")
    print(f"état final: {out['final_state_hash']} · replay db: {DB_PATH}")

    if args.db_out:
        import shutil
        shutil.copy(DB_PATH, args.db_out)
        print(f"db copiée → {args.db_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
