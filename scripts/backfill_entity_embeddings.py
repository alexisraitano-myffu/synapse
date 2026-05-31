#!/usr/bin/env python3
"""
Backfill `entities.embedding` for entities that have none (SYN-60).

The Dream Cycle's `step6_vectorize` embeds every touched entity at the end of a
run, so new entities are always vectorized. This one-shot script covers the
historical tail — entities created before that step existed, or any whose
embed failed mid-cycle.

Idempotent: by default only entities with a NULL embedding are processed, so
re-running is safe and cheap. Pass `--all` to force a full re-embed (e.g. after
changing EMBEDDING_MODEL — the entity equivalent of `reembed.py`).

Usage:
    python scripts/backfill_entity_embeddings.py            # missing only
    python scripts/backfill_entity_embeddings.py --all      # re-embed everything
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db import get_connection, cursor_to_dicts, init_db
from embeddings import embed_text
from entity_search import entity_embedding_text


def backfill(force_all: bool = False) -> int:
    init_db()
    conn = get_connection()
    try:
        where = "" if force_all else "WHERE embedding IS NULL"
        entities = cursor_to_dicts(conn.execute(
            f"SELECT id, canonical_name, type, aliases, attributes, summary "
            f"FROM entities {where}"
        ))
        if not entities:
            print("Nothing to backfill — every entity already has an embedding.")
            return 0

        scope = "all" if force_all else "missing"
        print(f"Embedding {len(entities)} entit{'y' if len(entities) == 1 else 'ies'} ({scope})…")
        done = 0
        for entity in entities:
            try:
                vec_bytes = embed_text(entity_embedding_text(entity))
                with conn:
                    conn.execute(
                        "UPDATE entities SET embedding=? WHERE id=?",
                        (vec_bytes, entity["id"]),
                    )
                done += 1
            except Exception as exc:  # one bad row shouldn't abort the backfill
                print(f"  ! skipped '{entity['canonical_name']}': {exc}")
        print(f"Done — {done}/{len(entities)} entit{'y' if done == 1 else 'ies'} embedded.")
        return done
    finally:
        conn.close()


if __name__ == "__main__":
    backfill(force_all="--all" in sys.argv[1:])
