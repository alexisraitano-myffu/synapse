"""
SYN-111 golden corpus, step 1: record real classifications.

Replays every processed capture of the production inbox through the CURRENT
Python classifier (one live Haiku call each, day_context=None for
reproducibility, DB context blocks from the production database) and freezes
the (capture, classified) pairs into `$SYNAPSE_GOLDEN_DIR/corpus.json`.

The frozen `classified` dicts are the INPUT of the routing-parity tests —
classification itself is nondeterministic (LLM), so parity is asserted on
routing, never on re-classification.

Personal data: the corpus stays in ~/.synapse/golden/, never in a repo.

Usage (backend venv, .env provides the key):
    python -m scripts.golden.golden_classify [--force] [--limit N]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

from scripts.golden.golden_lib import GOLDEN_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    corpus_path = GOLDEN_DIR / "corpus.json"
    if corpus_path.exists() and not args.force:
        print(f"{corpus_path} exists — use --force to re-record (costs Haiku calls "
              f"and CHANGES the frozen reference).")
        return 1

    from anthropic_client import get_client
    from config import CLAUDE_MODEL
    from db import get_connection, cursor_to_dicts
    from dream_cycle.cycle import step1_classify

    client = get_client()
    conn = get_connection()
    try:
        entries = cursor_to_dicts(conn.execute(
            "SELECT id, content, source, created_at FROM inbox "
            "WHERE processed_at IS NOT NULL AND status = 'processed' "
            "ORDER BY created_at"
        ))
        if args.limit:
            entries = entries[: args.limit]
        print(f"Classifying {len(entries)} capture(s) with {CLAUDE_MODEL}…")

        recorded = []
        for i, entry in enumerate(entries, 1):
            try:
                classified = step1_classify(entry, client, conn=conn, day_context=None)
            except Exception as exc:  # noqa: BLE001 — record the failure, keep going
                print(f"  ! [{i}/{len(entries)}] id={entry['id']} classify failed: {exc}")
                continue
            recorded.append({
                "capture_id": entry["id"],
                "content": entry["content"],
                "created_at": entry["created_at"],
                "classified": classified,
            })
            print(f"  [{i}/{len(entries)}] id={entry['id']} → "
                  f"{classified.get('input_type')} "
                  f"(entities={len(classified.get('entities') or [])})")
    finally:
        conn.close()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps({
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": CLAUDE_MODEL,
        "entries": recorded,
    }, ensure_ascii=False, indent=1))
    print(f"\n{len(recorded)} entrée(s) → {corpus_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
