import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from config import CLAUDE_MODEL
from db import get_connection, init_db
from embeddings import embed_text

# ── Claude client ─────────────────────────────────────────────────────────────

def _get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


# ── Phase 1 — Filtrage ────────────────────────────────────────────────────────

_SYSTEM_FILTER = """\
You are a knowledge extraction agent for a personal second-brain system.

Given a list of raw inbox entries (thoughts, meeting notes, web clippings), your job is to:
1. Extract distinct, meaningful pieces of information as atomic notes
2. Merge entries that cover the same topic into a single note
3. Discard pure noise (test entries, greetings, incomplete fragments with no value)
4. Write content in clean, structured markdown
5. IMPORTANT: always write title and content in the same language as the source entries — never translate

Return ONLY a valid JSON array. Each element must have exactly these keys:
- "title": concise descriptive title, 5-10 words
- "content": clean markdown content preserving all important details
- "source_ids": array of inbox entry IDs this note was derived from

If all entries are noise with nothing worth keeping, return an empty array [].
Return JSON only — no explanation, no code fences.\
"""


def phase1_filter(entries: list[dict], client: anthropic.Anthropic) -> list[dict]:
    """Call Claude Haiku to extract structured atomic notes from raw inbox entries."""
    if not entries:
        return []

    payload = json.dumps(
        [{"id": e["id"], "content": e["content"], "source": e["source"]} for e in entries],
        ensure_ascii=False,
        indent=2,
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_FILTER,
                "cache_control": {"type": "ephemeral"},  # prompt caching — cheaper on repeat runs
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Extract atomic notes from these inbox entries:\n\n{payload}",
            }
        ],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if Claude wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()

    if not raw:
        print("  ⚠ Claude returned an empty response — skipping.")
        return []

    notes = json.loads(raw)

    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    print(f"  → {len(notes)} note(s) extracted from {len(entries)} entries")
    print(f"  → tokens: {usage.input_tokens} in / {usage.output_tokens} out"
          + (f" / {cache_read} from cache" if cache_read else ""))
    return notes


# ── Phase 2 — Synthèse ────────────────────────────────────────────────────────

def phase2_synthesize(notes: list[dict], entry_ids: list[int]) -> list[int]:
    """Write extracted notes to atomic_notes and mark inbox entries processed."""
    conn = get_connection()
    note_ids: list[int] = []
    now = datetime.now(timezone.utc).isoformat()

    try:
        with conn:
            for note in notes:
                conn.execute(
                    "INSERT INTO atomic_notes (title, content, source_ids) VALUES (?, ?, ?)",
                    (
                        note["title"],
                        note["content"],
                        json.dumps(note.get("source_ids", [])),
                    ),
                )
                note_ids.append(conn.last_insert_rowid())

            for eid in entry_ids:
                conn.execute(
                    "UPDATE inbox SET processed_at = ? WHERE id = ?",
                    (now, eid),
                )

        print(f"  → {len(note_ids)} note(s) written to atomic_notes")
        print(f"  → {len(entry_ids)} inbox entries marked as processed")
        return note_ids
    finally:
        conn.close()


# ── Phase 3 — Vectorisation ───────────────────────────────────────────────────

def phase3_vectorize(note_ids: list[int], client: anthropic.Anthropic) -> None:
    """Extract semantic concepts via Claude and store hash-projected vectors in sqlite-vec."""
    conn = get_connection()
    vectorized = 0

    try:
        for note_id in note_ids:
            row = conn.execute(
                "SELECT title, content FROM atomic_notes WHERE id = ?", (note_id,)
            ).fetchone()
            if not row:
                continue
            title, content = row

            vec_bytes = embed_text(f"Title: {title}\n{content}", client)
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO atomic_notes_vec(rowid, embedding) VALUES (?, ?)",
                    (note_id, vec_bytes),
                )
            vectorized += 1

        print(f"  → {vectorized} note(s) vectorized")
    finally:
        conn.close()


# ── Orchestrateur ─────────────────────────────────────────────────────────────

def run_cycle() -> None:
    print("═" * 52)
    print("  SYNAPSE  ·  Dream Cycle")
    print("═" * 52)

    client = _get_client()
    init_db()

    # Load unprocessed inbox entries
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id, content, source, created_at "
            "FROM inbox WHERE processed_at IS NULL ORDER BY created_at"
        )
        cols = [d[0] for d in cur.description]
        entries = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    if not entries:
        print("\n  Inbox empty — nothing to process.")
        print("═" * 52)
        return

    print(f"\n  {len(entries)} unprocessed inbox entr{'y' if len(entries) == 1 else 'ies'} found\n")

    # ── Phase 1
    print("▸ Phase 1  —  Filtering & Extraction")
    notes = phase1_filter(entries, client)

    entry_ids = [e["id"] for e in entries]

    if not notes:
        print("  Nothing worth keeping — marking entries processed.")
        conn = get_connection()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            for eid in entry_ids:
                conn.execute("UPDATE inbox SET processed_at = ? WHERE id = ?", (now, eid))
        conn.close()
        print("═" * 52)
        return

    # ── Phase 2
    print("\n▸ Phase 2  —  Synthesis")
    note_ids = phase2_synthesize(notes, entry_ids)

    # ── Phase 3
    print("\n▸ Phase 3  —  Vectorization")
    phase3_vectorize(note_ids, client)

    print("\n" + "═" * 52)
    print(f"  Done  ·  {len(note_ids)} note(s) added to memory")
    print("═" * 52 + "\n")
