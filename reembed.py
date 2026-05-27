#!/usr/bin/env python3
"""
Re-embed all atomic_notes into atomic_notes_vec with the current local model.

Run this after changing EMBEDDING_MODEL — old vectors live in a different space
and must be regenerated, or vector search mixes incompatible embeddings.
Idempotent: uses INSERT OR REPLACE keyed on the note id.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from db import get_connection, cursor_to_dicts, init_db
from embeddings import embed_text


def reembed() -> None:
    init_db()
    conn = get_connection()
    try:
        notes = cursor_to_dicts(conn.execute("SELECT id, title, content FROM atomic_notes"))
        if not notes:
            print("No atomic_notes to re-embed.")
            return

        print(f"Re-embedding {len(notes)} note(s) with the local model…")
        for note in notes:
            text = f"Title: {note['title']}\n{note['content']}" if note.get("title") else note["content"]
            vec_bytes = embed_text(text)
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO atomic_notes_vec(rowid, embedding) VALUES (?, ?)",
                    (note["id"], vec_bytes),
                )
        print(f"Done — {len(notes)} note(s) re-embedded.")
    finally:
        conn.close()


if __name__ == "__main__":
    reembed()
