#!/usr/bin/env python3
"""
Re-embed every stored vector with the current local model: atomic_notes into
atomic_notes_vec, entities.embedding (composite text) and resources.embedding.

Run this after changing EMBEDDING_MODEL or the model files' truncation
(SYN-118) — old vectors live in a different space and must be regenerated, or
vector search mixes incompatible embeddings. Idempotent: notes use INSERT OR
REPLACE keyed on the note id, entities/resources overwrite their BLOB column.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core_store import get_brain, get_store
from db import get_connection, cursor_to_dicts, init_db
from embeddings import embed_text_chunks


def reembed() -> None:
    init_db()
    conn = get_connection()
    try:
        notes = cursor_to_dicts(conn.execute("SELECT id, title, content FROM atomic_notes"))
        print(f"Re-embedding {len(notes)} note(s)…")
        for note in notes:
            text = f"Title: {note['title']}\n{note['content']}" if note.get("title") else note["content"]
            get_store().upsert_note_vectors(note["id"], embed_text_chunks(text))

        # Entities: the core rebuilds the composite text (name/type/aliases/
        # attributes/summary) itself and only embeds already-vectorized rows'
        # peers the same way the cycle does.
        entity_ids = [r["id"] for r in cursor_to_dicts(conn.execute("SELECT id FROM entities"))]
        print(f"Re-embedding {len(entity_ids)} entity(ies)…")
        done = get_brain().vectorize_entities(entity_ids)

        resources = cursor_to_dicts(conn.execute("SELECT id, title, summary FROM resources"))
        print(f"Re-embedding {len(resources)} resource(s)…")
        for res in resources:
            text = f"{res.get('title') or ''}\n{res.get('summary') or ''}"
            frames = b"".join(embed_text_chunks(text))
            get_store().set_resource_embedding(res["id"], frames)

        print(f"Done — {len(notes)} note(s), {done} entity(ies), {len(resources)} resource(s).")
    finally:
        conn.close()


if __name__ == "__main__":
    reembed()
