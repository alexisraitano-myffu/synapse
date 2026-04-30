"""
Shared embedding logic used by both the Dream Cycle and the MCP server.

Strategy: ask Claude Haiku to extract semantic concepts, then project them
into a 384-dim hash vector. Both query and notes use the same projection,
so cosine similarity is meaningful within this space.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import anthropic
import sqlite_vec

from config import CLAUDE_MODEL, EMBEDDING_DIM


def concepts_to_vector(concepts: list[str]) -> bytes:
    """Project a list of semantic concepts into a normalised 384-dim float vector."""
    vec = [0.0] * EMBEDDING_DIM
    for concept in concepts:
        digest = hashlib.md5(concept.lower().strip().encode()).hexdigest()
        for offset in (0, 8):
            idx = int(digest[offset: offset + 8], 16) % EMBEDDING_DIM
            vec[idx] += 1.0

    magnitude = sum(x * x for x in vec) ** 0.5
    if magnitude > 0:
        vec = [x / magnitude for x in vec]
    return sqlite_vec.serialize_float32(vec)


def embed_text(text: str, client: anthropic.Anthropic) -> bytes:
    """Call Claude Haiku to extract concepts from text, return serialised vector."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract 12 lowercase single-word semantic concepts from this text. "
                    "Return only a JSON array of strings, no other text.\n\n"
                    f"{text[:600]}"
                ),
            }
        ],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    concepts: list[str] = json.loads(raw)
    return concepts_to_vector(concepts)
