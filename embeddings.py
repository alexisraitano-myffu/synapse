"""
Shared embedding logic used by both the Dream Cycle and the MCP server.

Strategy: a local fastembed (ONNX) sentence-transformer model. Runs fully
offline after a one-time model download (~220 MB) — no API call, no PyTorch.
Vectors are L2-normalized so the sqlite-vec `vec0` L2 distance stays in [0, 2]
and is monotonic with cosine similarity, which keeps the downstream
`score = 1 - distance/2` mapping valid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import sqlite_vec

from config import EMBEDDING_DIM, EMBEDDING_MODEL

# Lazy singleton — loading the model is expensive, do it once per process.
_model = None


def _get_model():
    global _model
    if _model is None:
        import warnings
        from fastembed import TextEmbedding  # imported lazily to keep startup cheap
        # This model uses mean pooling (the correct modern default); silence the
        # one-time migration warning so it doesn't pollute the MCP server logs.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*mean pooling.*")
            _model = TextEmbedding(EMBEDDING_MODEL)
    return _model


def embed_text(text: str, client=None) -> bytes:
    """
    Embed text into a serialized, L2-normalized 384-dim float vector.

    `client` is accepted for backward compatibility with the previous
    API-based implementation but is ignored — embedding is now fully local.
    """
    model = _get_model()
    vec = list(next(model.embed([text])))

    magnitude = sum(x * x for x in vec) ** 0.5
    if magnitude > 0:
        vec = [x / magnitude for x in vec]

    if len(vec) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding model returned {len(vec)} dims, expected {EMBEDDING_DIM}. "
            f"Check EMBEDDING_MODEL ({EMBEDDING_MODEL}) matches EMBEDDING_DIM."
        )

    return sqlite_vec.serialize_float32(vec)
