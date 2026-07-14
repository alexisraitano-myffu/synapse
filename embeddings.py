"""
Shared embedding logic used by both the Dream Cycle and the MCP server.

Since SYN-111 the model runs inside the Rust core (`synapse_core.Embedder`,
ONNX runtime, fully offline): one model in memory for the whole process, and
the vectors are bit-identical to the core's own internal embeds (merge
fallback, note vectorization). The model files are DATA in
`~/.synapse/models/…` (or SYNAPSE_MODEL_DIR) — same files on desktop and
mobile. Vectors stay L2-normalized so the sqlite-vec `vec0` L2 distance is
monotonic with cosine and `score = 1 - distance/2` remains valid.
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


from config import EMBEDDING_DIM, EMBEDDING_MODEL

def embed_text(text: str, client=None) -> bytes:
    """
    Embed text into a serialized, L2-normalized 384-dim float vector.

    `client` is accepted for backward compatibility with the previous
    API-based implementation but is ignored — embedding is now fully local.
    """
    from core_store import get_embedder

    embedder = get_embedder()
    if embedder is None:
        raise EnvironmentError(
            "Fichiers du modèle d'embedding introuvables — attendus dans "
            "~/.synapse/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q "
            "(ou SYNAPSE_MODEL_DIR)."
        )
    # Le core garantit 384-d L2-normalisé (mêmes checks qu'ici avant).
    vec = embedder.embed(text)

    # Packed little-endian float32 — byte-identical to what the old
    # serialize_float32 helper produced; the DB format doesn't change.
    return struct.pack(f"<{len(vec)}f", *vec)


def embed_text_chunks(text: str) -> list[bytes]:
    """SYN-118: one serialized vector per ~128-token window of `text`.

    A short text returns exactly one element, byte-identical to
    `embed_text(text)`. Long texts (weekly digests, long captures) get one
    vector per window so search can match their tail: store them with
    `Storage.upsert_note_vectors` (notes) or concatenated (resources)."""
    from core_store import get_embedder

    embedder = get_embedder()
    if embedder is None:
        raise EnvironmentError(
            "Fichiers du modèle d'embedding introuvables — attendus dans "
            "~/.synapse/models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q "
            "(ou SYNAPSE_MODEL_DIR)."
        )
    return [struct.pack(f"<{len(v)}f", *v) for v in embedder.embed_chunks(text)]
