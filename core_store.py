"""
Access to the Rust core's storage substrate (SYN-110 / T1).

The compiled core (`synapse_core`, the synapse-core PyO3 wheel) owns the
SQLite schema and every vector read/write: the vec0 KNN over atomic_notes and
the entity/resource embedding columns + similarity scans. Python keeps its own
apsw connections for all non-vector SQL against the same database file.

One `Storage` handle per database path, cached for the process: opening runs
the idempotent schema init/migration, and the handle serializes its internal
connection, so per-request re-opens would only add overhead. Keyed by path
(not a singleton) because the test suite points `db.DB_PATH` at a fresh
temporary database per test.
"""

import os
from pathlib import Path

from synapse_core import Brain, Embedder, Storage

_stores: dict[str, Storage] = {}
_brains: dict[str, Brain] = {}
_embedder: Embedder | None = None
_embedder_loaded = False


def model_dir() -> Path | None:
    """Dossier des fichiers modèle d'embedding (donnée, jamais dans un repo) :
    SYNAPSE_MODEL_DIR, sinon l'emplacement standard ~/.synapse/models/…"""
    candidates = []
    env = os.getenv("SYNAPSE_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / ".synapse" / "models"
                      / "paraphrase-multilingual-MiniLM-L12-v2-onnx-Q")
    for c in candidates:
        if (c / "tokenizer.json").exists():
            return c
    return None


def get_embedder() -> Embedder | None:
    """L'embedder du core — UN par processus (le modèle pèse ~235 Mo), partagé
    par embed_text et tous les Brains. None si les fichiers modèle manquent
    (les chemins qui embarquent dégradent alors comme avant : skip silencieux
    des fallbacks, note non vectorisée)."""
    global _embedder, _embedder_loaded
    if not _embedder_loaded:
        d = model_dir()
        _embedder = Embedder(str(d)) if d else None
        _embedder_loaded = True
    return _embedder


def get_brain() -> Brain:
    """Le cerveau du core (routing + classif) pour la base courante."""
    import db  # late import: db.DB_PATH is monkeypatched by the test fixtures

    path = str(db.DB_PATH)
    brain = _brains.get(path)
    if brain is None:
        brain = Brain(path, embedder=get_embedder())
        _brains[path] = brain
    return brain


def get_store() -> Storage:
    """The core storage handle for the current database path."""
    import db  # late import: db.DB_PATH is monkeypatched by the test fixtures

    path = str(db.DB_PATH)
    store = _stores.get(path)
    if store is None:
        store = Storage(path)
        _stores[path] = store
    return store
