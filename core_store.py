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


_MODEL_REPO = "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
_MODEL_FILES = ("model_optimized.onnx", "tokenizer.json", "config.json",
                "special_tokens_map.json", "tokenizer_config.json")


def model_dir() -> Path | None:
    """Dossier des fichiers modèle d'embedding (donnée, jamais dans un repo) :
    SYNAPSE_MODEL_DIR, sinon l'emplacement standard ~/.synapse/models/… —
    téléchargé automatiquement au premier besoin s'il manque (le comportement
    historique de fastembed ; sans ça une machine vierge n'embedde jamais,
    silencieusement)."""
    candidates = []
    env = os.getenv("SYNAPSE_MODEL_DIR")
    if env:
        candidates.append(Path(env))
    default = (Path.home() / ".synapse" / "models"
               / "paraphrase-multilingual-MiniLM-L12-v2-onnx-Q")
    candidates.append(default)
    for c in candidates:
        if (c / "tokenizer.json").exists():
            return c
    return _download_model(default)


def _download_model(dest: Path) -> Path | None:
    """Récupère les fichiers modèle depuis Hugging Face (~130 Mo, une fois).
    Écrit dans un dossier temporaire puis renomme : jamais de dossier partiel
    (tokenizer.json présent = complet, c'est le marqueur de model_dir)."""
    import shutil
    import urllib.request

    tmp = dest.with_name(dest.name + ".download")
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        print(f"[model] downloading embedding model to {dest} (~130 MB, one-time)…",
              flush=True)
        for f in _MODEL_FILES:
            url = f"https://huggingface.co/{_MODEL_REPO}/resolve/main/{f}"
            with urllib.request.urlopen(url, timeout=120) as r, \
                    open(tmp / f, "wb") as out:
                shutil.copyfileobj(r, out)
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
        print("[model] embedding model ready", flush=True)
        return dest
    except Exception as exc:  # offline / HF down: degrade like a missing dir
        print(f"[model] download failed ({exc}); embeddings disabled for this run",
              flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
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
