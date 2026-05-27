from pathlib import Path
import os

# ~/.synapse/ by default — override via SYNAPSE_HOME env var
BASE_DIR = Path(os.getenv("SYNAPSE_HOME", Path.home() / ".synapse"))
DB_PATH = BASE_DIR / "synapse.db"

EMBEDDING_DIM = 384

# Local embedding model (fastembed / ONNX — runs offline, no API call).
# 384-dim multilingual (~50 languages incl. French) to match EMBEDDING_DIM.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Claude model used only for Dream Cycle reasoning (classification / extraction),
# no longer for embeddings.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
