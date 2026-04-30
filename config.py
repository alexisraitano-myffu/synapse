from pathlib import Path
import os

# ~/.synapse/ by default — override via SYNAPSE_HOME env var
BASE_DIR = Path(os.getenv("SYNAPSE_HOME", Path.home() / ".synapse"))
DB_PATH = BASE_DIR / "synapse.db"

EMBEDDING_DIM = 384

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
