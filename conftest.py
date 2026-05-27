"""
Shared pytest fixtures for Synapse.

`isolated_db` points every DB helper at a fresh temp SQLite file so tests never
touch the real ~/.synapse database and never leak state between tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Redirect config.DB_PATH / db.DB_PATH to a temp file and init the schema."""
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path))

    import config as cfg_mod
    import db as db_mod

    new_db_path = tmp_path / "synapse.db"
    monkeypatch.setattr(cfg_mod, "DB_PATH", new_db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", new_db_path)

    db_mod.init_db()
    return new_db_path
