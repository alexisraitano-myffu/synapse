"""Lightweight persistent key/value store at ~/.synapse/config.json.

Used to let the desktop app push its Anthropic API key into the backend at
runtime (so testers don't have to maintain a .env file). The env var always
wins so local dev is unaffected.
"""
import json
import os
import uuid
from pathlib import Path

from config import BASE_DIR

CONFIG_PATH = BASE_DIR / "config.json"


def _load() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (ValueError, OSError):
        return {}


def _save(data: dict) -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(CONFIG_PATH)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def get_anthropic_key() -> str | None:
    """Env var wins (dev override), then config.json, then None."""
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    return _load().get("anthropic_api_key")


def set_anthropic_key(key: str) -> None:
    data = _load()
    data["anthropic_api_key"] = key
    _save(data)


def has_anthropic_key() -> bool:
    return bool(get_anthropic_key())


def get_owner_entity_id() -> str | None:
    """Point 2 — the entity that IS the user (« moi »). First-person captures
    (je/mon/moi) resolve to this entity instead of a phantom 'auteur'. Single-user
    backend → one owner, kept in config.json."""
    return _load().get("owner_entity_id")


def set_owner_entity_id(entity_id: str | None) -> None:
    data = _load()
    if entity_id:
        data["owner_entity_id"] = entity_id
    else:
        data.pop("owner_entity_id", None)
    _save(data)


def get_instance_id() -> str:
    """Stable identity of THIS backend's database (SYN-73). Generated once and
    persisted, so a replica can detect it's now talking to a different master /
    a fresh DB (instance_id changed → its sync cursor is invalid → full resync)."""
    data = _load()
    iid = data.get("instance_id")
    if not iid:
        iid = uuid.uuid4().hex
        data["instance_id"] = iid
        _save(data)
    return iid
