"""
SYN-21 — real resource pipeline: fetch a URL, extract readable text, summarise
it with Haiku, and store it (searchable via its embedded summary).

T5 (SYN-114): the pipeline lives in the core (`resources.rs`) — URL scan, the
dependency-free HTML extraction, the network fetch (ureq), the LLM summary
(prompt = data `prompts/resource-summary.md`, snippet fallback offline/on
error) and the idempotent-per-URL store + embed. Everything runs on the
Brain's OWN connection with network + LLM before the DB write — call these
OUTSIDE any host transaction. This module keeps the historical signatures.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import synapse_core

from core_store import get_brain


def extract_urls(text: str) -> list[str]:
    """All http(s) URLs in a capture, de-duplicated, order-preserving."""
    return list(synapse_core.extract_urls(text or ""))


class _TextExtractor:
    """Compat shim over the core's HTML extraction (title + visible text,
    script/style/nav/… subtrees dropped). Feed then read `.title`/`.text`."""

    def __init__(self):
        self._html: list[str] = []

    def feed(self, html: str) -> None:
        self._html.append(html)

    def _page(self) -> dict:
        return json.loads(synapse_core.extract_page("".join(self._html)))

    @property
    def title(self) -> str:
        return self._page()["title"]

    @property
    def text(self) -> str:
        return self._page()["text"]


def fetch_and_extract(url: str, *, timeout: float = 10.0) -> dict | None:
    """GET the URL and return {title, text}. None on any network/parse failure
    (the caller treats a fetch miss as non-fatal)."""
    raw = synapse_core.fetch_and_extract(url, timeout)
    return json.loads(raw) if raw else None


def _config_args(client) -> dict:
    """LLM kwargs for the core resource calls — `client is None` (offline,
    tests) or a missing key mean no LLM: the summary falls back to a snippet."""
    if client is None:
        return {}
    from config import CLAUDE_MODEL
    from dream_cycle.cycle import PROMPTS_DIR, _TODAY, _llm_args
    try:
        key, base_url, fuel = _llm_args()
    except EnvironmentError:
        return {}
    return {"model": CLAUDE_MODEL, "api_key": key, "prompts_dir": str(PROMPTS_DIR),
            "today": _TODAY, "base_url": base_url, "fuel_token": fuel}


def process_resource(url: str, conn, client, *, capture_id=None, verbose=False) -> str | None:
    """Fetch → extract → summarise → store one URL. Idempotent on the URL (an
    already-stored link is skipped). Returns the resource id, or None if the
    fetch failed. The core runs network + LLM BEFORE the DB write, on ITS OWN
    connection — call outside `with conn:` (`conn` kept for the signature)."""
    rid = get_brain().process_resource(
        url, capture_id=str(capture_id) if capture_id is not None else None,
        **_config_args(client))
    if verbose:
        if rid:
            print(f"    [resource] stored/kept '{url}'")
        else:
            print(f"    [resource] fetch failed for '{url}'")
    return rid


def process_capture_resources(content: str, conn, client, *, capture_id=None,
                              verbose=False) -> list[str]:
    """Process every URL found in a capture. Each is independent — one failure
    never blocks the others (or the rest of the cycle)."""
    ids = json.loads(get_brain().process_capture_resources(
        content or "", capture_id=str(capture_id) if capture_id is not None else None,
        **_config_args(client)))
    if verbose and ids:
        print(f"    [resource] {len(ids)} ressource(s) stockée(s)/retrouvée(s)")
    return ids
