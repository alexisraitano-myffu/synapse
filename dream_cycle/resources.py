"""
SYN-21 — real resource pipeline: fetch a URL, extract readable text, summarise
it with Haiku, and store it (searchable via its embedded summary).

HTML extraction is stdlib-only (html.parser strips tags + script/style/nav) —
no trafilatura/lxml dependency. The text is rough, but Haiku summarises it well,
and the cleaned `content` is good enough for V1. Swap in trafilatura later if the
stored content quality matters.
"""

import re
import sys
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from config import CLAUDE_MODEL
from db import cursor_to_dicts, first_row
from embeddings import embed_text

URL_RE = re.compile(r'https?://[^\s<>"\'\)\]]+')
_SKIP_TAGS = {"script", "style", "noscript", "head", "nav", "footer", "header", "svg"}
_MAX_CONTENT = 50_000  # cap stored text — articles can be huge


def extract_urls(text: str) -> list[str]:
    """All http(s) URLs in a capture, de-duplicated, order-preserving."""
    seen, out = set(), []
    for u in URL_RE.findall(text or ""):
        u = u.rstrip(".,;)")  # trailing punctuation isn't part of the URL
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title, self._in_title = "", False
        self._skip = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip == 0:
            t = data.strip()
            if t:
                self._chunks.append(t)

    @property
    def text(self) -> str:
        return re.sub(r"\s+\n", "\n", "\n".join(self._chunks)).strip()


def fetch_and_extract(url: str, *, timeout: float = 10.0) -> dict | None:
    """GET the URL and return {title, text}. None on any network/parse failure
    (the caller treats a fetch miss as non-fatal)."""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": "SynapseBot/1.0 (personal memory)"})
        resp.raise_for_status()
    except Exception:
        return None
    if "html" not in resp.headers.get("content-type", "text/html"):
        # non-HTML (PDF, etc.) — out of scope for V1, store raw text if textual
        text = resp.text if resp.text else ""
        return {"title": url, "text": text[:_MAX_CONTENT]} if text else None
    parser = _TextExtractor()
    parser.feed(resp.text)
    text = parser.text[:_MAX_CONTENT]
    if not text:
        return None
    return {"title": (parser.title.strip() or url), "text": text}


_SUMMARY_SYSTEM = (
    "Tu résumes une ressource web pour une mémoire personnelle. En 2-4 phrases, "
    "en français, donne l'essentiel : de quoi ça parle, l'idée clé, pourquoi c'est "
    "notable. Pas de méta-commentaire, pas de 'cet article'. Juste le fond."
)


def summarize(client, title: str, text: str) -> str:
    """Haiku summary of the extracted text. Falls back to a truncated snippet if
    no client (offline) or on error."""
    if client is None:
        return text[:300]
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=300,
            system=_SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": f"Titre : {title}\n\n{text[:8000]}"}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return text[:300]


def process_resource(url: str, conn, client, *, capture_id=None, verbose=False) -> str | None:
    """Fetch → extract → summarise → store one URL. Idempotent on the URL (an
    already-stored link is skipped). Returns the resource id, or None if skipped
    / fetch failed. Network + LLM happen BEFORE the DB write (no lock held)."""
    existing = first_row(conn.execute("SELECT id FROM resources WHERE url = ?", (url,)))
    if existing:
        if verbose:
            print(f"    [resource] '{url}' already stored — skip")
        return existing["id"]

    extracted = fetch_and_extract(url)
    if not extracted:
        if verbose:
            print(f"    [resource] fetch failed for '{url}'")
        return None

    summary = summarize(client, extracted["title"], extracted["text"])
    try:
        embedding = embed_text(f"{extracted['title']}\n{summary}")
    except Exception:
        embedding = None

    rid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute(
            "INSERT INTO resources (id, type, source, url, title, content, summary, "
            " embedding, fetched_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, "url", str(capture_id) if capture_id is not None else None,
             url, extracted["title"], extracted["text"], summary, embedding, now),
        )
    if verbose:
        print(f"    [resource] stored '{extracted['title'][:60]}' ({url})")
    return rid


def process_capture_resources(content: str, conn, client, *, capture_id=None,
                              verbose=False) -> list[str]:
    """Process every URL found in a capture. Each is independent — one failure
    never blocks the others (or the rest of the cycle)."""
    ids = []
    for url in extract_urls(content):
        try:
            rid = process_resource(url, conn, client, capture_id=capture_id, verbose=verbose)
            if rid:
                ids.append(rid)
        except Exception as exc:
            if verbose:
                print(f"    [resource] error on '{url}': {exc}")
    return ids
