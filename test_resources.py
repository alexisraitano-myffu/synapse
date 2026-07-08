"""
Offline tests for SYN-21 — resource fetch + summary pipeline.

T5: the pipeline (URL scan, HTML extraction, fetch, store) lives in the core,
so the network seam is a local http.server stub instead of a monkeypatch —
the tests exercise the REAL fetch path. Storage/search run for real
(embeddings are local); no LLM (client=None → snippet-fallback summary).
"""

import http.server
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


def test_extract_urls_dedups_and_strips_punctuation():
    from dream_cycle.resources import extract_urls
    urls = extract_urls(
        "voir https://exemple.fr/article. puis https://exemple.fr/article encore, "
        "et http://x.io/y)"
    )
    assert urls.count("https://exemple.fr/article") == 1
    assert "http://x.io/y" in urls


def test_text_extractor_skips_script_grabs_title():
    from dream_cycle.resources import _TextExtractor
    html = ("<html><head><title>Mon Titre</title><style>x{}</style></head>"
            "<body><script>bad()</script><p>Bonjour le monde</p>"
            "<nav>menu</nav></body></html>")
    p = _TextExtractor()
    p.feed(html)
    assert p.title.strip() == "Mon Titre"
    assert "Bonjour le monde" in p.text
    assert "bad()" not in p.text and "menu" not in p.text


@pytest.fixture
def html_stub():
    """Local HTTP server: `/page` serves the configured HTML, anything else
    404s. Yields (base_url, set_page)."""
    state = {"html": ""}

    class _Stub(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/page":
                self.send_error(404)
                return
            body = state["html"].encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def set_page(html):
        state["html"] = html

    yield f"http://127.0.0.1:{server.server_port}", set_page
    server.shutdown()


def test_process_resource_stores_and_is_idempotent(isolated_db, html_stub):
    import dream_cycle.resources as R
    from db import get_connection
    base, set_page = html_stub
    set_page("<html><head><title>Article Exemple</title></head>"
             "<body><p>Un texte sur les pandas roux et la grimpe.</p></body></html>")
    conn = get_connection()
    try:
        rid1 = R.process_resource(f"{base}/page", conn, client=None)
        rid2 = R.process_resource(f"{base}/page", conn, client=None)
        rows = conn.execute(
            "SELECT url, title, summary, embedding FROM resources").fetchall()
    finally:
        conn.close()
    assert rid1 and rid1 == rid2, "same URL must not be stored twice"
    assert len(rows) == 1
    assert rows[0][0] == f"{base}/page"
    assert rows[0][1] == "Article Exemple"
    assert "pandas roux" in rows[0][2]          # no client → snippet fallback
    assert rows[0][3] is not None, "summary should be embedded for search"


def test_failed_fetch_stores_nothing(isolated_db, html_stub):
    import dream_cycle.resources as R
    from db import get_connection
    base, _ = html_stub
    conn = get_connection()
    try:
        rid = R.process_resource(f"{base}/dead-link", conn, client=None)  # 404
        n = conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
    finally:
        conn.close()
    assert rid is None and n == 0


def test_stored_resource_is_searchable(isolated_db, html_stub):
    import dream_cycle.resources as R
    from db import get_connection
    from embeddings import embed_text
    from entity_search import search_resources_by_vector
    base, set_page = html_stub
    set_page("<html><head><title>Pandas roux</title></head>"
             "<body><p>Les pandas roux vivent dans l'Himalaya.</p></body></html>")
    conn = get_connection()
    try:
        ids = R.process_capture_resources(f"regarde {base}/page",
                                          conn, client=None)
        results = search_resources_by_vector(conn, embed_text("panda roux himalaya"), limit=5)
    finally:
        conn.close()
    assert ids, "the URL in the capture should be fetched and stored"
    assert results, "the resource should be retrievable by similarity"
    assert "/page" in (results[0]["url"] or "")
