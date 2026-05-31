"""
Offline tests for SYN-21 — resource fetch + summary pipeline.

The network fetch is monkeypatched; HTML extraction (stdlib) and storage/search
run for real (embeddings are local).
"""

import sys
from pathlib import Path

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


def test_process_resource_stores_and_is_idempotent(isolated_db, monkeypatch):
    import dream_cycle.resources as R
    from db import get_connection
    monkeypatch.setattr(R, "fetch_and_extract",
                        lambda url, **k: {"title": "Article Exemple",
                                          "text": "Un texte sur les pandas roux et la grimpe."})
    conn = get_connection()
    try:
        rid1 = R.process_resource("https://exemple.fr/article", conn, client=None)
        rid2 = R.process_resource("https://exemple.fr/article", conn, client=None)
        rows = conn.execute(
            "SELECT url, title, summary, embedding FROM resources").fetchall()
    finally:
        conn.close()
    assert rid1 and rid1 == rid2, "same URL must not be stored twice"
    assert len(rows) == 1
    assert rows[0][0] == "https://exemple.fr/article"
    assert rows[0][3] is not None, "summary should be embedded for search"


def test_failed_fetch_stores_nothing(isolated_db, monkeypatch):
    import dream_cycle.resources as R
    from db import get_connection
    monkeypatch.setattr(R, "fetch_and_extract", lambda url, **k: None)
    conn = get_connection()
    try:
        rid = R.process_resource("https://dead.link/x", conn, client=None)
        n = conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
    finally:
        conn.close()
    assert rid is None and n == 0


def test_stored_resource_is_searchable(isolated_db, monkeypatch):
    import dream_cycle.resources as R
    from db import get_connection
    from embeddings import embed_text
    from entity_search import search_resources_by_vector
    monkeypatch.setattr(R, "fetch_and_extract",
                        lambda url, **k: {"title": "Pandas roux",
                                          "text": "Les pandas roux vivent dans l'Himalaya."})
    conn = get_connection()
    try:
        R.process_capture_resources("regarde https://exemple.fr/pandas",
                                    conn, client=None)
        results = search_resources_by_vector(conn, embed_text("panda roux himalaya"), limit=5)
    finally:
        conn.close()
    assert results, "the resource should be retrievable by similarity"
    assert "exemple.fr/pandas" in (results[0]["url"] or "")
