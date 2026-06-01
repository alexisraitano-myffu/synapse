"""
Offline tests for SYN-70 — cluster labels (Haiku, cached) + convex hulls.

The Haiku call is stubbed (no API key needed); the cache and hull geometry are
exercised directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class _FakeClient:
    """Stands in for anthropic.Anthropic — returns a fixed JSON label map and
    counts how many times the model was called."""

    def __init__(self, mapping: dict):
        import json
        self._text = json.dumps(mapping)
        self.calls = 0

    @property
    def messages(self):
        return self

    def create(self, **_kw):
        self.calls += 1
        block = type("Block", (), {"text": self._text})()
        return type("Resp", (), {"content": [block]})()


def _ent(nid, label, comm, *, x=0.0, y=0.0):
    return {"id": nid, "kind": "entity", "label": label, "community_id": comm,
            "memory_strength": 1.0, "degree": 1, "x": x, "y": y}


def test_convex_hull_of_a_square():
    from graph_clusters import _convex_hull
    hull = _convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])  # interior pt dropped
    assert {tuple(p) for p in hull} == {(0, 0), (1, 0), (1, 1), (0, 1)}


def test_convex_hull_degenerate():
    from graph_clusters import _convex_hull
    assert _convex_hull([(2, 3)]) == [[2, 3]]                 # single point
    assert _convex_hull([(0, 0), (1, 1)]) == [[0, 0], [1, 1]]  # segment


def test_labels_are_generated_then_cached(isolated_db):
    from db import get_connection
    from graph_clusters import build_clusters
    nodes = [_ent("e1", "Guitare", 0, x=0, y=0), _ent("e2", "Piano", 0, x=1, y=0)]
    conn = get_connection()
    try:
        fake = _FakeClient({"0": "Musique"})
        c1 = build_clusters(conn, nodes, client_factory=lambda: fake, model="m")
        assert c1[0]["label"] == "Musique"
        assert c1[0]["community_id"] == 0 and c1[0]["size"] == 2
        assert fake.calls == 1

        # same defining entities → cache hit, the model is NOT called again
        fake2 = _FakeClient({"0": "Autre"})
        c2 = build_clusters(conn, nodes, client_factory=lambda: fake2, model="m")
        assert c2[0]["label"] == "Musique"  # cached, not the new stub value
        assert fake2.calls == 0
    finally:
        conn.close()


def test_no_client_falls_back_without_caching(isolated_db):
    from db import get_connection
    from graph_clusters import build_clusters
    nodes = [_ent("e1", "Guitare", 0)]
    conn = get_connection()
    try:
        c = build_clusters(conn, nodes, client_factory=lambda: None, model="m")
        assert c[0]["label"] == "Cluster 0"  # generic fallback
        # fallback is not cached → a later run with a key still gets to label it
        cached = conn.execute("SELECT COUNT(*) FROM cluster_labels").fetchone()[0]
        assert cached == 0
    finally:
        conn.close()
