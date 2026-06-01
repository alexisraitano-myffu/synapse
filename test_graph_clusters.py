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
    nodes = [_ent("e1", "Guitare", 0, x=0, y=0), _ent("e2", "Piano", 0, x=1, y=0),
             _ent("e3", "Violon", 0, x=0, y=1)]
    conn = get_connection()
    try:
        fake = _FakeClient({"0": "Musique"})
        c1 = build_clusters(conn, nodes, client_factory=lambda: fake, model="m")
        assert c1[0]["label"] == "Musique"
        assert c1[0]["community_id"] == 0 and c1[0]["size"] == 3
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
    nodes = [_ent("e1", "Guitare", 0), _ent("e2", "Piano", 0), _ent("e3", "Violon", 0)]
    conn = get_connection()
    try:
        c = build_clusters(conn, nodes, client_factory=lambda: None, model="m")
        assert c[0]["label"] == "Cluster 0"  # generic fallback
        # fallback is not cached → a later run with a key still gets to label it
        cached = conn.execute("SELECT COUNT(*) FROM cluster_labels").fetchone()[0]
        assert cached == 0
    finally:
        conn.close()


def test_tiny_communities_are_not_forced_into_clusters(isolated_db):
    """A 1- or 2-node community is an orphan, not a zone — it must be dropped."""
    from db import get_connection
    from graph_clusters import build_clusters
    nodes = [
        _ent("a1", "Guitare", 0, x=0, y=0), _ent("a2", "Piano", 0, x=1, y=0),
        _ent("a3", "Violon", 0, x=0, y=1),                       # community 0: size 3 → kept
        _ent("b1", "Tennis", 1, x=5, y=5), _ent("b2", "Padel", 1, x=6, y=5),  # size 2 → dropped
        _ent("c1", "Orphan", 2, x=9, y=9),                       # size 1 → dropped
    ]
    conn = get_connection()
    try:
        c = build_clusters(conn, nodes, client_factory=lambda: None, model="m")
        assert [cl["community_id"] for cl in c] == [0]
        assert c[0]["size"] == 3 and len(c[0]["hull"]) >= 3
    finally:
        conn.close()
