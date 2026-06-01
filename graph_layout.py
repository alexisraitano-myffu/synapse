"""
SYN-69 — ForceAtlas2 layout + persisted node positions for the living map.

Positions live in the `node_positions` table keyed by the /graph node id (an
entity uuid, or 'n:<id>' for an atomic_note). Design goals:

- **Stable**: a read returns the persisted positions verbatim, so two openings of
  the map look identical (no random re-arrangement).
- **Incremental**: a node with no stored position is placed near its community
  centroid (plus a deterministic jitter) and persisted, *without* moving any
  node already on the map.
- **On-demand recompute**: a full ForceAtlas2 pass over the whole map runs only
  when explicitly requested (`full=True`), e.g. `/graph?relayout=true`.

ForceAtlas2 comes from networkx (`nx.forceatlas2_layout`, pure-Python) — no extra
C-extension dependency, consistent with the SYN-68 clustering choice. Intra-
community edges are weighted up so communities cohere spatially.
"""

import math
import os
import zlib

from db import cursor_to_dicts

_INTRA_COMMUNITY_PULL = 3.0  # intra-cluster edges tug harder than bridges

# Semantic layout (SYN-64): soft springs between entities whose embeddings are close,
# so vector-similar nodes drift together even without an explicit relation. Layout-only —
# these edges are never returned to the client. Tunable via env; kept gentle so real
# relations still dominate the structure.
_SEMANTIC_K = int(os.environ.get("SYNAPSE_SEMANTIC_K", "4"))             # neighbours per entity
_SEMANTIC_MIN_SCORE = float(os.environ.get("SYNAPSE_SEMANTIC_MIN_SCORE", "0.80"))  # cosine floor
_SEMANTIC_WEIGHT = float(os.environ.get("SYNAPSE_SEMANTIC_WEIGHT", "0.45"))        # vs ~1.0 relations
_SEMANTIC_MAX_NODES = int(os.environ.get("SYNAPSE_SEMANTIC_MAX_NODES", "800"))     # O(n²) guard


def _read_positions(conn) -> dict:
    return {
        r["node_id"]: (r["x"], r["y"])
        for r in cursor_to_dicts(conn.execute(
            "SELECT node_id, x, y FROM node_positions"))
    }


def _write_positions(conn, pos: dict) -> None:
    """Upsert positions. Caller owns the transaction."""
    for nid, (x, y) in pos.items():
        conn.execute(
            "INSERT INTO node_positions (node_id, x, y, updated_at) "
            "VALUES (?,?,?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(node_id) DO UPDATE SET "
            "  x=excluded.x, y=excluded.y, updated_at=CURRENT_TIMESTAMP",
            (nid, float(x), float(y)))


def _build_graph(nodes: list[dict], edges: list[dict]):
    import networkx as nx
    g = nx.Graph()
    ids = {n["id"] for n in nodes}
    comm = {n["id"]: n.get("community_id") for n in nodes}
    g.add_nodes_from(ids)
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a in ids and b in ids and a != b:
            w = float(e.get("confidence") or 1.0)
            if comm.get(a) is not None and comm.get(a) == comm.get(b):
                w *= _INTRA_COMMUNITY_PULL
            if g.has_edge(a, b):
                g[a][b]["weight"] += w
            else:
                g.add_edge(a, b, weight=w)
    return g


def semantic_edges(conn, nodes: list[dict]) -> list[dict]:
    """Layout-only soft edges between entities with close embeddings (top-K cosine).

    Returns edge dicts `{from, to, confidence, semantic: True}` — `confidence` is the
    spring weight (`_SEMANTIC_WEIGHT × cosine`). Entities only (notes have no entity
    embedding); empty on any missing dependency / oversized graph so layout still works."""
    ent_ids = [n["id"] for n in nodes if n.get("kind") == "entity"]
    if len(ent_ids) < 3 or len(ent_ids) > _SEMANTIC_MAX_NODES:
        return []
    try:
        import numpy as np
        from entity_search import deserialize_vec
    except Exception:
        return []
    want = set(ent_ids)
    rows = [r for r in cursor_to_dicts(conn.execute(
        "SELECT id, embedding FROM entities "
        "WHERE embedding IS NOT NULL AND merged_into_id IS NULL AND status='active'"
    )) if r["id"] in want]
    if len(rows) < 3:
        return []
    ids = [r["id"] for r in rows]
    try:
        mat = np.array([deserialize_vec(r["embedding"]) for r in rows], dtype=np.float32)
    except (ValueError, TypeError):
        return []                                       # ragged dims (model changed mid-flight)
    if mat.ndim != 2 or mat.shape[0] != len(ids):
        return []
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms                                   # cosine = dot on unit vectors
    sims = mat @ mat.T
    np.fill_diagonal(sims, -1.0)                         # never pick self
    k = min(_SEMANTIC_K, len(ids) - 1)
    edges: list[dict] = []
    seen: set = set()
    for a in range(len(ids)):
        for b in np.argpartition(-sims[a], k - 1)[:k]:
            b = int(b)
            score = float(sims[a, b])
            if score < _SEMANTIC_MIN_SCORE:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": ids[a], "to": ids[b],
                          "confidence": round(_SEMANTIC_WEIGHT * score, 4), "semantic": True})
    return edges


def _full_layout(nodes: list[dict], edges: list[dict]) -> dict:
    import networkx as nx
    g = _build_graph(nodes, edges)
    if g.number_of_nodes() == 0:
        return {}
    pos = nx.forceatlas2_layout(g, weight="weight", seed=42)
    return {nid: (float(xy[0]), float(xy[1])) for nid, xy in pos.items()}


def _jitter(node_id: str) -> tuple[float, float]:
    """Deterministic small offset from a stable hash of the id (crc32, not the
    process-salted builtin hash → identical across runs / resume-safe)."""
    h = zlib.crc32(str(node_id).encode())
    ang = (h % 360) * math.pi / 180.0
    r = 0.05 + (h % 17) * 0.005
    return r * math.cos(ang), r * math.sin(ang)


def _place_incremental(nodes: list[dict], existing: dict) -> dict:
    """Position nodes missing from `existing` near their community centroid,
    leaving placed nodes untouched. Falls back to the global centroid for a
    community with no placed members yet (or the origin if the map is empty)."""
    comm_of = {n["id"]: n.get("community_id") for n in nodes}
    sums: dict = {}
    gx = gy = gk = 0.0
    for nid, (x, y) in existing.items():
        gx += x; gy += y; gk += 1
        c = comm_of.get(nid)
        if c is not None:
            sx, sy, k = sums.get(c, (0.0, 0.0, 0))
            sums[c] = (sx + x, sy + y, k + 1)
    centroids = {c: (sx / k, sy / k) for c, (sx, sy, k) in sums.items()}
    global_centroid = (gx / gk, gy / gk) if gk else (0.0, 0.0)
    new: dict = {}
    for n in nodes:
        nid = n["id"]
        if nid in existing:
            continue
        cx, cy = centroids.get(comm_of.get(nid), global_centroid)
        dx, dy = _jitter(nid)
        new[nid] = (cx + dx, cy + dy)
    return new


def ensure_positions(conn, nodes: list[dict], edges: list[dict], *, full: bool = False,
                     semantic: bool = False) -> dict:
    """Return {node_id: {'x':, 'y':}} for every node, persisting as needed.

    full=True forces a ForceAtlas2 recompute of the whole map (and rewrites every
    position). Otherwise only nodes without a stored position are placed
    incrementally; already-placed nodes are returned byte-identical.

    semantic=True adds embedding-kNN soft edges to the layout graph (SYN-64) — only
    meaningful on a full recompute, so it's ignored on the incremental path."""
    existing = _read_positions(conn)
    if full or not existing:
        extra = semantic_edges(conn, nodes) if semantic else []
        pos = _full_layout(nodes, edges + extra)
        if pos:
            with conn:
                _write_positions(conn, pos)
        merged = pos
    else:
        missing = _place_incremental(nodes, existing)
        if missing:
            with conn:
                _write_positions(conn, missing)
        merged = {**existing, **missing}
    return {nid: {"x": x, "y": y} for nid, (x, y) in merged.items()}
