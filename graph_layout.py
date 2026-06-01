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
import zlib

from db import cursor_to_dicts

_INTRA_COMMUNITY_PULL = 3.0  # intra-cluster edges tug harder than bridges


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


def ensure_positions(conn, nodes: list[dict], edges: list[dict], *, full: bool = False) -> dict:
    """Return {node_id: {'x':, 'y':}} for every node, persisting as needed.

    full=True forces a ForceAtlas2 recompute of the whole map (and rewrites every
    position). Otherwise only nodes without a stored position are placed
    incrementally; already-placed nodes are returned byte-identical."""
    existing = _read_positions(conn)
    if full or not existing:
        pos = _full_layout(nodes, edges)
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
