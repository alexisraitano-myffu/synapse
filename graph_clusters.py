"""
SYN-70 — cluster labels (Haiku) + hulls for the living map.

Each community on the map is a named, shaded region. Two pieces:

- **Label**: a short human name ("Musique", "Stack Synapse") from Haiku, fed the
  cluster's top entities. Cached in `cluster_labels` keyed by a stable signature
  of those *defining* entities (not the volatile community index) — so a label is
  reused as long as its characteristic entities do, and a single batched call
  fills every cache miss. No key / a failed call → a generic "Cluster N" fallback
  that is NOT cached (it upgrades to a real label once a key is available).
- **Hull**: a convex polygon around the cluster's node positions, via Andrew's
  monotone chain — pure-Python, no shapely/C-extension (keeps the .dmg lean).
  A concave / alpha shape can refine this later if the visuals call for it.

Cost: labels are batched (one call for all misses) and cached, so this adds a
fraction of a cent per relabel on the user's own key — negligible vs the Dream
Cycle, which calls Haiku on every capture.
"""

import hashlib
import json

from db import cursor_to_dicts

_TOP_N = 8  # defining entities per cluster (fed to Haiku + hashed into the signature)

# A community smaller than this isn't a "zone" — it's an orphan (or a lone pair).
# We don't force it into a named region: a convex hull needs ≥3 points to read as
# an area anyway, and a 1–2 node label would just land on top of the node itself.
MIN_CLUSTER_SIZE = 3


def _node_score(n: dict) -> float:
    return (n.get("memory_strength") or 0.0) * (n.get("degree", 0) + 1)


def _by_community(nodes: list[dict]) -> dict:
    groups: dict = {}
    for n in nodes:
        c = n.get("community_id")
        if c is not None:
            groups.setdefault(c, []).append(n)
    return groups


def _top_entities(members: list[dict]) -> list[str]:
    ents = sorted((m for m in members if m.get("kind") == "entity"),
                  key=_node_score, reverse=True)
    return [e["label"] for e in ents if e.get("label")][:_TOP_N]


def _signature(names: list[str]) -> str:
    return hashlib.sha1("|".join(sorted(names)).encode("utf-8")).hexdigest()


def _cached_labels(conn, signatures: set) -> dict:
    if not signatures:
        return {}
    qs = ",".join("?" * len(signatures))
    return {r["signature"]: r["label"] for r in cursor_to_dicts(conn.execute(
        f"SELECT signature, label FROM cluster_labels WHERE signature IN ({qs})",
        tuple(signatures)))}


def _generate_labels(client, model: str, to_label: dict) -> dict:
    """One batched Haiku call: {community_id: [entities]} → {community_id: label}.
    Returns {} on any failure so the caller falls back to a generic label."""
    payload = [{"id": cid, "entities": ents} for cid, ents in to_label.items()]
    prompt = (
        "Voici des grappes d'entités issues d'un graphe de connaissances personnel. "
        "Donne à CHAQUE grappe un label court (1 à 3 mots, dans la langue des "
        "entités) qui nomme le thème commun. Réponds UNIQUEMENT en JSON : un objet "
        '{"<id>": "<label>"} sans aucun texte autour.\n\n'
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        resp = client.messages.create(
            model=model, max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        return {cid: data[str(cid)] for cid in to_label if data.get(str(cid))}
    except Exception:
        return {}


def _convex_hull(points: list[tuple]) -> list[list]:
    """Andrew's monotone chain. Returns the hull [[x,y],...]; for ≤2 unique points
    returns the points themselves (a point or a segment)."""
    pts = sorted({(round(x, 4), round(y, 4)) for x, y in points})
    if len(pts) <= 2:
        return [[x, y] for x, y in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [[x, y] for x, y in lower[:-1] + upper[:-1]]


def build_clusters(conn, nodes: list[dict], *, client_factory=None, model: str | None = None) -> list[dict]:
    """Return [{community_id, label, size, hull}] for the map. Labels are cached
    (one batched Haiku call for misses); hulls come from node x/y, so the caller
    must have run layout first."""
    groups = {cid: m for cid, m in _by_community(nodes).items()
              if len(m) >= MIN_CLUSTER_SIZE}
    if not groups:
        return []

    tops = {cid: _top_entities(members) for cid, members in groups.items()}
    sigs = {cid: _signature(names) for cid, names in tops.items() if names}

    cached = _cached_labels(conn, set(sigs.values()))
    labels = {cid: cached[sig] for cid, sig in sigs.items() if sig in cached}
    missing = {cid: tops[cid] for cid, sig in sigs.items() if sig not in cached}

    if missing and client_factory is not None:
        try:
            client = client_factory()
        except Exception:
            client = None
        generated = _generate_labels(client, model, missing) if client else {}
        if generated:
            with conn:
                for cid, lbl in generated.items():
                    conn.execute(
                        "INSERT INTO cluster_labels (signature, label, updated_at) "
                        "VALUES (?,?,CURRENT_TIMESTAMP) "
                        "ON CONFLICT(signature) DO UPDATE SET "
                        "  label=excluded.label, updated_at=CURRENT_TIMESTAMP",
                        (sigs[cid], lbl))
            labels.update(generated)

    clusters = []
    for cid, members in groups.items():
        pts = [(n["x"], n["y"]) for n in members if "x" in n and "y" in n]
        clusters.append({
            "community_id": cid,
            "label": labels.get(cid) or f"Cluster {cid}",
            "size": len(members),
            "hull": _convex_hull(pts) if pts else [],
        })
    clusters.sort(key=lambda c: c["community_id"])
    return clusters
