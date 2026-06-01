"""
Throwaway synthetic seeder for the living map (SYN-64 dogfood).

Injects ~150 entities in 10 dense thematic communities + inter-theme bridges +
~50 atomic notes, with varied memory_strength / last_mentioned so the map shows
real density, regions, salience and decay — WITHOUT calling the Dream Cycle.

Idempotent: synthetic rows are prefixed (`syn-` ids, notes tagged
`source_ids='syn-demo'`) and wiped on every run. `--clean` removes them and exits.

    python -m scripts.seed_demo_map           # (re)seed
    python -m scripts.seed_demo_map --clean    # remove all synthetic rows

NOTE: writes into the configured SYNAPSE_HOME (default ~/.synapse). Test data only.
"""

import math
import random
import sys
from datetime import date, timedelta

sys.path.insert(0, ".")

from db import get_connection

TODAY = date(2026, 6, 1)
TAU = 30.0  # Ebbinghaus τ in days (mirrors dream_cycle/decay.py)
random.seed(64)

# theme label -> [(name, type), ...]
THEMES = {
    "Famille": [
        ("Maman", "person"), ("Papa", "person"), ("Léa", "person"), ("Tom", "person"),
        ("Mamie Jeanne", "person"), ("Papi Robert", "person"), ("Gilou", "animal"),
        ("Oncle Marc", "person"), ("Tante Sophie", "person"), ("Hugo", "person"),
        ("Emma", "person"), ("Maison de Bordeaux", "place"),
    ],
    "Datadog": [
        ("Datadog", "organization"), ("Sarah", "person"), ("Kevin", "person"),
        ("Équipe Backend", "concept"), ("Migration Postgres", "project"), ("Slack", "concept"),
        ("Sprint Q2", "concept"), ("Julien", "person"), ("Datadog APM", "concept"),
        ("Oncall", "concept"), ("Revue de code", "concept"),
    ],
    "Startups IA": [
        ("Mistral", "organization"), ("Anthropic", "organization"), ("OpenAI", "organization"),
        ("Hugging Face", "organization"), ("LangChain", "project"), ("Projet RAG", "project"),
        ("Embeddings", "concept"), ("Claude", "concept"), ("Fine-tuning", "concept"),
        ("vLLM", "project"), ("Quantization", "concept"),
    ],
    "Tennis": [
        ("Tennis", "concept"), ("Club Roland", "place"), ("Raquette Wilson", "concept"),
        ("Nadal", "person"), ("Match du dimanche", "concept"), ("Padel", "concept"),
        ("Coach Antoine", "person"), ("Tournoi local", "concept"), ("Cordage", "concept"),
    ],
    "Musique": [
        ("Guitare", "concept"), ("Piano", "concept"), ("Spotify", "organization"),
        ("Radiohead", "concept"), ("Daft Punk", "concept"), ("Concert Olympia", "concept"),
        ("Vinyles", "concept"), ("Fender", "organization"), ("Solfège", "concept"),
    ],
    "Voyage": [
        ("Lisbonne", "place"), ("Japon", "place"), ("Tokyo", "place"), ("Kyoto", "place"),
        ("Airbnb", "organization"), ("Portugal", "place"), ("Road trip", "concept"),
        ("Passeport", "concept"), ("Billet avion", "concept"),
    ],
    "Santé": [
        ("Dr Martin", "person"), ("Sommeil", "concept"), ("Course à pied", "concept"),
        ("Méditation", "concept"), ("Nutrition", "concept"), ("Ostéo", "person"),
        ("Magnésium", "concept"), ("Hydratation", "concept"),
    ],
    "Amis Lyon": [
        ("Lyon", "place"), ("Paul", "person"), ("Marie", "person"), ("Le Sucre", "place"),
        ("Camille", "person"), ("Soirée jeux", "concept"), ("Théo", "person"),
        ("Brunch dimanche", "concept"), ("Quai du Rhône", "place"),
    ],
    "Cuisine": [
        ("Curry maison", "concept"), ("Marché", "place"), ("Basilic", "concept"),
        ("Pâtes fraîches", "concept"), ("Four", "concept"), ("Thermomix", "concept"),
        ("Pain au levain", "concept"), ("Huile d'olive", "concept"),
    ],
    "Synapse": [
        ("Synapse", "project"), ("Kotlin", "concept"), ("Compose", "concept"),
        ("FastAPI", "concept"), ("SQLite", "concept"), ("Dream Cycle", "concept"),
        ("Backend Synapse", "project"), ("App mobile", "project"), ("mDNS", "concept"),
    ],
}

# explicit cross-theme bridges (serendipity edges)
BRIDGES = [
    ("Embeddings", "Backend Synapse", "utilisé_par"),
    ("Claude", "Dream Cycle", "alimente"),
    ("Anthropic", "Claude", "édite"),
    ("Marie", "Maman", "amie_de"),
    ("Paul", "Tennis", "joue_au"),
    ("Kevin", "Guitare", "joue_de"),
    ("Lyon", "Course à pied", "lieu_de"),
    ("Kotlin", "App mobile", "écrite_en"),
    ("Japon", "Concert Olympia", "souvenir_de"),
    ("Sarah", "Méditation", "pratique"),
]

INTRA_PREDICATES = ["lié_à", "connaît", "travaille_avec", "fait_partie_de", "évoque"]
FACT_TEMPLATES = {
    "person": [("contexte", "rencontré·e récemment"), ("note", "à recontacter")],
    "place": [("type", "lieu"), ("note", "à revisiter")],
    "project": [("statut", "en cours"), ("priorité", "haute")],
    "organization": [("secteur", "tech"), ("note", "à suivre")],
    "concept": [("catégorie", "idée"), ("note", "à creuser")],
    "animal": [("espèce", "chien"), ("note", "trop mignon")],
}
NOTE_TEMPLATES = [
    "Pensé à {a} aujourd'hui, ça m'a rappelé {b}.",
    "Discussion intéressante autour de {a} et {b}.",
    "Note rapide : {a} — à creuser avec {b}.",
    "Bon moment avec {a}.",
    "Idée en passant sur {a}, lien possible avec {b}.",
]


def _ms_for(days_ago: int) -> float:
    return round(math.exp(-days_ago / TAU), 4)


def clean(conn):
    with conn:
        conn.execute("DELETE FROM relations WHERE id LIKE 'synr-%'")
        conn.execute("DELETE FROM facts WHERE id LIKE 'synf-%'")
        conn.execute("DELETE FROM entities WHERE id LIKE 'syn-%'")
        conn.execute("DELETE FROM atomic_notes WHERE source_ids = 'syn-demo'")
    print("Synthetic rows removed.")


def seed(conn):
    from embeddings import embed_text                 # local ONNX model (loads on first call)
    clean(conn)
    name_to_id: dict[str, str] = {}
    rels: list[tuple] = []
    facts: list[tuple] = []
    ridx = fidx = 0

    with conn:
        # entities
        for theme, members in THEMES.items():
            for j, (name, etype) in enumerate(members):
                eid = f"syn-{abs(hash(name)) % 10**8}-{j}"
                name_to_id[name] = eid
                days = random.choice([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 120])
                lm = (TODAY - timedelta(days=days)).isoformat()
                conn.execute(
                    "INSERT INTO entities (id, type, canonical_name, mention_count, "
                    "persistence_value, summary, last_mentioned, status, memory_strength, embedding) "
                    "VALUES (?,?,?,?,?,?,?, 'active', ?, ?)",
                    (eid, etype, name, random.randint(1, 12), random.randint(2, 5),
                     f"{name} — {theme}", lm, _ms_for(days), embed_text(f"{name}. {theme}")),
                )
                for pred, val in random.sample(FACT_TEMPLATES[etype], k=1):
                    conn.execute(
                        "INSERT INTO facts (id, entity_id, predicate, value, confidence) VALUES (?,?,?,?,?)",
                        (f"synf-{fidx}", eid, pred, val, round(random.uniform(0.6, 0.95), 2)))
                    fidx += 1

            # intra-theme relations: each entity wired to 2 others in the theme
            ids = [name_to_id[n] for n, _ in members]
            for a in ids:
                for b in random.sample([x for x in ids if x != a], k=min(2, len(ids) - 1)):
                    conn.execute(
                        "INSERT INTO relations (id, entity_from, predicate, entity_to, confidence) VALUES (?,?,?,?,?)",
                        (f"synr-{ridx}", a, random.choice(INTRA_PREDICATES), b, round(random.uniform(0.55, 0.9), 2)))
                    ridx += 1

        # inter-theme bridges
        for a, b, pred in BRIDGES:
            if a in name_to_id and b in name_to_id:
                conn.execute(
                    "INSERT INTO relations (id, entity_from, predicate, entity_to, confidence) VALUES (?,?,?,?,?)",
                    (f"synr-{ridx}", name_to_id[a], pred, name_to_id[b], round(random.uniform(0.5, 0.8), 2)))
                ridx += 1

        # atomic notes — mostly within a theme to reinforce clusters
        for _ in range(50):
            theme = random.choice(list(THEMES))
            pool = [n for n, _ in THEMES[theme]]
            k = random.randint(1, min(3, len(pool)))
            picked = random.sample(pool, k=k)
            a = picked[0]; b = picked[1] if len(picked) > 1 else random.choice(pool)
            content = random.choice(NOTE_TEMPLATES).format(a=a, b=b)
            days = random.choice([1, 2, 4, 7, 12, 20, 33, 60, 95])
            lr = (TODAY - timedelta(days=days)).isoformat()
            import json as _json
            conn.execute(
                "INSERT INTO atomic_notes (title, content, summary, entities_mentioned, "
                "memory_strength, last_reactivated_at, source_ids) "
                "VALUES (?,?,?,?,?,?, 'syn-demo')",
                (content[:60], content, content, _json.dumps(picked, ensure_ascii=False),
                 _ms_for(days), lr))

    ents = conn.execute("SELECT COUNT(*) FROM entities WHERE id LIKE 'syn-%'").fetchone()[0]
    rcnt = conn.execute("SELECT COUNT(*) FROM relations WHERE id LIKE 'synr-%'").fetchone()[0]
    ncnt = conn.execute("SELECT COUNT(*) FROM atomic_notes WHERE source_ids='syn-demo'").fetchone()[0]
    print(f"Seeded: {ents} entities, {rcnt} relations, {ncnt} notes (synthetic).")


if __name__ == "__main__":
    conn = get_connection()
    try:
        if "--clean" in sys.argv:
            clean(conn)
        else:
            seed(conn)
    finally:
        conn.close()
