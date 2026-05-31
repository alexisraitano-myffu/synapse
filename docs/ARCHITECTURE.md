# Synapse — Spec technique actuelle (prod)

> Architecture réelle du système : ce qui tourne aujourd'hui. Les pistes restantes sont listées en §9–§10.

Philosophie : **« capture passive, traitement actif »**, **100 % local-first, 0 % cloud**. On capture tout (texte), une IA (Claude Haiku) nettoie/relie/structure, une base locale rend le tout consultable et mémorisable durablement.

---

## 1. Topologie de déploiement

Un **cerveau** (Mac Mini, toujours allumé) ; les autres appareils sont des **répliques en lecture + une outbox de captures**. Aucun cloud : synchro sur le LAN, ou via Tailscale (réseau privé chiffré) en déplacement.

```mermaid
flowchart LR
    Phone["📱 Téléphone<br/>réplique lecture + outbox"]
    Air["💻 MacBook Air<br/>réplique lecture + outbox"]
    Mini["🖥 Mac Mini — CERVEAU<br/>DB canonique + Dream Cycle<br/>(seul processeur / écrivain)"]

    Phone <-->|"LAN / Tailscale"| Mini
    Air <-->|"LAN / Tailscale"| Mini

    Phone -.->|"captures (UUID) → outbox"| Mini
    Air -.->|"captures (UUID) → outbox"| Mini
    Mini -.->|"état dérivé (lecture seule)"| Phone
    Mini -.->|"état dérivé (lecture seule)"| Air
```

Règles :
- **Un seul processeur** (le Mini) fait tourner le Dream Cycle et écrit l'état dérivé (entités/faits/notes). Évite la divergence multi-maître.
- Les **captures** remontent de partout, append-only, **clé = UUID** → conflit-free.
- L'**état dérivé** redescend en lecture seule → flux à sens unique, rien à fusionner.
- Chaque appareil garde une **copie locale complète** → consultation **hors-ligne, partout**. Seule la *transformation* des nouvelles captures attend de joindre le Mini.

---

## 2. Flux de traitement — le Dream Cycle

Un seul cycle, par entrée d'inbox. Claude classe l'entrée puis route selon `input_type`.

```mermaid
flowchart TD
    Capture["Capture (texte)"] --> Inbox[("inbox<br/>processed_at = NULL")]
    Inbox --> Classify{"① CLASSIFY (Haiku)<br/>input_type + entités + persistance 1–5"}

    Classify -->|fact| Resolve["② RESOLVE<br/>dédup alias · dates relatives→absolues"]
    Classify -->|episodic| Episodic["write_episodic_note<br/>→ atomic_notes (+ vecteur)"]
    Classify -->|ephemeral| Intentions[("intentions · TTL 48h")]
    Classify -->|"URL détectée"| ResourceFetch["fetch + extraction + résumé Haiku<br/>→ resources (cherchable) · SYN-21"]

    Resolve --> Create["entité créée sur mention<br/>(garde-fou MIN_ENTITY_PERSISTENCE)"]
    Create --> Score["③ SCORE — compute_confidence<br/>explicit .5 + contexte .3 + répétition ≤.2 + bonus persistance"]
    Score --> Route{"④ ROUTE des FAITS par confiance"}

    Create --> Entities[("entities (nœuds)")]
    Route -->|"> 0.85"| Facts[("facts (confirmés)")]
    Route -->|"0.5 – 0.85"| Pending[("pending_facts")]
    Route -->|"< 0.5"| Review[("review_queue")]
    Facts --> Entities

    Pending --> Validate["⑤ VALIDATION COMPORTEMENTALE<br/>corroboration → promotion si >0.85"]
    Validate -->|promu| Facts

    Entities --> Vectorize["⑥ VECTORIZE → entities.embedding (fastembed local)"]
    Episodic --> Store[["search_memory / get_entity"]]
    Entities --> Store
```

Code : [dream_cycle/cycle.py](../dream_cycle/cycle.py). Déclenché par `python -m dream_cycle` ou le tool MCP `run_dream_cycle`.

---

## 3. Les leviers réglables (vision fonctionnelle)

| Levier | Où | Valeur | Effet |
|---|---|---|---|
| Barème persistance 1–5 | prompt classify | rubrique fixe | définit permanent ↔ bruit ; nourrit confiance + (futur) oubli |
| Poids de confiance | `compute_confidence` | .5 / .3 / ≤.2 / bonus | ↑ consolide vite (+ faux positifs) ; ↓ prudent |
| **`MIN_ENTITY_PERSISTENCE`** | `step4_route` | **2** | garde-fou anti-pollution : ↑ = moins d'entités (plus strict) ; 1 = crée pour tout ce qui est mentionné |
| Seuil consolidation `T_high` (faits) | `step4_route` | 0.85 | un **fait** > 0.85 est confirmé direct ; sinon pending (l'entité, elle, est créée sur mention) |
| Seuil pending `T_pending` | `step4_route` | 0.5 | borne basse « à valider » vs digest |
| TTL intentions | `handle_intentions` | 48h | durée des rappels éphémères |
| **memory_strength decay** | `dream_cycle/decay.py` | τ = `SYNAPSE_DECAY_TAU_DAYS` (30j) | oubli gracieux Ebbinghaus (SYN-19) — actif |
| **Seuil merge embedding** | `_propose_merge_by_embedding` | `SYNAPSE_MERGE_EMBEDDING_THRESHOLD` (0.85) | fusion auto de doublons (SYN-61) |
| **Prédicats single-valued** | `facts_store` | liste statique | last-writes-wins / obsolescence (SYN-37) |
| Confiance validation manuelle | `validate_fact` | 0.95 | certitude quand l'utilisateur confirme |
| Seuil distance graphe | visualiseur | 1.1 | densité du graphe |
| Modèle d'embedding | `config.py` | MiniLM multilingue 384-d | qualité/langue de la similarité |

> ℹ️ Depuis le découplage : l'**entité** est créée dès la 1ʳᵉ mention (si elle passe `MIN_ENTITY_PERSISTENCE` ou est dans une relation). Ses **faits**, eux, restent en pending tant qu'ils n'atteignent pas 0.85 (1ʳᵉ mention ≈ 0.75) → confirmés à la 2ᵉ mention ou par validation manuelle.

---

## 4. Modèle de données

SQLite (`~/.synapse/synapse.db`), ouvert via `apsw`, extension `sqlite-vec`. Schéma : [db/__init__.py](../db/__init__.py).

| Table | Rôle | Niveau mémoire |
|---|---|---|
| `inbox` | captures brutes (`processed_at` NULL → à traiter) | working (TTL 7j *prévu*) |
| `atomic_notes` (+`atomic_notes_vec`) | mémoire épisodique ; cols `summary`, `entities_mentioned`, `memory_strength` | épisodique |
| `entities` / `facts` / `relations` | graphe sémantique (ids = UUID) | sémantique (∞) |
| `pending_facts` / `review_queue` | faits à valider / digest | — |
| `intentions` | éphémère (TTL 48h) | — |
| `validation_events` | journal append-only des validations (durable, réplicable) | — |
| `cycle_runs` | 1 ligne par run du cycle (stats `/dream-cycle/last`) | — |
| `resources` | URL fetchées + résumé + embedding (SYN-21) | sémantique |
| `entity_merge_proposals` | file de dédup d'entités (SYN-39/61) | — |
| `entity_type_proposals` / `active_entity_types` | vocab de types dynamique (SYN-58) | — |
| `project_entries` / `project_state` / `project_state_versions` | agrégat projet (SYN-40) | — |
| `knowledge_graph` | legacy, **inutilisé** | — |

Colonnes de cycle de vie : `entities.status` (active/pending/archived) + `archived_at` ; `facts.archived_at`/`obsoleted_at`/`obsoleted_by` (SYN-37/58/59) ; `atomic_notes.last_reactivated_at` (decay SYN-19). Vues de lecture filtrent par défaut (`status='active'`, `archived_at IS NULL`, `obsoleted_at IS NULL`).

Embeddings : **fastembed local** (ONNX, `paraphrase-multilingual-MiniLM-L12-v2`, 384-d, L2-normalisé). Pas d'appel API pour embedder. Notes dans `atomic_notes_vec` (vec0) ; entités en BLOB (`entities.embedding`) recherchées par cosinus manuel.

---

## 5. Déclenchement du cycle — garde-fous

Le cycle est **idempotent** (ne traite que l'inbox non traitée) → sûr à relancer. On déclenche **par condition, pas par horloge**.

```mermaid
flowchart TD
    T["Trigger : sur capture (debounce) · périodique · manuel · réveil"] --> L{"lock libre ?"}
    L -->|non| S1["skip — un cycle tourne déjà"]
    L -->|oui| I{"inbox non vide<br/>OU forcé ?"}
    I -->|non| S2["skip — rien à faire"]
    I -->|oui| R["run cycle → écrit cycle_runs (last_run + stats)"]
    R --> U["libère le lock"]
```

- **Sur arrivée de captures** (principal) : ✅ implémenté — scheduler interne à l'API, debounce `SYNAPSE_CYCLE_DEBOUNCE_SECONDS` (déf. 120s), activé par `SYNAPSE_AUTO_CYCLE=1`. Un batch synchronisé = un cycle. Sert aussi de rattrapage au démarrage (entrées en file → run après le debounce).
- **Filet périodique** (`launchd` ~3h) : complément hors-process, no-op si inbox vide → rattrapage si l'API n'a pas tourné.
- **Maintenance nocturne** (futur) : 1×/jour (decay, compression, digest) ; inoffensif si manqué.
- **Manuel** : bouton « Déclencher maintenant ».
- Verrous : **lock mono-instance** + **`cycle_runs.last_run`**. Sur macOS, `launchd` > `cron` (rattrape au réveil). Seul le Mini planifie.

---

## 6. Outils MCP (existant)

`add_to_inbox` · `search_memory` (vecteur notes + entités, fusion par score ; fallback texte) · `list_recent` · `run_dream_cycle` · `get_entity` · `list_pending` · `validate_fact`. Code : [mcp_server/server.py](../mcp_server/server.py).

---

## 7. API HTTP (implémentée — `api/app.py`, `python -m api`)

Sur le Mini (FastAPI, port 8000), auth **bearer token** (`SYNAPSE_API_TOKEN` ; auth désactivée si non défini = dev), LAN/Tailscale. Les 10 endpoints sont implémentés ; les formes de réponse incluent des champs réservés non encore remplis (`memory_strength`, etc.). Contrat machine : [`openapi.json`](../openapi.json).

| Endpoint | Rôle |
|---|---|
| `GET /health` | ping + statut (pour l'indicateur « Mac · 12ms ») |
| `POST /capture` | capture ; **idempotent sur `id` (UUID client)** ; body `{id, device_id, captured_at, content, type, source}` |
| `GET /feed?limit=` | captures récentes + **statut** (queued / processed / failed) |
| `GET /graph?mode=ego&entity=&depth=` | nœuds (entités) + arêtes (relations), `mention_count`, type, `memory_strength` |
| `GET /entity/{id}` | détail entité : facts, relations, aliases, summary, stats |
| `GET /pending` | faits à valider : question lisible + **citation source** + confiance |
| `POST /pending/{id}/validate` | `{confirmed, correction?}` → stocké comme **événement** |
| `POST /dream-cycle/run` | déclenche le cycle (avec lock) |
| `GET /dream-cycle/last` | dernier run : date, nb notes, nb entités, nb pending (écran Réglages) |
| `GET /changes?since=<cursor>` | réplication : descend l'état dérivé mis à jour vers les répliques |

---

## 8. Modèle de synchronisation

```mermaid
sequenceDiagram
    participant P as Téléphone (outbox)
    participant M as Mac Mini (API)
    participant C as Dream Cycle
    P->>P: capture hors-ligne → outbox (id = UUID)
    Note over P,M: retour sur le LAN (ou Tailscale)
    P->>M: POST /capture (idempotent sur id)
    M->>M: INSERT inbox si id nouveau
    M-->>C: déclenche (debounce + lock)
    C->>C: classify → route → vectorize
    P->>M: GET /changes?since=cursor
    M-->>P: état dérivé à jour (graphe / faits / notes)
```

Décisions verrouillées (rendent le multi-Mac possible plus tard, sans le coûter maintenant) :
1. Chaque capture porte `id` (UUID client) + `device_id` + `captured_at`.
2. `POST /capture` **idempotent** sur l'`id` (reprise offline sans doublon).
3. Les **validations sont des événements** append-only (pas un simple UPDATE) → survivent à une reconstruction, se répliquent.
4. L'état dérivé est **reconstructible** depuis inbox + événements de validation.

---

## 9. État d'implémentation

**Implémenté** : Dream Cycle unifié (routing **non-exclusif**) · création d'entités sur mention + garde-fou · embeddings locaux · `search_memory` notes + entités + **ressources** · API HTTP (33 endpoints) + modèle de sync · résilience par entrée · tests hors-ligne (73 verts).

**Batch graphe d'entités (shippé 2026-05-31)** :

| Domaine | Livré | Ticket |
|---|---|---|
| Sémantique | vectorisation entités (cosine partagé) · suggestions `/entity/{id}/similar` | SYN-60, SYN-62 |
| Dédup | merge proposals substring + fallback embedding | SYN-39, SYN-61 |
| Types | vocab `entity.type` extensible via pending + garde-fou projet | SYN-58 |
| Faits | conflit last-writes-wins (obsolescence) + archive/obsolète manuel | SYN-37, SYN-59 |
| Mémoire | **decay `memory_strength`** (Ebbinghaus) | SYN-19 |
| Ressources | **fetch + résumé d'URL** → `resources` cherchable | SYN-21 |

**Pistes restantes** :

| Domaine | Piste |
|---|---|
| Traitement | résolution de coréférence (fenêtre de contexte récent) · multi-format (image / vision) |
| Projets | refinement actif via MCP · exhumation · élagage dégressif de l'historique de synthèse (SYN-40 future) |
| Mémoire | TTL inbox · compression des `atomic_notes` éteintes · digest périodique de la `review_queue` |
| App | toggle « voir entités archivées » · retouches navigation |

---

## 10. Roadmap

Directions backend (sans dates) :
- ~~Oubli gracieux (`memory_strength` Ebbinghaus)~~ ✅ SYN-19 · ~~Ressources (fetch + résumé d'URL)~~ ✅ SYN-21.
- **Compression** — compresser/archiver les `atomic_notes` éteintes (sous le seuil de decay), PDF resources.
- **Coréférence** — résoudre pronoms/références via une fenêtre de contexte récent.
- **Projets (SYN-40 future)** — refinement actif via agent MCP, exhumation, élagage dégressif de l'historique de synthèse.
- **Digest** — remonter les éléments faible-confiance de `review_queue`.
- **Découverte LAN** — mDNS/Bonjour pour que les clients trouvent le serveur sans URL manuelle.

Les clients (mobile/desktop) vivent dans un projet séparé et consomment cette API HTTP.
