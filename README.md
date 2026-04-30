# Synapse — Phase B : MCP Server

Serveur MCP local qui expose la mémoire sémantique de Synapse directement dans Claude Code et Claude Desktop.

## Stack

| Composant | Choix | Raison |
|-----------|-------|--------|
| Stockage | SQLite + `sqlite-vec` | Local-first, zéro serveur, vecteurs inclus |
| Embeddings | `fastembed` (ONNX) | 130 MB, pas de PyTorch, ~30 ms/requête |
| Protocole | MCP (stdio) | Natif Claude Code/Desktop |

> **Note :** `sqlite-vec` est le successeur officiel de `sqlite-vss`, plus léger et activement maintenu.

---

## Installation

```bash
cd ~/Synapse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le modèle d'embedding (~130 MB) se télécharge automatiquement au premier appel `search_memory`.

---

## Lancer le serveur (test manuel)

```bash
source .venv/bin/activate
python mcp_server/server.py
```

Le serveur écoute sur **stdio** — c'est le transport attendu par Claude.

---

## Intégration Claude Desktop

Ajouter dans `~/Library/Application Support/Claude/claude_desktop_config.json` :

```json
{
  "mcpServers": {
    "synapse": {
      "command": "/Users/alexisraitano/Synapse/.venv/bin/python",
      "args": ["/Users/alexisraitano/Synapse/mcp_server/server.py"]
    }
  }
}
```

Redémarrer Claude Desktop. L'icône outil apparaît dans la barre de composition.

## Intégration Claude Code

```bash
claude mcp add synapse \
  /Users/alexisraitano/Synapse/.venv/bin/python \
  /Users/alexisraitano/Synapse/mcp_server/server.py
```

---

## Outils MCP exposés

| Outil | Description |
|-------|-------------|
| `add_to_inbox` | Capture rapide (texte brut → inbox) |
| `search_memory` | Recherche sémantique dans les notes traitées |
| `list_recent` | Affiche les dernières entrées inbox non traitées |

### Exemple d'usage dans Claude

```
Souviens-toi que notre réunion du 24 avril a décidé d'utiliser sqlite-vec.
→ add_to_inbox("Décision archi : sqlite-vec retenu le 2026-04-24", source="meeting")

Qu'avons-nous décidé sur la base de données ?
→ search_memory("décision base de données vectorielle")
```

---

## Tester sans Claude (script Python)

```python
from db import init_db, get_connection
import sqlite_vec

init_db()
conn = get_connection()

# Ajouter une note directement dans atomic_notes + vecteur (bypass Dream Cycle)
from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
vec = list(next(model.embed(["Test note on sqlite-vec architecture"])))
blob = sqlite_vec.serialize_float32(vec)

conn.execute("INSERT INTO atomic_notes (title, content) VALUES (?, ?)",
             ("Test", "sqlite-vec est le moteur vectoriel de Synapse"))
note_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
conn.execute("INSERT INTO atomic_notes_vec(rowid, embedding) VALUES (?, ?)", (note_id, blob))
conn.commit()
print("Note insérée, id =", note_id)
```

---

## Roadmap — Phases suivantes

- **Phase C** : Dream Cycle — agent nocturne (FastAPI + Claude Haiku) qui dépile l'inbox
- **Phase D** : Graphe atomique — visualisation D3.js des nœuds de connaissance
- **Phase E** : Captures — extension Chrome + app mobile (bouton vocal → Whisper)
