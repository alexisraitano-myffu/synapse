# Synapse — Accès Mobile (Phase E)

Capture d'informations depuis Android vers Synapse via HTTP.

---

## Architecture

```
Android (HTTP Shortcuts)
        ↓  HTTPS
Cloudflare Tunnel
        ↓  HTTP
api/server.py :8765 (FastAPI)
        ↓
~/.synapse/synapse.db  →  Dream Cycle  →  atomic_notes
```

---

## Démarrage rapide

### 1. Configurer .env

Ouvrir `.env` à la racine du projet et renseigner votre clé Anthropic :

```
ANTHROPIC_API_KEY=sk-ant-...
API_TOKEN=synapse_6d8eecef28688b0ccd2ec0fbe6bd7900
API_HOST=0.0.0.0
API_PORT=8765
```

> Le token est déjà généré. Ne le partagez pas.

### 2. Lancer le serveur complet

```bash
chmod +x start_synapse.sh
./start_synapse.sh
```

Ce script :
- Active le venv
- Lance `api/server.py` en arrière-plan (port 8765)
- Lance le Dream Cycle toutes les heures en arrière-plan
- Lance le tunnel Cloudflare au premier plan (affiche l'URL publique)

### 3. Lancer le serveur seul (sans tunnel)

```bash
source .venv/bin/activate
python api/server.py
```

### 4. Lancer le tunnel seul

```bash
chmod +x setup_tunnel.sh
./setup_tunnel.sh
```

> L'URL change à chaque redémarrage (version gratuite).
> Pour une URL fixe : `cloudflared login` (compte Cloudflare gratuit).

---

## Endpoints API

Tous les endpoints sauf `/health` requièrent :
```
Authorization: Bearer synapse_6d8eecef28688b0ccd2ec0fbe6bd7900
```

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Statut serveur + compteurs |
| POST | `/inbox` | Capture texte |
| POST | `/inbox/audio` | Upload audio → transcription Claude |
| POST | `/inbox/image` | Upload image → description Claude |
| GET | `/pending` | Liste des faits à valider |
| POST | `/validate/{id}` | Valider ou rejeter un fait |

### Exemples curl

```bash
BASE_URL="https://xxxxx.trycloudflare.com"
TOKEN="synapse_6d8eecef28688b0ccd2ec0fbe6bd7900"

# Health check
curl $BASE_URL/health

# Capture texte
curl -X POST $BASE_URL/inbox \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "test depuis Android", "type": "text", "source": "android"}'

# Upload audio
curl -X POST $BASE_URL/inbox/audio \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@note.m4a"

# Upload image
curl -X POST $BASE_URL/inbox/image \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@photo.jpg"

# Pending facts
curl $BASE_URL/pending \
  -H "Authorization: Bearer $TOKEN"

# Valider un fact
curl -X POST $BASE_URL/validate/FACT_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"confirmed": true}'
```

---

## Installation Android — HTTP Shortcuts

### Étape 1 : Installer l'app

Installer **HTTP Shortcuts** depuis le Play Store :
> https://play.google.com/store/apps/details?id=ch.rmy.android.http_shortcuts

### Étape 2 : Importer la config

1. Ouvrir HTTP Shortcuts
2. Menu (⋮) → **Import**
3. Sélectionner le fichier `http_shortcuts_config.json`
4. Les 4 raccourcis apparaissent dans la catégorie "Synapse"

### Étape 3 : Configurer les variables

Dans HTTP Shortcuts → **Variables** :

- `base_url` → L'URL Cloudflare affichée au démarrage (ex: `https://abc123.trycloudflare.com`)
- `api_token` → `synapse_6d8eecef28688b0ccd2ec0fbe6bd7900` (déjà pré-rempli)

### Étape 4 : Ajouter à l'écran d'accueil

1. Long press sur chaque raccourci → **Place on Home Screen**
2. Ou créer un widget HTTP Shortcuts regroupant les 4

### Étape 5 : Test end-to-end

1. Appuyer sur "Note rapide"
2. Saisir : `test depuis Android`
3. Sur le Mac : `python run_cycle.py`
4. Dans Claude Code : `get_entity("android")` ou `search_memory("test")`

---

## Flux de capture audio

1. HTTP Shortcuts enregistre l'audio via le micro
2. Envoie le fichier en multipart à `/inbox/audio`
3. `api/server.py` encode en base64 et appelle Claude Haiku
4. La transcription est stockée dans `inbox`
5. Le Dream Cycle extrait les entités et faits

---

## Sécurité

- Le token Bearer est stocké dans HTTP Shortcuts Variables (chiffré sur l'appareil)
- Le tunnel Cloudflare est en HTTPS — le trafic entre Android et Cloudflare est chiffré
- Le segment local (Cloudflare → Mac) est en HTTP sur loopback uniquement
- `/health` est public intentionnellement (monitoring, pas de données sensibles)
- Renouveler le token : modifier `API_TOKEN` dans `.env` et dans HTTP Shortcuts Variables

---

## Roadmap future

### Phase E2 — PWA offline-first
Progressive Web App installable depuis le navigateur Android.
- Capture offline avec sync au retour en ligne (Service Worker + IndexedDB)
- Interface web minimaliste pour voir la liste `inbox` et les entités
- Déployable sur Cloudflare Pages (gratuit)

### Phase E3 — App Android native (Kotlin)
- Base de données SQLite locale (Room) pour stockage offline complet
- Sync bidirectionnelle avec le Mac via l'API REST
- Widget Android pour capture rapide sans ouvrir l'app
- Enregistrement audio natif + transcription locale (Whisper.cpp)

### Phase E4 — Sync multi-device via CRDTs
- Conflict-free Replicated Data Types (CRDTs) pour fusionner les bases SQLite
- Protocole de sync peer-to-peer ou via relay minimal
- Convergence sans serveur central — chaque device est autoritaire sur ses captures
- Librairie candidate : `cr-sqlite` (extension SQLite pour CRDTs)

### Phase E5 — Offre cloud optionnelle
- Backend Synapse hébergé (Fly.io ou Railway) pour les utilisateurs sans Mac toujours allumé
- Auth forte (OAuth2 + PKCE)
- Sync chiffrée de bout en bout (clé maître côté client)
- Plan gratuit : inbox seul. Plan pro : Dream Cycle hébergé + accès API
