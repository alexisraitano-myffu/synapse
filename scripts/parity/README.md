# Harnais de parité modèles (SYN-171)

Valider un modèle candidat **avant** de l'intégrer, et pouvoir rejouer la mesure
d'une commande. Trois fois de suite — Gemma E4B, le `.litertlm` mobile, E2B — la
décision s'est prise sur un harnais jeté après usage. Celui-ci est versionné.

## Deux étages

| | Ce que ça répond | Coût |
| -- | -- | -- |
| **Étage 1 — `gate`** | Le modèle est-il *utilisable* ? | ~12 appels |
| **Étage 2 — `full`** | Le modèle est-il *bon* ? | tout le corpus × 6 prompts |

L'étage 1 ne mesure pas la qualité. Il cherche quatre vices rédhibitoires et
s'arrête au premier trouvé, parce qu'aucun n'est rattrapable par l'intelligence
du modèle :

1. **Avale-t-il le prompt ?** Le classifieur fait ~4 700 tokens. Une fenêtre trop
   courte le tronque *en silence* — le modèle paraît stupide alors qu'il n'a
   jamais reçu les règles. C'est ce qui disqualifie Gemini Nano (2 048 à 4 096
   tokens de fenêtre totale).
2. **Rend-il du JSON exploitable ?** Valide, non tronqué. Un modèle qui casse le
   parsing casse le pipeline, quelle que soit sa justesse.
3. **Respecte-t-il l'énumération fermée** d'`input_type` ?
4. **Ne perd-il rien ?** Une capture marquée `drop_guard` doit laisser une trace
   **durable** : note, entrée projet, fait ou relation. Une intention éphémère ne
   compte pas — elle expire en 48 h, et c'est précisément le mode d'échec
   historique (« Répondre à l'e-mail de Vincent » classé éphémère puis perdu).

## Usage

```bash
ollama serve &                                   # pour un modèle local
python -m scripts.parity.gate anthropic:claude-haiku-4-5-20251001    # la référence
python -m scripts.parity.gate ollama:qwen2.5:3b-instruct-q4_K_M
python -m scripts.parity.gate ollama:llama3.2:3b --json out.json
python -m scripts.parity.gate ollama:qwen2.5:3b --prompt /chemin/classifier-compact.md
```

Le code de sortie vaut 1 en cas de NO-GO : utilisable en CI.

## Ce qui rend une mesure opposable

* **Contexte figé.** Types d'entité builtin, auteur figé, `today=2026-07-13`. Le
  harnais de juillet lisait les types et projets dans la base vivante
  `~/.synapse` : son résultat dépendait de l'état de la mémoire ce jour-là et
  n'était pas rejouable ailleurs. Ici, seuls le prompt et le modèle varient.
* **Empreinte de contexte.** Chaque exécution imprime une empreinte SHA-256
  courte des blocs système. Deux mesures ne se comparent que si leurs empreintes
  coïncident — sinon on compare deux énoncés différents.
* **Le prompt réel.** Les blocs sont assemblés dans l'ordre du core
  (`Brain::build_classify_params`), avec le `cache_control` sur le premier bloc,
  comme en production.

## Pièges déjà payés

* **`usage.input_tokens` d'Anthropic exclut le cache.** Au deuxième appel, le
  classifieur bascule dans `cache_read_input_tokens` et `input_tokens` retombe à
  ~200. Le gate a rendu un faux NO-GO là-dessus à sa première exécution. On somme
  les trois compteurs.
* **`num_ctx` d'Ollama vaut 2048 par défaut** selon les modèles : il tronque le
  début du prompt sans rien dire. On le fixe explicitement (8192) *et* on relit
  `prompt_eval_count` pour vérifier ce que le modèle a réellement reçu.
* **Ne pas conclure sur la latence depuis une machine 8 Go.** SYN-124 a mesuré
  76 s/capture dominées par 6,2 Go de swap. La justesse, elle, se mesure sans
  réserve.
* **Un modèle à raisonnement** consomme son budget de sortie en `thinking` avant
  de répondre : on regarde `stop_reason`, pas la seule présence de texte.

## Corpus

`corpus.py`, **100 % synthétique** — ce dépôt est public. Les captures réelles
restent accessibles en local, jamais versionnées.

* `GATE_CASES` — 12 cas, un par mode de défaillance.
* `HARD_CASES` — les 29 cas durs de SYN-124, portés depuis le document Linear où
  ils ne survivaient que recopiés.
* `ATOMICITY_CASES` — la règle SYN-98 (extraction **par information**), qu'aucun
  test ne couvrait.
* `AMBIGUOUS` — les cas que le prompt lui-même ne tranche pas. Observés, exclus
  du décompte d'échec. Aujourd'hui : `e4`, raté par tous les modèles mesurés,
  Haiku compris.

Les labels dérivent strictement de `classifier.md`. Un cas que le prompt ne
tranche pas n'est pas un échec du modèle : c'est un défaut du prompt.
