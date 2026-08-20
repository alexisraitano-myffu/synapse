# Harnais de parité modèles (SYN-171)

Valider un modèle candidat **avant** de l'intégrer, et pouvoir rejouer la mesure
d'une commande. Trois fois de suite — Gemma E4B, le `.litertlm` mobile, E2B — la
décision s'est prise sur un harnais jeté après usage. Celui-ci est versionné.

## Trois étages

| | Ce que ça répond | Coût |
| -- | -- | -- |
| **Étage 1 — `gate`** | Le modèle est-il *utilisable* ? | ~12 appels |
| **Étage 2 — `baseline`** | Le modèle est-il *bon* ? | 58 cas |
| **Étage 3 — `scenario`** | La règle tient-elle *en contexte* ? | 5 scénarios × 5 passes |

Les deux premiers mesurent une capture seule. Le troisième mesure ce que le
CONTEXTE fait à la décision — et c'est là qu'on a trouvé des règles vertes aux
deux premiers étages et fausses en production. Voir « Étage 3 » plus bas.

L'étage 1 ne mesure pas la qualité. Il cherche quatre vices rédhibitoires et
s'arrête au premier trouvé, parce qu'aucun n'est rattrapable par l'intelligence
du modèle :

1. **Avale-t-il le prompt ?** Le classifieur fait ~4 700 tokens. Une fenêtre trop
   courte le tronque *en silence* — le modèle paraît stupide alors qu'il n'a
   jamais reçu les règles. C'est ce qui disqualifie Gemini Nano (2 048 à 4 096
   tokens de fenêtre totale).
2. **Rend-il du JSON exploitable ?** Valide, non tronqué. Un modèle qui casse le
   parsing casse le pipeline, quelle que soit sa justesse.
3. **Respecte-t-il l'énumération fermée** d'`atomic_note_kind`
   (`note|task|event|episode`) ? C'est ce champ qui décide du stockage, de la
   décroissance et de l'affichage : une valeur inventée fait dégrader la note en
   « note » par le core, donc perd une tâche en silence. (Portait sur `input_type`
   jusqu'au 20/08 — ce champ ne pilotait rien et a été retiré.)
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
* **Un `diff` non nul ne prouve rien tant qu'on n'a pas mesuré le plancher.**
  Deux passes du MÊME prompt à température 0 divergent encore : 2 cas sur 58 au
  20/08/2026 sur Haiku, uniquement sur les compteurs `facts`/`relations`, jamais
  sur une branche de routage. Avant de lire un écart de comptage comme un effet
  du prompt, rejouer la baseline contre elle-même. Un basculement de branche,
  lui, n'est jamais du bruit.
* **Une règle en prose a des effets de second ordre.** Le 20/08, trois écritures
  successives de la même règle ont chacune cassé une règle voisine — la note
  d'atomicité, puis la déduction, puis la tâche encore due absorbée par
  l'épisode. Aucune n'était fausse ; toutes déplaçaient un a priori. D'où la
  boucle : écrire, mesurer les 58 cas, lire le diff cas par cas.

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

## Étage 3 — le mode scénario

`python -m scripts.parity.scenario <modèle>`

Les étages 1 et 2 classent une capture **dans le vide**. La production, elle,
ajoute la mémoire de travail (SYN-93) : le fil des captures récentes, précédé de
« ⚠ n'extrais rien de ce bloc ». Cette consigne est respectée — rien n'est extrait
du bloc — et pourtant **le bloc déplace la décision prise sur la capture
courante**.

Découvert le 2026-08-20 en installant Synapse sur le Mac de dev : deux règles
mesurées 100 % stables en appel isolé se comportaient autrement dans le vrai
cycle, et à chaque fois la note disparaissait — avec une confiance qui montait à
1,0, donc sans même atteindre « À valider ». Les deux défaillances se cumulent au
pire endroit : la note est perdue ET la perte est marquée certaine.

Ce que ce mode mesure n'est pas la justesse mais la **stabilité** : chaque
scénario est rejoué `repeat` fois et on compte les branches obtenues. Une règle
qui sort 3 fois sur 5 n'est pas une règle, quel que soit son score sur 58 cas.

**Chaque scénario embarque ses témoins** — le même texte sans fil, et le même
texte avec un fil sans rapport. Sans eux on attribuerait au contexte ce qui
pourrait n'être qu'une capture difficile. C'est le témoin qui a permis d'établir
qu'une **seule ligne** du fil suffisait, et laquelle.

L'en-tête du bloc est **importé de `dream_cycle/cycle.py`**, jamais recopié : un
harnais qui n'envoie pas exactement ce que la prod envoie ne mesure pas la prod —
et c'est précisément ce trou qui a laissé passer ce défaut pendant des semaines.
