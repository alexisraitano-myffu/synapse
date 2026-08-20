"""SYN-171 — le schéma de sortie du classifieur, pour le décodage contraint.

Pourquoi ce fichier existe. Mesuré le 2026-08-19 : Gemma E2B passait 10 des 12
cas du gate et échouait les DEUX cas « action à faire », toujours de la même
façon — il écrivait `input_type="task"`, une valeur qui appartenait à
`atomic_note_kind`. Le contenu, lui, était juste : la note était là, son `kind`
correct, l'entité extraite. Seul le nom du champ dérapait.

Le prompt avertissait DÉJÀ de ce piège, nommément et en majuscules. Un
avertissement plus long n'avait aucune raison de faire mieux : on ne persuade pas
un modèle de 2 milliards de paramètres, on contraint son décodage. Ollama accepte
un JSON Schema dans `format` et n'échantillonne alors que des continuations
valides, ce qui rend cette classe d'erreur **impossible par construction**.

`input_type` a été retiré le 2026-08-20 — il ne pilotait rien. Ce mode de
défaillance précis a donc disparu avec le champ qui le rendait possible : on ne
confond plus deux champs quand il n'en reste qu'un. Le schéma contraint garde tout
son intérêt pour les autres dérapages de forme.

⚠️ Une mesure sous contrainte ne dit pas la même chose qu'une mesure libre : elle
mesure la justesse du modèle, plus sa capacité à respecter un format. Les deux
comptent, mais séparément — d'où le drapeau `--schema` plutôt qu'un défaut.

Le schéma reproduit la forme déclarée en tête de `classifier.md`. Le garder
synchrone avec elle fait partie du contrat de ce harnais.
"""
from __future__ import annotations

_FACT = {
    "type": "object",
    "properties": {
        "predicate": {"type": "string"},
        "value": {"type": "string"},
        "persistence_value": {"type": "integer"},
        "evidence_strength": {"type": "string", "enum": ["explicit", "hedged", "implicit"]},
        "category": {"type": "string"},
    },
    "required": ["predicate", "value", "persistence_value", "evidence_strength", "category"],
}

_TYPE_PROPOSAL = {
    "type": ["object", "null"],
    "properties": {"value": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["value", "reason"],
}

_ENTITY = {
    "type": "object",
    "properties": {
        "canonical_name": {"type": "string"},
        "type": {"type": "string"},
        "type_proposal": _TYPE_PROPOSAL,
        "aliases": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": ["string", "null"]},
        "attributes": {"type": "object"},
        "facts": {"type": "array", "items": _FACT},
    },
    # ⚠️ Même leçon qu'au niveau racine ci-dessous, et elle n'y avait PAS été
    # appliquée : mesuré le 2026-08-20, E2B contraint a produit ZÉRO fait sur les
    # 59 cas (Haiku en produit sur 18) — non pas parce qu'il ne sait pas en
    # extraire, mais parce que `facts` était facultatif dans l'entité et que le
    # décodage contraint prend toujours la sortie la moins chère. Un `summary`
    # laissé sans type est ressorti en objet `{"value": …}`, et `type_proposal`
    # rempli là où le type était déjà actif. Une exigence partielle ne mesure pas
    # le modèle : elle mesure ce qu'on l'a autorisé à ne pas faire.
    "required": ["canonical_name", "type", "type_proposal", "aliases",
                 "summary", "attributes", "facts"],
}

_RELATION = {
    "type": "object",
    "properties": {
        "from": {"type": "string"},
        "predicate": {"type": "string"},
        "to": {"type": "string"},
        "confidence": {"type": "number"},
    },
    # `confidence` requis : c'est lui qui porte l'arbitrage « déduction à 0,6 vs
    # énoncé à 1,0 ». Facultatif, le modèle l'omet et la règle devient invisible.
    "required": ["from", "predicate", "to", "confidence"],
}

_PROJECT_ENTRY = {
    "type": "object",
    "properties": {
        "project_canonical": {"type": "string"},
        "content": {"type": "string"},
        "is_new": {"type": "boolean"},
    },
    "required": ["project_canonical", "content", "is_new"],
}

# Les deux énumérations que le prompt déclare fermées. Ce sont elles qui portent
# tout l'intérêt du décodage contraint : le reste du schéma n'est là que pour que
# la contrainte reste cohérente avec la forme attendue.
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "atomic_note": {"type": ["string", "null"]},
        # Toujours une des trois valeurs, jamais null. Le core l'ignore de toute
        # façon quand `atomic_note` est vide (`routing.rs:196` ne l'utilise que
        # pour une note non vide), donc l'exiger ne coûte rien et ferme la porte
        # au null qui, autorisé, dégradait une tâche en note.
        "atomic_note_kind": {"type": "string", "enum": ["note", "task", "event", "episode"]},
        "event_date": {"type": ["string", "null"]},
        "event_recurring": {"type": "boolean"},
        "is_ephemeral": {"type": "boolean"},
        "classification_confidence": {"type": "number"},
        "project_entries": {"type": "array", "items": _PROJECT_ENTRY},
        "entities": {"type": "array", "items": _ENTITY},
        "relations": {"type": "array", "items": _RELATION},
        "summary": {"type": ["string", "null"]},
    },
    # ⚠️ TOUS les champs déclarés sont requis, et ce n'est pas du zèle.
    # Mesuré le 2026-08-19 : avec seulement trois champs requis, Qwen contraint
    # n'en émettait plus que 4 à 5 sur 13 — il omettait `atomic_note_kind`,
    # `is_ephemeral`, `event_date`, `relations`… Un schéma permissif ne se contente
    # pas de ne rien garantir : il AUTORISE le modèle à moins produire, et les
    # absences se lisaient ensuite comme des erreurs de jugement. Un schéma qui
    # n'exige pas la forme complète mesure autre chose que ce qu'on croit.
    "required": [
        "language", "atomic_note", "atomic_note_kind", "event_date",
        "event_recurring", "is_ephemeral", "classification_confidence",
        "project_entries", "entities", "relations", "summary",
    ],
}
