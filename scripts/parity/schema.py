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
    "required": ["predicate", "value"],
}

_ENTITY = {
    "type": "object",
    "properties": {
        "canonical_name": {"type": "string"},
        "type": {"type": "string"},
        "type_proposal": {},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "summary": {},
        "attributes": {"type": "object"},
        "facts": {"type": "array", "items": _FACT},
    },
    "required": ["canonical_name", "type"],
}

_RELATION = {
    "type": "object",
    "properties": {
        "from": {"type": "string"},
        "predicate": {"type": "string"},
        "to": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["from", "predicate", "to"],
}

_PROJECT_ENTRY = {
    "type": "object",
    "properties": {
        "project_canonical": {"type": "string"},
        "content": {"type": "string"},
        "is_new": {"type": "boolean"},
    },
    "required": ["project_canonical", "content"],
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
