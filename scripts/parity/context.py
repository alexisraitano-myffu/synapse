"""SYN-171 — le contexte envoyé au modèle, figé et reproductible.

**Une seule source de vérité.** Les blocs déterministes (types actifs, auteur,
date figée, chargement du prompt, distillation de la sortie) vivent déjà dans
`scripts/lang_harness.py` depuis SYN-121. On les réimporte au lieu d'en faire
une copie : deux définitions du contexte dériveraient, et deux mesures prises
sous des contextes différents ne se comparent pas — ce qui ôterait au harnais
la seule chose qu'il apporte.

Différence assumée avec le harnais de juillet (archivé dans le doc Linear
« Benchmark — Gemma 4 E4B vs Claude Haiku ») : celui-là lisait les types et les
projets **dans la base vivante** `~/.synapse`. Le résultat dépendait donc de
l'état de la mémoire de la machine ce jour-là, et n'était pas rejouable
ailleurs. Ici le contexte est statique, comme dans `lang_harness` : ce qu'on
mesure ne dépend plus que du prompt et du modèle.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.lang_harness import (  # noqa: F401 — réexports volontaires
    _CORE_CLASSIFIER as CORE_CLASSIFIER,
    _OWNER as OWNER,
    _TODAY as TODAY,
    _distill as distill,
    _load_prompt as load_prompt,
    _static_owner_block as static_owner_block,
    _static_types_block as static_types_block,
)

# Budget de sortie du classifieur, tel que le core le fixe (`llm.rs`). Relevé de
# 1536 à 4096 en juin (SYN-77) après des classifications tronquées en silence.
CLASSIFY_MAX_TOKENS = 4096


def classifier_system(prompt_path: Path | None = None) -> list[str]:
    """Les blocs système du classifieur, dans l'ordre où le core les assemble
    (`Brain::build_classify_params`) : prompt, types actifs, auteur."""
    prompt = load_prompt(prompt_path or CORE_CLASSIFIER)
    return [prompt, static_types_block(), static_owner_block()]


# ── Mode scénario (SYN-171) ─────────────────────────────────────────────────
#
# Les deux fonctions ci-dessous n'existent QUE pour le mode scénario. Les étages
# 1 et 2 continuent d'appeler `classifier_system` : leurs baselines restent donc
# comparables à celles d'avant, et l'ajout ne les déplace pas d'un iota.
#
# Pourquoi elles existent. Mesuré le 2026-08-20 : deux règles vérifiées 100 %
# stables en appel isolé se comportent AUTREMENT dans le vrai cycle — la note
# d'anniversaire disparaît (et sa confiance passe de 0,55 à 1,0, donc plus
# d'arbitrage) et l'épisode d'une course faite disparaît aussi. La cause n'est
# pas le prompt : c'est la MÉMOIRE DE TRAVAIL (SYN-93), un bloc que la prod
# ajoute et que le harnais n'envoyait jamais. Un harnais qui ne peut pas
# reproduire la prod ne peut pas la valider.

# Importé de la prod, JAMAIS recopié : une deuxième définition dériverait, et un
# harnais qui n'envoie pas exactement ce que la prod envoie ne mesure pas la prod.
# C'est précisément ce trou qui a laissé passer le défaut de la mémoire de travail.
from dream_cycle.cycle import _WM_HEADER  # noqa: E402

# Horodatages FIGÉS : la prod les tire de l'horloge, mais une mesure qui change
# de contexte à chaque exécution ne se compare à rien.
_WM_STAMPS = ["09:05", "09:17", "09:24", "09:38", "09:41", "09:46", "09:50"]


def working_memory_block(prior: list[str], current: str) -> str:
    """Le bloc mémoire de travail, au format EXACT de `cycle.py::_build_day_context`
    (en-tête, `[horodatage · phase] texte`), horodatages figés. `prior` = les
    captures déjà consolidées du fil ; `current` = celle qu'on classe."""
    lines = [_WM_HEADER]
    for i, text in enumerate(prior):
        ts = _WM_STAMPS[min(i, len(_WM_STAMPS) - 2)]
        lines.append(f"[{TODAY} {ts} · consolidated] {' '.join(text.split())}")
    lines.append(f"[{TODAY} {_WM_STAMPS[-1]} · pending] {' '.join(current.split())}")
    return "\n".join(lines)


# Le harnais n'envoyait pas non plus le bloc projets, que la prod ajoute
# toujours. Statique ici, et volontairement NON VIDE : un contexte sans projet
# ne teste pas le rattachement, qui est justement une des décisions que le
# contexte vivant peut déplacer.
def static_projects_block() -> str:
    return (
        "[EXISTING PROJECTS — use their exact canonical_name to attach]\n"
        "- Climbing\n"
        "- Synapse"
    )


def scenario_system(current: str, prior: list[str],
                    prompt_path: Path | None = None) -> list[str]:
    """Les blocs système tels que la PRODUCTION les assemble : prompt, mémoire de
    travail, types actifs, projets, auteur (`Brain::build_classify_params`).
    Sans capture antérieure, la prod n'émet pas de bloc mémoire (`_build_day_context`
    rend None dès que le fil tient en une ligne) — on fait pareil."""
    prompt = load_prompt(prompt_path or CORE_CLASSIFIER)
    blocks = [prompt]
    if prior:
        blocks.append(working_memory_block(prior, current))
    blocks += [static_types_block(), static_projects_block(), static_owner_block()]
    return blocks


def fingerprint(blocks: list[str]) -> str:
    """Empreinte courte du contexte complet. Deux mesures ne se comparent que si
    leurs empreintes coïncident — c'est ce qui rend un verdict opposable."""
    joined = "\n\n".join(blocks).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:12]


def parse_classify(text: str, stop_reason: str | None) -> dict | None:
    """Parse local, miroir de `parse_classify_text` côté core : une troncature
    n'est jamais un JSON exploitable, et la clôture ```…``` est tolérée.
    Renvoie None si la sortie n'est pas du JSON — c'est un résultat, pas une erreur."""
    if stop_reason == "max_tokens":
        return None
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if "```" in raw:
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
    try:
        parsed = raw and __import__("json").loads(raw)
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None
