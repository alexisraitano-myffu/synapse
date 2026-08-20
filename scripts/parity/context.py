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
