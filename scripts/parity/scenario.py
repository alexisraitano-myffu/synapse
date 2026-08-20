"""SYN-171 mode scénario — mesurer ce que le CONTEXTE fait à la décision.

Les étages 1 et 2 classent une capture dans le vide et rendent un verdict sur le
prompt. La production, elle, ajoute la mémoire de travail (SYN-93) : le fil des
captures récentes. Le 2026-08-20, deux règles vérifiées 100 % stables en appel
isolé se sont révélées instables dans le vrai cycle — et à chaque fois la note
disparaissait. Un harnais incapable de reproduire la prod ne peut pas la valider.

Ce que ce mode mesure n'est donc PAS la justesse mais la STABILITÉ : chaque
scénario est rejoué `repeat` fois et on compte les branches obtenues. Une règle
qui sort 3 fois sur 5 n'est pas une règle, quel que soit son score sur 58 cas.

Chaque scénario embarque son témoin : le même texte sans fil, et le même texte
avec un fil sans rapport. Sans eux, on attribuerait au contexte ce qui pourrait
n'être qu'une capture difficile.

Usage :
    python -m scripts.parity.scenario anthropic:claude-haiku-4-5-20251001
    python -m scripts.parity.scenario ollama:qwen2.5:3b-instruct-q4_K_M --repeat 3
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from scripts.parity.context import (
    CLASSIFY_MAX_TOKENS,
    fingerprint,
    parse_classify,
    scenario_system,
)
from scripts.parity.corpus import SCENARIO_CASES
from scripts.parity.providers import call


def _branch(parsed: dict | None) -> tuple:
    """La branche prise, réduite à ce qui change l'issue pour l'utilisateur :
    y a-t-il une note, de quel type, et l'incertitude a-t-elle survécu (une
    confiance ≥ 0.7 fait sauter la file « À valider »)."""
    if not parsed:
        return ("illisible", None, None)
    note = parsed.get("atomic_note")
    has_note = bool(note) and str(note).strip().lower() not in ("", "null", "none")
    raw = parsed.get("atomic_note_kind")
    kind = (raw if isinstance(raw, str) and raw else "note") if has_note else None
    conf = parsed.get("classification_confidence")
    return (has_note, kind, conf)


def _matches(branch: tuple, expect: dict) -> bool:
    has_note, kind, conf = branch
    if has_note == "illisible":
        return False
    if "note" in expect and has_note != expect["note"]:
        return False
    if expect.get("kind") and kind != expect["kind"]:
        return False
    if "confidence_below" in expect:
        # L'incertitude EST le livrable ici : sans elle, rien n'atteint la file
        # de validation et la question est répondue toute seule.
        if not isinstance(conf, (int, float)) or conf >= expect["confidence_below"]:
            return False
    return True


def run(model: str, repeat_override: int | None, temperature: float) -> int:
    print(f"modèle   : {model}")
    print("mesure   : stabilité de la branche sous contexte, pas justesse")
    worst = 0
    for case in SCENARIO_CASES:
        wm = case.get("wm") or []
        blocks = scenario_system(case["text"], wm)
        repeat = repeat_override or case.get("repeat", 5)
        label = f"{len(wm)} capture(s) au fil" if wm else "capture seule"
        print(f"\n  {case['id']}  ({label}, empreinte {fingerprint(blocks)})")
        print(f"    {case['text']!r}")

        branches = Counter()
        for _ in range(repeat):
            reply = call(model, blocks, case["text"], CLASSIFY_MAX_TOKENS,
                         temperature=temperature)
            branches[_branch(parse_classify(reply.text, reply.stop_reason))] += 1

        ok = sum(n for b, n in branches.items() if _matches(b, case["expect"]))
        rate = ok / repeat
        worst = max(worst, repeat - ok)
        for branch, n in branches.most_common():
            has_note, kind, conf = branch
            desc = ("illisible" if has_note == "illisible"
                    else f"note={'oui' if has_note else 'NON'} kind={kind} conf={conf}")
            print(f"      {n}/{repeat}  {'✓' if _matches(branch, case['expect']) else '✗'}  {desc}")
        verdict = "STABLE" if rate == 1 else ("INSTABLE" if ok else "TOUJOURS FAUX")
        print(f"    → {verdict} ({ok}/{repeat} sur la branche attendue) — {case['why']}")

    print()
    if worst == 0:
        print("VERDICT : chaque scénario tient sa branche sous contexte.")
        return 0
    print(f"VERDICT : au pire {worst} passe(s) hors branche. Le contexte déplace")
    print("          des décisions que le prompt tranche pourtant seul —")
    print("          ce n'est pas un défaut de modèle, c'est un défaut de contexte.")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SYN-171 — parité sous contexte (mode scénario)")
    ap.add_argument("model", help="provider:modèle")
    ap.add_argument("--repeat", type=int, default=None,
                    help="force le nombre de passes (défaut : celui du scénario)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = reproductible (défaut). Le défaut de contexte se voit À 0.")
    args = ap.parse_args(argv)
    return run(args.model, args.repeat, args.temperature)


if __name__ == "__main__":
    sys.exit(main())
