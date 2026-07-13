"""SYN-121 — harnais classification multilingue (garde-fou anti-régression).

Classifie en ISOLATION un jeu de captures FR/EN (`scripts/lang_dataset.json`) avec
un contexte 100 % déterministe (prompt + types builtin + owner figé + date figée),
extrait les signaux de routage, et sauve un snapshot JSON. Un mode `compare` diffe
deux snapshots et flagge les régressions — c'est ce qui prouve que la bascule du
prompt en base anglaise (SYN-119) ne dégrade pas la qualité FR de prod.

Pourquoi un contexte figé (conn=None) plutôt que la vraie DB : reproductibilité.
La qualité mesurée ne doit dépendre que du PROMPT, pas de l'état de ~/.synapse
(projets/types/owner du moment). On mime donc les blocs de contexte en statique.

Usage :
    python -m scripts.lang_harness run   --lang fr   --label fr-before
    python -m scripts.lang_harness run   --lang both --label baseline
    python -m scripts.lang_harness compare fr-before fr-after
    python -m scripts.lang_harness compare baseline.fr baseline.en   # EN-vs-FR parité

Nécessite ANTHROPIC_API_KEY (ou un token fuel) dans l'environnement / .env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Le script tourne comme `python -m scripts.lang_harness` depuis la racine du repo.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO / ".env")

from config import CLAUDE_MODEL  # noqa: E402
from anthropic_client import get_client  # noqa: E402

_DATASET = _REPO / "scripts" / "lang_dataset.json"
_SNAP_DIR = _REPO / "scripts" / "lang_snapshots"
_TODAY = "2026-07-13"  # figé : rend la résolution des dates relatives reproductible
_OWNER = "Alexis"

# Prompt-as-data (SYN-111) : le classifieur est un .md versionné dans synapse-core,
# lu à l'exécution par le core (substitution de {today}). Le harnais teste CE fichier
# directement — pas la wheel — pour rester utilisable sans toolchain Rust.
_CORE_CLASSIFIER = Path(
    os.getenv("SYNAPSE_CLASSIFIER_MD",
              str(_REPO.parent / "synapse-core" / "prompts" / "classifier.md")))


def _load_prompt(path: Path) -> str:
    """Charge un prompt .md et substitue {today} — miroir de ce que fait le core."""
    return path.read_text(encoding="utf-8").rstrip("\n").replace("{today}", _TODAY)


def _parse_classify_text(text: str, content_len: int, stop_reason: str | None) -> dict:
    """Parse local (le core l'implémente en Rust) : garde max_tokens, strip fence, JSON."""
    if stop_reason == "max_tokens":
        raise ValueError(f"classification tronquée (max_tokens) — {content_len} chars")
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)

# Contexte statique — mime _load_active_types_block / _load_owner_block sans DB.
_BUILTIN_TYPES = ["person", "place", "project", "concept", "organization", "animal"]


def _static_types_block() -> str:
    return (
        "[TYPES D'ENTITÉ ACTIFS — choisis EXACTEMENT l'un d'eux pour `type`]\n"
        + ", ".join(_BUILTIN_TYPES)
        + "\nAucun ne convient ? → type=\"concept\" + type_proposal "
        "{\"value\": \"<type_snake>\", \"reason\": \"...\"}."
    )


def _static_owner_block() -> str:
    return (
        f"[AUTEUR — l'utilisateur de ce second cerveau]\n"
        f"L'auteur des captures est « {_OWNER} ». Toute référence à la PREMIÈRE "
        f"PERSONNE (je, j', me, moi, mon, ma, mes / I, me, my) le désigne. Utilise "
        f"EXACTEMENT le canonical_name « {_OWNER} » pour les faits et relations le "
        f"concernant. Ne crée JAMAIS d'entité générique « auteur »/« User »/« moi »."
    )


def _classify(client, text: str, prompt: str) -> dict:
    """Un appel classify isolé, contexte déterministe. Renvoie le dict brut."""
    system = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": _static_types_block()},
        {"type": "text", "text": _static_owner_block()},
    ]
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    return _parse_classify_text(resp.content[0].text, len(text), resp.stop_reason)


def _distill(raw: dict) -> dict:
    """Réduit la sortie classifieur aux signaux de routage qu'on surveille."""
    note = raw.get("atomic_note")
    facts = []
    for ent in raw.get("entities", []) or []:
        for f in ent.get("facts", []) or []:
            facts.append({
                "entity": ent.get("canonical_name"),
                "predicate": f.get("predicate"),
                "evidence_strength": f.get("evidence_strength"),
                "persistence_value": f.get("persistence_value"),
                "category": f.get("category"),
            })
    relations = [
        {"from": r.get("from"), "predicate": r.get("predicate"),
         "to": r.get("to"), "confidence": r.get("confidence")}
        for r in raw.get("relations", []) or []
    ]
    projects = [
        {"canonical": p.get("project_canonical"), "is_new": p.get("is_new")}
        for p in raw.get("project_entries", []) or []
    ]
    return {
        "input_type": raw.get("input_type"),
        "has_note": note is not None,
        "atomic_note_kind": raw.get("atomic_note_kind") if note is not None else None,
        "atomic_note": note,
        "event_date": raw.get("event_date"),
        "event_recurring": raw.get("event_recurring"),
        "is_ephemeral": raw.get("is_ephemeral"),
        "classification_confidence": raw.get("classification_confidence"),
        "language": raw.get("language"),  # présent seulement après SYN-119
        "entities": [e.get("canonical_name") for e in raw.get("entities", []) or []],
        "entity_types": [e.get("type") for e in raw.get("entities", []) or []],
        "facts": facts,
        "relations": relations,
        "project_entries": projects,
        "summary": raw.get("summary"),
    }


def _prompt_fingerprint(prompt: str) -> str:
    import hashlib
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def cmd_run(args) -> None:
    data = json.loads(_DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    langs = ["fr", "en"] if args.lang == "both" else [args.lang]
    prompt_path = Path(args.prompt) if args.prompt else _CORE_CLASSIFIER
    prompt = _load_prompt(prompt_path)
    client = get_client()
    _SNAP_DIR.mkdir(exist_ok=True)
    fp = _prompt_fingerprint(prompt)
    print(f"prompt: {prompt_path}  (fp={fp})")

    for lang in langs:
        results = {}
        print(f"\n=== classify · lang={lang} · prompt={fp} · {len(cases)} captures ===")
        for c in cases:
            text = c[lang]
            try:
                raw = _classify(client, text, prompt)
                dist = _distill(raw)
                err = None
            except Exception as exc:  # noqa: BLE001 — on capture pour ne pas tout stopper
                dist, err = {}, f"{type(exc).__name__}: {exc}"
            entry = {"text": text, "note": c.get("note"), "expect": c.get("expect", {}),
                     "result": dist, "error": err}
            results[c["id"]] = entry
            _print_case(c["id"], text, dist, err)
        label = f"{args.label}.{lang}" if args.lang == "both" else args.label
        out = _SNAP_DIR / f"{label}.json"
        out.write_text(json.dumps(
            {"meta": {"lang": lang, "prompt_fingerprint": fp, "model": CLAUDE_MODEL,
                      "today": _TODAY, "label": label},
             "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ snapshot écrit : {out.relative_to(_REPO)}")


def _print_case(cid: str, text: str, dist: dict, err: str | None) -> None:
    if err:
        print(f"  ❌ {cid:22} {text[:40]!r:44} ERREUR {err}")
        return
    kind = dist.get("atomic_note_kind") or "—"
    flags = []
    if dist.get("has_note"):
        flags.append(f"note:{kind}")
    if dist.get("is_ephemeral"):
        flags.append("ephemeral")
    if dist.get("event_date"):
        flags.append(f"date:{dist['event_date']}")
    if dist.get("facts"):
        flags.append(f"facts:{len(dist['facts'])}")
    if dist.get("relations"):
        flags.append(f"rel:{len(dist['relations'])}")
    if dist.get("project_entries"):
        flags.append(f"proj:{len(dist['project_entries'])}")
    if dist.get("language"):
        flags.append(f"lang:{dist['language']}")
    conf = dist.get("classification_confidence")
    conf_s = f"conf:{conf}" if conf is not None else ""
    print(f"  · {cid:22} type={str(dist.get('input_type')):9} "
          f"{' '.join(flags):42} {conf_s}")


# ── compare ──────────────────────────────────────────────────────────────────

_EVIDENCE_RANK = {"explicit": 3, "hedged": 2, "implicit": 1}


def _load_snap(label: str) -> dict:
    p = _SNAP_DIR / f"{label}.json"
    if not p.exists():
        sys.exit(f"snapshot introuvable : {p.relative_to(_REPO)}")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_compare(args) -> None:
    """Diffe deux snapshots (avant/après, ou FR/EN). Flagge les régressions de routage."""
    a, b = _load_snap(args.before), _load_snap(args.after)
    print(f"\n=== compare  {args.before}  →  {args.after} ===")
    print(f"    prompt {a['meta']['prompt_fingerprint']} → {b['meta']['prompt_fingerprint']}"
          f"   (lang {a['meta']['lang']} → {b['meta']['lang']})\n")
    regressions, warnings = 0, 0
    for cid, ca in a["cases"].items():
        cb = b["cases"].get(cid)
        if cb is None:
            print(f"  ? {cid}: absent du snapshot cible")
            continue
        ra, rb = ca.get("result", {}), cb.get("result", {})
        issues = _diff_case(ra, rb)
        if not issues:
            print(f"  ✅ {cid}")
            continue
        for sev, msg in issues:
            marker = "❌" if sev == "reg" else "⚠️ "
            print(f"  {marker} {cid}: {msg}")
            if sev == "reg":
                regressions += 1
            else:
                warnings += 1
    print(f"\n=== {regressions} régression(s), {warnings} avertissement(s) ===")
    if regressions:
        sys.exit(1)


def _note_written(r: dict) -> bool:
    """Statut EFFECTIF après routage (_process_entry cycle.py) : une note kind=note
    est écrite seulement si non-éphémère ; une task/event est durable (toujours écrite).
    C'est ce qui compte — pas le simple has_note de la sortie brute du classifieur."""
    eph = bool(r.get("is_ephemeral")) or r.get("input_type") == "ephemeral"
    return bool(r.get("has_note")) and (
        not eph or r.get("atomic_note_kind") in ("task", "event"))


def _diff_case(a: dict, b: dict) -> list[tuple[str, str]]:
    """Renvoie [(severity, message)]. severity ∈ {reg, warn}. Vide = OK."""
    issues: list[tuple[str, str]] = []

    # Perte d'une note EFFECTIVEMENT écrite = la régression Haiku historique
    # (inclut le cas « kind=note marqué is_ephemeral=true » que le routage drop).
    if _note_written(a) and not _note_written(b):
        issues.append(("reg", f"note effective perdue (kind={a.get('atomic_note_kind')}, "
                              f"is_ephemeral={b.get('is_ephemeral')})"))
    if a.get("atomic_note_kind") != b.get("atomic_note_kind") and a.get("has_note"):
        issues.append(("warn",
                       f"kind {a.get('atomic_note_kind')} → {b.get('atomic_note_kind')}"))

    # Une action durable qui bascule en pur éphémère sans note = drop.
    if (not a.get("is_ephemeral")) and b.get("is_ephemeral") and not b.get("has_note"):
        issues.append(("reg", "bascule en ephemeral sans note (action perdue)"))

    if a.get("input_type") != b.get("input_type"):
        issues.append(("warn",
                       f"input_type {a.get('input_type')} → {b.get('input_type')}"))

    # Perte d'échéance / de date d'événement.
    if a.get("event_date") and not b.get("event_date"):
        issues.append(("warn", f"event_date perdue ({a.get('event_date')})"))

    # Downgrade d'evidence_strength (explicit → hedged → implicit) sur un prédicat.
    ea = {(f["entity"], f["predicate"]): f.get("evidence_strength") for f in a.get("facts", [])}
    for f in b.get("facts", []):
        k = (f["entity"], f["predicate"])
        if k in ea:
            ra, rb = _EVIDENCE_RANK.get(ea[k], 0), _EVIDENCE_RANK.get(f.get("evidence_strength"), 0)
            if rb < ra:
                issues.append(("warn",
                               f"evidence {k[1]} {ea[k]} → {f.get('evidence_strength')}"))

    # Chute du nombre de faits / relations / projets extraits.
    for field, lbl in [("facts", "faits"), ("relations", "relations"),
                       ("project_entries", "projets")]:
        na, nb = len(a.get(field, [])), len(b.get(field, []))
        if nb < na:
            issues.append(("warn", f"{lbl} {na} → {nb}"))

    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="classifie le dataset et sauve un snapshot")
    r.add_argument("--lang", choices=["fr", "en", "both"], default="both")
    r.add_argument("--label", required=True, help="nom du snapshot (ex: fr-before)")
    r.add_argument("--prompt", default=None,
                   help="chemin du classifier.md à tester (défaut: synapse-core/prompts/classifier.md)")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("compare", help="diffe deux snapshots")
    c.add_argument("before")
    c.add_argument("after")
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
