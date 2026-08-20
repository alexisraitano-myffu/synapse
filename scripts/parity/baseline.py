"""SYN-171 — figer une baseline, puis diffé deux baselines.

Le gate dit « utilisable / inutilisable ». Ce module-ci ne juge rien : il
enregistre ce qu'un modèle répond sur TOUT le corpus, avec l'empreinte du
contexte, pour qu'on puisse changer le prompt et prouver ce qui a bougé.

C'est l'outil qui manquait aux trois décisions de modèle prises depuis juillet :
à chaque fois on a mesuré l'après sans avoir gardé l'avant.

    python -m scripts.parity.baseline run anthropic:claude-haiku-4-5-20251001 \
        --label avant-arbitrages
    python -m scripts.parity.baseline diff avant-arbitrages apres-arbitrages
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.parity import context, providers  # noqa: E402
from scripts.parity.corpus import (  # noqa: E402
    ADVERSARIAL_CASES, ATOMICITY_CASES, GATE_CASES, HARD_CASES,
)
from scripts.parity.paths import path_of  # noqa: E402
from scripts.parity.schema import CLASSIFY_SCHEMA  # noqa: E402

SNAP_DIR = _REPO / "scripts" / "parity" / "baselines"

SETS = {
    "gate": GATE_CASES,
    "hard": HARD_CASES,
    "atomicity": ATOMICITY_CASES,
    "adversarial": ADVERSARIAL_CASES,
}

# Les axes qu'on diffe. Volontairement les signaux de ROUTAGE seulement : le
# texte d'une note varie d'une exécution à l'autre sans que ce soit une
# régression, alors qu'une branche qui bascule en est toujours une (ou une
# correction, et c'est justement ce qu'on veut voir).
DIFFED = ("has_note", "kind", "ephemeral", "facts", "relations", "projects")


def cmd_run(args) -> int:
    system = context.classifier_system(Path(args.prompt) if args.prompt else None)
    fp = context.fingerprint(system)
    schema = CLASSIFY_SCHEMA if args.schema else None
    sets = args.sets.split(",") if args.sets else list(SETS)

    print(f"modèle   : {args.model}")
    print(f"contexte : {sum(len(b) for b in system)} caractères · empreinte {fp}")
    print(f"corpus   : {', '.join(sets)}\n")

    out: dict = {"model": args.model, "fingerprint": fp, "label": args.label,
                 "schema_constrained": bool(schema),
                 "temperature": args.temperature, "cases": {}}
    for set_name in sets:
        for case in SETS[set_name]:
            reply = providers.call(args.model, system, case["text"],
                                   context.CLASSIFY_MAX_TOKENS, args.num_ctx, schema,
                                   args.temperature)
            parsed = context.parse_classify(reply.text, reply.stop_reason)
            rec = path_of(parsed)
            rec.update(set=set_name, text=case["text"], latency_s=reply.latency_s,
                       confidence=(parsed or {}).get("classification_confidence"),
                       prompt_tokens=reply.prompt_tokens)
            out["cases"][case["id"]] = rec
            mark = "·" if rec.get("parsed") else "✗"
            print(f"  {mark} {case['id']:22} "
                  f"note={str(rec.get('has_note')):5} kind={str(rec.get('kind')):6} "
                  f"f={rec.get('facts')} r={rec.get('relations')}", flush=True)

    # Combien de la fenêtre le prompt a mangé. Le gate vérifie déjà qu'il ENTRE
    # (`prompt_tokens < num_ctx`) — mais entrer ne suffit pas : il faut aussi
    # qu'il reste de la place pour ÉCRIRE. Avec `--context-shift`, un prompt qui
    # occupe presque toute la fenêtre fait évincer ses propres premières lignes
    # pendant la génération : le modèle répond sans les règles qu'on lui a
    # données, et la mesure impute au modèle un défaut de fenêtre.
    seen = [r["prompt_tokens"] for r in out["cases"].values() if r.get("prompt_tokens")]
    if seen:
        worst = max(seen)
        libre = args.num_ctx - worst
        alerte = "  ⚠ SOUS le budget de sortie" if libre < context.CLASSIFY_MAX_TOKENS else ""
        print(f"\nfenêtre  : prompt ≤ {worst} tokens sur num_ctx={args.num_ctx} "
              f"→ {libre} libres pour écrire (budget {context.CLASSIFY_MAX_TOKENS}){alerte}")
        out["prompt_tokens_max"] = worst

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAP_DIR / f"{args.label}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ baseline écrite : {path.relative_to(_REPO)}  ({len(out['cases'])} cas)")
    return 0


def cmd_diff(args) -> int:
    a = json.loads((SNAP_DIR / f"{args.before}.json").read_text(encoding="utf-8"))
    b = json.loads((SNAP_DIR / f"{args.after}.json").read_text(encoding="utf-8"))

    if a["fingerprint"] != b["fingerprint"]:
        print(f"⚠ empreintes de contexte différentes : {a['fingerprint']} → "
              f"{b['fingerprint']} — c'est attendu si le prompt a changé, "
              f"suspect sinon.")
    if a["model"] != b["model"]:
        print(f"⚠ modèles différents : {a['model']} → {b['model']} — le diff mélange "
              f"alors l'effet du prompt et celui du modèle.")

    ids = sorted(set(a["cases"]) | set(b["cases"]))
    changed = 0
    for cid in ids:
        ca, cb = a["cases"].get(cid), b["cases"].get(cid)
        if ca is None:
            print(f"  + {cid:22} cas neuf")
            continue
        if cb is None:
            print(f"  − {cid:22} cas disparu")
            continue
        deltas = [f"{k}: {ca.get(k)!r} → {cb.get(k)!r}"
                  for k in DIFFED if ca.get(k) != cb.get(k)]
        if deltas:
            changed += 1
            print(f"  ~ {cid:22} ({ca.get('set')})")
            for d in deltas:
                print(f"      {d}")
    total = len(ids)
    print(f"\n{changed} cas modifiés sur {total}. "
          f"{'Aucune bascule de branche.' if not changed else ''}")
    print("Un changement n'est pas une régression : c'est à toi de dire, cas par "
          "cas, si la nouvelle branche est celle que tu voulais.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="SYN-171 — baselines de parité")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="jouer le corpus et figer une baseline")
    r.add_argument("model")
    r.add_argument("--label", required=True)
    r.add_argument("--prompt")
    r.add_argument("--sets", help=f"sous-ensembles séparés par des virgules ({', '.join(SETS)})")
    r.add_argument("--schema", action="store_true")
    r.add_argument("--num-ctx", type=int, default=providers.DEFAULT_NUM_CTX)
    r.add_argument("--temperature", type=float, default=0.0)
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("diff", help="comparer deux baselines")
    d.add_argument("before")
    d.add_argument("after")
    d.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
