"""SYN-171 — étage 5 : le classifieur en DEUX appels, mesuré comme un seul.

Pourquoi. Mesuré le 2026-08-20 sur les 59 cas, tableau 2×2 complet (prompt v14/v23 ×
schéma contraint/libre) : E2B émet 33-34 notes sous le prompt v14 et 22 sous le v23,
identiquement avec et sans schéma. Le schéma est disculpé, l'abondance de faits aussi
(v14 sans schéma produit 56 faits ET garde 33 notes). C'est le prompt seul — et
précisément les répétitions retirées à la compaction, dont « la note n'est jamais
absorbée », martelée à quatre endroits.

L'hypothèse testée ici : ces répétitions ne sont nécessaires que parce que les sorties
se CONCURRENCENT dans un appel unique. Séparées en deux appels, l'invariant n'est plus
une consigne — l'appel graphe n'a pas de champ `atomic_note`, il ne peut pas le mettre
à null. La règle disparaît du prompt au lieu d'être répétée.

Les deux appels sont volontairement INDÉPENDANTS : l'extracteur ne reçoit pas la
décision du routeur. C'est le test pur. Un enchaînement rendrait de la cohérence mais
rétablirait le couplage qu'on cherche justement à supprimer — et on ne pourrait plus
dire si le gain vient de la séparation ou de l'ordre.

    python -m scripts.parity.split run <modèle> --label deux-appels [--schema]

La sortie est FUSIONNÉE au format d'un appel unique, pour que `path_of`, les baselines
et `baseline diff` marchent sans la moindre modification : un résultat qui ne se compare
pas aux mesures d'hier ne vaut rien.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.parity import context, providers  # noqa: E402
from scripts.parity.baseline import SETS, SNAP_DIR  # noqa: E402
from scripts.parity.paths import path_of  # noqa: E402
from scripts.parity.schema import CLASSIFY_SCHEMA  # noqa: E402

# Les deux moitiés du prompt (`note.md` et `graph.md`), fournies par
# SYNAPSE_SPLIT_PROMPTS_DIR.
#
# Elles vivent volontairement HORS du repo tant que l'expérience n'est pas
# tranchée : les prompts de production sont dans synapse-core, et y déposer une
# variante non validée ferait deux sources de vérité — exactement ce que SYN-111
# a fermé. Le jour où le découpage est adopté, elles rejoignent synapse-core avec
# le reste et cette variable disparaît.
SPLIT_DIR = Path(os.environ.get("SYNAPSE_SPLIT_PROMPTS_DIR", ""))


def _require_split_dir() -> Path:
    if not SPLIT_DIR.name or not (SPLIT_DIR / "note.md").is_file():
        raise SystemExit(
            "SYNAPSE_SPLIT_PROMPTS_DIR doit pointer vers un dossier contenant "
            "note.md et graph.md (les deux moitiés du classifieur). Elles ne sont "
            "pas versionnées ici : voir SYN-171."
        )
    return SPLIT_DIR

# Découpe du schéma racine par appartenance, pas par recopie : les deux sous-schémas
# se dérivent de CLASSIFY_SCHEMA, donc ils ne peuvent pas dériver de lui.
_NOTE_FIELDS = ("language", "atomic_note", "atomic_note_kind", "event_date",
                "event_recurring", "is_ephemeral", "classification_confidence", "summary")
_GRAPH_FIELDS = ("language", "entities", "relations", "project_entries")


def _subschema(fields: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {k: CLASSIFY_SCHEMA["properties"][k] for k in fields},
        "required": list(fields),
    }


NOTE_SCHEMA = _subschema(_NOTE_FIELDS)
GRAPH_SCHEMA = _subschema(_GRAPH_FIELDS)


def _system(prompt_file: str) -> list[str]:
    """Prompt de la moitié + l'échafaudage, dans l'ordre où le core l'assemble.
    L'échafaudage est le MÊME pour les deux appels : c'est ce qui rend le surcoût
    d'entrée additif et non multiplicatif (mesuré : ~+10 %, pas ×2)."""
    prompt = context.load_prompt(_require_split_dir() / prompt_file)
    return [prompt, context.static_types_block(), context.static_owner_block()]


def classify_split(model: str, text: str, schema: bool, temperature: float) -> tuple[dict | None, dict]:
    """Les deux appels, fusionnés au format d'un appel unique.

    Retourne (fusion, diag). `fusion` vaut None si AUCUNE des deux moitiés n'a produit
    de JSON — une seule moitié perdue laisse l'autre exploitable, ce qui est justement
    une propriété du découpage qu'on veut voir dans les chiffres, pas masquer.
    """
    a = providers.call(model, _system("note.md"), text, context.CLASSIFY_MAX_TOKENS,
                       providers.DEFAULT_NUM_CTX, NOTE_SCHEMA if schema else None, temperature)
    b = providers.call(model, _system("graph.md"), text, context.CLASSIFY_MAX_TOKENS,
                       providers.DEFAULT_NUM_CTX, GRAPH_SCHEMA if schema else None, temperature)
    note = context.parse_classify(a.text, a.stop_reason)
    graph = context.parse_classify(b.text, b.stop_reason)
    diag = {"note_parsed": note is not None, "graph_parsed": graph is not None,
            "latency_s": round(a.latency_s + b.latency_s, 2),
            "prompt_tokens": max(a.prompt_tokens or 0, b.prompt_tokens or 0)}
    if note is None and graph is None:
        return None, diag
    merged: dict = {}
    merged.update(note or {})
    # Le graphe ne peut pas écraser la note : il n'a aucune clé en commun avec elle
    # hormis `language`, et la moitié note fait autorité dessus (elle a lu le texte
    # pour l'écrire dedans).
    for k in ("entities", "relations", "project_entries"):
        merged[k] = (graph or {}).get(k) or []
    merged.setdefault("language", (graph or {}).get("language"))
    return merged, diag


def cmd_run(args) -> int:
    schema = bool(args.schema)
    sets = args.sets.split(",") if args.sets else list(SETS)
    fp_note = context.fingerprint(_system("note.md"))
    fp_graph = context.fingerprint(_system("graph.md"))
    print(f"modèle   : {args.model}")
    print(f"appel 1  : {sum(len(b) for b in _system('note.md'))} car · empreinte {fp_note}")
    print(f"appel 2  : {sum(len(b) for b in _system('graph.md'))} car · empreinte {fp_graph}")
    print(f"décodage : {'schéma contraint' if schema else 'libre'}\n")

    out: dict = {"model": args.model, "fingerprint": f"{fp_note}+{fp_graph}",
                 "label": args.label, "schema_constrained": schema,
                 "temperature": args.temperature, "split": True, "cases": {}}
    demi = 0
    for set_name in sets:
        for case in SETS[set_name]:
            merged, diag = classify_split(args.model, case["text"], schema, args.temperature)
            rec = path_of(merged)
            rec.update(set=set_name, text=case["text"], **diag,
                       confidence=(merged or {}).get("classification_confidence"))
            out["cases"][case["id"]] = rec
            if not (diag["note_parsed"] and diag["graph_parsed"]):
                demi += 1
            mark = "·" if rec.get("parsed") else "✗"
            half = "" if diag["note_parsed"] and diag["graph_parsed"] else "  ⚠ moitié perdue"
            print(f"  {mark} {case['id']:22} "
                  f"note={str(rec.get('has_note')):5} kind={str(rec.get('kind')):6} "
                  f"f={rec.get('facts')} r={rec.get('relations')}{half}", flush=True)

    if demi:
        print(f"\n⚠ {demi} cas où une seule des deux moitiés a répondu — c'est le coût "
              f"propre au découpage, il ne doit pas se lire comme une erreur de jugement.")
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAP_DIR / f"{args.label}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ baseline écrite : {path.relative_to(_REPO)}  ({len(out['cases'])} cas)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--label", required=True)
    r.add_argument("--sets", default=None)
    r.add_argument("--schema", action="store_true")
    r.add_argument("--temperature", type=float, default=0.0)
    r.set_defaults(func=cmd_run)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
