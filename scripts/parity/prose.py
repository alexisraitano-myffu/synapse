"""SYN-171 — les CINQ AUTRES prompts. Ceux qui écrivent, pas celui qui lit.

Le classifieur rend du JSON : on compare des branches. Les cinq autres rendent de
la PROSE, et jusqu'ici aucun garde-fou ne les couvrait — alors que ce sont eux qui
produisent ce que l'utilisateur LIT : la fiche d'une entité, la synthèse d'un
projet, le digest du lundi matin, le résumé d'un lien.

On ne peut pas comparer deux proses. Mais chacun de ces prompts ÉNONCE des
contraintes, et une contrainte énoncée se vérifie. C'est la seule règle de ce
fichier, et elle est stricte : **tout contrôle ici cite la ligne du prompt dont
il dérive**. Un contrôle qui exprimerait mon goût plutôt que le prompt ferait
échouer un modèle sur une exigence que personne ne lui a formulée.

Ce qui échappe par construction — est-ce que la synthèse est BONNE, agréable,
utile — n'est pas mesuré. On mesure qu'elle respecte ce qu'on lui a demandé.
C'est moins ambitieux et c'est opposable.

Usage :
    python -m scripts.parity.prose anthropic:claude-haiku-4-5-20251001
    python -m scripts.parity.prose ollama:qwen2.5:3b-instruct-q4_K_M --only digest
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from scripts.parity.context import CORE_CLASSIFIER, load_prompt
from scripts.parity.providers import call

PROMPTS_DIR = CORE_CLASSIFIER.parent


# ── outils de mesure, tous déterministes ────────────────────────────────────

def sentences(text: str) -> list[str]:
    """Découpe en phrases. Les abréviations courantes sont protégées, sinon
    « M. Dupont » compterait pour deux et un contrôle « 1 à 2 phrases »
    échouerait sur une sortie correcte."""
    t = re.sub(r"\b(M|Mme|Dr|St|etc|cf|p|ex)\.", r"\1<POINT>", text.strip())
    parts = [p.strip() for p in re.split(r"[.!?]+(?=\s|$)", t) if p.strip()]
    return [p.replace("<POINT>", ".") for p in parts]


def words(text: str) -> int:
    return len(text.split())


_FR_MARKERS = {"le", "la", "les", "des", "une", "un", "du", "de", "et", "est",
               "sur", "avec", "pour", "dans", "cette", "ses", "son", "aux", "au"}
_EN_MARKERS = {"the", "a", "an", "of", "and", "is", "on", "with", "for", "in",
               "this", "its", "his", "her", "to", "at", "are", "was"}


def lang_of(text: str) -> str:
    """FR ou EN par comptage de mots-outils. Volontairement rustique : pas de
    dépendance, et une détection déterministe vaut mieux qu'une bibliothèque de
    plus dans un harnais dont tout l'intérêt est d'être rejouable."""
    toks = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ']+", text.lower())
    fr = sum(t in _FR_MARKERS for t in toks)
    en = sum(t in _EN_MARKERS for t in toks)
    if fr == en:
        return "indécis"
    return "fr" if fr > en else "en"


def absent(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t.lower() in low]


def body_only(text: str) -> str:
    """Le corps, titres markdown retirés. Indispensable pour TIMELESS : digest.md
    PRESCRIT le titre « ## Cette semaine ». Balayer les titres ferait échouer le
    modèle sur une chaîne que le prompt lui impose d'écrire — le contrôle
    sanctionnerait alors l'obéissance."""
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def present(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t.lower() not in low]


# Formulations relatives interdites par la règle TIMELESS, qui apparaît
# textuellement dans resummary.md ET digest.md. Un résumé relu dans six mois doit
# rester vrai : « la semaine prochaine » y sera faux.
_META = ["cet article", "this article", "l'article", "the article",
         "ce document", "this document"]

# La règle TIMELESS existe pour une raison énoncée : « the digest will be reread
# months later ». Ce qu'elle interdit vraiment, ce sont les relatifs FLOTTANTS,
# ceux qui deviennent FAUX à la relecture.
_RELATIFS = ["next week", "la semaine prochaine", "tomorrow", "demain", "soon",
             "bientôt", "recently", "récemment", "just now", "à l'instant",
             "yesterday", "hier", "aujourd'hui", "today",
             "le mois prochain", "next month"]

# « Cette semaine » est un cas à part, et seulement dans le digest : le document
# EST une semaine, le prompt lui impose ce titre, et la formule y est
# co-référente de son propre sujet — relue dans six mois elle désigne toujours la
# même semaine, donc elle ne ment pas. Sur une fiche d'entité, en revanche, elle
# flotte : rien n'ancre « cette semaine », et le résumé devient faux.
_RELATIFS_FICHE = _RELATIFS + ["this week", "cette semaine"]


# ── les cas ─────────────────────────────────────────────────────────────────
#
# Chaque `checks` est une liste de (libellé, fonction(sortie) -> str | None).
# La fonction rend None si la contrainte tient, sinon le motif de l'échec.
# Le libellé cite la contrainte du prompt : c'est ce qui rend le verdict
# opposable plutôt qu'une opinion sur le style.

def _resummary_cases() -> list[dict]:
    sys_fr = load_prompt(PROMPTS_DIR / "resummary.md").replace(
        "{language}",
        "write the summary in fr (ISO 639-1) — the dominant language of this "
        "entity's captures. Never translate to another language.")
    sys_en = load_prompt(PROMPTS_DIR / "resummary.md").replace(
        "{language}",
        "write the summary in en (ISO 639-1) — the dominant language of this "
        "entity's captures. Never translate to another language.")
    # Forme EXACTE du message utilisateur construit par `summaries.rs::resummarize`.
    user_fr = ("Entity: Nadia Belkacem (type person)\n"
               "Facts:\n"
               "- works_at : Cabinet Orsay\n"
               "- lives_in : Bordeaux\n"
               "- has_birthday : 1991-04-12\n"
               "Relations:\n"
               "- sibling_of → Karim Belkacem")
    return [
        dict(id="resum-fr", system=sys_fr, user=user_fr, max_tokens=1024,
             checks=[
                 ("1 à 2 phrases", lambda o: None if 1 <= len(sentences(o)) <= 2
                  else f"{len(sentences(o))} phrases"),
                 ("langue = fr", lambda o: None if lang_of(o) == "fr"
                  else f"détecté {lang_of(o)}"),
                 ("TIMELESS (aucun relatif, même ancré)",
                  lambda o: None if not absent(o, _RELATIFS_FICHE)
                  else f"relatif : {absent(o, _RELATIFS_FICHE)}"),
                 ("texte seul, pas de markdown",
                  lambda o: None if not re.search(r"^\s*[#\-*]|\*\*", o) else "markdown détecté"),
                 # N'invente rien : rien dans le message ne dit sa profession ni
                 # son âge. Un modèle qui les déduit du nom du cabinet invente.
                 ("aucune invention",
                  lambda o: None if not absent(o, ["avocate", "avocat", "lawyer", "médecin"])
                  else f"inventé : {absent(o, ['avocate', 'avocat', 'lawyer', 'médecin'])}"),
             ]),
        # Même matière, consigne de langue opposée : c'est le contrôle qui prouve
        # que le modèle SUIT la directive au lieu de suivre la langue des faits.
        dict(id="resum-en", system=sys_en, user=user_fr, max_tokens=1024,
             checks=[
                 ("1 à 2 phrases", lambda o: None if 1 <= len(sentences(o)) <= 2
                  else f"{len(sentences(o))} phrases"),
                 ("langue = en malgré une matière FR",
                  lambda o: None if lang_of(o) == "en" else f"détecté {lang_of(o)}"),
                 ("TIMELESS (aucun relatif, même ancré)",
                  lambda o: None if not absent(o, _RELATIFS_FICHE)
                  else f"relatif : {absent(o, _RELATIFS_FICHE)}"),
             ]),
    ]


def _resource_cases() -> list[dict]:
    system = load_prompt(PROMPTS_DIR / "resource-summary.md")
    corps_en = ("Spaced repetition schedules reviews at growing intervals. The method rests on "
                "the spacing effect: recall is stronger when study sessions are separated in "
                "time than when they are massed together. Modern implementations adapt the "
                "interval to the learner's answer, lengthening it after a success and shortening "
                "it after a lapse.")
    corps_fr = ("La répétition espacée programme les révisions à intervalles croissants. La "
                "méthode repose sur l'effet d'espacement : le rappel est plus solide quand les "
                "sessions sont séparées dans le temps que lorsqu'elles sont massées. Les "
                "implémentations modernes adaptent l'intervalle à la réponse de l'apprenant.")
    common = [
        ("2 à 4 phrases", lambda o: None if 2 <= len(sentences(o)) <= 4
         else f"{len(sentences(o))} phrases"),
        # « No meta-commentary, no "this article" » — textuel dans le prompt.
        ("pas de méta-commentaire",
         lambda o: None if not absent(o, _META) else f"méta : {absent(o, _META)}"),
    ]
    return [
        dict(id="resource-en", system=system, max_tokens=1024,
             user=f"Title: Spaced repetition, a primer\n\n{corps_en}",
             checks=common + [("langue = celle de la ressource (en)",
                               lambda o: None if lang_of(o) == "en" else f"détecté {lang_of(o)}")]),
        # Le miroir FR est le seul contrôle qui distingue « suit la ressource »
        # de « répond toujours en anglais parce que le prompt est en anglais ».
        dict(id="resource-fr", system=system, max_tokens=1024,
             user=f"Title: La répétition espacée, une introduction\n\n{corps_fr}",
             checks=common + [("langue = celle de la ressource (fr)",
                               lambda o: None if lang_of(o) == "fr" else f"détecté {lang_of(o)}")]),
    ]


def _project_cases() -> list[dict]:
    sum_sys = load_prompt(PROMPTS_DIR / "project-summary.md")
    ref_sys = load_prompt(PROMPTS_DIR / "project-refinement.md")
    # Forme de `summaries.rs::synthesize_project` (branche « synthèse actuelle »).
    sum_user = (
        "Project: Rénovation appartement\n\n"
        "Current synthesis:\n---\n"
        "## Objectif\nRefaire la cuisine et la salle de bain.\n\n"
        "## Décisions\n- Cuisine en chêne clair.\n---\n\n"
        "New entry to integrate:\nDevis du plombier reçu, 3 200 €.\n\n"
        "Faits actifs du projet (durable data, to be reflected verbatim in the synthesis):\n"
        "- budget_total : 18 500 €\n- date_livraison : 2027-02-15"
    )
    ref_user = (
        "Project: Rénovation appartement\n\n"
        "All entries in chronological order:\n"
        "[2026-01-04 10:00:00] Objectif : refaire la cuisine et la salle de bain.\n\n"
        "[2026-02-11 10:00:00] Budget envisagé : 12 000 €.\n\n"
        "[2026-03-02 10:00:00] Budget envisagé : 12 000 €.\n\n"
        "[2026-04-19 10:00:00] On part sur du chêne clair pour la cuisine.\n\n"
        "[2026-05-30 10:00:00] Devis du plombier reçu, 3 200 €.\n\n"
        "Faits actifs du projet (durable data, to be reflected verbatim in the synthesis):\n"
        "- budget_total : 18 500 €"
    )
    no_preamble = (
        "commence directement (pas de préambule)",
        lambda o: None if not re.match(r"^\s*(voici|here is|here's|bien sûr|sure|d'accord)",
                                       o.strip(), re.I) else "préambule détecté",
    )
    # Miroir ANGLAIS. `project-summary.md` et `project-refinement.md` ne disaient
    # RIEN de la langue jusqu'au 20/08 — seuls prompts sur six dans ce cas, alors
    # que SYN-119 pose que la sortie suit la langue du contenu. Haiku s'en sortait
    # par bon sens ; Qwen et Llama viennent de démontrer qu'ils n'en ont pas
    # (l'un répond EN sur matière FR, l'autre FR sur directive EN). La règle a été
    # ajoutée aux deux prompts, ce qui rend ce contrôle exigible.
    sum_user_en = (
        "Project: Kitchen renovation\n\n"
        "Current synthesis:\n---\n"
        "## Goal\nRedo the kitchen and the bathroom.\n\n"
        "## Decisions\n- Light oak kitchen.\n---\n\n"
        "New entry to integrate:\nPlumber quote received, 3,200 EUR.\n\n"
        "Faits actifs du projet (durable data, to be reflected verbatim in the synthesis):\n"
        "- budget_total : 18,500 EUR"
    )
    return [
        dict(id="project-summary-en", system=sum_sys, user=sum_user_en, max_tokens=2048,
             checks=[
                 ("markdown (titres ##)", lambda o: None if "##" in o else "aucun titre ##"),
                 no_preamble,
                 ("langue = celle de la matière (en), malgré un prompt FR",
                  lambda o: None if lang_of(o) == "en" else f"détecté {lang_of(o)}"),
                 ("le fait actif est repris tel quel",
                  lambda o: None if not present(o, ["18,500"]) else "budget_total absent"),
             ]),
        dict(id="project-summary", system=sum_sys, user=sum_user, max_tokens=2048,
             checks=[
                 ("markdown (titres ##)", lambda o: None if "##" in o else "aucun titre ##"),
                 ("~500 mots max", lambda o: None if words(o) <= 550 else f"{words(o)} mots"),
                 no_preamble,
                 # « les faits actifs font foi sur la prose des entrées » : la
                 # valeur durable doit apparaître TELLE QUELLE.
                 ("le fait actif est repris tel quel",
                  lambda o: None if not present(o, ["18 500"]) else "budget_total absent"),
             ]),
        dict(id="project-refinement", system=ref_sys, user=ref_user, max_tokens=3072,
             checks=[
                 ("markdown (titres ##)", lambda o: None if "##" in o else "aucun titre ##"),
                 ("500-800 mots max", lambda o: None if words(o) <= 850 else f"{words(o)} mots"),
                 no_preamble,
                 ("le fait actif est repris tel quel",
                  lambda o: None if not present(o, ["18 500"]) else "budget_total absent"),
                 # « déduplique » : l'entrée budget figure DEUX fois à l'identique
                 # dans la timeline. Elle ne doit pas ressortir deux fois.
                 ("déduplication", lambda o: None if o.count("12 000") <= 1
                  else f"« 12 000 » répété {o.count('12 000')} fois"),
                 # ⚠ AMBIGUÏTÉ DU PROMPT, pas défaut du modèle. `project-refinement.md`
                 # demande d'« élaguer le périmé » ET de « préserver l'historique des
                 # décisions importantes ». Une révision de budget relève des deux :
                 # Haiku écrit « 18 500 € (révisé à la hausse depuis 12 000 €) », ce
                 # qui est défendable. Le contrôle qui exigeait la disparition de
                 # 12 000 € exprimait MA préférence — retiré. Ce qui reste exigible
                 # sans ambiguïté : le fait actif figure, et il figure comme le
                 # montant COURANT (contrôlé au-dessus). Trancher « historique ou
                 # élagage » est une décision produit, à porter dans le prompt avant
                 # de pouvoir la contrôler ici.
             ]),
    ]


def _digest_cases() -> list[dict]:
    system = load_prompt(PROMPTS_DIR / "digest.md")
    import json as _json
    # Une VRAIE semaine. La matière maigre était une erreur de fixture : le prompt
    # demande ~250-400 mots ET interdit de meubler (« if a section is empty, say so
    # briefly rather than padding »). Sur trois lignes de matière, ces deux règles
    # se contredisent, et le contrôle de longueur sanctionnait le modèle pour avoir
    # obéi à la bonne. Une contrainte ne se mesure que dans les conditions où elle
    # s'applique.
    semaine_fr = _json.dumps({
        "week_start": "2026-08-17",
        "new_entities": [
            {"canonical_name": "Nadia Belkacem", "type": "person"},
            {"canonical_name": "Cabinet Orsay", "type": "organization"},
            {"canonical_name": "Théo Marchand", "type": "person"},
            {"canonical_name": "Rénovation appartement", "type": "project"},
            {"canonical_name": "Bordeaux", "type": "place"},
        ],
        "new_facts": [
            {"entity": "Nadia Belkacem", "predicate": "works_at", "value": "Cabinet Orsay"},
            {"entity": "Nadia Belkacem", "predicate": "lives_in", "value": "Bordeaux"},
            {"entity": "Théo Marchand", "predicate": "has_birthday", "value": "1988-11-03"},
            {"entity": "Rénovation appartement", "predicate": "budget_total", "value": "18 500 €"},
        ],
        "new_notes": [
            {"title": "Réunions du lundi", "kind": "note",
             "content": "Je pense qu'on devrait arrêter les réunions du lundi : "
                        "personne n'y décide rien avant le mardi de toute façon."},
            {"title": "Devis plomberie", "kind": "task",
             "content": "Comparer le devis du plombier avec un deuxième avis."},
            {"title": "Escalade avec Théo", "kind": "episode",
             "content": "Séance d'escalade avec Théo, j'ai enfin passé le 6b+."},
            {"title": "Sur la mémoire espacée", "kind": "note",
             "content": "L'effet d'espacement marche parce qu'il force la "
                        "reconstruction, pas la relecture."},
        ],
        "trends": [
            {"canonical_name": "Nadia Belkacem", "mention_count": 6},
            {"canonical_name": "Rénovation appartement", "mention_count": 4},
            {"canonical_name": "Théo Marchand", "mention_count": 3},
        ],
        "upcoming_events": [
            {"title": "Salon Vivatech", "event_date": "2026-08-24"},
            {"title": "Rendez-vous dentiste", "event_date": "2026-08-26"},
            {"title": "Anniversaire de Théo Marchand", "event_date": "2026-11-03"},
        ],
        "open_tasks": [
            {"title": "Répondre à l'e-mail de Vincent"},
            {"title": "Déclarer mes revenus à l'URSSAF"},
            {"title": "Comparer le devis du plombier"},
        ],
        "validated_count": 5,
    }, ensure_ascii=False, indent=2)
    return [
        dict(id="digest-fr", system=system, max_tokens=3072,
             user=f"Material for the week of 2026-08-17:\n\n{semaine_fr}",
             checks=[
                 # « Start directly with the first section heading. Do not add an H1 »
                 ("commence par un titre ##, sans H1",
                  lambda o: None if o.strip().startswith("## ") else
                  f"commence par {o.strip()[:24]!r}"),
                 ("~250-400 mots", lambda o: None if 150 <= words(o) <= 520
                  else f"{words(o)} mots"),
                 ("langue = celle du matériau (fr), titres inclus",
                  lambda o: None if lang_of(o) == "fr" else f"détecté {lang_of(o)}"),
                 ("TIMELESS : aucun relatif FLOTTANT dans le corps",
                  lambda o: None if not absent(body_only(o), _RELATIFS)
                  else f"relatif : {absent(body_only(o), _RELATIFS)}"),
                 # « No invention: mention only what is in the JSON. »
                 ("aucune invention",
                  lambda o: None if not absent(o, ["Vivatech 2027", "Paris", "lundi 24"])
                  else f"inventé : {absent(o, ['Vivatech 2027', 'Paris', 'lundi 24'])}"),
                 ("les deux sections sont là",
                  lambda o: None if o.count("## ") >= 2 else f"{o.count('## ')} section(s)"),
                 # « No greetings, no filler. »
                 ("pas de salutation",
                  lambda o: None if not absent(o, ["bonjour", "salut", "hello", "bonne semaine"])
                  else "salutation détectée"),
             ]),
    ]


def all_cases() -> list[dict]:
    return (_resummary_cases() + _resource_cases()
            + _project_cases() + _digest_cases())


def run(model: str, only: str | None, temperature: float) -> int:
    cases = [c for c in all_cases() if not only or only in c["id"]]
    if not cases:
        print(f"aucun cas ne correspond à {only!r}")
        return 2
    print(f"modèle : {model}")
    print(f"cas    : {len(cases)} — les cinq prompts qui écrivent de la prose\n")
    failed = 0
    for case in cases:
        reply = call(model, [case["system"]], case["user"], case["max_tokens"],
                     temperature=temperature)
        out = (reply.text or "").strip()
        if reply.truncated:
            print(f"  ✗ {case['id']:20} sortie tronquée (budget épuisé)")
            failed += 1
            continue
        if not out:
            print(f"  ✗ {case['id']:20} sortie vide")
            failed += 1
            continue
        problems = [(label, why) for label, check in case["checks"]
                    if (why := check(out)) is not None]
        mark = "✓" if not problems else "✗"
        print(f"  {mark} {case['id']:20} {len(case['checks']) - len(problems)}"
              f"/{len(case['checks'])} contraintes")
        for label, why in problems:
            print(f"        · {label} → {why}")
        if problems:
            failed += 1
            print(f"        sortie : {out[:160]!r}")
    print()
    if failed:
        print(f"VERDICT : {failed} cas sur {len(cases)} violent une contrainte ÉNONCÉE")
        print("          par leur prompt. Ce n'est pas un jugement de style.")
        return 1
    print(f"VERDICT : {len(cases)}/{len(cases)} — chaque prompt tient ses propres règles.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SYN-171 — les cinq prompts qui écrivent")
    ap.add_argument("model", help="provider:modèle")
    ap.add_argument("--only", help="filtre sur l'id (resum, resource, project, digest)")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args(argv)
    return run(args.model, args.only, args.temperature)


if __name__ == "__main__":
    sys.exit(main())
