"""SYN-171 — le corpus étiqueté. Synthétique, donc committable dans ce dépôt public.

Les labels dérivent STRICTEMENT de `classifier.md` : ce sont les règles écrites
du prompt, pas des préférences. Un cas dont le prompt ne tranche pas n'a rien à
faire ici (voir `AMBIGUOUS` plus bas).

Champs, tous optionnels — un champ absent = axe non vérifié pour ce cas :
    note        True/False : la capture doit-elle produire une atomic_note
    kind        note | task | event | episode, vérifié seulement si une note est produite
    ephemeral   valeur attendue de is_ephemeral
    drop_guard  True : cette capture NE DOIT PAS disparaître (note, intention
                ou entrée projet — au moins une trace). C'est le garde-fou du
                bug historique « action terse classée éphémère puis droppée ».
    rel         fragment attendu dans un prédicat de relation
    proj        "new" | "existing" : une entrée projet est attendue
    facts_min   nombre minimal de faits attendus (atomicité, SYN-98)
"""
from __future__ import annotations

# Raté par TOUS les modèles mesurés depuis juillet 2026, Haiku compris : le
# prompt lui-même ne tranche pas entre « anniversaire = note event récurrente »
# et « anniversaire = fait daté ». Gardé pour l'observer, exclu du décompte
# d'échec tant que le prompt reste ambigu.
AMBIGUOUS = {"e4"}


# ---------------------------------------------------------------------------
# ÉTAGE 1 — le gate. Douze cas, un par mode de défaillance rédhibitoire.
# Objectif : rendre un NO-GO en minutes, pas en soirée (leçon SYN-154).
# ---------------------------------------------------------------------------
GATE_CASES = [
    # — Le modèle avale-t-il le prompt et rend-il du JSON exploitable ? —
    # (vérifié sur les douze : validité JSON, troncature, énumération fermée)

    # — drop_guard : les quatre familles qu'on ne doit jamais perdre —
    dict(id="g-task-addressed", text="Répondre à l'e-mail de Vincent",
         note=True, kind="task", drop_guard=True),
    dict(id="g-event-nominal", text="Salon Vivatech le 24",
         note=True, kind="event", drop_guard=True),
    dict(id="g-note-reflexive",
         text="Je pense qu'on devrait arrêter de faire des réunions le lundi",
         note=True, kind="note", ephemeral=False),
    dict(id="g-ephemeral-trivial", text="Acheter du pain",
         note=False, ephemeral=True),

    # — Les natures de capture qui n'aboutissent PAS à une note —
    dict(id="g-type-fact", text="Marie a un chat qui s'appelle Gipsy",
         note=False),
    # Lot 2 puis arbitrage du 20/08 : un épisode vécu garde sa note, MAIS un
    # footing solitaire commenté d'une humeur n'a rien où revenir. Même texte que
    # `ep2` — les deux doivent rester d'accord, ils ont divergé une journée.
    dict(id="g-type-episodic", text="Went for a run this morning, felt good",
         note=False),
    # Le prompt ne parle plus d'URL depuis le retrait d'`input_type` (20/08) :
    # les ressources sont pilotées par le scan de liens en aval, jamais par le
    # classifieur. Reste vérifiable, et ça suffit : deux mots d'appréciation ne
    # font pas une prise de position (critère b), donc pas de note — et surtout
    # pas d'entité inventée pour le domaine.
    dict(id="g-type-resource",
         text="https://example.com/article super intéressant sur la mémoire",
         note=False),
    dict(id="g-type-ephemeral", text="Acheter un harnais",
         note=False, ephemeral=True),

    # — Fait contre relation : l'anti-redite de juin 2026 —
    dict(id="g-relation", text="Audric est le cousin d'Alexis",
         note=False, rel="cousin"),

    # — Projet : la frontière avec une simple tâche —
    dict(id="g-project-new", text="Nouveau projet : rénovation de l'appartement",
         proj="new"),

    # — Anglais : le prompt est EN-base depuis SYN-119, il doit tenir —
    dict(id="g-english-task", text="Reply to Léna's email about the contract",
         note=True, kind="task", drop_guard=True),

    # — Atomicité (SYN-98) : la capture MIXTE doit produire la note event ET les
    #   faits, jamais l'un À LA PLACE de l'autre. C'est ce qui a fait disparaître
    #   un anniversaire de « À venir » pendant des semaines.
    #   Formulation reprise TELLE QUELLE de `mixed-birthday-facts` dans
    #   `scripts/lang_dataset.json` : la vérité-terrain est déjà versionnée et
    #   validée là-bas, en réécrire une variante ferait diverger deux garde-fous
    #   censés protéger la même règle.
    dict(id="g-atomicity-mixed",
         text="C'est l'anniversaire de Yanis le 23 juillet, Yanis est le fils de "
              "Marc et Julie donc mon neveu et le frère de Léna",
         note=True, kind="event", drop_guard=True, facts_min=1),
]


# ---------------------------------------------------------------------------
# ÉTAGE 2 — les 29 cas durs. Portés depuis le harnais de SYN-124 (juillet 2026),
# qui ne survivait que recopié dans un document Linear.
# ---------------------------------------------------------------------------
HARD_CASES = [
    # TASK : action adressée ou engagement — NE DOIT PAS être droppée en éphémère
    dict(id="t1", text="Répondre à l'e-mail de Vincent", note=True, kind="task", drop_guard=True),
    dict(id="t2", text="Déclarer mes revenus à l'URSSAF", note=True, kind="task", drop_guard=True),
    dict(id="t3", text="Parler à Vincent de l'appartement", note=True, kind="task", drop_guard=True),
    dict(id="t4", text="Envoyer la facture à efcsn", note=True, kind="task", drop_guard=True),
    dict(id="t5", text="Reply to Léna's email about the contract", note=True, kind="task", drop_guard=True),
    dict(id="t6", text="Appeler le dentiste avant le 20", note=True, kind="task", drop_guard=True),
    dict(id="t7", text="Je dois préparer la démo pour Vivatech", note=True, kind="task", drop_guard=True),

    # EVENT : syntagme nominal sans verbe — doit quand même produire une note event
    dict(id="e1", text="Salon Vivatech le 24", note=True, kind="event", drop_guard=True),
    dict(id="e2", text="Rendez-vous dentiste mardi", note=True, kind="event", drop_guard=True),
    dict(id="e3", text="Dentist appointment next Tuesday", note=True, kind="event", drop_guard=True),
    dict(id="e4", text="L'anniversaire de Léa est le 16 juin", note=True, kind="event", drop_guard=True),

    # EPHEMERAL trivial : ni destinataire, ni date, ni enjeu — intention seule
    dict(id="p1", text="Acheter du pain", note=False, ephemeral=True),
    dict(id="p2", text="Acheter un harnais", note=False, ephemeral=True),
    dict(id="p3", text="Comprar pan", note=False, ephemeral=True),

    # NOTE réflexive : pensée durable — kind note, JAMAIS éphémère
    dict(id="n1", text="Je pense qu'on devrait arrêter de faire des réunions le lundi",
         note=True, kind="note", ephemeral=False),
    dict(id="n2", text="C'est marrant comme les gens surestiment le court terme "
                       "et sous-estiment le long terme", note=True, kind="note", ephemeral=False),
    dict(id="n3", text="I realized I focus too much on tools instead of outcomes",
         note=True, kind="note", ephemeral=False),

    # PROJET contre TASK
    dict(id="j1", text="J'ai un projet d'escalade pour faire un 7a", proj="existing"),
    dict(id="j2", text="Nouveau projet : rénovation de l'appartement", proj="new"),
    dict(id="j3", text="Je veux apprendre le japonais", proj="new"),

    # FAIT contre RELATION : objet = entité nommée → la relation seule
    dict(id="r1", text="Audric est le cousin d'Alexis", note=False, rel="cousin"),
    dict(id="r2", text="Pierre travaille chez Acme", note=False, rel="work"),

    # FAIT sur autrui : pas de note
    dict(id="f1", text="Marie a un chat qui s'appelle Gipsy", note=False),
    dict(id="f2", text="Ma mère a un nouveau chat", note=False),

    # HEDGED : formulation prudente
    dict(id="h1", text="Pierre déménage probablement à Lyon", note=False),
    dict(id="h2", text="Léa a sans doute adopté un chien", note=False),

    # RESOURCE
    # ⚠ INSTABLE depuis le retrait d'`input_type` (20/08) : `is_ephemeral`
    # oscille (~2 fois sur 3 à true) sur une capture de lien. Conséquence réelle
    # mais mineure — une intention à 48 h de trop, jamais une perte. Asserté ici
    # pour que la dérive reste comptée au lieu d'être oubliée ; volontairement PAS
    # dans le gate, où un cas instable rendrait l'étage 1 capricieux.
    dict(id="u1", text="https://example.com/article super intéressant sur la mémoire",
         note=False, ephemeral=False),

    # EPISODIC : action déjà vécue, routinière → pas de note (+ progrès projet)
    # Les deux côtés de la frontière épisode, arbitrée le 20/08 : ce n'est pas
    # « vécu ou pas », c'est « y a-t-il quelque chose à y revenir ». Une personne,
    # un lieu, un accomplissement, une première fois — sinon rien.
    dict(id="ep1", text="J'ai été escalader avec Alexis aujourd'hui et j'ai réussi mon 6b+",
         note=True, kind="episode", proj="existing"),
    dict(id="ep2", text="Went for a run this morning, felt good", note=False),
]

# Ajouts SYN-171 : l'atomicité n'était couverte par aucun cas de SYN-124.
ATOMICITY_CASES = [
    # ⚠️ « né le » n'est PAS « c'est l'anniversaire de ». Mesuré le 2026-08-19 :
    # Haiku (prompt v8) en tire `has_birthday` + `lives_in` + `nephew_of`, mais
    # AUCUNE note event. Ce n'est pas forcément un défaut : SYN-97 fait remonter
    # les faits `has_birthday`/`born_on` dans « À venir » par le chemin des faits,
    # côté digest. Reste à vérifier que le NotificationPlanner de l'app fait de
    # même — sinon la date est visible dans le digest et muette en notification.
    # Cas gardé en observation, hors décompte tant que la question n'est pas tranchée.
    dict(id="a1", text="Marc est né le 3 mars, c'est le neveu de Julie et il vit à Nantes",
         facts_min=2, drop_guard=True),
    dict(id="a2", text="Réunion avec Léna le 12 septembre pour parler du contrat Acme, "
                       "elle vient d'être promue directrice",
         note=True, kind="event", drop_guard=True, facts_min=1),
    dict(id="a3", text="Julie m'a dit qu'elle déménageait à Bordeaux en janvier "
                       "et qu'elle cherchait un architecte",
         drop_guard=True, facts_min=1),
]


# ---------------------------------------------------------------------------
# CORPUS ADVERSE — les cas que le prompt ne traite nulle part, ou traite mal.
#
# Ils viennent des arbitrages du 2026-08-19/20. Plusieurs portent des attentes
# que le scoring actuel ne sait PAS encore vérifier (`needs_review`,
# `no_entity`, `forbidden_value`) : ils sont là quand même, pour observer ce que
# les modèles font quand personne ne leur dit rien. Un cas qu'on n'a pas encore
# joué est un cas dont on ne sait rien.
# ---------------------------------------------------------------------------
ADVERSARIAL_CASES = [
    # — La négation. Absente du prompt. Un routage par signaux booléens
    #   produirait une tâche fantôme : `action-à-faire` reste vrai.
    dict(id="x-negation", text="Je ne vais finalement pas appeler le dentiste",
         note=False,
         why="Aucune tâche ne doit naître d'une action annulée."),

    # — L'acteur. Il y a une action à faire, mais elle n'est pas celle de l'auteur.
    dict(id="x-reported-speech",
         text="Marie m'a dit qu'elle devait appeler le dentiste",
         why="Tâche de Marie, pas de l'auteur : rattachée à la fiche de Marie."),

    # — Temps mêlés. Mesuré le 2026-08-19 : Haiku garde la tâche future et PERD
    #   l'appel déjà passé. Arbitrage : la timeline doit survivre.
    dict(id="x-mixed-tense",
         text="J'ai appelé le dentiste ce matin, il faut que je rappelle jeudi",
         drop_guard=True,
         why="Deux sorties attendues : l'épisode vécu ET la tâche datée."),

    # — Passé habituel : première personne, passé, mais AUCUN moment situé.
    #   La règle « déjà vécu ⇒ episodic » exige les trois conditions ; celle-ci
    #   n'en remplit que deux, et le prompt ne dit pas quoi faire.
    dict(id="x-habitual-past", text="Je faisais du piano quand j'étais petit",
         facts_min=1,
         why="Biographie durable, pas un moment : le fait doit exister."),

    # — Épisode pur (cas 3 de l'arbitrage) : ne vaut que comme moment.
    #   Aujourd'hui : aucune note, seule la capture brute reste dans l'inbox.
    dict(id="x-pure-episode", text="J'ai mangé chez Léa hier",
         drop_guard=True,
         why="La timeline mérite une trace durable."),

    # — LA frontière : passé vécu ET course triviale. Arbitré le 20/08 — une
    #   corvée solitaire ne mérite PAS de note permanente (la règle (f) du lot 2
    #   était trop large). Ce qui reste garanti, et c'est l'essentiel : elle ne
    #   redevient jamais une intention à 48 h. Une course FAITE n'est pas à faire.
    dict(id="x-past-errand", text="J'ai acheté du pain ce matin",
         note=False, ephemeral=False,
         why="Rien à y revenir : pas de note. Mais faite, donc jamais un rappel."),

    # — Anniversaire, les trois formulations (arbitrage 7).
    dict(id="x-birthday-party", text="La fête d'anniversaire de Yanis est le 12 juin",
         note=True, kind="event", drop_guard=True,
         why="Mention de fête ⇒ événement."),
    dict(id="x-birthday-birth", text="Yanis est né le 12 juin 1990",
         facts_min=1,
         why="Mention de naissance ⇒ fait daté."),
    dict(id="x-birthday-bare", text="Le 12 juin c'est l'anniversaire de Yanis",
         needs_review=True,
         why="Ni fête ni naissance : indécidable, doit passer par « À valider »."),

    # — Verbes d'assistance (arbitrage 5). « aller à » est un verbe, et pourtant
    #   c'est un événement : le critère n'est pas « y a-t-il un verbe » mais
    #   « l'auteur agit-il SUR quelque chose, ou assiste-t-il à quelque chose ».
    dict(id="x-attend-verb", text="Je vais au salon Vivatech le 12",
         note=True, kind="event", drop_guard=True,
         why="Assister à une occurrence ⇒ event, malgré le verbe."),
    dict(id="x-attend-noun", text="J'ai la fête de Pierre le 20",
         note=True, kind="event", drop_guard=True,
         why="Occurrence possédée, pas action exécutée."),

    # — Création d'entité animal, gouvernée par la persistance (arbitrage 3).
    dict(id="x-pet-owned", text="Mon chat s'appelle Gipsy",
         entity_expected="Gipsy",
         why="Animal du foyer : persistant, mérite son nœud."),
    dict(id="x-pet-incidental",
         text="J'ai vu un ours au zoo qui s'appelait Balthazar",
         no_entity="Balthazar",
         why="Croisé une fois : sous le seuil de persistance. Pas dramatique s'il est créé."),

    # — Le discriminant de la frontière épisode : solitaire ET routinier ne
    #   suffit PAS à exclure. Une PREMIÈRE FOIS mérite sa note, sans personne ni
    #   lieu. Sans ce cas, « solitaire ⇒ pas de note » passerait pour la règle,
    #   et on perdrait exactement les épisodes qui comptent.
    dict(id="x-episode-first-time",
         text="J'ai couru mon premier semi-marathon dimanche",
         note=True, kind="episode",
         why="Une première fois est un accomplissement : il y a de quoi y revenir."),

    # — Invention (arbitrage 13). Qwen a produit `has_breed = \"Domestic cat
    #   (domestic shorthair or domestic longhair)\"` sur cette phrase, qui ne dit
    #   rien de la race. La non-invention n'est exigée aujourd'hui que des
    #   RÉSUMÉS, jamais des faits.
    dict(id="x-no-invention", text="Marie a un chat qui s'appelle Gipsy",
         forbidden_value="shorthair",
         why="Aucun fait ne doit énoncer ce que la capture ne dit pas."),
]


# ---------------------------------------------------------------------------
# MODE SCÉNARIO — la capture N'EST PAS SEULE.
#
# Les étages 1 et 2 classent une capture dans le vide. La production, elle,
# ajoute la MÉMOIRE DE TRAVAIL (SYN-93) : le fil des captures récentes, avec la
# consigne explicite « n'extrais rien de ce bloc ». Cette consigne est respectée
# à la lettre — rien n'est extrait du bloc — et pourtant le bloc DÉPLACE la
# décision prise sur la capture courante.
#
# Mesuré le 2026-08-20 sur l'installation réelle, deux fois, sur deux règles
# différentes. C'est ce qui rend ce mode nécessaire : ces deux cas passent les
# étages 1 et 2 sans broncher, et échouent en prod.
#
# `wm` = les captures antérieures du fil, dans l'ordre. `repeat` = le nombre de
# passes : la défaillance est une INSTABILITÉ (3 fois sur 5, pas 5 sur 5), donc
# une seule passe ne la voit pas. `expect` = la branche attendue, celle que le
# même prompt produit de façon 100 % stable quand la capture est seule.
# ---------------------------------------------------------------------------
SCENARIO_CASES = [
    # Arbitrage 7 — la date nue doit atteindre « À valider ». Seule : note event
    # + confiance 0,55, 11 fois sur 11. Dans le fil : la note disparaît ET la
    # confiance monte à 1,0, donc plus rien ne demande d'arbitrer. Les deux
    # défaillances se cumulent au pire endroit.
    dict(id="s-birthday-in-thread",
         text="L'anniversaire de Yuki est le 4 mars",
         wm=["L'anniversaire de Manon est le 4 mars",
             "Hier j'ai déjeuné avec Manon au Petit Bar, il faut que je la rappelle vendredi",
             "L'anniversaire de Sofia est le 4 mars"],
         expect=dict(note=True, kind="event", confidence_below=0.6),
         repeat=5,
         why="Trois quasi-jumelles au fil. Tombait en prod sous v10 ; tient depuis\n              le retrait d'input_type — un a priori de moins vers « simple fait »."),

    # Lot 2 — une course DÉJÀ FAITE est un épisode. Seule : épisode, 3 fois sur
    # 3. Dans le fil : aucune note.
    dict(id="s-done-errand-in-thread",
         text="J'ai acheté du pain ce matin",
         wm=["La fête d'anniversaire de Yanis est le 12 juin",
             "Marie m'a dit qu'elle devait appeler le dentiste",
             "J'ai mangé chez Léa hier"],
         expect=dict(note=False),
         repeat=5,
         why="Le contexte avait raison avant le prompt : à côté d'un épisode plus riche "
              "au fil, l'ordinaire cesse de valoir une note — et c'est le bon appel "
              "(arbitré le 20/08). La règle (f) a été assouplie pour dire la même chose "
              "SANS dépendre du fil ; ce cas garde l'invariant : même réponse seule ou au fil."),

    # Témoin — même capture, fil SANS RAPPORT. Sert à distinguer « la mémoire de
    # travail déstabilise » de « ces captures contiennent des quasi-jumelles ».
    # Sans ce témoin, on attribuerait au mauvais coupable.
    dict(id="s-birthday-neutral-thread",
         text="L'anniversaire de Yuki est le 4 mars",
         wm=["Réunion budget repoussée à lundi",
             "Le train de 7h12 était supprimé"],
         expect=dict(note=True, kind="event", confidence_below=0.6),
         repeat=5,
         why="Témoin : mémoire de travail présente, mais rien qui ressemble."),

    dict(id="s-done-errand-alone",
         text="J'ai acheté du pain ce matin",
         wm=[],
         expect=dict(note=False),
         repeat=5,
         why="Témoin : la MÊME réponse que dans le fil — c'est ça, l'invariant."),

    # Témoin — capture seule, aucun bloc mémoire. C'est la mesure de référence
    # des étages 1 et 2 : elle DOIT être stable, sinon le reste ne veut rien dire.
    dict(id="s-birthday-alone",
         text="L'anniversaire de Yuki est le 4 mars",
         wm=[],
         expect=dict(note=True, kind="event", confidence_below=0.6),
         repeat=5,
         why="Témoin : la branche de référence, mesurée 11/11 stable."),
]
