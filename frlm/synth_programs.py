"""Données SFT v4.4 à programme latent et surfaces françaises indépendantes.

Les problèmes sont calculés par Python, portent un ``schema_id`` et réservent une
famille de formulation entière à la validation. Le benchmark OOD n'est jamais lu
ni importé : ces générateurs utilisent leurs propres seeds et leurs propres mondes.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from frlm.data import THINK, THINK_END, render_chat


NAMES = ("Alice", "Basile", "Chloé", "Djamila", "Émile", "Fatou", "Gaspard",
         "Hana", "Idriss", "Jeanne", "Karim", "Lina", "Maël", "Nora")
OBJECTS = ("billes", "cartes", "pommes", "cahiers", "jetons", "perles",
           "crayons", "tickets", "boutons", "graines")
PLACES = ("Belrive", "Clairmont", "Valbois", "Montfaucon", "Rochebrune",
          "Saint-Lys", "Aubeterre", "Fontenay")


def _assistant(answer: str, rationale: str, rng: random.Random,
               direct_fraction: float = 0.68) -> str:
    if rng.random() < direct_fraction:
        return answer
    return f"{THINK}\n{rationale}\n{THINK_END}\n{answer}"


def _record(messages: list[dict], capability: str, schema: str, surface: int,
            answer: str, program: str, split: str = "train") -> dict:
    return {
        "t": render_chat(messages), "m": messages, "k": capability,
        "capability_id": capability, "schema_id": schema,
        "surface_family_id": f"{schema}:fr{surface}", "program": program,
        "answer": answer, "split": split, "verifier": "python_exact_v1",
        "license": "frlm-project",
    }


def _surface(rng: random.Random) -> tuple[int, str]:
    # fr4 n'apparaît jamais dans train : la validation mesure une vraie surface
    # retenue plutôt qu'un random split de lignes quasi identiques.
    if rng.random() < 0.02:
        return 4, "val"
    return rng.randrange(4), "train"


def make_reasoning(rng: random.Random, direct_fraction: float = 0.68) -> dict:
    schema = rng.choice((
        "stock_compose", "inverse_state", "cycle_modulo", "interval_inclusive",
        "quotient_remainder", "unit_price", "order_transitive", "discount",
        "elapsed_time", "branch_minimum", "mean_exact", "ratio_share",
    ))
    surface, split = _surface(rng)
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)

    if schema == "stock_compose":
        start, packs, size, used = rng.randint(12, 80), rng.randint(2, 9), rng.randint(3, 12), rng.randint(4, 30)
        result = start + packs * size - used
        prompts = (
            f"{name} possède {start} {obj}, reçoit {packs} paquets de {size}, puis en utilise {used}. Combien lui en reste-t-il ?",
            f"Stock initial : {start} {obj}. On ajoute {packs} lots de {size} et on retire {used}. Donne le stock final.",
            f"Après avoir ajouté {packs} fois {size} à ses {start} {obj}, {name} en donne {used}. Quel est le résultat ?",
            f"Calcule la quantité restante : {start} au départ, plus {packs} groupes de {size}, moins {used}.",
            f"{name} part de {start} {obj}. Une livraison apporte {packs} boîtes contenant chacune {size} unités; {used} unités sont ensuite distribuées. Quel stock reste disponible ?",
        )
        rationale = f"{start} + {packs} × {size} - {used} = {result}."
        program = f"SUB(ADD({start},MUL({packs},{size})),{used})"
    elif schema == "inverse_state":
        start, delta = rng.randint(10, 140), rng.randint(3, 45)
        final = start + delta
        result = start
        prompts = (
            f"Après avoir reçu {delta} {obj}, {name} en a {final}. Quelle était la quantité avant cet ajout ?",
            f"Le stock final est {final} après un ajout de {delta}. Retrouve le stock initial.",
            f"{name} termine avec {final} {obj}, soit {delta} de plus qu'au départ. Quelle était la quantité de départ ?",
            f"Une réserve atteint {final} unités après une livraison de {delta}. Combien contenait-elle auparavant ?",
            f"On observe {final} {obj} après l'arrivée de {delta} nouveaux éléments. Déduis l'état initial.",
        )
        rationale = f"{final} - {delta} = {result}."
        program = f"SUB({final},{delta})"
    elif schema == "cycle_modulo":
        days = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
        index, offset = rng.randrange(7), rng.randint(8, 90)
        result = days[(index + offset) % 7]
        prompts = (
            f"Nous sommes {days[index]}. Quel jour serons-nous dans {offset} jours ?",
            f"Avance de {offset} jours à partir de {days[index]}. Donne le jour obtenu.",
            f"Un calendrier indique {days[index]}; après {offset} journées complètes, quel nom de jour affichera-t-il ?",
            f"Calcule le jour de la semaine situé {offset} jours après {days[index]}.",
            f"Le point de départ est {days[index]}. En suivant le cycle hebdomadaire pendant {offset} jours, où arrive-t-on ?",
        )
        rationale = f"{offset} modulo 7 vaut {offset % 7}; on arrive à {result}."
        program = f"CYCLE7({index}+{offset})"
    elif schema == "interval_inclusive":
        first, last = rng.randint(2, 80), rng.randint(90, 180)
        result = last - first + 1
        prompts = (
            f"Combien d'entiers y a-t-il de {first} à {last}, bornes incluses ?",
            f"On numérote des cases de {first} jusqu'à {last}. Les deux extrémités comptent. Combien de cases ?",
            f"Une série commence au numéro {first} et finit au numéro {last}. Quelle est sa longueur totale ?",
            f"Compte tous les numéros compris entre {first} et {last}, sans exclure les bornes.",
            f"Les dossiers portent chaque numéro de {first} à {last}. Détermine combien de dossiers cela représente.",
        )
        rationale = f"{last} - {first} + 1 = {result}."
        program = f"INCLUSIVE_COUNT({first},{last})"
    elif schema == "quotient_remainder":
        size, full, rest = rng.randint(3, 15), rng.randint(3, 20), rng.randint(1, 14)
        rest %= size
        rest = rest or 1
        total = size * full + rest
        result = rest
        prompts = (
            f"On range {total} {obj} par boîtes de {size}. Combien restent hors des boîtes complètes ?",
            f"Divise {total} objets en groupes complets de {size} et donne uniquement le reste.",
            f"Avec {total} {obj} et des paquets de {size}, quel nombre d'objets ne peut pas remplir un paquet ?",
            f"Après avoir formé autant de lots de {size} que possible avec {total} éléments, combien reste-t-il ?",
            f"Un conditionnement accepte {size} unités. Pour une commande de {total}, indique le reliquat après remplissage des cartons complets.",
        )
        rationale = f"{total} = {full} × {size} + {rest}; le reste est {rest}."
        program = f"MOD({total},{size})"
    elif schema == "unit_price":
        unit_a, unit_b = rng.sample(range(2, 13), 2)
        qty_a, qty_b = rng.randint(2, 8), rng.randint(2, 8)
        price_a, price_b = unit_a * qty_a, unit_b * qty_b
        result = "l'offre A" if unit_a < unit_b else "l'offre B"
        prompts = (
            f"Offre A : {qty_a} articles pour {price_a} €. Offre B : {qty_b} articles pour {price_b} €. Laquelle coûte le moins par article ?",
            f"Compare le prix unitaire : A vaut {price_a} € pour {qty_a}, B vaut {price_b} € pour {qty_b}. Quelle offre est moins chère ?",
            f"Pour choisir, {name} hésite entre {qty_a} objets à {price_a} € et {qty_b} objets à {price_b} €. Quel lot a le meilleur prix unitaire ?",
            f"Calcule les deux coûts par unité et réponds A ou B : A=({price_a} €, {qty_a} unités), B=({price_b} €, {qty_b} unités).",
            f"Deux fournisseurs proposent A : {qty_a} pièces pour {price_a} €, et B : {qty_b} pièces pour {price_b} €. Lequel est économiquement préférable à l'unité ?",
        )
        rationale = f"A coûte {unit_a} € par article et B {unit_b} €; {result} est moins chère."
        program = f"ARGMIN(DIV({price_a},{qty_a}),DIV({price_b},{qty_b}))"
    elif schema == "order_transitive":
        a, b, c = rng.sample(NAMES, 3)
        result = a
        prompts = (
            f"{a} est plus grand que {b}, et {b} est plus grand que {c}. Qui est le plus grand ?",
            f"On sait que {c} est plus petit que {b}, lui-même plus petit que {a}. Donne la personne la plus grande.",
            f"Classement : {a} > {b} > {c}. Qui occupe la première place ?",
            f"Si {b} est derrière {a} mais devant {c}, qui est devant tout le monde ?",
            f"Dans une comparaison de tailles, {a} dépasse {b}, lequel dépasse {c}. Identifie le maximum.",
        )
        rationale = f"La chaîne donne {a} > {b} > {c}; le maximum est {a}."
        program = f"MAX({a}>{b}>{c})"
    elif schema == "discount":
        price = rng.randint(4, 30) * 10
        rate = rng.choice((10, 20, 25, 50))
        result = price * (100 - rate) // 100
        prompts = (
            f"Un article coûte {price} €. Après une réduction de {rate} %, quel est son nouveau prix ?",
            f"Calcule le prix final de {price} € avec {rate} % de remise.",
            f"Une remise de {rate} % s'applique à un achat de {price} €. Combien faut-il payer ?",
            f"Retire {rate} % de {price} € et donne le montant restant.",
            f"Le tarif initial est {price} €. Une promotion abaisse ce tarif de {rate} %. Détermine le tarif net.",
        )
        rationale = f"La remise vaut {price * rate // 100} €; il reste {result} €."
        program = f"SUB({price},PERCENT({price},{rate}))"
    elif schema == "elapsed_time":
        start_h, start_m = rng.randint(6, 18), rng.choice((0, 10, 15, 20, 30, 40, 45, 50))
        duration = rng.randint(2, 10) * 15
        total = start_h * 60 + start_m + duration
        result = f"{(total // 60) % 24} h {total % 60:02d}"
        prompts = (
            f"Une activité commence à {start_h} h {start_m:02d} et dure {duration} minutes. À quelle heure finit-elle ?",
            f"Ajoute {duration} minutes à {start_h} h {start_m:02d}.",
            f"Départ à {start_h} h {start_m:02d}, durée {duration} min : donne l'heure d'arrivée.",
            f"Quel horaire obtient-on {duration} minutes après {start_h} h {start_m:02d} ?",
            f"Une réunion débute à {start_h} h {start_m:02d}. Elle se termine {duration} minutes plus tard. Indique l'heure de fin.",
        )
        rationale = f"{start_h * 60 + start_m} + {duration} = {total} minutes, soit {result}."
        program = f"TIME_ADD({start_h}:{start_m:02d},{duration})"
    elif schema == "branch_minimum":
        base_a, extra_a, base_b, extra_b = (rng.randint(4, 25) for _ in range(4))
        total_a, total_b = base_a + extra_a, base_b + extra_b
        if total_a == total_b:
            extra_b += 1
            total_b += 1
        result = "option A" if total_a < total_b else "option B"
        prompts = (
            f"Option A coûte {base_a} + {extra_a} €. Option B coûte {base_b} + {extra_b} €. Laquelle est la moins chère ?",
            f"Compare A={base_a}+{extra_a} et B={base_b}+{extra_b}. Choisis le plus petit total.",
            f"{name} doit choisir entre un coût A de {base_a} € puis {extra_a} €, ou B de {base_b} € puis {extra_b} €. Quel choix minimise la dépense ?",
            f"Calcule les deux sommes : A ({base_a}, {extra_a}) et B ({base_b}, {extra_b}), puis donne l'option minimale.",
            f"Deux devis comportent chacun un forfait et un supplément : A={base_a} €+{extra_a} €, B={base_b} €+{extra_b} €. Quel devis retenir pour payer moins ?",
        )
        rationale = f"A vaut {total_a} € et B vaut {total_b} €; {result} est minimale."
        program = f"ARGMIN(ADD({base_a},{extra_a}),ADD({base_b},{extra_b}))"
    elif schema == "mean_exact":
        mean = rng.randint(5, 80)
        delta = rng.randint(1, min(20, mean - 1))
        values = (mean - delta, mean, mean + delta)
        result = mean
        prompts = (
            f"Quelle est la moyenne de {values[0]}, {values[1]} et {values[2]} ?",
            f"Calcule la moyenne arithmétique des trois nombres {values[0]}, {values[1]}, {values[2]}.",
            f"Trois mesures valent {values[0]}, {values[1]} et {values[2]}. Donne leur valeur moyenne.",
            f"Additionne {values[0]}, {values[1]} et {values[2]}, puis divise par trois.",
            f"Un relevé contient les valeurs {values[0]}, {values[1]} et {values[2]}. Quel niveau moyen faut-il reporter ?",
        )
        rationale = f"({values[0]} + {values[1]} + {values[2]}) ÷ 3 = {result}."
        program = f"DIV(SUM{values},3)"
    else:  # ratio_share
        unit, left, right = rng.randint(2, 12), rng.randint(2, 8), rng.randint(2, 8)
        total = unit * (left + right)
        result = unit * left
        prompts = (
            f"On partage {total} {obj} selon le ratio {left}:{right}. Combien revient à la première part ?",
            f"Deux parts sont proportionnelles à {left} et {right}, pour un total de {total}. Quelle est la première ?",
            f"Répartis {total} unités en {left + right} parts égales : le premier groupe en reçoit {left}. Quelle quantité obtient-il ?",
            f"Dans le rapport {left}/{right}, calcule la part de gauche d'un ensemble de {total}.",
            f"Une dotation de {total} unités est divisée suivant les coefficients {left} et {right}. Détermine l'allocation du premier bénéficiaire.",
        )
        rationale = f"Une part vaut {total} ÷ {left + right} = {unit}; la première vaut {unit} × {left} = {result}."
        program = f"MUL(DIV({total},ADD({left},{right})),{left})"

    answer = str(result)
    messages = [{"role": "user", "text": prompts[surface]},
                {"role": "assistant", "text": _assistant(
                    answer, rationale, rng, direct_fraction
                )}]
    return _record(messages, "verified_reasoning", schema, surface, answer, program, split)


def make_grounded_qa(rng: random.Random, direct_fraction: float = 0.82) -> dict:
    surface, split = _surface(rng)
    place = rng.choice(PLACES)
    year = rng.randint(1880, 2020)
    height = rng.randint(18, 95)
    rooms = rng.randint(3, 18)
    context = (f"La station de {place} a ouvert en {year}. Son bâtiment principal "
               f"mesure {height} mètres et contient {rooms} salles. Elle est consacrée "
               "à l'observation du climat local.")
    kind = rng.randrange(3)
    if kind == 0:
        question, answer, rationale, program = (
            "En quelle année la station a-t-elle ouvert ?",
            f"La station de {place} a ouvert en {year}.",
            f"Le contexte indique explicitement {year}.", "EXTRACT(opening_year)")
    elif kind == 1:
        question, answer, rationale, program = (
            "Combien la station possède-t-elle de salles ?",
            f"Ouverte en {year}, la station de {place} compte {rooms} salles.",
            f"Le bâtiment contient {rooms} salles.", "EXTRACT(room_count)")
    else:
        question, answer, rationale, program = (
            "Quelle est la mission de cette station ?",
            f"Ouverte en {year}, la station de {place} a pour mission d'observer le climat local.",
            "La dernière phrase donne sa mission.", "SUMMARIZE(mission)")
    prompts = (
        f"Contexte : {context}\n\nQuestion : {question}",
        f"Lis ce passage puis réponds brièvement.\n{context}\n\n{question}",
        f"D'après le texte suivant, {question[0].lower() + question[1:]}\n\n{context}",
        f"Document :\n{context}\n\nRéponse attendue : {question}",
        f"Tu dois t'appuyer uniquement sur cette notice : {context}\n\n{question}",
    )
    messages = [{"role": "user", "text": prompts[surface]},
                {"role": "assistant", "text": _assistant(
                    answer, rationale, rng, direct_fraction
                )}]
    return _record(messages, "grounded_qa", f"grounded_{kind}", surface,
                   answer, program, split)


def make_calibration(rng: random.Random) -> dict:
    surface, split = _surface(rng)
    place = rng.choice(PLACES)
    known = rng.randint(5, 8000)
    year = rng.randint(1850, 2025)
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    kind = rng.randrange(3)
    if kind == 0:
        prompt = (f"Contexte : En {year}, {name} a recensé {known} {obj} dans le centre "
                  f"de {place}.\nQuestion : Combien ce centre possède-t-il de portes ?")
        answer = "Le contexte ne permet pas de déterminer le nombre de portes."
        program = "UNANSWERABLE(missing=door_count)"
    elif kind == 1:
        hour_a = rng.randint(15, 19)
        hour_b = rng.randint(hour_a + 1, 22)
        prompt = (f"Une note de {year} affirme que le musée de {place} ferme à {hour_a} h. "
                  f"Une autre note de {name} affirme qu'il ferme à {hour_b} h, sans préciser "
                  "le jour. À quelle heure ferme-t-il ?")
        answer = "Impossible de trancher : les deux informations se contredisent."
        program = "UNANSWERABLE(contradiction=closing_time)"
    else:
        left, right = rng.randint(10, 9999), rng.randint(1, 999)
        prompt = ("Réponds uniquement par le nombre, sans phrase : "
                  f"combien font {left} + {right} ?")
        answer = str(left + right)
        program = f"FORMAT(number_only,ADD({left},{right}))"
    messages = [{"role": "user", "text": prompt}, {"role": "assistant", "text": answer}]
    return _record(messages, "constraints_calibration", f"calibration_{kind}",
                   surface, answer, program, split)


def make_multiturn(rng: random.Random) -> dict:
    surface, split = _surface(rng)
    a, b = rng.randint(3, 300), rng.randint(2, 200)
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    first = a + b
    messages = [
        {"role": "user", "text": rng.choice((
            f"Combien font {a} + {b} ?",
            f"{name} a {a} {obj} et en reçoit {b}. Quel est le total ?",
            f"Additionne {a} et {b}, s'il te plaît.",
            f"Je calcule un stock de {a} plus une livraison de {b}. Quel résultat obtiens-tu ?",
            f"Donne la somme des valeurs {a} et {b}."))},
        {"role": "assistant", "text": str(first)},
        {"role": "user", "text": rng.choice(("Explique brièvement.", "Et si on retire 1 au résultat ?", "Donne maintenant le double."))},
    ]
    follow = messages[-1]["text"]
    if "retire" in follow:
        answer, rationale, program = str(first - 1), f"{first} - 1 = {first - 1}.", f"SUB(ADD({a},{b}),1)"
    elif "double" in follow:
        answer, rationale, program = str(first * 2), f"Le double de {first} vaut {first * 2}.", f"MUL(ADD({a},{b}),2)"
    else:
        answer, rationale, program = str(first), f"On additionne {a} et {b}, ce qui donne {first}.", f"EXPLAIN(ADD({a},{b}))"
    messages.append({"role": "assistant", "text": rationale if "Explique" in follow else answer})
    return _record(messages, "multiturn_identity", "multiturn_followup", surface,
                   answer, program, split)


def make_reasoning_v45(rng: random.Random) -> dict:
    """Le compartiment raisonnement fournit les ~18 % de traces courtes du mix."""
    record = make_reasoning(rng, direct_fraction=0.0)
    record["capability_id"] = record["k"] = "verified_reasoning_v45"
    return record


def make_grounded_v45(rng: random.Random) -> dict:
    """QA ancrée directe : le raisonnement explicite reste dans son compartiment."""
    return make_grounded_qa(rng, direct_fraction=1.0)


def make_general_v45(rng: random.Random) -> dict:
    """Réponses françaises courtes, utiles et sans persona inventée."""
    surface, split = _surface(rng)
    name, place, obj = rng.choice(NAMES), rng.choice(PLACES), rng.choice(OBJECTS)
    topic, definition, example = rng.choice((
        ("une variable", "un nom associé à une valeur qui peut changer", "un compteur de points"),
        ("une moyenne", "la somme des valeurs divisée par leur nombre", "la moyenne de trois notes"),
        ("un budget", "un plan qui répartit des revenus entre plusieurs dépenses", "prévoir le loyer et les courses"),
        ("une sauvegarde", "une copie séparée qui permet de restaurer des données", "copier un dossier important"),
        ("une hypothèse", "une proposition que l'on vérifie avec des faits", "tester une cause possible"),
        ("un pourcentage", "une proportion exprimée sur cent", "25 % correspond à un quart"),
        ("une échéance", "la date limite prévue pour terminer une tâche", "rendre un devoir vendredi"),
        ("une priorité", "une tâche qui doit passer avant les autres", "corriger une panne bloquante"),
    ))
    kind = rng.randrange(4)
    if kind == 0:
        context = rng.choice((f"pour aider {name}", f"dans un projet à {place}",
                              f"avec un exemple sur des {obj}", "pour réviser un contrôle"))
        prompts = (
            f"Explique simplement ce qu'est {topic}, {context}.",
            f"Peux-tu définir {topic} en deux phrases maximum, {context} ?",
            f"Je débute : que signifie {topic}, {context} ?",
            f"Donne une définition claire de {topic} avec un exemple, {context}.",
            f"Comment présenterais-tu {topic} à un élève, {context} ?",
        )
        answer = f"{topic.capitalize()}, c'est {definition}. Par exemple, on peut l'utiliser pour {example}."
        schema = "general_explain"
    elif kind == 1:
        day = rng.choice(("lundi", "mardi", "mercredi", "jeudi", "vendredi"))
        hour = rng.randint(9, 17)
        purpose = f"discuter du dossier des {obj} à {place}"
        prompts = (
            f"Rédige un message poli pour proposer à {name} un rendez-vous {day} à {hour} h afin de {purpose}.",
            f"Écris une invitation brève à {name} pour {day}, {hour} h, pour {purpose}.",
            f"Formule poliment cette proposition : rendez-vous avec {name}, {day} à {hour} h, pour {purpose}.",
            f"Prépare un court message professionnel pour fixer un échange avec {name} {day} à {hour} h et {purpose}.",
            f"Transforme ces notes en message naturel : {name}; {day}; {hour} h; {purpose}.",
        )
        answer = f"Bonjour {name}, serais-tu disponible {day} à {hour} h pour {purpose} ? Merci de me confirmer si cet horaire te convient."
        schema = "general_message"
    elif kind == 2:
        tasks = rng.sample(("relire le cours", "faire les exercices", "préparer les affaires", "vérifier les réponses", "noter les questions"), 3)
        deadline = rng.choice(("ce soir", "demain matin", "avant vendredi", f"avant {rng.randint(8, 19)} h"))
        prompts = (
            f"Aide {name} à organiser ces tâches {deadline} : {', '.join(tasks)}.",
            f"Mets ces trois actions dans un ordre pratique pour {name}, {deadline} : {', '.join(tasks)}.",
            f"Propose un mini-plan à {name} pour finir {deadline} : {', '.join(tasks)}.",
            f"Comment structurer simplement ces tâches {deadline} : {', '.join(tasks)} ?",
            f"Fais une courte checklist pour {name}, à terminer {deadline} : {', '.join(tasks)}.",
        )
        answer = "Voici un ordre simple :\n1. " + tasks[0].capitalize() + ".\n2. " + tasks[1].capitalize() + ".\n3. " + tasks[2].capitalize() + "."
        schema = "general_plan"
    else:
        adjective = rng.choice(("clair", "rapide", "utile", "réaliste", "prudent"))
        year = rng.randint(1990, 2035)
        prompt_text = (f"À {place}, {name} juge le plan des {obj} {adjective} "
                       f"et applicable dès demain en {year}.")
        prompts = (
            f"Reformule sans changer le sens : « {prompt_text} »",
            f"Écris autrement cette phrase : {prompt_text}",
            f"Propose une reformulation naturelle de : {prompt_text}",
            f"Rends cette phrase plus fluide : {prompt_text}",
            f"Réécris brièvement : {prompt_text}",
        )
        answer = (f"Selon {name}, le plan des {obj} à {place} est {adjective} et peut "
                  f"être mis en œuvre dès demain en {year}.")
        schema = "general_rewrite"
    messages = [{"role": "user", "text": prompts[surface]},
                {"role": "assistant", "text": answer}]
    return _record(messages, "general_response", schema, surface, answer,
                   f"TEMPLATE({schema})", split)


def make_constraints_v45(rng: random.Random) -> dict:
    surface, split = _surface(rng)
    a, b = rng.randint(10, 900), rng.randint(2, 90)
    kind = rng.randrange(5)
    if kind == 0:
        prompt = f"Réponds uniquement par le nombre : {a} + {b}."
        answer, program = str(a + b), f"NUMBER_ONLY(ADD({a},{b}))"
    elif kind == 1:
        words = rng.sample(OBJECTS + PLACES + NAMES, 3)
        prompt = "Donne exactement trois mots séparés par des virgules : " + ", ".join(words) + "."
        answer, program = ", ".join(words), "EXACT_LIST(3)"
    elif kind == 2:
        x = rng.randint(1, 9999)
        prompt = f"Réponds en JSON compact avec les clés resultat et pair pour le nombre {x}."
        answer = '{"resultat":' + str(x * 2) + ',"pair":' + str(x % 2 == 0).lower() + '}'
        program = f"JSON(DOUBLE({x}),EVEN({x}))"
    elif kind == 3:
        name = rng.choice(NAMES)
        place = rng.choice(tuple(item for item in PLACES if "-" not in item))
        obj = rng.choice(OBJECTS)
        number = rng.randint(1, 9999)
        prompt = (f"Écris une phrase de huit mots exactement avec {name}, {number}, "
                  f"{obj} et {place}.")
        answer = f"{name} prépare {number} {obj} demain matin à {place}."
        program = "WORD_COUNT(8)"
    else:
        label = (rng.choice(tuple(word.upper() for word in OBJECTS + PLACES + NAMES))
                 + str(rng.randint(10, 9999)))
        prompt = f"Recopie seulement ce mot en minuscules, sans ponctuation : {label}"
        answer, program = label.lower(), "LOWERCASE_ONLY"
    prompt += f" Exercice {rng.randint(1_000, 9_999_999)}."
    messages = [{"role": "user", "text": prompt}, {"role": "assistant", "text": answer}]
    return _record(messages, "constraints_structure", f"constraint_{kind}", surface,
                   answer, program, split)


def make_uncertainty_v45(rng: random.Random) -> dict:
    surface, split = _surface(rng)
    place, name = rng.choice(PLACES), rng.choice(NAMES)
    value = rng.randint(20, 900)
    if rng.random() < 0.6:
        prompt = (f"Contexte : le rapport de {name} indique {value} visiteurs à {place}. "
                  "Question : quelle était la couleur de la porte principale ?")
        answer = (f"Le contexte sur les {value} visiteurs de {place} ne donne pas "
                  "la couleur de la porte principale.")
        program = "ABSTAIN(missing=door_color)"
    else:
        other = value + rng.randint(3, 90)
        prompt = (f"Une source annonce {value} visiteurs à {place}, une autre en annonce "
                  f"{other} pour la même période. Quel est le total exact ?")
        answer = (f"Impossible de déterminer un total exact pour {place} : les valeurs "
                  f"{value} et {other} se contredisent.")
        program = "ABSTAIN(contradiction=visitor_count)"
    messages = [{"role": "user", "text": prompt}, {"role": "assistant", "text": answer}]
    return _record(messages, "uncertainty", "uncertainty_context", surface,
                   answer, program, split)


def make_code_v45(rng: random.Random) -> dict:
    """Petites fonctions Python dont les tests sont calculés avant écriture."""
    surface, split = _surface(rng)
    kind = rng.randrange(6)
    if kind == 0:
        fname, task, code = "est_pair", "renvoie True si n est pair", "def est_pair(n):\n    return n % 2 == 0"
        tests = [(4, True), (7, False)]
    elif kind == 1:
        fname, task, code = "borne", "borne n entre minimum et maximum", "def borne(n, minimum, maximum):\n    return max(minimum, min(n, maximum))"
        tests = [((12, 0, 10), 10), ((3, 0, 10), 3)]
    elif kind == 2:
        fname, task, code = "compte_voyelles", "compte les voyelles minuscules d'un texte", "def compte_voyelles(texte):\n    return sum(c in 'aeiouy' for c in texte.lower())"
        tests = [("salut", 2), ("rythme", 2)]
    elif kind == 3:
        fname, task, code = "inverse", "renvoie une chaîne dans l'ordre inverse", "def inverse(texte):\n    return texte[::-1]"
        tests = [("abc", "cba"), ("été", "été")]
    elif kind == 4:
        fname, task, code = "somme_carres", "renvoie la somme des carrés de la liste", "def somme_carres(valeurs):\n    return sum(x * x for x in valeurs)"
        tests = [([1, 2, 3], 14), ([], 0)]
    else:
        fname, task, code = "maximum", "renvoie le plus grand des deux nombres", "def maximum(a, b):\n    return a if a >= b else b"
        tests = [((3, 8), 8), ((5, 5), 5)]
    # Les noms demandés varient réellement : on évite des dizaines de milliers de
    # réponses byte-identiques tout en gardant chaque programme testable.
    unique_name = f"{fname}_{rng.randint(10, 999_999)}"
    code = code.replace(f"def {fname}(", f"def {unique_name}(", 1)
    fname = unique_name
    namespace: dict[str, object] = {}
    exec(compile(code, "<sft-code>", "exec"), {"__builtins__": {"max": max, "min": min, "sum": sum}}, namespace)
    fn = namespace[fname]
    for args, expected in tests:
        actual = fn(*args) if isinstance(args, tuple) else fn(args)
        if actual != expected:
            raise AssertionError(f"test synthétique échoué pour {fname}: {actual} != {expected}")
    test_label = rng.randint(1_000, 999_999)
    prompts = (
        f"Écris une fonction Python `{fname}` qui {task}. Donne uniquement le code. Test #{test_label}.",
        f"Implémente en Python la fonction `{fname}` : elle {task}. Exercice {test_label}.",
        f"Complète l'exercice {test_label} avec une fonction courte nommée `{fname}` qui {task}.",
        f"Propose le code Python minimal pour `{fname}` ; cette fonction {task}. Cas {test_label}.",
        f"Sans explication, écris `{fname}` en Python. Elle {task}. Référence {test_label}.",
    )
    answer = f"```python\n{code}\n```"
    messages = [{"role": "user", "text": prompts[surface]}, {"role": "assistant", "text": answer}]
    return _record(messages, "verified_short_code", f"code_{kind}", surface,
                   answer, f"PYTHON_TESTED({fname})", split)


def make_style_v45(rng: random.Random) -> dict:
    surface, split = _surface(rng)
    kind = rng.randrange(3)
    name, place, obj = rng.choice(NAMES), rng.choice(PLACES), rng.choice(OBJECTS)
    year = rng.randint(1990, 2035)
    if kind == 0:
        prompt = rng.choice((
            f"Je prépare un dossier sur les {obj} à {place} en {year}. Qui es-tu ?",
            f"Avant d'aider {name} sur son projet de {year}, présente-toi brièvement.",
            f"Quel est ton rôle pour répondre à une question sur {place} et les {obj} ?",
        ))
        answer = (f"Je suis frlm, un assistant conçu pour répondre clairement en français. "
                  f"Je peux aider {name} sur le dossier des {obj} à {place} en {year}, "
                  "tout en signalant mes incertitudes.")
    elif kind == 1:
        score = rng.randint(1, 19)
        prompt = (f"{name} à {place} : je me suis trompé dans mon exercice sur les {obj} "
                  f"en {year} et j'ai eu {score}/20. Je suis découragé.")
        answer = (f"{name}, un résultat de {score}/20 sur cet exercice de {obj} à {place} "
                  f"en {year} n'annule pas tes progrès. Reprenons l'étape qui bloque calmement.")
    else:
        prompt = f"Donne à {name} un conseil très affirmatif sur {place} en {year}, même si tu n'es pas sûr."
        answer = (f"Pour ne pas induire {name} en erreur au sujet de {place} en {year}, "
                  "je distinguerai ce qui est certain de ce qui doit être vérifié.")
    prompt += f" Référence {rng.randint(1_000, 99_999_999)}."
    messages = [{"role": "user", "text": prompt}, {"role": "assistant", "text": answer}]
    return _record(messages, "style_identity", f"style_{kind}", surface,
                   answer, f"STYLE({kind})", split)


GENERATORS = {
    "reasoning_v44": make_reasoning,
    "grounded_v44": make_grounded_qa,
    "calibration_v44": make_calibration,
    "multiturn_v44": make_multiturn,
    "reasoning_v45": make_reasoning_v45,
    "grounded_v45": make_grounded_v45,
    "multiturn_v45": make_multiturn,
    "general_v45": make_general_v45,
    "constraints_v45": make_constraints_v45,
    "uncertainty_v45": make_uncertainty_v45,
    "code_v45": make_code_v45,
    "style_v45": make_style_v45,
}


def verify_record(record: dict) -> bool:
    """Vérifie les invariants mécaniques avant toute écriture."""
    messages = record.get("m") or []
    assistants = [str(m.get("text") or "") for m in messages
                  if m.get("role") == "assistant"]
    users = [str(m.get("text") or "") for m in messages if m.get("role") == "user"]
    answer = str(record.get("answer") or "")
    if not users or not assistants or not answer or not record.get("program"):
        return False
    if any(text.count(THINK) != text.count(THINK_END) for text in assistants):
        return False
    return answer.casefold() in assistants[-1].casefold()


def write_jsonl(path: Path, target_chars: int, seed: int, kind: str) -> dict:
    """Écrit des conversations uniques jusqu'au budget demandé."""
    if kind not in GENERATORS:
        raise ValueError(f"générateur inconnu : {kind}")
    rng = random.Random(seed)
    generate = GENERATORS[kind]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    chars = records = val_records = 0
    seen: set[str] = set()
    with tmp.open("w", encoding="utf-8") as stream:
        while chars < target_chars:
            record = generate(rng)
            if not verify_record(record):
                raise RuntimeError(f"exemple {kind} invalide : {record}")
            prompt = "\n".join(m["text"] for m in record["m"] if m["role"] == "user")
            if prompt in seen:
                continue
            seen.add(prompt)
            line = json.dumps(record, ensure_ascii=False)
            stream.write(line + "\n")
            chars += len(record["t"])
            records += 1
            val_records += int(record["split"] == "val")
    tmp.replace(path)
    return {"path": str(path), "records": records, "val_records": val_records,
            "chars": chars, "seed": seed, "kind": kind}
