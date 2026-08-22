"""
synth.py — Générateur local de problèmes de maths et de logique en français.

Pourquoi générer plutôt que télécharger ?
  À 30-60M de params, un modèle n'apprend pas à calculer en regardant du web : il
  faut des MILLIONS d'exemples corrects, gradués, avec les étapes rédigées. Aucun
  dataset français de cette taille n'existe — mais l'arithmétique se génère : la
  solution est calculée en Python, donc JAMAIS fausse (contrairement au modèle qui
  l'apprendra). C'est le levier n°1 du "raisonnement léger" à petite échelle.

Deux formats de sortie :
  mode "pretrain" : texte type manuel scolaire (énoncé + solution rédigée),
                    à mélanger dans le pré-entraînement et le midtrain ;
  mode "sft"      : conversations user/assistant avec <think> court et correct —
                    la longueur du think est PROPORTIONNELLE à la difficulté,
                    exactement la discipline qui manquait à la v1.

Tout est déterministe à partir de la graine : reproductible, streamable, gratuit.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

THINK = "<think>"
THINK_END = "</think>"

# --------------------------------------------------------------------------------------
# Matériel français : prénoms, objets (avec genre pour l'accord), unités
# --------------------------------------------------------------------------------------
PRENOMS = [
    "Léa", "Hugo", "Emma", "Louis", "Jade", "Gabriel", "Chloé", "Raphaël", "Manon",
    "Arthur", "Camille", "Jules", "Inès", "Lucas", "Zoé", "Adam", "Lina", "Nathan",
    "Sarah", "Tom", "Nora", "Paul", "Alice", "Sacha", "Eva", "Noah", "Juliette",
    "Marius", "Rose", "Simon", "Anna", "Malo", "Clara", "Rayan", "Margaux", "Elio",
]

# (singulier, pluriel, genre 'm'/'f', prix unitaire plausible min-max en euros)
OBJETS = [
    ("bille", "billes", "f", (1, 3)),
    ("carte", "cartes", "f", (1, 4)),
    ("livre", "livres", "m", (5, 20)),
    ("cahier", "cahiers", "m", (2, 6)),
    ("stylo", "stylos", "m", (1, 5)),
    ("bonbon", "bonbons", "m", (1, 2)),
    ("pomme", "pommes", "f", (1, 3)),
    ("gâteau", "gâteaux", "m", (2, 8)),
    ("image", "images", "f", (1, 3)),
    ("timbre", "timbres", "m", (1, 4)),
    ("fleur", "fleurs", "f", (2, 5)),
    ("ballon", "ballons", "m", (3, 9)),
    ("crayon", "crayons", "m", (1, 3)),
    ("perle", "perles", "f", (1, 2)),
    ("coquillage", "coquillages", "m", (1, 2)),
    # v2.2 : le modèle déraillait sur tout objet hors liste (« poires » !) —
    # on élargit le monde pour décorréler la procédure du vocabulaire
    ("poire", "poires", "f", (1, 3)),
    ("orange", "oranges", "f", (1, 3)),
    ("cerise", "cerises", "f", (1, 2)),
    ("noix", "noix", "f", (2, 4)),
    ("œuf", "œufs", "m", (1, 2)),
    ("autocollant", "autocollants", "m", (1, 2)),
]

# (singulier, pluriel, catégorie singulière, catégorie plurielle)
ANIMAUX = [
    ("chat", "chats", "un animal", "des animaux"),
    ("merle", "merles", "un oiseau", "des oiseaux"),
    ("truite", "truites", "un poisson", "des poissons"),
    ("chêne", "chênes", "un arbre", "des arbres"),
    ("rosier", "rosiers", "une plante", "des plantes"),
    ("frelon", "frelons", "un insecte", "des insectes"),
    ("dauphin", "dauphins", "un mammifère", "des mammifères"),
    ("lézard", "lézards", "un reptile", "des reptiles"),
]

PRENOMS_M = {"Hugo", "Louis", "Gabriel", "Raphaël", "Arthur", "Jules", "Lucas", "Adam",
             "Nathan", "Tom", "Paul", "Sacha", "Noah", "Marius", "Simon", "Malo",
             "Rayan", "Elio"}


def _il(prenom: str) -> str:
    return "il" if prenom in PRENOMS_M else "elle"


def _obj(rng):
    return rng.choice(OBJETS)


def _qty(o, n):
    """'15 billes' / '1 bille' avec le bon accord."""
    return f"{n} {o[0] if n == 1 else o[1]}"


def _de(o):
    """'de billes' mais 'd'images' — élision devant voyelle."""
    pl = o[1]
    return f"d'{pl}" if pl[0] in "aeiouéèêhœ" else f"de {pl}"


# --------------------------------------------------------------------------------------
# Étapes de calcul rédigées (toujours exactes : calculées, pas imitées)
# --------------------------------------------------------------------------------------
def steps_add(a: int, b: int) -> list[str]:
    if a < 10 and b < 10:
        return [f"{a} + {b} = {a + b}"]
    # décomposition par centaines/dizaines/unités — la méthode qu'on veut lui apprendre
    parts = []
    total, cur = a, a
    for unit in (100, 10, 1):
        chunk = (b // unit) % 10 * unit if unit > 1 else b % 10
        if chunk:
            parts.append(f"{cur} + {chunk} = {cur + chunk}")
            cur += chunk
    return parts if parts else [f"{a} + {b} = {a + b}"]


def steps_sub(a: int, b: int) -> list[str]:
    if b < 10:
        return [f"{a} − {b} = {a - b}"]
    parts, cur = [], a
    for unit in (100, 10, 1):
        chunk = (b // unit) % 10 * unit if unit > 1 else b % 10
        if chunk:
            parts.append(f"{cur} − {chunk} = {cur - chunk}")
            cur -= chunk
    return parts if parts else [f"{a} − {b} = {a - b}"]


def steps_mul(a: int, b: int) -> list[str]:
    if a <= 10 and b <= 10:
        return [f"{a} × {b} = {a * b}"]
    # distributivité sur les dizaines : 23 × 7 -> 20×7 + 3×7
    big, small = (a, b) if a >= b else (b, a)
    tens, ones = big - big % 10, big % 10
    if tens and ones:
        return [f"{tens} × {small} = {tens * small}",
                f"{ones} × {small} = {ones * small}",
                f"{tens * small} + {ones * small} = {a * b}"]
    return [f"{a} × {b} = {a * b}"]


def steps_div(a: int, b: int) -> list[str]:
    q = a // b
    return [f"{b} × {q} = {a}, donc {a} ÷ {b} = {q}"]


# --------------------------------------------------------------------------------------
# Générateurs de problèmes.
# Chacun renvoie un dict : {q, steps, a, ans, level}
#   q     : l'énoncé (question) — v2.2 : PLUSIEURS formulations par concept, pour
#           décorréler la compétence du gabarit (le bench OOD a montré 8/40 sinon)
#   steps : les étapes rédigées (list[str]) — deviendront le <think> en SFT
#   a     : la phrase de réponse finale
#   ans   : la réponse canonique VÉRIFIABLE (int, ou str pour oui/pair/prénom) —
#           c'est la récompense du RL : calculée en Python, jamais fausse
#   level : 1 (facile) à 3 (plus dur) — pilote la longueur du think
# --------------------------------------------------------------------------------------
def p_calc(rng: random.Random) -> dict:
    """Calcul pur : « Calcule : 47 + 385 »."""
    op = rng.choice("+-×÷")
    lvl = rng.choice((1, 1, 2, 2, 3))
    hi = {1: 20, 2: 100, 3: 999}[lvl]
    if op == "+":
        a, b = rng.randint(2, hi), rng.randint(2, hi)
        st, res = steps_add(a, b), a + b
    elif op == "-":
        a, b = sorted((rng.randint(2, hi), rng.randint(2, hi)), reverse=True)
        st, res = steps_sub(a, b), a - b
    elif op == "×":
        m = {1: 10, 2: 12, 3: 99}[lvl]
        a, b = rng.randint(2, m), rng.randint(2, 12)
        st, res = steps_mul(a, b), a * b
    else:
        b = rng.randint(2, 12)
        q = rng.randint(2, {1: 10, 2: 12, 3: 30}[lvl])
        a, res = b * q, q
        st = steps_div(a, b)
    sym = {"+": "+", "-": "−", "×": "×", "÷": "÷"}[op]
    # v2.2 : symboles « machine » (* et /) parfois — le modèle partait en vrille
    # sur « 10*50 = » faute de les avoir jamais vus
    sym_vu = sym if rng.random() < 0.7 else {"+": "+", "−": "-", "×": "*", "÷": "/"}[sym]
    imperatif = {"+": f"Additionne {a} et {b}.", "−": f"Retire {b} de {a}.",
                 "×": f"Multiplie {a} par {b}.", "÷": f"Divise {a} par {b}."}[sym]
    gabarits = [f"Calcule : {a} {sym_vu} {b}",
                f"Calcule : {a} {sym_vu} {b}",
                f"Combien font {a} {sym_vu} {b} ?",
                f"{a} {sym_vu} {b} = ?",
                f"{a} {sym_vu} {b} =",
                f"Quel est le résultat de {a} {sym_vu} {b} ?",
                imperatif]
    if sym == "−":
        gabarits.append(f"Quelle est la différence entre {a} et {b} ?")
    if sym == "÷":
        gabarits.append(f"Combien de fois {b} dans {a} ?")
    return {"q": rng.choice(gabarits), "steps": st,
            "a": f"{a} {sym} {b} = {res}", "ans": res, "level": lvl}


def p_reste(rng: random.Random) -> dict:
    """« Léa a 24 billes, elle en donne 9… » — soustraction en contexte."""
    o = _obj(rng)
    p1, p2 = rng.sample(PRENOMS, 2)
    lvl = rng.choice((1, 2, 2, 3))
    n = rng.randint(8, {1: 20, 2: 60, 3: 300}[lvl])
    k = rng.randint(2, n - 1)
    verbe = rng.choice((f"en donne {k} à {p2}", f"en perd {k}", f"en offre {k} à {p2}"))
    il = _il(p1).capitalize()
    fin = rng.choice((f"Combien {_de(o)} lui reste-t-il ?",
                      f"Combien {_de(o)} a-t-{_il(p1)} maintenant ?",
                      "Combien lui en reste-t-il ?"))
    return {"q": f"{p1} a {_qty(o, n)}. {il} {verbe}. {fin}",
            "steps": [f"{p1} part avec {n} {o[1]} et en {'perd' if 'perd' in verbe else 'donne'} {k}."]
            + steps_sub(n, k),
            "a": f"Il lui reste {_qty(o, n - k)}.", "ans": n - k, "level": lvl}


def p_gain(rng: random.Random) -> dict:
    """Addition en contexte, parfois en deux temps (gagne puis re-gagne)."""
    o = _obj(rng)
    p1 = rng.choice(PRENOMS)
    lvl = rng.choice((1, 2, 2, 3))
    hi = {1: 15, 2: 50, 3: 250}[lvl]
    n, k = rng.randint(3, hi), rng.randint(2, hi)
    il = _il(p1)
    forme = rng.random()
    if forme < 0.12:
        # elliptique : « 9 de plus que 37, ça fait combien ? » (échec OOD v2)
        return {"q": f"{k} de plus que {n}, ça fait combien ?",
                "steps": steps_add(n, k),
                "a": f"{k} de plus que {n}, ça fait {n + k}.", "ans": n + k, "level": 1}
    if forme < 0.24:
        # comparatif : « X en a k de plus que Y »
        p2 = rng.choice([p for p in PRENOMS if p != p1])
        return {"q": f"{p1} a {_qty(o, n)}. {p2} en a {k} de plus. "
                     f"Combien {_de(o)} {p2} a-t-{_il(p2)} ?",
                "steps": steps_add(n, k),
                "a": f"{p2} a {_qty(o, n + k)}.", "ans": n + k, "level": 2}
    deux_temps = lvl >= 2 and rng.random() < 0.4
    if deux_temps:
        j = rng.randint(2, 30)
        q = (f"{p1} a {_qty(o, n)}. Le matin, on lui en donne {k}, "
             f"et l'après-midi encore {j}. Combien {_de(o)} a-t-{il} maintenant ?")
        st = steps_add(n, k) + steps_add(n + k, j)
        return {"q": q, "steps": st, "a": f"{p1} a maintenant {_qty(o, n + k + j)}.",
                "ans": n + k + j, "level": 3}
    return {"q": f"{p1} a {_qty(o, n)} et en reçoit {k} de plus. Combien {_de(o)} a-t-{il} en tout ?",
            "steps": steps_add(n, k),
            "a": f"{p1} a {_qty(o, n + k)} en tout.", "ans": n + k, "level": lvl}


def _un(o) -> str:
    return "Une" if o[2] == "f" else "Un"


def _eur(n: int) -> str:
    return f"{n} euro" if n == 1 else f"{n} euros"


def p_achat(rng: random.Random) -> dict:
    """Prix unitaire × quantité, avec parfois le rendu de monnaie."""
    o = _obj(rng)
    prix = rng.randint(*o[3])
    n = rng.randint(2, 12)
    total = prix * n
    lvl = 2 if total <= 60 else 3
    rendu = rng.random() < 0.3 and total < 200
    if rendu:
        billet = next(b for b in (10, 20, 50, 100, 200) if b >= total)
        return {"q": (f"{_un(o)} {o[0]} coûte {_eur(prix)}. {rng.choice(PRENOMS)} achète "
                      f"{_qty(o, n)} et paie avec un billet de {billet} euros. "
                      f"Combien lui rend-on ?"),
                "steps": steps_mul(n, prix) + [f"Le total est {total} euros."]
                + steps_sub(billet, total),
                "a": f"On lui rend {billet - total} euros.", "ans": billet - total, "level": 3}
    q = rng.choice((f"{_un(o)} {o[0]} coûte {_eur(prix)}. Combien coûtent {_qty(o, n)} ?",
                    f"{_un(o)} {o[0]} coûte {_eur(prix)}. Quel est le prix de {_qty(o, n)} ?",
                    f"J'achète {_qty(o, n)} à {_eur(prix)} pièce. Combien je paie en tout ?"))
    return {"q": q, "steps": steps_mul(n, prix),
            "a": f"{_qty(o, n).capitalize()} coûtent {total} euros.", "ans": total, "level": lvl}


def p_partage(rng: random.Random) -> dict:
    o = _obj(rng)
    k = rng.choice((2, 3, 4, 5, 6, 8))
    part = rng.randint(2, 15)
    n = k * part
    q = rng.choice((f"On partage {_qty(o, n)} équitablement entre {k} enfants. "
                    f"Combien chaque enfant en reçoit-il ?",
                    f"{k} enfants se partagent {_qty(o, n)} à parts égales. "
                    f"Combien chacun en reçoit-il ?",
                    f"{n} {o[1]} pour {k} enfants, à parts égales : combien chacun ?"))
    return {"q": q, "steps": steps_div(n, k),
            "a": f"Chaque enfant reçoit {_qty(o, part)}.", "ans": part, "level": 2}


def p_double_moitie(rng: random.Random) -> dict:
    o = _obj(rng)
    p1, p2 = rng.sample(PRENOMS, 2)
    if rng.random() < 0.5:
        n = rng.randint(3, 40)
        if rng.random() < 0.35:
            # elliptique : « Le double de 16 ? » — le modèle v2 répondait au hasard
            q = rng.choice((f"Le double de {n} ?", f"Quel est le double de {n} ?",
                            f"Combien font deux fois {n} ?"))
            return {"q": q, "steps": [f"Le double de {n}, c'est {n} × 2 = {2 * n}."],
                    "a": f"Le double de {n} est {2 * n}.", "ans": 2 * n, "level": 1}
        return {"q": f"{p1} a {_qty(o, n)}. {p2} en a le double. Combien {p2} en a-t-il ?",
                "steps": [f"Le double de {n}, c'est {n} × 2 = {2 * n}."],
                "a": f"{p2} a {_qty(o, 2 * n)}.", "ans": 2 * n, "level": 1}
    n = 2 * rng.randint(2, 40)
    if rng.random() < 0.35:
        q = rng.choice((f"La moitié de {n} ?", f"Quelle est la moitié de {n} ?"))
        return {"q": q, "steps": [f"La moitié de {n}, c'est {n} ÷ 2 = {n // 2}."],
                "a": f"La moitié de {n} est {n // 2}.", "ans": n // 2, "level": 1}
    return {"q": f"{p1} a {_qty(o, n)}. {p2} en a la moitié. Combien {p2} en a-t-il ?",
            "steps": [f"La moitié de {n}, c'est {n} ÷ 2 = {n // 2}."],
            "a": f"{p2} a {_qty(o, n // 2)}.", "ans": n // 2, "level": 1}


def p_comparaison(rng: random.Random) -> dict:
    o = _obj(rng)
    p1, p2 = rng.sample(PRENOMS, 2)
    a, b = rng.randint(5, 90), rng.randint(5, 90)
    while a == b:
        b = rng.randint(5, 90)
    gagnant = p1 if a > b else p2
    return {"q": f"{p1} a {_qty(o, a)} et {p2} en a {b}. Qui en a le plus, et de combien ?",
            "steps": [f"On compare : {max(a, b)} > {min(a, b)}."] + steps_sub(max(a, b), min(a, b)),
            "a": f"C'est {gagnant} qui en a le plus, avec {abs(a - b)} de plus.",
            "ans": abs(a - b), "level": 2}


def p_suite(rng: random.Random) -> dict:
    pas = rng.choice((2, 3, 4, 5, 10, 11, 25))
    debut = rng.randint(1, 30)
    seq = [debut + i * pas for i in range(4)]
    nxt = seq[-1] + pas
    if rng.random() < 0.25:
        return {"q": f"Quel nombre vient après {seq[-1]} quand on compte de {pas} en {pas} ?",
                "steps": [f"{seq[-1]} + {pas} = {nxt}"],
                "a": f"Après {seq[-1]}, on trouve {nxt}.", "ans": nxt, "level": 1}
    q = rng.choice(("Complète la suite : " + ", ".join(map(str, seq)) + ", … ?",
                    "Quel est le nombre suivant : " + ", ".join(map(str, seq)) + " ?"))
    return {"q": q,
            "steps": [f"D'un nombre au suivant, on ajoute {pas}.",
                      f"{seq[-1]} + {pas} = {nxt}"],
            "a": f"Le nombre suivant est {nxt}.", "ans": nxt, "level": 2}


def p_conversion(rng: random.Random) -> dict:
    kind = rng.choice(("sem", "h", "eur", "diz"))
    if kind == "sem":
        n = rng.randint(2, 12)
        q = rng.choice((f"Combien y a-t-il de jours dans {n} semaines ?",
                        f"Convertis {n} semaines en jours.",
                        f"{n} semaines, ça fait combien de jours ?"))
        return {"q": q, "steps": ["Une semaine compte 7 jours."] + steps_mul(n, 7),
                "a": f"Il y a {n * 7} jours dans {n} semaines.", "ans": n * 7, "level": 2}
    if kind == "h":
        n = rng.randint(2, 10)
        q = rng.choice((f"Combien y a-t-il de minutes dans {n} heures ?",
                        f"{n} heures, ça fait combien de minutes ?"))
        return {"q": q, "steps": ["Une heure compte 60 minutes.", f"{n} × 60 = {n * 60}"],
                "a": f"Il y a {n * 60} minutes dans {n} heures.", "ans": n * 60, "level": 2}
    if kind == "eur":
        n = rng.randint(2, 20)
        return {"q": f"Combien de centimes font {n} euros ?",
                "steps": ["Un euro vaut 100 centimes.", f"{n} × 100 = {n * 100}"],
                "a": f"{n} euros font {n * 100} centimes.", "ans": n * 100, "level": 1}
    n = rng.randint(2, 9)
    return {"q": f"Combien font {n} dizaines ?",
            "steps": [f"Une dizaine vaut 10, donc {n} × 10 = {n * 10}."],
            "a": f"{n} dizaines font {n * 10}.", "ans": n * 10, "level": 1}


def p_syllogisme(rng: random.Random) -> dict:
    sing, plur, cat, cat_pl = rng.choice(ANIMAUX)
    noms_m = ("Pico", "Félix", "Nino", "Rex", "Titi")
    noms_f = ("Bella", "Mimi", "Luna")
    nom = rng.choice(noms_m + noms_f)
    il = "il" if nom in noms_m else "elle"
    return {"q": f"Tous les {plur} sont {cat_pl}. {nom} est un {sing}. "
                 f"{nom} est-{il} {cat} ?",
            "steps": [f"{nom} est un {sing}, et tous les {plur} sont {cat_pl}."],
            "a": f"Oui, {nom} est {cat}.", "ans": "oui", "level": 1}


def p_transitivite(rng: random.Random) -> dict:
    p1, p2, p3 = rng.sample(PRENOMS, 3)
    adj, plus, moins = rng.choice((
        ("âgé", "le plus âgé", "le plus jeune"),
        ("grand", "le plus grand", "le plus petit"),
        ("rapide", "le plus rapide", "le plus lent"),
    ))
    demande_max = rng.random() < 0.5
    cible = p1 if demande_max else p3
    return {"q": f"{p1} est plus {adj} que {p2}. {p2} est plus {adj} que {p3}. "
                 f"Qui est {plus if demande_max else moins} ?",
            "steps": [f"{p1} passe devant {p2}, et {p2} passe devant {p3}.",
                      f"L'ordre est donc : {p1}, puis {p2}, puis {p3}."],
            "a": f"C'est {cible} qui est {plus if demande_max else moins}.",
            "ans": cible, "level": 2}


def p_parite(rng: random.Random) -> dict:
    n = rng.randint(10, 999)
    pair = n % 2 == 0
    return {"q": f"{n} est-il un nombre pair ou impair ?",
            "steps": [f"Un nombre est pair si son dernier chiffre est 0, 2, 4, 6 ou 8. "
                      f"{n} se termine par {n % 10}."],
            "a": f"{n} est un nombre {'pair' if pair else 'impair'}.",
            "ans": "pair" if pair else "impair", "level": 1}


# --- v2.1 : les trous de distribution découverts en testant le modèle à mi-run ---
_MANGEABLES = {"pomme", "bonbon", "gâteau", "poire", "orange", "cerise", "noix", "œuf"}
# (1ère personne, 3ème personne) — la conjugaison change entre « je » et « il »
_V_PERTE = [("mange", "mange"), ("donne", "donne"), ("perds", "perd"), ("offre", "offre")]
_V_GAIN = [("trouve", "trouve"), ("reçois", "reçoit"), ("achète", "achète"),
           ("récupère", "récupère")]


def p_etat(rng: random.Random) -> dict:
    """Suivi d'état : « J'ai 15 pommes, j'en mange 7 et j'en récupère 3… ».

    2-3 opérations successives sur un même stock, en 1ère OU 3ème personne.
    Le modèle testé à mi-run additionnait les deux premiers nombres venus :
    il n'avait jamais vu un inventaire évoluer en plusieurs temps.
    """
    o = _obj(rng)
    je = rng.random() < 0.4
    p1 = rng.choice(PRENOMS)
    n_ops = rng.choice((2, 2, 3))
    cur = rng.randint(8, 40)
    depart = cur
    steps = [f"Au départ : {_qty(o, cur)}."]
    morceaux = []
    for i in range(n_ops):
        perte = rng.random() < 0.5 and cur > 3
        if perte:
            k = rng.randint(1, min(cur - 1, 20))
            vje, vil = rng.choice([v for v in _V_PERTE
                                   if v[0] != "mange" or o[0] in _MANGEABLES])
            steps += steps_sub(cur, k)
            cur -= k
        else:
            k = rng.randint(2, 20)
            vje, vil = rng.choice(_V_GAIN)
            steps += steps_add(cur, k)
            cur += k
        morceaux.append(f"j'en {vje} {k}" if je else f"en {vil} {k}")
    if je:
        recit = ", puis ".join(morceaux)
        fin = rng.choice((f"Combien {_de(o)} ai-je maintenant ?", "Combien j'en ai maintenant ?",
                          f"Combien {_de(o)} me reste-t-il ?"))
        q = f"J'ai {_qty(o, depart)}. {recit[0].upper()}{recit[1:]}. {fin}"
        a = f"J'ai maintenant {_qty(o, cur)}."
    else:
        il = _il(p1)
        recit = f"{il.capitalize()} " + ", puis ".join(morceaux)
        fin = rng.choice((f"Combien {_de(o)} a-t-{il} maintenant ?",
                          f"Combien {_de(o)} a-t-{il} à la fin ?"))
        q = f"{p1} a {_qty(o, depart)}. {recit}. {fin}"
        a = f"{p1} a maintenant {_qty(o, cur)}."
    return {"q": q, "steps": steps, "a": a, "ans": cur, "level": 2 if n_ops == 2 else 3}


def p_somme_longue(rng: random.Random) -> dict:
    """« Calcule : 1 + 2 + 3 + 4 + 100 » — réduction terme à terme d'une somme longue."""
    n_termes = rng.randint(4, 6)
    termes = [rng.randint(1, 20) for _ in range(n_termes)]
    if rng.random() < 0.5:
        termes[rng.randrange(n_termes)] = rng.choice((50, 100, 200))
    steps, cur = [], termes[0]
    for t in termes[1:]:
        steps.append(f"{cur} + {t} = {cur + t}")
        cur += t
    expr = " + ".join(map(str, termes))
    return {"q": f"Calcule : {expr}", "steps": steps,
            "a": f"{expr} = {cur}", "ans": cur, "level": 2 if n_termes <= 4 else 3}


_CONTENANTS = [("boîte", "boîtes"), ("paquet", "paquets"), ("sac", "sacs"),
               ("caisse", "caisses"), ("rangée", "rangées")]


def p_groupes(rng: random.Random) -> dict:
    """« 3 boîtes de 12 crayons » : reconnaître qu'un groupement = multiplication."""
    o = _obj(rng)
    c = rng.choice(_CONTENANTS)
    n = rng.randint(2, 9)
    m = rng.randint(3, 25)
    p1 = rng.choice(PRENOMS)
    forme = rng.random()
    if forme < 0.33:
        q = f"{p1} a {n} {c[1]} de {m} {o[1]}. Combien {_de(o)} a-t-{_il(p1)} en tout ?"
        a = f"{p1} a {n * m} {o[1]} en tout."
    elif forme < 0.66:
        q = f"Il y a {n} {c[1]} de {m} {o[1]}. Combien {_de(o)} y a-t-il en tout ?"
        a = f"Il y a {n * m} {o[1]} en tout."
    else:
        # « Chaque table a 4 chaises. Il y a 7 tables. » — la structure inversée
        # du bench OOD que le modèle v2 ratait
        q = (f"Chaque {c[0]} contient {m} {o[1]}. Il y a {n} {c[1]}. "
             f"Combien {_de(o)} en tout ?")
        a = f"Il y a {n * m} {o[1]} en tout."
    return {"q": q,
            "steps": [f"{n} {c[1]} de {m} {o[1]}, c'est {n} × {m}."] + steps_mul(n, m),
            "a": a, "ans": n * m, "level": 2}


def p_expr(rng: random.Random) -> dict:
    """« (2 * 50) / 2 » — deux opérations, parenthèses, symboles machine.

    Le modèle v2 partait en radotage GSM8K sur ce genre d'entrée : il n'avait
    jamais vu ni les parenthèses ni la priorité des opérations.
    """
    forme = rng.choice(("pm_mul", "pm_div", "mul_div"))
    if forme == "pm_mul":                       # (a ± b) × c
        c = rng.randint(2, 9)
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        if rng.random() < 0.5:
            inner, isym = a + b, "+"
        else:
            a, b = max(a, b), min(a, b)
            if a == b:
                a += 1
            inner, isym = a - b, "−"
        res = inner * c
        steps = [f"D'abord la parenthèse : {a} {isym} {b} = {inner}."] + steps_mul(inner, c)
        canon = f"({a} {isym} {b}) × {c}"
    elif forme == "pm_div":                     # (a + b) ÷ c, division exacte
        c = rng.randint(2, 9)
        q_ = rng.randint(2, 20)
        inner = c * q_
        a = rng.randint(1, inner - 1)
        b = inner - a
        res = q_
        steps = [f"D'abord la parenthèse : {a} + {b} = {inner}."] + steps_div(inner, c)
        canon = f"({a} + {b}) ÷ {c}"
    else:                                       # (a × b) ÷ c, c divise a
        c = rng.randint(2, 9)
        a = c * rng.randint(1, 10)
        b = rng.randint(2, 12)
        inner = a * b
        res = inner // c
        steps = steps_mul(a, b) + steps_div(inner, c)
        canon = f"({a} × {b}) ÷ {c}"
    vu = canon.replace("×", "*").replace("÷", "/").replace("−", "-") if rng.random() < 0.5 else canon
    q = rng.choice((f"Calcule : {vu}", f"{vu} = ?", f"{vu} =", f"Combien font {vu} ?"))
    return {"q": q, "steps": steps, "a": f"{canon} = {res}", "ans": res, "level": 3}


_CONTENANTS_EAU = ["baignoire", "piscine", "citerne", "cuve"]   # toutes féminines


def p_liquide(rng: random.Random) -> dict:
    """« Ma baignoire contient 300 litres. J'en vide 100, puis j'en ajoute 50. »

    Suivi d'état version liquides : unité « litres », verbes vider/ajouter,
    grands nombres ronds — exactement le test de la baignoire raté par la v2.
    """
    contenant = rng.choice(_CONTENANTS_EAU)
    cur = 10 * rng.randint(5, 50)
    depart = cur
    n_ops = rng.choice((2, 2, 3))
    steps = [f"Au départ : {cur} litres."]
    morceaux = []
    for _ in range(n_ops):
        perte = rng.random() < 0.5 and cur > 30
        if perte:
            k = 10 * rng.randint(1, (cur - 10) // 10)
            v = rng.choice(("vide", "retire"))
            steps += steps_sub(cur, k)
            cur -= k
        else:
            k = 10 * rng.randint(1, 15)
            v = rng.choice(("ajoute", "verse"))
            steps += steps_add(cur, k)
            cur += k
        morceaux.append(f"j'en {v} {k}")
    recit = ", puis ".join(morceaux)
    recit = recit[0].upper() + recit[1:]
    fin = rng.choice(("Combien de litres reste-t-il dedans ?",
                      "Combien de litres y a-t-il maintenant ?",
                      f"Combien de litres dans ma {contenant} ?"))
    q = f"Ma {contenant} contient {depart} litres d'eau. {recit}. {fin}"
    return {"q": q, "steps": steps,
            "a": f"Il y a maintenant {cur} litres dans la {contenant}.",
            "ans": cur, "level": 2 if n_ops == 2 else 3}


# --------------------------------------------------------------------------------------
# v4 : les familles du « jamais-enseigné ». Le bench OOD du 2026-08-21 a mesuré
# 1/10 sur les concepts absents du corpus (heures, fractions, « x fois plus »,
# périmètre, âges, arrondi de boîtes, énoncés inversés). On les enseigne.
# RÈGLE D'HYGIÈNE : le bench_ood actuel devient de fait in-distribution — les
# familles secrètes du futur bench_ood v2 ne doivent JAMAIS apparaître ici.
# --------------------------------------------------------------------------------------
_EN_LETTRES = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
               "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize"]


def p_heure(rng: random.Random) -> dict:
    """Heures qui avancent — passage de minuit au niveau 3."""
    lvl = rng.choice((1, 1, 2, 3))
    if lvl == 3:
        h = rng.randint(18, 23)
        d = rng.randint(25 - h, 11)          # force le passage de minuit
    else:
        h = rng.randint(6, 20)
        d = rng.randint(1, min(9, 23 - h))
    fin = (h + d) % 24
    duree = f"{d} heure" + ("s" if d > 1 else "")
    i = rng.randrange(4)
    q = [f"Il est {h} h. Quelle heure sera-t-il dans {duree} ?",
         f"Un film commence à {h} h et dure {duree}. À quelle heure se termine-t-il ?",
         f"Le train part à {h} h et le trajet dure {duree}. À quelle heure arrive-t-il ?",
         f"Le magasin ouvre à {h} h et reste ouvert {duree}. À quelle heure ferme-t-il ?"][i]
    a = [f"Il sera {fin} h.", f"Le film se termine à {fin} h.",
         f"Le train arrive à {fin} h.", f"Le magasin ferme à {fin} h."][i]
    steps = [f"{h} + {d} = {h + d}"]
    if h + d >= 24:
        steps.append(f"une journée compte 24 heures : {h + d} − 24 = {fin}")
    return {"q": q, "steps": steps, "a": a, "ans": fin, "level": lvl}


def p_par_jour(rng: random.Random) -> dict:
    """Débit quotidien × durée — et « une semaine = 7 jours »."""
    p = rng.choice(PRENOMS)
    n = rng.choice((2, 3, 4, 5, 6, 8, 10, 12))
    verbe, sg, plu = rng.choice((("lit", "page", "pages"), ("écrit", "ligne", "lignes"),
                                 ("plante", "graine", "graines"), ("ramasse", "coquillage", "coquillages")))
    en_semaine = rng.random() < 0.5
    j = 7 if en_semaine else rng.randint(3, 9)
    quand = "en une semaine" if en_semaine else f"en {j} jours"
    res = n * j
    q = rng.choice((f"{p} {verbe} {n} {plu if n > 1 else sg} par jour. Combien de {plu} {quand} ?",
                    f"Chaque jour, {p} {verbe} {n} {plu if n > 1 else sg}. Combien {quand} ?"))
    steps = (["une semaine compte 7 jours"] if en_semaine else []) + steps_mul(n, j)
    return {"q": q, "steps": steps, "a": f"Cela fait {res} {plu} {quand}.",
            "ans": res, "level": 2}


def p_age(rng: random.Random) -> dict:
    """Âges : année de naissance, « x ans de plus », somme d'âges."""
    p = rng.choice(PRENOMS)
    mode = rng.random()
    if mode < 0.45:                          # année de naissance
        naiss = rng.randint(1950, 2020)
        annee = rng.randint(max(naiss + 4, 2021), 2035)
        age = annee - naiss
        e = "" if p in PRENOMS_M else "e"
        q = rng.choice((f"{p} est né{e} en {naiss}. Quel âge a-t-{_il(p)} en {annee} ?",
                        f"{p} est né{e} en {naiss}. Quel est son âge en {annee} ?"))
        return {"q": q, "steps": [f"{annee} − {naiss} = {age}"],
                "a": f"{p} a {age} ans.", "ans": age, "level": 2}
    if mode < 0.75:                          # écart d'âge
        n, k = rng.randint(4, 60), rng.randint(1, 10)
        plus = rng.random() < 0.6
        lien, pos = rng.choice((("frère", "son"), ("sœur", "sa"), ("cousin", "son"),
                                ("cousine", "sa"), ("voisin", "son"), ("voisine", "sa")))
        res = n + k if plus else n - k
        que_lui = "que lui" if p in PRENOMS_M else "qu'elle"
        q = (f"{p} a {n} ans. {pos.capitalize()} {lien} a {k} ans de "
             f"{'plus' if plus else 'moins'} {que_lui}. Quel âge a {pos} {lien} ?")
        st = steps_add(n, k) if plus else steps_sub(n, k)
        return {"q": q, "steps": st, "a": f"{pos.capitalize()} {lien} a {res} ans.",
                "ans": res, "level": 2}
    x, y = rng.randint(25, 55), rng.randint(25, 55)   # somme d'âges
    duo = rng.choice((("Papa", "maman"), ("Grand-père", "grand-mère"), ("Le père", "la mère")))
    q = rng.choice((f"{duo[0]} a {x} ans et {duo[1]} a {y} ans. Quelle est la somme de leurs âges ?",
                    f"{duo[0]} a {x} ans et {duo[1]} a {y} ans. Quel âge ont-ils à eux deux ?"))
    return {"q": q, "steps": steps_add(x, y),
            "a": f"À eux deux, ils ont {x + y} ans.", "ans": x + y, "level": 2}


def p_fois_plus(rng: random.Random) -> dict:
    """« x fois plus / x fois moins » — multiplication déguisée en comparaison."""
    p1, p2 = rng.sample(PRENOMS, 2)
    o = _obj(rng)
    k, n = rng.randint(2, 6), rng.randint(2, 12)
    res = k * n
    if rng.random() < 0.25:                  # fois moins : division
        q = (f"{p1} a {k} fois moins {_de(o)} que {p2}. "
             f"{p2} en a {res}. Combien {p1} en a-t-{_il(p1)} ?")
        return {"q": q, "steps": steps_div(res, k),
                "a": f"{p1} a {_qty(o, n)}.", "ans": n, "level": 3}
    q = rng.choice((f"{p1} a {k} fois plus {_de(o)} que {p2}. "
                    f"{p2} en a {n}. Combien {p1} en a-t-{_il(p1)} ?",
                    f"{p2} a {_qty(o, n)}. {p1} en a {k} fois plus. "
                    f"Combien {_de(o)} a {p1} ?"))
    return {"q": q, "steps": steps_mul(k, n),
            "a": f"{p1} a {_qty(o, res)}.", "ans": res, "level": 2}


def p_fraction(rng: random.Random) -> dict:
    """Moitié, tiers, quart, trois quarts — toujours sur un total divisible."""
    label, den, num = rng.choice((("la moitié", 2, 1), ("le tiers", 3, 1),
                                  ("un tiers", 3, 1), ("le quart", 4, 1),
                                  ("un quart", 4, 1), ("les trois quarts", 4, 3)))
    unit = rng.randint(2, 15)
    total, res = den * unit, num * unit
    steps = [f"{total} ÷ {den} = {unit}"]
    if num > 1:
        steps.append(f"{unit} × {num} = {res}")
    if rng.random() < 0.5:                   # calcul pur
        q = rng.choice((f"Combien font {label} de {total} ?",
                        f"Quel est {label} de {total} ?"))
        return {"q": q, "steps": steps,
                "a": f"{label.capitalize()} de {total}, c'est {res}.", "ans": res, "level": 2}
    o = _obj(rng)                            # en contexte
    p = rng.choice(PRENOMS)
    q = rng.choice((f"{p} a {_qty(o, total)}. {_il(p).capitalize()} en donne {label}. "
                    f"Combien en donne-t-{_il(p)} ?",
                    f"Sur {_qty(o, total)}, {label} {'sont' if num * unit > 1 else 'est'} "
                    f"dans un panier. Combien dans le panier ?"))
    return {"q": q, "steps": steps, "a": f"Cela fait {_qty(o, res)}.",
            "ans": res, "level": 2}


def p_perimetre(rng: random.Random) -> dict:
    """Périmètre du carré et du rectangle."""
    if rng.random() < 0.6:
        c = rng.randint(2, 15)
        res = 4 * c
        q = rng.choice((f"Chaque côté d'un carré mesure {c} cm. Quel est son périmètre ?",
                        f"Quel est le périmètre d'un carré de {c} cm de côté ?"))
        return {"q": q, "steps": ["un carré a 4 côtés égaux", f"4 × {c} = {res}"],
                "a": f"Le périmètre est de {res} cm.", "ans": res, "level": 2}
    lo, la = sorted((rng.randint(2, 12), rng.randint(3, 15)))
    res = 2 * (lo + la)
    q = f"Un rectangle mesure {la} cm de long et {lo} cm de large. Quel est son périmètre ?"
    return {"q": q, "steps": ["le périmètre, c'est deux fois longueur plus largeur",
                              f"{la} + {lo} = {la + lo}", f"2 × {la + lo} = {res}"],
            "a": f"Le périmètre est de {res} cm.", "ans": res, "level": 3}


def p_monnaie(rng: random.Random) -> dict:
    """Argent : billets à compter, ou achat + monnaie rendue (deux opérations)."""
    if rng.random() < 0.4:
        k, val = rng.randint(2, 5), rng.choice((5, 10, 20, 50))
        res = k * val
        q = rng.choice((f"J'ai {k} billets de {val} euros. Combien d'argent ai-je en tout ?",
                        f"Dans ma tirelire il y a {k} billets de {val} euros. Quelle somme cela fait-il ?"))
        return {"q": q, "steps": steps_mul(k, val),
                "a": f"Cela fait {_eur(res)} en tout.", "ans": res, "level": 2}
    n, prix = rng.randint(2, 5), rng.randint(1, 5)
    cout = n * prix
    billet = min(b for b in (5, 10, 20, 50) if b > cout)
    rendu = billet - cout
    art = rng.choice((("croissant", "croissants"), ("cahier", "cahiers"),
                      ("stylo", "stylos"), ("pain au chocolat", "pains au chocolat")))
    q = (f"J'achète {n} {art[1]} à {_eur(prix)} pièce. Je paie avec un billet de "
         f"{billet} euros. " + rng.choice(("On me rend combien ?", "Combien me rend-on ?")))
    return {"q": q, "steps": steps_mul(n, prix) + steps_sub(billet, cout),
            "a": f"On me rend {_eur(rendu)}.", "ans": rendu, "level": 3}


def p_lettres(rng: random.Random) -> dict:
    """Nombres écrits en toutes lettres — « trois plus cinq »."""
    op = rng.choice(("plus", "plus", "moins", "fois"))
    if op == "fois":
        x, y = rng.randint(2, 6), rng.randint(2, 6)
        res, sym = x * y, "×"
    elif op == "moins":
        x, y = sorted((rng.randint(0, 16), rng.randint(0, 16)), reverse=True)
        res, sym = x - y, "−"
    else:
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        res, sym = x + y, "+"
    lx, ly = _EN_LETTRES[x], _EN_LETTRES[y]
    q = rng.choice((f"Combien font {lx} {op} {ly} ?",
                    f"Que vaut {lx} {op} {ly} ?",
                    f"{lx.capitalize()} {op} {ly}, ça fait combien ?"))
    return {"q": q,
            "steps": [f"{lx} = {x} et {ly} = {y}", f"{x} {sym} {y} = {res}"],
            "a": f"{lx.capitalize()} {op} {ly} font {res}.", "ans": res, "level": 1}


def p_inverse(rng: random.Random) -> dict:
    """Énoncé inversé : retrouver l'état INITIAL — le piège qui a coulé le bench."""
    o = _obj(rng)
    p = rng.choice(PRENOMS)
    il = _il(p)
    k, reste = rng.randint(2, 30), rng.randint(2, 60)
    if rng.random() < 0.5:                   # x − k = reste  ->  départ = reste + k
        depart = reste + k
        evt = rng.choice((f"en a perdu {k}", f"en a donné {k}", f"en a vendu {k}"))
        q = rng.choice((f"{p} avait des {o[1]}. {il.capitalize()} {evt}, et il lui en "
                        f"reste {reste}. Combien en avait-{il} au départ ?",
                        f"Après avoir perdu {k} {o[1]}, {p} en a encore {reste}. "
                        f"Combien en avait-{il} avant ?"))
        steps = [f"le nombre de départ a diminué de {k} pour arriver à {reste}",
                 f"on remonte : {reste} + {k} = {depart}"]
    else:                                    # x + k = total  ->  départ = total − k
        total = reste + k
        depart = reste
        q = rng.choice((f"Après avoir reçu {k} {o[1]}, {p} en a {total}. "
                        f"Combien en avait-{il} avant ?",
                        f"{p} a ramassé {k} {o[1]} et en possède maintenant {total}. "
                        f"Combien en avait-{il} au début ?"))
        steps = [f"le nombre de départ a augmenté de {k} pour arriver à {total}",
                 f"on redescend : {total} − {k} = {depart}"]
    return {"q": q, "steps": steps,
            "a": f"{p} en avait {depart} au départ.", "ans": depart, "level": 3}


def p_boites(rng: random.Random) -> dict:
    """Division « pour de vrai » : quand ça ne tombe pas juste, on arrondit AU-DESSUS."""
    taille = rng.choice((4, 6, 10, 12))
    obj, sg, plu, chacun = rng.choice((("œufs", "boîte", "boîtes", "chacune"),
                                       ("gâteaux", "paquet", "paquets", "chacun"),
                                       ("bouteilles", "carton", "cartons", "chacun")))
    k = rng.randint(3, 8)
    exact = rng.random() < 0.5
    besoin = taille * k + (0 if exact else rng.randint(1, taille - 1))
    res = k if exact else k + 1
    q = (f"Il me faut {besoin} {obj}. Les {plu} contiennent {taille} {obj} {chacun}. "
         f"Combien de {plu} dois-je acheter ?")
    if exact:
        steps = steps_div(besoin, taille)
    else:
        steps = [f"{taille} × {k} = {taille * k} : pas assez, il manque {besoin - taille * k} {obj}",
                 f"il faut donc {res} {plu} ({taille} × {res} = {taille * res}, assez pour {besoin})"]
    return {"q": q, "steps": steps, "a": f"Il faut acheter {res} {plu}.",
            "ans": res, "level": 3}


GENERATEURS = [
    (p_calc, 0.18), (p_reste, 0.10), (p_gain, 0.09), (p_achat, 0.09),
    (p_partage, 0.06), (p_double_moitie, 0.05), (p_comparaison, 0.06),
    (p_suite, 0.05), (p_conversion, 0.05), (p_syllogisme, 0.04),
    (p_transitivite, 0.04), (p_parite, 0.01),
    (p_etat, 0.08), (p_somme_longue, 0.05), (p_groupes, 0.05),
    (p_expr, 0.05), (p_liquide, 0.04),          # v2.2 : les trous du bench OOD
    # v4 : le jamais-enseigné (rng.choices normalise — part totale ~27%)
    (p_heure, 0.04), (p_par_jour, 0.03), (p_age, 0.04), (p_fois_plus, 0.04),
    (p_fraction, 0.05), (p_perimetre, 0.02), (p_monnaie, 0.04), (p_lettres, 0.03),
    (p_inverse, 0.06), (p_boites, 0.03),
]
_FNS = [g for g, _ in GENERATEURS]
_W = [w for _, w in GENERATEURS]


def make_problem(rng: random.Random) -> dict:
    return rng.choices(_FNS, weights=_W, k=1)[0](rng)


# --------------------------------------------------------------------------------------
# Rendu
# --------------------------------------------------------------------------------------
_CADRES = [  # variété de mises en forme pour le pré-entraînement
    ("Exercice : {q}\nSolution : {s}\n{a}", "\n"),
    ("Question : {q}\nRéponse : {s} {a}", " ; "),
    ("{q}\n{s}\n{a}", "\n"),
    ("Problème. {q}\nOn raisonne étape par étape : {s}\nConclusion : {a}", " ; "),
]


def render_pretrain(rng: random.Random, n_par_doc: int = 6) -> str:
    """Un « document » façon page de manuel : plusieurs exercices corrigés."""
    blocs = []
    for _ in range(rng.randint(2, n_par_doc)):
        pb = make_problem(rng)
        cadre, sep = rng.choice(_CADRES)
        blocs.append(cadre.format(q=pb["q"], s=sep.join(pb["steps"]), a=pb["a"]))
    return "\n\n".join(blocs)


def render_sft(rng: random.Random) -> list[dict]:
    """Une conversation : think court et correct, proportionnel à la difficulté.

    - niveau 1 : ~30 % de réponses SANS think (bloc vide) -> apprend qu'une question
      triviale n'exige pas de réflexion, et respecte le mode /think off ;
    - sinon : think = les étapes, rien de plus. Pas de remplissage rituel.
    """
    pb = make_problem(rng)
    sans_think = pb["level"] == 1 and rng.random() < 0.3
    if sans_think:
        reponse = f"{THINK}\n\n{THINK_END}\n{pb['a']}"
    else:
        pense = "\n".join(pb["steps"])
        reponse = f"{THINK}\n{pense}\n{THINK_END}\n{pb['a']}"
    return [{"role": "user", "text": pb["q"]}, {"role": "assistant", "text": reponse}]


# --------------------------------------------------------------------------------------
# Écriture jsonl (même format que data.download_source : {"t": ..., "m": ...})
# --------------------------------------------------------------------------------------
def write_jsonl(out_path: Path, char_budget: int, seed: int = 0, mode: str = "pretrain") -> dict:
    """Génère jusqu'au budget de caractères. Déterministe à graine fixée."""
    rng = random.Random(seed * 7_919 + (1 if mode == "sft" else 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs, n_chars = 0, 0
    with out_path.open("w", encoding="utf-8") as f:
        while n_chars < char_budget:
            if mode == "sft":
                msgs = render_sft(rng)
                text = "".join(m["text"] for m in msgs)
                rec = {"t": text, "m": msgs}
            else:
                text = render_pretrain(rng)
                rec = {"t": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_docs += 1
            n_chars += len(text)
    return {"source": f"synth-{mode}", "docs": n_docs, "chars": n_chars}


if __name__ == "__main__":
    # aperçu rapide : python synth.py
    rng = random.Random(0)
    print("=== PRETRAIN ===")
    print(render_pretrain(rng))
    print("\n=== SFT ===")
    for _ in range(3):
        for m in render_sft(rng):
            print(f"[{m['role']}] {m['text']}")
        print("---")
