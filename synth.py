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
    return f"d'{pl}" if pl[0] in "aeiouéèêh" else f"de {pl}"


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
# Chacun renvoie un dict : {q, steps, a, level}
#   q     : l'énoncé (question)
#   steps : les étapes rédigées (list[str]) — deviendront le <think> en SFT
#   a     : la phrase de réponse finale
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
    return {"q": f"Calcule : {a} {sym} {b}", "steps": st,
            "a": f"{a} {sym} {b} = {res}", "level": lvl}


def p_reste(rng: random.Random) -> dict:
    """« Léa a 24 billes, elle en donne 9… » — soustraction en contexte."""
    o = _obj(rng)
    p1, p2 = rng.sample(PRENOMS, 2)
    lvl = rng.choice((1, 2, 2, 3))
    n = rng.randint(8, {1: 20, 2: 60, 3: 300}[lvl])
    k = rng.randint(2, n - 1)
    verbe = rng.choice((f"en donne {k} à {p2}", f"en perd {k}", f"en offre {k} à {p2}"))
    il = _il(p1).capitalize()
    return {"q": f"{p1} a {_qty(o, n)}. {il} {verbe}. Combien {_de(o)} lui reste-t-il ?",
            "steps": [f"{p1} part avec {n} {o[1]} et en {'perd' if 'perd' in verbe else 'donne'} {k}."]
            + steps_sub(n, k),
            "a": f"Il lui reste {_qty(o, n - k)}.", "level": lvl}


def p_gain(rng: random.Random) -> dict:
    """Addition en contexte, parfois en deux temps (gagne puis re-gagne)."""
    o = _obj(rng)
    p1 = rng.choice(PRENOMS)
    lvl = rng.choice((1, 2, 2, 3))
    hi = {1: 15, 2: 50, 3: 250}[lvl]
    n, k = rng.randint(3, hi), rng.randint(2, hi)
    deux_temps = lvl >= 2 and rng.random() < 0.4
    il = _il(p1)
    if deux_temps:
        j = rng.randint(2, 30)
        q = (f"{p1} a {_qty(o, n)}. Le matin, on lui en donne {k}, "
             f"et l'après-midi encore {j}. Combien {_de(o)} a-t-{il} maintenant ?")
        st = steps_add(n, k) + steps_add(n + k, j)
        return {"q": q, "steps": st, "a": f"{p1} a maintenant {_qty(o, n + k + j)}.",
                "level": 3}
    return {"q": f"{p1} a {_qty(o, n)} et en reçoit {k} de plus. Combien {_de(o)} a-t-{il} en tout ?",
            "steps": steps_add(n, k),
            "a": f"{p1} a {_qty(o, n + k)} en tout.", "level": lvl}


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
        return {"q": (f"Un {o[0]} coûte {prix} euros. {rng.choice(PRENOMS)} achète "
                      f"{_qty(o, n)} et paie avec un billet de {billet} euros. "
                      f"Combien lui rend-on ?"),
                "steps": steps_mul(n, prix) + [f"Le total est {total} euros."]
                + steps_sub(billet, total),
                "a": f"On lui rend {billet - total} euros.", "level": 3}
    return {"q": f"Un {o[0]} coûte {prix} euros. Combien coûtent {_qty(o, n)} ?",
            "steps": steps_mul(n, prix),
            "a": f"{_qty(o, n).capitalize()} coûtent {total} euros.", "level": lvl}


def p_partage(rng: random.Random) -> dict:
    o = _obj(rng)
    k = rng.choice((2, 3, 4, 5, 6, 8))
    part = rng.randint(2, 15)
    n = k * part
    return {"q": f"On partage {_qty(o, n)} équitablement entre {k} enfants. "
                 f"Combien chaque enfant en reçoit-il ?",
            "steps": steps_div(n, k),
            "a": f"Chaque enfant reçoit {_qty(o, part)}.", "level": 2}


def p_double_moitie(rng: random.Random) -> dict:
    o = _obj(rng)
    p1, p2 = rng.sample(PRENOMS, 2)
    n = rng.randint(3, 40)
    if rng.random() < 0.5:
        return {"q": f"{p1} a {_qty(o, n)}. {p2} en a le double. Combien {p2} en a-t-il ?",
                "steps": [f"Le double de {n}, c'est {n} × 2 = {2 * n}."],
                "a": f"{p2} a {_qty(o, 2 * n)}.", "level": 1}
    n = 2 * rng.randint(2, 40)
    return {"q": f"{p1} a {_qty(o, n)}. {p2} en a la moitié. Combien {p2} en a-t-il ?",
            "steps": [f"La moitié de {n}, c'est {n} ÷ 2 = {n // 2}."],
            "a": f"{p2} a {_qty(o, n // 2)}.", "level": 1}


def p_comparaison(rng: random.Random) -> dict:
    o = _obj(rng)
    p1, p2 = rng.sample(PRENOMS, 2)
    a, b = rng.randint(5, 90), rng.randint(5, 90)
    while a == b:
        b = rng.randint(5, 90)
    gagnant = p1 if a > b else p2
    return {"q": f"{p1} a {_qty(o, a)} et {p2} en a {b}. Qui en a le plus, et de combien ?",
            "steps": [f"On compare : {max(a, b)} > {min(a, b)}."] + steps_sub(max(a, b), min(a, b)),
            "a": f"C'est {gagnant} qui en a le plus, avec {abs(a - b)} de plus.", "level": 2}


def p_suite(rng: random.Random) -> dict:
    pas = rng.choice((2, 3, 4, 5, 10, 11, 25))
    debut = rng.randint(1, 30)
    seq = [debut + i * pas for i in range(4)]
    nxt = seq[-1] + pas
    return {"q": "Complète la suite : " + ", ".join(map(str, seq)) + ", … ?",
            "steps": [f"D'un nombre au suivant, on ajoute {pas}.",
                      f"{seq[-1]} + {pas} = {nxt}"],
            "a": f"Le nombre suivant est {nxt}.", "level": 2}


def p_conversion(rng: random.Random) -> dict:
    kind = rng.choice(("sem", "h", "eur", "diz"))
    if kind == "sem":
        n = rng.randint(2, 12)
        return {"q": f"Combien y a-t-il de jours dans {n} semaines ?",
                "steps": ["Une semaine compte 7 jours."] + steps_mul(n, 7),
                "a": f"Il y a {n * 7} jours dans {n} semaines.", "level": 2}
    if kind == "h":
        n = rng.randint(2, 10)
        return {"q": f"Combien y a-t-il de minutes dans {n} heures ?",
                "steps": ["Une heure compte 60 minutes.", f"{n} × 60 = {n * 60}"],
                "a": f"Il y a {n * 60} minutes dans {n} heures.", "level": 2}
    if kind == "eur":
        n = rng.randint(2, 20)
        return {"q": f"Combien de centimes font {n} euros ?",
                "steps": ["Un euro vaut 100 centimes.", f"{n} × 100 = {n * 100}"],
                "a": f"{n} euros font {n * 100} centimes.", "level": 1}
    n = rng.randint(2, 9)
    return {"q": f"Combien font {n} dizaines ?",
            "steps": [f"Une dizaine vaut 10, donc {n} × 10 = {n * 10}."],
            "a": f"{n} dizaines font {n * 10}.", "level": 1}


def p_syllogisme(rng: random.Random) -> dict:
    sing, plur, cat, cat_pl = rng.choice(ANIMAUX)
    noms_m = ("Pico", "Félix", "Nino", "Rex", "Titi")
    noms_f = ("Bella", "Mimi", "Luna")
    nom = rng.choice(noms_m + noms_f)
    il = "il" if nom in noms_m else "elle"
    return {"q": f"Tous les {plur} sont {cat_pl}. {nom} est un {sing}. "
                 f"{nom} est-{il} {cat} ?",
            "steps": [f"{nom} est un {sing}, et tous les {plur} sont {cat_pl}."],
            "a": f"Oui, {nom} est {cat}.", "level": 1}


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
            "a": f"C'est {cible} qui est {plus if demande_max else moins}.", "level": 2}


def p_parite(rng: random.Random) -> dict:
    n = rng.randint(10, 999)
    pair = n % 2 == 0
    return {"q": f"{n} est-il un nombre pair ou impair ?",
            "steps": [f"Un nombre est pair si son dernier chiffre est 0, 2, 4, 6 ou 8. "
                      f"{n} se termine par {n % 10}."],
            "a": f"{n} est un nombre {'pair' if pair else 'impair'}.", "level": 1}


# --- v2.1 : les trous de distribution découverts en testant le modèle à mi-run ---
_MANGEABLES = {"pomme", "bonbon", "gâteau"}
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
        q = f"J'ai {_qty(o, depart)}. {recit[0].upper()}{recit[1:]}. Combien {_de(o)} ai-je maintenant ?"
        a = f"J'ai maintenant {_qty(o, cur)}."
    else:
        il = _il(p1)
        recit = f"{il.capitalize()} " + ", puis ".join(morceaux)
        q = f"{p1} a {_qty(o, depart)}. {recit}. Combien {_de(o)} a-t-{il} maintenant ?"
        a = f"{p1} a maintenant {_qty(o, cur)}."
    return {"q": q, "steps": steps, "a": a, "level": 2 if n_ops == 2 else 3}


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
            "a": f"{expr} = {cur}", "level": 2 if n_termes <= 4 else 3}


_CONTENANTS = [("boîte", "boîtes"), ("paquet", "paquets"), ("sac", "sacs"),
               ("caisse", "caisses"), ("rangée", "rangées")]


def p_groupes(rng: random.Random) -> dict:
    """« 3 boîtes de 12 crayons » : reconnaître qu'un groupement = multiplication."""
    o = _obj(rng)
    c = rng.choice(_CONTENANTS)
    n = rng.randint(2, 9)
    m = rng.randint(3, 25)
    p1 = rng.choice(PRENOMS)
    if rng.random() < 0.5:
        q = f"{p1} a {n} {c[1]} de {m} {o[1]}. Combien {_de(o)} a-t-{_il(p1)} en tout ?"
        a = f"{p1} a {n * m} {o[1]} en tout."
    else:
        q = f"Il y a {n} {c[1]} de {m} {o[1]}. Combien {_de(o)} y a-t-il en tout ?"
        a = f"Il y a {n * m} {o[1]} en tout."
    return {"q": q,
            "steps": [f"{n} {c[1]} de {m} {o[1]}, c'est {n} × {m}."] + steps_mul(n, m),
            "a": a, "level": 2}


GENERATEURS = [
    (p_calc, 0.18), (p_reste, 0.10), (p_gain, 0.09), (p_achat, 0.09),
    (p_partage, 0.06), (p_double_moitie, 0.05), (p_comparaison, 0.06),
    (p_suite, 0.05), (p_conversion, 0.05), (p_syllogisme, 0.04),
    (p_transitivite, 0.04), (p_parite, 0.01),
    (p_etat, 0.08), (p_somme_longue, 0.05), (p_groupes, 0.05),
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
