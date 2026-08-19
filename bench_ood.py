# --------------------------------------------------------------------------------------
# Benchmark hors-distribution (OOD) : 40 problèmes écrits À LA MAIN, zéro gabarit de
# synth.py, + 8 faits inédits. Mesure la GÉNÉRALISATION, pas la maîtrise du format.
#
# Trois niveaux, score séparé :
#   reformulé    — mêmes maths, phrasé jamais vu à l'entraînement
#   contexte     — situations inédites (bus, gares, âges, lecture...)
#   concept      — notions jamais enseignées (× fois plus, tiers, heures, dates...)
#
# Usage : python bench_ood.py            (mêmes modèles que bench_vs.py, déjà en cache)
# --------------------------------------------------------------------------------------
import re
import time
from pathlib import Path

import torch

from bench_vs import CONCURRENTS, ModeleHF, NotreModele, dernier_nombre

ROOT = Path(__file__).parent

# (niveau, question, réponse attendue) — réponses vérifiées à la main.
PROBLEMES = [
    # ------------------------------------------------------------ reformulé
    ("reformulé", "Si j'ai 14 billes et que tu m'en donnes 9, combien en ai-je ?", "23"),
    ("reformulé", "Quel est le résultat de 45 moins 17 ?", "28"),
    ("reformulé", "Additionne 23 et 39.", "62"),
    ("reformulé", "Retire 8 de 30.", "22"),
    ("reformulé", "Chaque table a 4 chaises. Il y a 7 tables. Combien de chaises ?", "28"),
    ("reformulé", "Trois amis se partagent 27 bonbons à parts égales. Combien chacun en reçoit-il ?", "9"),
    ("reformulé", "Le double de 16 ?", "32"),
    ("reformulé", "La moitié de 90 ?", "45"),
    ("reformulé", "9 de plus que 37, ça fait combien ?", "46"),
    ("reformulé", "Quelle est la différence entre 80 et 46 ?", "34"),
    ("reformulé", "Quel nombre vient après 15 quand on compte de 5 en 5 ?", "20"),
    ("reformulé", "1000 moins 1 ?", "999"),
    ("reformulé", "Combien de fois 6 dans 42 ?", "7"),
    ("reformulé", "5 équipes de 11 joueurs. Combien de joueurs en tout ?", "55"),
    ("reformulé", "De 13 pour aller à 21, combien faut-il ajouter ?", "8"),
    # ------------------------------------------------------------ contexte neuf
    ("contexte", "J'avais 50 euros. Après avoir acheté un jeu à 34 euros, combien me reste-t-il ?", "16"),
    ("contexte", "Un bus transporte 28 passagers. À l'arrêt, 12 descendent et 5 montent. Combien de passagers restent dans le bus ?", "21"),
    ("contexte", "Sur 25 élèves, 11 sont des filles. Combien y a-t-il de garçons ?", "14"),
    ("contexte", "J'achète 3 croissants à 2 euros pièce. Je paie avec 10 euros. On me rend combien ?", "4"),
    ("contexte", "Un fermier a 12 poules. Chaque poule pond 2 œufs. Combien d'œufs au total ?", "24"),
    ("contexte", "Tom lit 10 pages par jour. Combien de pages lit-il en une semaine ?", "70"),
    ("contexte", "Il y a 24 heures dans une journée. Combien d'heures dans 3 journées ?", "72"),
    ("contexte", "Un livre fait 120 pages. J'en ai lu 45. Combien de pages me reste-t-il à lire ?", "75"),
    ("contexte", "Le train part avec 100 passagers. À la première gare, 23 descendent. À la deuxième, 41 descendent. Combien reste-t-il de passagers ?", "36"),
    ("contexte", "Emma a 7 ans. Son frère a 3 ans de plus qu'elle. Quel âge a son frère ?", "10"),
    ("contexte", "J'ai 15 pommes rouges et 8 pommes vertes. Combien de pommes en tout ?", "23"),
    ("contexte", "Nina a 40 stickers. Elle en colle la moitié. Combien lui en reste-t-il ?", "20"),
    ("contexte", "Un kilo de pommes coûte 3 euros. Combien coûtent 2 kilos ?", "6"),
    ("contexte", "Papa a 40 ans et maman a 38 ans. Quelle est la somme de leurs âges ?", "78"),
    ("contexte", "Un escargot avance de 2 mètres par jour. Quelle distance parcourt-il en 5 jours ?", "10"),
    # ------------------------------------------------------------ concept neuf
    ("concept", "Combien font trois plus cinq ?", "8"),
    ("concept", "Marie a 3 fois plus de livres que Paul. Paul en a 8. Combien Marie en a-t-elle ?", "24"),
    ("concept", "Il me faut 60 œufs. Les boîtes contiennent 12 œufs chacune. Combien de boîtes dois-je acheter ?", "5"),
    ("concept", "Un film commence à 20 h et dure 2 heures. À quelle heure se termine-t-il ?", "22"),
    ("concept", "Léo est né en 2010. Quel âge a-t-il en 2026 ?", "16"),
    ("concept", "Un carré a des côtés de 6 cm. Quel est son périmètre ?", "24"),
    ("concept", "Un crayon coûte 1 euro et une gomme coûte 2 euros. Combien coûtent 2 crayons et 1 gomme ?", "4"),
    ("concept", "Dans une classe de 30 élèves, un tiers porte des lunettes. Combien d'élèves portent des lunettes ?", "10"),
    ("concept", "Il est 15 h. Quelle heure sera-t-il dans 4 heures ?", "19"),
    ("concept", "J'ai deux billets de 20 euros. Combien d'argent ai-je en tout ?", "40"),
]

# faits inédits (complétion brute, notation regex + audit manuel)
FAITS = [
    ("La capitale de l'Espagne est", r"\bMadrid\b"),
    ("La capitale de l'Allemagne est", r"\bBerlin\b"),
    ("Une année compte", r"douze|12|365"),
    ("Le fromage est fabriqué à partir de", r"\blait\b"),
    ("La neige est de couleur", r"\bblanc"),
    ("Le miel est produit par les", r"\babeille"),
    ("En hiver, il fait généralement", r"\bfroid\b"),
    ("Un carré possède quatre", r"côtés|angles|sommets"),
]

NIVEAUX = ("reformulé", "contexte", "concept")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modeles = [NotreModele(ROOT / "runs" / "fr-v2", ROOT / "data-v2", device)]
    for repo, nom in CONCURRENTS:
        try:
            modeles.append(ModeleHF(repo, nom, device))
        except Exception as e:
            print(f"[!] {repo} indisponible ({e}) — ignoré.")

    print(f"[i] OOD : {len(PROBLEMES)} problèmes inédits "
          f"({', '.join(f'{n} {sum(1 for x in PROBLEMES if x[0] == n)}' for n in NIVEAUX)}) "
          f"· {len(FAITS)} faits inédits\n")

    lignes, details = [], []
    for m in modeles:
        t0 = time.time()
        par_niveau = {n: [0, 0] for n in NIVEAUX}  # [ok, total]
        for niveau, q, attendu in PROBLEMES:
            try:
                brut = m.repondre(q)
            except Exception as e:
                brut = f"<erreur : {e}>"
            rep = dernier_nombre(brut)
            bon = rep == attendu
            par_niveau[niveau][0] += bon
            par_niveau[niveau][1] += 1
            details.append((m.nom, niveau, q, attendu,
                            f"{rep!r} · texte : {brut.strip()[:200]}", bon))

        ok_faits = 0
        for amorce, motif in FAITS:
            try:
                gen = m.completer(amorce)
            except Exception:
                gen = ""
            bon = re.search(motif, gen, re.IGNORECASE) is not None
            ok_faits += bon
            details.append((m.nom, "fait", amorce, motif, gen.strip()[:100], bon))

        total_ok = sum(v[0] for v in par_niveau.values())
        scores = " · ".join(f"{n} {v[0]}/{v[1]}" for n, v in par_niveau.items())
        lignes.append((m.nom, par_niveau, total_ok, ok_faits))
        print(f"{m.nom:<42} {scores} · total {total_ok}/{len(PROBLEMES)} "
              f"· faits {ok_faits}/{len(FAITS)}  ({time.time()-t0:.0f}s)")

        if not isinstance(m, NotreModele):
            del m.model
            torch.cuda.empty_cache()

    rap = ROOT / "bench_ood_report.md"
    with rap.open("w", encoding="utf-8") as f:
        f.write("# Benchmark hors-distribution (problèmes 100% inédits)\n\n")
        f.write("| modèle | reformulé | contexte | concept | total | faits |\n")
        f.write("|---|---|---|---|---|---|\n")
        for nom, pn, tot, faits in lignes:
            cases = " | ".join(f"{pn[n][0]}/{pn[n][1]}" for n in NIVEAUX)
            f.write(f"| {nom} | {cases} | {tot}/{len(PROBLEMES)} | {faits}/{len(FAITS)} |\n")
        f.write("\n## Détails\n\n")
        for nom, niveau, q, attendu, obtenu, bon in details:
            f.write(f"- {'✅' if bon else '❌'} `{niveau}` **{nom}** — {q!r} → "
                    f"attendu {attendu!r}, obtenu {obtenu!r}\n")
    print(f"\n[i] rapport détaillé : {rap}")


if __name__ == "__main__":
    main()
