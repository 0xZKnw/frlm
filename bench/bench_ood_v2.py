# --------------------------------------------------------------------------------------
# Benchmark hors-distribution v2 — familles SECRÈTES.
#
# ⚠️  RÈGLE D'HYGIÈNE ABSOLUE : aucun de ces énoncés, ni aucune famille structurellement
#     équivalente, ne doit JAMAIS entrer dans synth.py ni dans un corpus d'entraînement
#     (pretrain, mid, SFT, RL, distillation). Le jour où une famille d'ici est enseignée,
#     elle est morte pour le bench — il faudra un v3.
#
# Pourquoi un v2 : les familles du bench v1 (heures, âges, fractions, périmètres,
# monnaie, ×-fois-plus, boîtes...) ont été ajoutées à synth.py pour le corpus v4 —
# le v1 mesure désormais de la révision, pas de la généralisation.
#
# Sept familles réellement inédites (jamais générées par synth.py, v1 comme v4) :
#   transitif    — ordres relationnels A>B, B>C : qui est le plus X ? (réponse = prénom)
#   piège        — info absente ou hors sujet : le modèle sait-il dire « 0 » ou
#                  « on ne peut pas savoir » au lieu d'halluciner un chiffre ?
#   intervalle   — fencepost : bornes incluses, poteaux, coupes de ruban
#   branches     — comparer DEUX calculs et choisir (réponse = camp gagnant)
#   reste        — division euclidienne : ce qui RESTE (boîtes/synth = quotient, jamais le reste)
#   cycle        — jours de la semaine, arithmétique modulaire en mots
#   composition  — opérations imbriquées en une phrase (le double du double...)
# + faits v2     — complétions factuelles inédites (aucun recouvrement avec v1/bench_vs)
#
# Protocole : notre modèle SFT joue en format chat natif ; les modèles de base jouent
# en few-shot 4 exemples (dont un exemple « réponse en mot » et un « on ne sait pas »,
# pour que le protocole soit connu de tous) ; les modèles instruct via leur chat template.
#
# Usage :
#   python -m bench.bench_ood_v2 --run fr-v4 --data-dir data-v4          # tout
#   python -m bench.bench_ood_v2 --run gpu1 --data-dir data-gpu1 --hf none
#   --hf : all (défaut) | none (nos modèles seuls)
# --------------------------------------------------------------------------------------
import json
import re
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_vs import ModeleHF, NotreModele, dernier_nombre  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# Concurrents — même classe de poids uniquement (~124-270M). Chaque entrée :
# (repo, nom, genre) avec genre "base" (few-shot) ou "instruct" (chat template).
# Les indisponibles sont ignorés proprement (modèles gated type Gemma : accepter
# la licence sur HF + hf auth login). NB : LFM2.5 exige transformers récent
# (arch hybride conv+GQA) — pip install -U transformers si "model type lfm2".
# --------------------------------------------------------------------------------------
CONCURRENTS_V2 = [
    # l'ancienne garde 2020-2021, même classe de poids
    ("asi/gpt-fr-cased-small", "GPT-fr small · 124M (Inria 2021)", "base"),
    ("antoinelouis/belgpt2", "BelGPT-2 · 124M (60 Go fr)", "base"),
    ("dbddv01/gpt2-french-small", "GPT-2 fr · 124M (transfert)", "base"),
    # les 2025-2026 de même taille — le plafond moderne du format ~250M
    ("google/gemma-3-270m", "Gemma 3 · 270M (gated)", "base"),
    ("LiquidAI/LFM2.5-230M-Base", "LFM2.5 Base · 230M (28T tok)", "base"),
    ("LiquidAI/LFM2.5-230M", "LFM2.5 · 230M (RL, 28T tok)", "instruct"),
]

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# type "num"   : résultat numérique unique dans la conclusion finale
# type "choix" : option unique dans une conclusion finale non ambiguë
# type "regex" : motif présent dans la réponse finale
PROBLEMES = [
    # ------------------------------------------------------------ transitif
    dict(cat="transitif", q="Léa est plus grande que Max. Max est plus grand que Zoé. Qui est le plus grand des trois ?",
         type="choix", options=["Léa", "Max", "Zoé"], attendu="Léa"),
    dict(cat="transitif", q="Paul court plus vite que Jules. Anna court plus vite que Paul. Qui est le plus rapide ?",
         type="choix", options=["Anna", "Paul", "Jules"], attendu="Anna"),
    dict(cat="transitif", q="Le chien est plus lourd que le chat. Le chat est plus lourd que le lapin. Qui est le plus léger ?",
         type="choix", options=["chien", "chat", "lapin"], attendu="lapin"),
    dict(cat="transitif", q="Marc est plus âgé que Lise. Lise est plus âgée que Tom. Qui est le plus jeune ?",
         type="choix", options=["Marc", "Lise", "Tom"], attendu="Tom"),
    dict(cat="transitif", q="La tour A est plus haute que la tour B. La tour C est plus basse que la tour B. Quelle tour est la plus haute ?",
         type="choix", options=["A", "B", "C"], attendu="A"),
    dict(cat="transitif", q="Zoé a plus de billes qu'Émile, et Émile en a plus que Sami. Qui en a le moins ?",
         type="choix", options=["Zoé", "Émile", "Sami"], attendu="Sami"),
    # ------------------------------------------------------------ piège
    dict(cat="piège", q="J'ai 5 pommes et 3 oranges dans mon panier. Combien y a-t-il de bananes dans le panier ?",
         type="regex", attendu=r"\b(0|zéro|aucune?|pas de banane)"),
    dict(cat="piège", q="Un train roule à 80 km/h pendant 2 heures. Quel âge a le conducteur ?",
         type="regex", attendu=r"(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)"),
    dict(cat="piège", q="Marie a 12 ans. Son chat s'appelle Félix. Quel âge a le chat ?",
         type="regex", attendu=r"(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)"),
    dict(cat="piège", q="J'achète une baguette à 1 euro et un croissant à 2 euros. Combien coûte le journal ?",
         type="regex", attendu=r"(ne (peux|peut|sait)|sais pas|impossible|pas possible|ne le dit pas|pas précisé|pas indiqué|inconnu|aucune information|ne permet pas)"),
    dict(cat="piège", q="Dans un sac, il y a 10 billes rouges. Combien de billes bleues y a-t-il dans le sac ?",
         type="regex", attendu=r"\b(0|zéro|aucune?|pas de bille bleue)"),
    dict(cat="piège", q="Le boulanger vend 30 croissants le matin et 20 l'après-midi. Combien de baguettes vend-il ?",
         type="regex", attendu=r"\b(0|zéro|aucune?)\b|(ne (peux|peut|sait)|sais pas|impossible|ne le dit pas|pas précisé|inconnu)"),
    # ------------------------------------------------------------ intervalle (fencepost)
    dict(cat="intervalle", q="Combien y a-t-il de nombres entiers de 4 à 9, en comptant 4 et 9 ?",
         type="num", attendu="6"),
    dict(cat="intervalle", q="Une clôture droite de 12 mètres a un poteau tous les 3 mètres, avec un poteau à chaque bout. Combien de poteaux ?",
         type="num", attendu="5"),
    dict(cat="intervalle", q="Je lis du chapitre 3 au chapitre 8 inclus. Combien de chapitres vais-je lire ?",
         type="num", attendu="6"),
    dict(cat="intervalle", q="Un immeuble a des étages numérotés de 0 à 6. Combien d'étages différents l'ascenseur dessert-il ?",
         type="num", attendu="7"),
    dict(cat="intervalle", q="Un escalier a 10 marches. Je suis sur la 4e marche. Combien de marches me reste-t-il à monter ?",
         type="num", attendu="6"),
    dict(cat="intervalle", q="On coupe un ruban en 5 morceaux. Combien de coupes faut-il faire ?",
         type="num", attendu="4"),
    # ------------------------------------------------------------ branches
    dict(cat="branches", q="Qu'est-ce qui coûte le plus cher : 3 stylos à 2 euros pièce, ou 2 cahiers à 4 euros pièce ?",
         type="choix", options=["cahier", "stylo"], attendu="cahier"),
    dict(cat="branches", q="Qui a le plus de fruits : Ali avec 4 sacs de 3 pommes, ou Bea avec 2 sacs de 7 pommes ?",
         type="choix", options=["Ali", "Bea"], attendu="Bea"),
    dict(cat="branches", q="Lise calcule 6 × 5. Nino calcule 4 × 8. Qui obtient le plus grand résultat ?",
         type="choix", options=["Lise", "Nino"], attendu="Nino"),
    dict(cat="branches", q="Un pack de 6 bouteilles coûte 12 euros. Une bouteille seule coûte 3 euros. Qu'est-ce qui est le moins cher par bouteille ?",
         type="choix", options=["pack", "seule"], attendu="pack"),
    dict(cat="branches", q="Tom a 25 billes et en gagne 5. Léo a 40 billes et en perd 15. Qui a le plus de billes maintenant ?",
         type="choix", options=["Tom", "Léo"], attendu="Tom"),
    # ------------------------------------------------------------ reste (division euclidienne)
    dict(cat="reste", q="On partage 23 bonbons entre 4 enfants, chacun en reçoit autant. Combien de bonbons restent ?",
         type="num", attendu="3"),
    dict(cat="reste", q="J'ai 17 œufs et des boîtes de 6. Je remplis uniquement des boîtes complètes. Combien d'œufs ne sont pas en boîte ?",
         type="num", attendu="5"),
    dict(cat="reste", q="Un ruban de 20 cm est coupé en morceaux de 6 cm. Quelle longueur de ruban reste-t-il, en cm ?",
         type="num", attendu="2"),
    dict(cat="reste", q="31 élèves montent dans des voitures de 4 places. Toutes les voitures sont pleines sauf la dernière. Combien d'élèves dans la dernière voiture ?",
         type="num", attendu="3"),
    dict(cat="reste", q="Il y a 50 chaises à ranger en rangées de 8. Combien de chaises ne forment pas une rangée complète ?",
         type="num", attendu="2"),
    # ------------------------------------------------------------ cycle (jours de la semaine)
    dict(cat="cycle", q="Nous sommes mardi. Quel jour serons-nous dans 3 jours ?",
         type="choix", options=JOURS, attendu="vendredi"),
    dict(cat="cycle", q="Hier, c'était dimanche. Quel jour serons-nous demain ?",
         type="choix", options=JOURS, attendu="mardi"),
    dict(cat="cycle", q="Nous sommes samedi. Quel jour étions-nous il y a 2 jours ?",
         type="choix", options=JOURS, attendu="jeudi"),
    dict(cat="cycle", q="Nous sommes vendredi. Quel jour serons-nous dans 7 jours ?",
         type="choix", options=JOURS, attendu="vendredi"),
    dict(cat="cycle", q="Nous sommes jeudi. Quel jour serons-nous dans 4 jours ?",
         type="choix", options=JOURS, attendu="lundi"),
    # ------------------------------------------------------------ composition
    dict(cat="composition", q="Quel est le double du double de 5 ?", type="num", attendu="20"),
    dict(cat="composition", q="Quelle est la moitié de la somme de 6 et 10 ?", type="num", attendu="8"),
    dict(cat="composition", q="Ajoute 3 au produit de 4 et 5. Combien ?", type="num", attendu="23"),
    dict(cat="composition", q="Soustrais 2 de la moitié de 18. Combien ?", type="num", attendu="7"),
    dict(cat="composition", q="Quel est le triple de la différence entre 9 et 4 ?", type="num", attendu="15"),
    dict(cat="composition", q="Prends 10, ajoute 5, puis divise le tout par 3. Combien ?", type="num", attendu="5"),
    dict(cat="composition", q="Quelle est la somme du double de 3 et du double de 4 ?", type="num", attendu="14"),
]

# faits inédits v2 — aucun recouvrement avec FAITS de bench_vs ni du bench v1
FAITS = [
    ("La capitale du Portugal est", r"\bLisbonne\b"),
    ("La capitale de la Belgique est", r"\bBruxelles\b"),
    ("Le mois de février compte généralement", r"\b28\b|vingt-huit"),
    ("Le contraire de gauche est la", r"\bdroite\b"),
    ("Les poissons vivent dans l'", r"\beau\b"),
    ("L'auteur des Misérables est", r"\bHugo\b"),
    ("Un vélo a deux", r"\broues?\b"),
    ("La Terre tourne autour du", r"\b[Ss]oleil\b"),
    ("Le premier mois de l'année est", r"\bjanvier\b"),
    ("Les feuilles des arbres sont généralement de couleur", r"\bvert"),
    ("Pour écrire, on utilise un stylo ou un", r"\bcrayon\b|\bclavier\b|\bfeutre\b|\bstylet\b"),
    ("Le lait provient principalement de la", r"\bvache\b"),
]

CATS = ("transitif", "piège", "intervalle", "branches", "reste", "cycle", "composition")

# Protocole few-shot pour les modèles de base : 4 exemples qui montrent les FORMATS
# de réponse possibles (nombre, mot, aveu d'ignorance) sans dévoiler aucune famille.
FEWSHOT_V2 = (
    "Question : Calcule : 12 + 7\nRéponse : 19\n\n"
    "Question : Léo est plus grand que Max. Qui est le plus grand ?\nRéponse : Léo\n\n"
    "Question : Un bus roule pendant 3 heures. Quel âge a le chauffeur ?\n"
    "Réponse : On ne peut pas le savoir.\n\n"
    "Question : Nous sommes lundi. Quel jour serons-nous demain ?\nRéponse : mardi\n\n"
)


# --------------------------------------------------------------------------------------
def _sans_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _dernier_nombre_v2(texte: str) -> str | None:
    """Comme dernier_nombre, mais « 0,5 » est UN nombre (≠ « 5 ») : pas de point
    gratuit quand le modèle répond une décimale dont la partie fractionnaire
    coïncide avec l'attendu."""
    t = texte.replace(" ", " ").replace(" ", " ")
    nombres = re.findall(r"\d+(?: \d{3})*(?:,\d+)?", t)
    if not nombres:
        return None
    return nombres[-1].replace(" ", "")


def _nettoyer_sortie(texte: str) -> str:
    texte = re.sub(r"<\|(?:im_start|im_end|endoftext)\|>", "\n", texte)
    texte = re.sub(r"</?think>", "\n", texte, flags=re.IGNORECASE)
    return texte.replace("\r", "\n").strip()


def _zone_finale(texte: str) -> tuple[str, bool]:
    """Isole une conclusion explicite, ou à défaut la dernière proposition.

    Le benchmark ne récompense plus un nom ou un nombre seulement cité dans un
    raisonnement contradictoire. Le booléen indique la présence d'un marqueur de
    conclusion explicite.
    """
    texte = _nettoyer_sortie(texte)
    marqueurs = list(re.finditer(
        r"(?is)\b(?:réponse|reponse|conclusion)\s*[:：=-]",
        texte,
    ))
    if marqueurs:
        zone = texte[marqueurs[-1].end():].strip()
        if zone:
            return zone, True
    lignes = [ligne.strip() for ligne in texte.splitlines() if ligne.strip()]
    zone = lignes[-1] if lignes else texte
    propositions = [part.strip() for part in re.split(r"(?<=[.!?])\s+", zone)
                    if part.strip()]
    zone = propositions[-1] if propositions else zone
    donc = re.match(r"(?is)^donc\b\s*[:,]?\s*", zone)
    return (zone[donc.end():] if donc else zone), bool(donc)


def _option_mentions(texte: str, options: list[str]) -> list[str]:
    normalise = _sans_accents(texte.casefold())
    return [option for option in options if re.search(
        r"(?<!\w)" + re.escape(_sans_accents(option.casefold())) + r"(?!\w)",
        normalise,
    )]


def noter_strict(prob: dict, texte: str) -> dict:
    """Retourne correct/incorrect/manual_review avec extraction auditable."""
    zone, explicite = _zone_finale(texte)
    if prob["type"] == "num":
        nombres = re.findall(r"(?<!\d)\d+(?:[ \u00a0\u202f]\d{3})*(?:,\d+)?(?!\d)", zone)
        uniques = list(dict.fromkeys(n.replace(" ", "").replace("\u00a0", "")
                                    .replace("\u202f", "") for n in nombres))
        if len(uniques) != 1:
            return {"status": "manual_review" if uniques else "incorrect",
                    "extracted": repr(uniques), "zone": zone,
                    "reason": "numeric_final_ambiguous" if uniques else "numeric_final_missing"}
        rep = uniques[0]
        return {"status": "correct" if rep == prob["attendu"] else "incorrect",
                "extracted": repr(rep), "zone": zone, "reason": "numeric_final_unique"}
    if prob["type"] == "regex":
        ok = re.search(prob["attendu"], zone, re.IGNORECASE) is not None
        return {"status": "correct" if ok else "incorrect",
                "extracted": zone[:60], "zone": zone, "reason": "regex_final"}

    mentions = _option_mentions(zone, prob["options"])
    if len(mentions) != 1:
        return {"status": "manual_review" if len(mentions) > 1 else "incorrect",
                "extracted": repr(mentions), "zone": zone,
                "reason": "choice_final_ambiguous" if mentions else "choice_final_missing"}
    rep = mentions[0]
    conclusion_naturelle = bool(re.search(
        r"(?i)(?:c['’]est|il s['’]agit de|\best\s+(?:le|la)\s+plus|\bobtient\s+le\s+plus)",
        zone,
    ))
    sortie_courte = len(zone.split()) <= 4
    if not (explicite or conclusion_naturelle or sortie_courte):
        return {"status": "manual_review", "extracted": repr(rep), "zone": zone,
                "reason": "choice_without_conclusion"}
    return {"status": "correct" if rep == prob["attendu"] else "incorrect",
            "extracted": repr(rep), "zone": zone, "reason": "choice_final_unique"}


def noter(prob: dict, texte: str) -> tuple[bool, str]:
    """Compatibilité : seul le statut strict `correct` vaut un point automatique."""
    result = noter_strict(prob, texte)
    return result["status"] == "correct", result["extracted"]


def noter_fait(motif: str, texte: str) -> dict:
    """Note seulement la première proposition d'une complétion factuelle."""
    propre = _nettoyer_sortie(texte)
    premiere = re.split(r"(?:\n|(?<=[.!?])\s+)", propre, maxsplit=1)[0].strip()
    ok = re.search(motif, premiere, re.IGNORECASE) is not None
    return {"status": "correct" if ok else "incorrect", "extracted": premiere[:100],
            "zone": premiere, "reason": "fact_first_proposition"}


class ModeleHFInstruct(ModeleHF):
    """Concurrent instruct : passe par son chat template, pas par le few-shot."""

    @torch.no_grad()
    def repondre(self, question: str, max_new: int = 160) -> str:
        msgs = [{"role": "user", "content": question}]
        # return_dict=True : selon la version de transformers, sans lui le retour
        # est tantôt un tenseur nu, tantôt un BatchEncoding sans .shape
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_dict=True,
                                           return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id or 0)
        n_in = enc["input_ids"].shape[1]
        return self.tok.decode(out[0, n_in:], skip_special_tokens=True)


def _patch_fewshot(m):
    """Impose le few-shot v2 (4 exemples multi-formats) aux modèles de base."""
    if isinstance(m, ModeleHFInstruct):
        return
    if isinstance(m, ModeleHF):
        orig = m._greedy

        def repondre(question, max_new=80):
            gen = orig(FEWSHOT_V2 + f"Question : {question}\nRéponse :", max_new)
            return gen.split("Question")[0]
        m.repondre = repondre
    elif getattr(m, "phase", "sft") != "sft":
        tok, sp, model, device = m.tok, m.sp, m.model, m.device

        @torch.no_grad()
        def repondre(question, max_new=80):
            texte = FEWSHOT_V2 + f"Question : {question}\nRéponse :"
            ids = torch.tensor([tok.encode(texte).ids], device=device)
            out = model.generate(ids, max_new_tokens=max_new, temperature=0.0,
                                 repetition_penalty=1.0, stop_ids=(sp["eot"],))
            gen = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=False)
            return gen.split("Question")[0]
        m.repondre = repondre


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Benchmark hors-distribution v2 (familles secrètes)")
    ap.add_argument("--run", default="fr-v4", help="nom du run (dossier dans runs/)")
    ap.add_argument("--data-dir", default="data-v4")
    ap.add_argument("--hf", default="all", choices=["all", "none"],
                    help="concurrents HF : tous, ou aucun (nos modèles seuls)")
    ap.add_argument("--stage", default=None,
                    help="phase à charger (sft, rl, rlaif, rlaif2...) ; "
                         "défaut = 1re trouvée parmi sft/mid/pretrain")
    ap.add_argument("--ckpt", default=None,
                    help="fichier précis (ckpt_best.pt / ckpt_latest.pt) ; "
                         "défaut = best puis latest")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    roster = CONCURRENTS_V2 if a.hf == "all" else []

    n_cat = {c: sum(1 for p in PROBLEMES if p['cat'] == c) for c in CATS}
    print(f"[i] OOD v2 : {len(PROBLEMES)} problèmes "
          f"({', '.join(f'{c} {n}' for c, n in n_cat.items())}) · {len(FAITS)} faits\n")

    lignes, details, raw_records = [], [], []

    # Un SEUL modèle en VRAM à la fois : chargé juste avant son tour, libéré juste
    # après. Charger tout le roster d'avance (comme le bench v1) sature les 8 Go
    # dès qu'un rival ≥0.5B est là, et fait planter TOUS les tours suivants en
    # silence (CUBLAS_STATUS_ALLOC_FAILED) — vécu le 2026-08-21.
    def evaluer(m):
        t0 = time.time()
        par_cat = {c: [0, 0] for c in CATS}
        reviews = {c: 0 for c in CATS}
        err_affichee = False   # 1re exception montrée en clair, les suivantes avalées
        for problem_index, prob in enumerate(PROBLEMES):
            try:
                brut = m.repondre(prob["q"])
            except Exception as e:
                brut = f"<erreur : {e}>"
                if not err_affichee:
                    import traceback
                    print(f"[!] {m.nom} — exception sur {prob['q']!r} :")
                    traceback.print_exc()
                    err_affichee = True
            scoring = noter_strict(prob, brut)
            bon = scoring["status"] == "correct"
            rep = scoring["extracted"]
            par_cat[prob["cat"]][0] += bon
            par_cat[prob["cat"]][1] += 1
            reviews[prob["cat"]] += int(scoring["status"] == "manual_review")
            details.append((m.nom, prob["cat"], prob["q"], str(prob["attendu"]),
                            f"{rep} · statut : {scoring['status']} · "
                            f"zone : {scoring['zone'][:100]} · texte : {brut.strip()[:200]}", bon))
            raw_records.append({
                "model": m.nom, "section": "reasoning", "item_index": problem_index,
                "category": prob["cat"], "prompt": prob["q"],
                "expected": str(prob["attendu"]), "extracted": rep,
                "raw_answer": brut, "auto_correct": bool(bon),
                "auto_status": scoring["status"], "auto_reason": scoring["reason"],
                "final_zone": scoring["zone"],
            })

        ok_faits = 0
        for fact_index, (amorce, motif) in enumerate(FAITS):
            try:
                gen = m.completer(amorce)
            except Exception:
                gen = ""
                if not err_affichee:
                    import traceback
                    print(f"[!] {m.nom} — exception sur le fait {amorce!r} :")
                    traceback.print_exc()
                    err_affichee = True
            scoring = noter_fait(motif, gen)
            bon = scoring["status"] == "correct"
            ok_faits += bon
            details.append((m.nom, "fait", amorce, motif,
                            f"{scoring['extracted']} · texte : {gen.strip()[:100]}", bon))
            raw_records.append({
                "model": m.nom, "section": "facts", "item_index": fact_index,
                "category": "fait", "prompt": amorce, "expected": motif,
                "extracted": scoring["extracted"], "raw_answer": gen,
                "auto_correct": bool(bon), "auto_status": scoring["status"],
                "auto_reason": scoring["reason"], "final_zone": scoring["zone"],
            })

        total_ok = sum(v[0] for v in par_cat.values())
        scores = " · ".join(f"{c} {v[0]}/{v[1]}" for c, v in par_cat.items())
        lignes.append((m.nom, par_cat, total_ok, ok_faits, reviews))
        print(f"{m.nom:<44} {scores}\n{'':<44} total {total_ok}/{len(PROBLEMES)} "
              f"· faits {ok_faits}/{len(FAITS)}  ({time.time()-t0:.0f}s)")

        del m.model
        if device == "cuda":
            torch.cuda.empty_cache()

    notre = NotreModele(ROOT / "runs" / a.run, ROOT / a.data_dir, device,
                        stage=a.stage, ckpt_name=a.ckpt)
    _patch_fewshot(notre)
    evaluer(notre)

    for repo, nom, genre in roster:
        try:
            cls = ModeleHFInstruct if genre == "instruct" else ModeleHF
            m = cls(repo, nom, device)
            _patch_fewshot(m)
        except Exception as e:
            print(f"[!] {repo} indisponible ({type(e).__name__}: {str(e)[:80]}) — ignoré.")
            if device == "cuda":
                torch.cuda.empty_cache()   # un chargement avorté peut laisser des miettes
            continue
        evaluer(m)

    # suffixe explicite : sans lui, un bench d'un ckpt RL écraserait le rapport
    # de référence du SFT (bench_ood_v2_fr-v4.md).
    suff = ""
    if a.stage:
        suff += "_" + a.stage
    if a.ckpt:
        suff += "_" + a.ckpt.replace("ckpt_", "").replace(".pt", "")
    rap = ROOT / "bench" / "reports" / f"bench_ood_v2_{a.run}{suff}.md"
    rap.parent.mkdir(parents=True, exist_ok=True)
    with rap.open("w", encoding="utf-8") as f:
        f.write("# Benchmark hors-distribution v2 (familles secrètes)\n\n")
        f.write("⚠️ Ces énoncés ne doivent jamais entrer dans un corpus d'entraînement.\n\n")
        f.write("| modèle | " + " | ".join(CATS) + " | total | faits |\n")
        f.write("|---" * (len(CATS) + 3) + "|\n")
        for nom, pc, tot, faits, reviews in lignes:
            cases = " | ".join(f"{pc[c][0]}/{pc[c][1]}" for c in CATS)
            f.write(f"| {nom} | {cases} | **{tot}/{len(PROBLEMES)}** | {faits}/{len(FAITS)} |\n")
            pending = sum(reviews.values())
            if pending:
                f.write(f"\n> {pending} réponse(s) `manual_review` comptée(s) fausse(s) "
                        "dans le total automatique strict.\n")
        f.write("\n## Détails\n\n")
        for nom, cat, q, attendu, obtenu, bon in details:
            f.write(f"- {'✅' if bon else '❌'} `{cat}` **{nom}** — {q!r} → "
                    f"attendu {attendu!r}, obtenu {obtenu!r}\n")
    raw_path = rap.with_suffix(".raw.json")
    raw_path.write_text(json.dumps({
        "schema": "frlm-ood-v2-raw-2-strict", "created_unix": time.time(),
        "run": a.run, "stage": a.stage, "checkpoint": a.ckpt,
        "records": raw_records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[i] rapport détaillé : {rap}")
    print(f"[i] réponses brutes pour correction manuelle : {raw_path}")


if __name__ == "__main__":
    main()
