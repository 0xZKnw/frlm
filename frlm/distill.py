# --------------------------------------------------------------------------------------
# Distillation sequence-level depuis un gros teacher (Kimi via l'API Moonshot,
# compatible OpenAI). Idée : le teacher écrit des réponses À LA PORTÉE d'un 58M
# (think de 1-2 lignes, vocabulaire CE2), et notre verifier Python jette tout ce
# qui est faux, trop long ou pollué. Le "learnability gap" est esquivé par la
# contrainte de style, la qualité est garantie par le filtre — pas par le teacher.
#
# Trois familles de prompts :
#   maths   (~60 %) : problèmes de synth.py -> la réponse canonique `ans` est connue,
#                     le verifier tranche mécaniquement (comme au RL).
#   consigne (~25 %) : problème + instruction vérifiable ("Donne uniquement le
#                     nombre", "en 8 mots max"...) -> le checker de rl.py tranche.
#   chat    (~15 %) : questions libres (identité, explications, petites consignes
#                     de style) -> pas vérifiable, donc filtres d'hygiène stricts
#                     et quota volontairement bas.
#
# Sortie : data-v2/raw/distill.jsonl, même format que synth.write_jsonl
# ({"t": texte, "m": messages}) -> repris tel quel par data.encode_sft.
#
# Usage (clé plateforme Moonshot, format OpenAI) :
#   $env:MOONSHOT_API_KEY = 'sk-...'
#   python -m frlm.distill --list-models
#   python -m frlm.distill --n 20000 --model kimi-k3
# Usage (clé plan Kimi Code sk-kimi-..., format Anthropic) :
#   python -m frlm.distill --api anthropic --base-url https://api.kimi.com/anthropic --n 200
#
# Reprise : relancer la même commande — le script recharge le jsonl existant,
# déduplique sur l'énoncé et complète jusqu'à --n.
# --------------------------------------------------------------------------------------
import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from frlm import synth
from frlm.rl import _DECORS, instructions_possibles, verifier

# --------------------------------------------------------------------------------------
# Prompts de chat libre — le trou que synth.py ne sait pas boucher : formulations
# naturelles, identité, explications. Non vérifiables -> quota bas + hygiène stricte.
# --------------------------------------------------------------------------------------
CHAT_PROMPTS = [
    # identité / conversation
    "Bonjour !", "Salut, ça va ?", "Qui es-tu ?", "Tu sais faire quoi ?",
    "Tu peux m'aider ?", "Comment tu t'appelles ?", "Tu es un robot ?",
    "Merci beaucoup !", "Au revoir !", "Tu parles anglais ?",
    "Qu'est-ce que tu sais faire en maths ?", "Tu peux compter jusqu'à combien ?",
    # explications simples (niveau primaire)
    "C'est quoi une addition ?", "C'est quoi une soustraction ?",
    "C'est quoi une multiplication ?", "C'est quoi une division ?",
    "Explique-moi la différence entre pair et impair.",
    "À quoi servent les tables de multiplication ?",
    "C'est quoi le double d'un nombre ?", "C'est quoi la moitié d'un nombre ?",
    "Explique-moi ce qu'est un périmètre.", "C'est quoi un nombre entier ?",
    "Pourquoi 2 + 2 font 4 ?", "Comment on fait une soustraction avec retenue ?",
    "C'est quoi le plus grand : 100 ou 99 ?", "Explique-moi les euros et les centimes.",
    "Combien de jours dans une semaine ? Explique.",
    "Combien de minutes dans une heure ? Explique.",
    # méta / pédagogie
    "Pose-moi un problème de calcul.", "Donne-moi un exemple de problème de maths.",
    "Explique-moi comment tu réfléchis.", "Donne-moi une astuce pour calculer de tête.",
    "Comment apprendre ses tables de multiplication ?",
    "Donne-moi un conseil pour être bon en maths.",
    # petites consignes de style (non chiffrées)
    "Raconte une histoire en deux phrases.", "Donne trois animaux de la ferme.",
    "Écris une phrase avec le mot pomme.", "Donne trois couleurs.",
    "Cite deux fruits rouges.", "Fais une phrase avec le mot école.",
    "Donne les jours de la semaine.", "Cite trois métiers.",
    "Écris une phrase de politesse.", "Donne deux exemples de nombres pairs.",
    # variété conversationnelle
    "Coucou !", "Bonsoir !", "Hello", "Tu vas bien ?", "T'es qui toi ?",
    "C'est quoi ton nom ?", "Tu peux faire quoi exactement ?",
    "Aide-moi avec mes devoirs de maths.", "J'ai un exercice de maths.",
    "Je comprends rien aux divisions, aide-moi.", "Les maths c'est dur.",
    "T'es fort en calcul ?", "On peut parler de quoi ensemble ?",
    "Merci !", "Super, merci beaucoup !", "T'es trop fort !", "Bravo !",
    "C'est faux ce que tu dis.", "T'es sûr de ta réponse ?", "Pourquoi ?",
    "Explique encore une fois.", "J'ai pas compris.", "Répète s'il te plaît.",
    "Vas-y continue.", "Donne un autre exemple.", "Encore un !",
    # explications supplémentaires (niveau primaire)
    "C'est quoi un rectangle ?", "C'est quoi un triangle ?",
    "Combien de côtés a un carré ?", "C'est quoi une fraction ?",
    "Ça veut dire quoi « la somme » ?", "Ça veut dire quoi « la différence » ?",
    "C'est quoi le résultat d'une multiplication ?", "À quoi sert une division ?",
    "Comment on compte de 10 en 10 ?", "C'est quoi un nombre à deux chiffres ?",
    "Quel est le plus petit nombre à trois chiffres ?",
    "Ça fait combien une dizaine ?", "Ça fait combien une centaine ?",
    "Comment savoir si un nombre est pair ?",
    "Comment on calcule 9 fois quelque chose facilement ?",
    "C'est quoi une retenue en addition ?", "Comment poser une addition ?",
    "Combien de mois dans une année ?", "Combien de secondes dans une minute ?",
    "Quelle est la différence entre un mètre et un centimètre ?",
    "Combien de grammes dans un kilo ?", "C'est quoi un litre ?",
    # consignes de style variées
    "Réponds-moi en une seule phrase : c'est quoi une addition ?",
    "En un mot : le contraire de grand ?", "En un mot : le contraire de rapide ?",
    "Donne trois mots qui riment avec chat.", "Cite quatre saisons.",
    "Donne deux animaux qui volent.", "Cite trois moyens de transport.",
    "Écris une question sur les maths.", "Pose-moi une devinette simple.",
    "Fais une phrase avec le mot soleil.", "Fais une phrase très courte.",
    "Compte de 2 en 2 jusqu'à 10.", "Compte à rebours depuis 5.",
    "Donne les voyelles.", "Épelle le mot maison.",
    "Dis bonjour de trois façons différentes.",
    "Donne un exemple de nombre impair.", "Cite deux formes géométriques.",
    "Invente un prénom pour un chat.", "Donne un synonyme de content.",
]

SYSTEM_TEACHER = """Tu génères des données d'entraînement pour un TOUT PETIT modèle de langage français (58 millions de paramètres, niveau école primaire). Tu joues le rôle de "frlm", un petit assistant français open-source. Ne mentionne JAMAIS Kimi, Moonshot ni aucune autre IA.

Réponds TOUJOURS exactement dans ce format :
<think>
(raisonnement de 1 à 2 lignes MAXIMUM, très simple, vocabulaire niveau CE2 ; laisse cette partie vide si la question est triviale ou purement conversationnelle)
</think>
(la réponse finale : 1 phrase courte et simple — ou juste le nombre si la consigne le demande)

Règles strictes :
- Français uniquement. Jamais d'anglais, jamais de chinois.
- Phrases courtes, mots simples. Pas de « Attendez », pas de vérifications multiples, pas de retour en arrière.
- Le raisonnement montre le calcul (ex. : 12 - 5 = 7), rien de plus.
- La réponse finale ne répète pas le raisonnement.
- Si une consigne de format est donnée (un seul mot, majuscules, 8 mots max...), respecte-la À LA LETTRE dans la réponse finale."""

_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RE_CJK = re.compile(r"[぀-ヿ一-鿿]")
_RE_INTERDIT = re.compile(r"kimi|moonshot|openai|chatgpt|claude|anthropic|gpt", re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Construction des tâches
# --------------------------------------------------------------------------------------
def construire_taches(n: int, rng: random.Random, frac_instr: float, frac_chat: float):
    """Chaque tâche : {kind, q, check} — check(final) -> bool (None = hygiène seule)."""
    taches = []
    for _ in range(n):
        u = rng.random()
        if u < frac_chat:
            q = rng.choice(CHAT_PROMPTS)
            taches.append({"kind": "chat", "q": q, "check": None})
        elif u < frac_chat + frac_instr:
            pb = synth.make_problem(rng)
            poss = instructions_possibles(pb)
            instr, chk = rng.choice(poss)
            taches.append({"kind": "consigne", "q": f"{pb['q']} {instr}", "check": chk})
        else:
            pb = synth.make_problem(rng)
            g, d = rng.choice(_DECORS)
            taches.append({"kind": "maths", "q": f"{g}{pb['q']}{d}",
                           "check": (lambda f, a=pb["ans"]: verifier(a, f))})
    return taches


# --------------------------------------------------------------------------------------
# Appel API (Moonshot, compatible OpenAI) + filtrage
# --------------------------------------------------------------------------------------
def appeler(session, base_url, key, model, q, temperature, max_tokens,
            api="openai", essais=5):
    """Deux dialectes : "openai" (/chat/completions, Bearer) et "anthropic"
    (/v1/messages, x-api-key) — les clés Kimi Code (sk-kimi-...) parlent Anthropic."""
    if api == "anthropic":
        url = f"{base_url}/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        corps = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                 "system": SYSTEM_TEACHER,
                 "messages": [{"role": "user", "content": q}]}
    else:
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        corps = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                 "messages": [{"role": "system", "content": SYSTEM_TEACHER},
                              {"role": "user", "content": q}]}
    derniere_erreur = "aucun essai"
    for i in range(essais):
        try:
            r = session.post(url, json=corps, timeout=120, headers=headers)
            if r.status_code == 429 or r.status_code >= 500:
                derniere_erreur = f"HTTP {r.status_code} (réessai)"
                time.sleep(2.0 * (i + 1) + random.random())
                continue
            if r.status_code >= 400:
                # 4xx = permanent (clé, modèle, URL) : inutile de réessayer
                return None, {}, f"HTTP {r.status_code} : {r.text[:300]}"
            j = r.json()
            if api == "anthropic":
                texte = "".join(b.get("text", "") for b in j.get("content", [])
                                if b.get("type") == "text")
                u = j.get("usage", {})
                usage = {"prompt_tokens": u.get("input_tokens", 0),
                         "completion_tokens": u.get("output_tokens", 0)}
            else:
                texte = j["choices"][0]["message"]["content"] or ""
                usage = j.get("usage", {})
            return texte, usage, None
        except requests.RequestException as e:
            derniere_erreur = f"{type(e).__name__}: {e}"
            time.sleep(2.0 * (i + 1))
    return None, {}, derniere_erreur


def filtrer(brut: str, tache: dict, max_think: int, max_final: int):
    """-> (texte_assistant, None) si gardé, sinon (None, raison_du_rejet)."""
    m = _RE_THINK.search(brut)
    if not m:
        return None, "sans_think"
    think = m.group(1).strip()
    final = brut[m.end():].strip()
    if not final:
        return None, "final_vide"
    if _RE_CJK.search(brut):
        return None, "cjk"
    if _RE_INTERDIT.search(brut):
        return None, "marque_interdite"
    if len(think) > max_think or think.count("\n") > 2:
        return None, "think_trop_long"
    borne = max_final * (3 if tache["kind"] == "chat" else 1)
    if len(final) > borne:
        return None, "final_trop_long"
    if tache["check"] is not None and not tache["check"](final):
        return None, "reponse_fausse"
    corps_think = f"\n{think}\n" if think else "\n\n"
    return f"{synth.THINK}{corps_think}{synth.THINK_END}\n{final}", None


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Distillation Kimi -> jsonl SFT filtré")
    ap.add_argument("--n", type=int, default=20000, help="exemples GARDÉS visés")
    ap.add_argument("--model", default="kimi-k3")
    ap.add_argument("--base-url", default="https://api.moonshot.ai/v1")
    ap.add_argument("--api", choices=("openai", "anthropic"), default="openai",
                    help="dialecte de l'API ; clé Kimi Code (sk-kimi-...) -> anthropic "
                         "avec --base-url https://api.kimi.com/anthropic")
    ap.add_argument("--key-env", default="MOONSHOT_API_KEY")
    ap.add_argument("--out", default="data-v2/raw/distill.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-think", type=int, default=220, help="caractères max du think")
    ap.add_argument("--max-final", type=int, default=160)
    ap.add_argument("--frac-instr", type=float, default=0.25)
    ap.add_argument("--frac-chat", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"[!] variable {args.key_env} absente. "
                         f"PowerShell : $env:{args.key_env} = 'sk-...'")

    session = requests.Session()
    if args.list_models:
        if args.api == "anthropic":
            r = session.get(f"{args.base_url}/v1/models", timeout=30,
                            headers={"x-api-key": key,
                                     "anthropic-version": "2023-06-01"})
        else:
            r = session.get(f"{args.base_url}/models", timeout=30,
                            headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        for mdl in r.json().get("data", []):
            print(" ", mdl.get("id"))
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # reprise : recharge l'existant, dédupe sur l'énoncé utilisateur
    vus, gardes = set(), 0
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for ligne in f:
                try:
                    rec = json.loads(ligne)
                    vus.add(rec["m"][0]["text"])
                    gardes += 1
                except Exception:
                    pass
        print(f"[i] reprise : {gardes} exemples déjà dans {out}")

    rng = random.Random(args.seed + gardes)      # graine décalée à chaque reprise
    rejets = {}
    tok_in = tok_out = 0
    verrou = threading.Lock()
    t0 = time.time()
    apercus = 0

    def traiter(tache):
        brut, usage, err = appeler(session, args.base_url, key, args.model, tache["q"],
                                   args.temperature, args.max_tokens, api=args.api)
        if brut is None:
            return tache, None, "erreur_api", usage, err
        texte, raison = filtrer(brut, tache, args.max_think, args.max_final)
        return tache, texte, raison, usage, None

    print(f"[i] cible {args.n} gardés · {args.model} · {args.workers} workers · "
          f"mix maths/consigne/chat = "
          f"{1 - args.frac_instr - args.frac_chat:.0%}/{args.frac_instr:.0%}/{args.frac_chat:.0%}")

    with out.open("a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        while gardes < args.n:
            lot = [t for t in construire_taches(min(64, (args.n - gardes) * 2), rng,
                                                args.frac_instr, args.frac_chat)
                   if t["q"] not in vus or t["kind"] == "chat"]
            if not lot:
                continue
            gardes_avant, derniere_erreur = gardes, None
            futs = [pool.submit(traiter, t) for t in lot]
            for fut in as_completed(futs):
                tache, texte, raison, usage, err = fut.result()
                if err:
                    derniere_erreur = err
                with verrou:
                    tok_in += usage.get("prompt_tokens", 0)
                    tok_out += usage.get("completion_tokens", 0)
                    if texte is None:
                        rejets[raison] = rejets.get(raison, 0) + 1
                        continue
                    if tache["q"] in vus and tache["kind"] != "chat":
                        continue
                    vus.add(tache["q"])
                    rec = {"t": tache["q"] + texte,
                           "m": [{"role": "user", "text": tache["q"]},
                                 {"role": "assistant", "text": texte}],
                           "k": tache["kind"]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()          # le fichier reflète le progrès en temps réel
                    gardes += 1
                    if apercus < 3:
                        apercus += 1
                        print(f"\n--- aperçu {apercus} [{tache['kind']}] ---\n"
                              f"U: {tache['q']}\nA: {texte}\n")
                    if gardes % 25 == 0:
                        dt = time.time() - t0
                        vit = gardes / max(dt, 1e-9)
                        rej = sum(rejets.values())
                        eta = (args.n - gardes) / max(vit, 1e-9)
                        print(f"  {gardes}/{args.n} gardés · {rej} rejetés "
                              f"({100 * gardes / max(1, gardes + rej):.0f}% de rendement) · "
                              f"{vit * 60:.0f}/min · tokens {tok_in / 1e3:.0f}k in "
                              f"{tok_out / 1e3:.0f}k out · ETA {eta / 60:.0f} min")
                if gardes >= args.n:
                    for f2 in futs:
                        f2.cancel()      # annule ce qui n'a pas encore démarré
                    break
            # bilan de lot : plus jamais de boucle muette quand tout échoue
            if gardes == gardes_avant:
                detail = " · ".join(f"{k} {v}" for k, v in rejets.items())
                print(f"  [!] lot de {len(lot)} : 0 gardé ({detail or 'aucune réponse'})")
                if derniere_erreur and rejets.get("erreur_api", 0) >= len(lot):
                    raise SystemExit(
                        f"[!] tous les appels API échouent — dernière erreur :\n"
                        f"    {derniere_erreur}\n"
                        f"    Vérifie --model (kimi-for-coding / k3-256k), --base-url "
                        f"et la clé dans {args.key_env}.")

    print(f"\n[✓] {gardes} exemples dans {out} en {(time.time() - t0) / 60:.1f} min")
    print(f"    tokens : {tok_in:,} in · {tok_out:,} out")
    if rejets:
        print("    rejets :", " · ".join(f"{k} {v}" for k, v in
                                         sorted(rejets.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
