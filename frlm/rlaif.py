"""rlaif.py — GRPO à juge LLM : la phase « usage quotidien ».

Pourquoi une phase de plus après le RL vérifiable ?
  Le RL de rl.py (50 steps, maths synth uniquement) a fiabilisé le calcul mais
  RÉTRÉCI la politique : entropie 0.377 -> 0.077 nats, et « Salut, qui es-tu ? »
  répondait « On me donne un petit nombre : 12 ». La KL ne protège que ce qu'elle
  échantillonne — un RL mono-tâche laisse dériver tout le reste. Et le vérificateur
  Python est aveugle sur la qualité : « 24 IMAGES EN MITÉ » (charabia conforme)
  était récompensé.

Les deux remèdes, ensemble :
  1. un POOL de prompts variés (data-v4/rlaif_prompts.jsonl) : calculs du
     quotidien, questions-pièges sans réponse (-> « je ne sais pas »), faits,
     chat/identité, consignes — plus un flux synth (synth_frac) pour garder
     l'ancrage maths. Tout ce qu'on veut préserver doit être échantillonné.
  2. un JUGE LLM (Claude, via fichiers) qui note ce que Python ne voit pas :
     think qui dérive vraiment vs décoratif, entités inventées, français propre.

Récompense d'un rollout : partie Python (selon le type, voir _py_reward)
  + judge_weight × score_juge (dans [0, 1]). L'avantage reste relatif au groupe
  (GRPO inchangé) ; l'échelle par type n'a donc aucune importance inter-groupe.

Leçons du run 1 (100 steps, 22/08) intégrées ici :
  - **think-fleuve** : sur un problème hors de portée, le modèle part en
    divagation façon R1 (« D'accord, voyons voir… Hmm ») qui invente des entités,
    sature la fenêtre et ne conclut JAMAIS. C'était noté 0 par le juge mais restait
    « gratuit » côté Python -> overlong reward shaping (rl.py) + pénalité de
    radotage ci-dessous, pour que ça devienne le pire choix du groupe.
  - **groupes muets** : jusqu'à 4 groupes sur 6 étaient unanimes (souvent 6
    rollouts au texte IDENTIQUE), donc jetés après avoir coûté génération ET
    jugement. Dynamic sampling (DAPO) : un groupe dont tous les textes sont
    identiques est mathématiquement muet (le juge leur donnera le même score) ->
    on le remplace AVANT de déranger le juge.
  - **doublons dans un même step** : le tirage uniforme avec remise a sorti deux
    fois le même prompt dans un même lot -> tirage en « paquet de cartes »
    mélangé, sans remise, qui garantit aussi que tout le pool est vu.

Protocole juge (répertoire runs/<run>/rlaif/judge/) :
  - le trainer écrit  pending_step_N.jsonl  : une ligne JSON par groupe
      {"gi", "type", "q", "ans", "attendu", "note",
       "rollouts": [{"i", "final", "texte", "py"}]}
    puis attend (poll 1 s, timeout judge_timeout -> ckpt + sortie propre) ;
  - le juge écrit  scores_step_N.json  :
      {"step": N, "scores": [[s, ...] × group_size] × nb_groupes}   (ordre gi,
    scores dans [0, 1], jusqu'à -1.5 pour annuler un faux positif de la regex
    py). Fichier partiel/invalide -> le trainer re-poll.
  Barème juge (affiné au fil du run 1, écrit aussi dans judge/RUBRIQUE.md) :
    1.00  réponse juste + think utile (ou vide légitime) + français propre
    0.95  juste mais think vide là où un calcul aurait aidé
    0.85  juste avec défaut mineur (verbosité, think décoratif sans rapport)
    0.55  juste mais l'entité dérive (« 63 Go de peinture ») ou consigne ratée
    0.30  bon premier pas d'un problème à deux opérations, arrêté en route
    0.15  vrai mais hors sujet, ou refus d'une question parfaitement répondable
    0.05  invention assurée, réponse qui contredit son propre think
    0.00  charabia, entités fabriquées, think-fleuve sans réponse finale
  Hiérarchie spécifique aux PIÈGES (question sans réponse possible) :
    0.85  refus explicite (« je ne peux pas savoir, l'énoncé ne le dit pas »)
    0.80  refus + redirection utile (« ça dépend de X, peux-tu préciser ? »)
    0.50  écho neutre qui n'invente rien (« La soupe est encore chaude. »)
    0.40  reconnaissance implicite de l'inconnue (« Sa mère a X ans »)
    0.05  invention chiffrée assurée (« Il lui reste 350 kilomètres »)
  RÈGLE DURE : un rollout qui n'émet aucune réponse après </think> vaut 0, quelle
  que soit la qualité apparente du raisonnement — on ne récompense jamais la
  divagation. Et le score juge doit corriger les faux positifs/négatifs de la
  regex Python (elle attrape un nombre juste noyé dans du charabia, et rate une
  bonne réponse qui se termine par un nombre de l'énoncé).

Sondes anti-rétrécissement : à chaque éval, 8 prompts fixes (chat, piège, calc,
fait) générés en greedy et déposés dans judge/probe_step_N.txt — à lire pour
vérifier que le quotidien ne dérive pas pendant qu'on optimise le reste.

Affichage : lignes simples (pas de rich.Live) — la phase est pilotée par un
agent qui lit des logs, pas un humain devant un tty.

Usage :  python run.py rlaif --run fr-v4                (reprendre : --resume)
         départ par défaut : runs/<run>/rl/ckpt_best.pt (--init-stage sft si le
         RL vérifiable a trop rétréci la politique), ancre KL = ce même ckpt.
"""

from __future__ import annotations

import json
import random
import re
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from frlm import config_from_dict, model_from_cfg
from frlm import data as D
from frlm import synth
from frlm.rl import (RLConfig, RLTrainer, _mots, dernier_nombre, partie_finale,
                     verifier)


# --------------------------------------------------------------------------------------
# Vérificateurs Python par type de prompt
# --------------------------------------------------------------------------------------
_RE_REFUS = re.compile(
    r"je ne (sais|peux)|ne (peut|permet) pas|impossible de (savoir|dire|répondre)"
    r"|pas (précisé|indiqué|possible de savoir)|aucune information|l'énoncé ne"
    r"|on ne (peut|sait) pas|je l'ignore|\binconnu", re.IGNORECASE)


def _chk_nombre_seul(final: str, ans) -> bool:
    return len(_mots(final)) == 1 and dernier_nombre(final) == str(int(ans))


def _chk_un_mot(final: str, ans) -> bool:
    return len(_mots(final)) == 1 and verifier(ans, final)


def _chk_oui_non(final: str, ans) -> bool:
    return " ".join(_mots(final)).lower() == str(ans).lower()


def _chk_max_mots(final: str, ans) -> bool:
    return len(_mots(final)) <= 8 and verifier(ans, final)


CHECKS = {"nombre_seul": _chk_nombre_seul, "un_mot": _chk_un_mot,
          "oui_non": _chk_oui_non, "max_mots": _chk_max_mots}


def _repetition(bloc: str, n: int = 4) -> float:
    """Part de n-grammes répétés d'UN bloc de texte, ramenée dans [0, 1].

    Calibré sur les sorties réelles du run 1 (unicité des 4-grammes mesurée) :
        1.000  explication propre, histoire correcte, réponse de calcul normale
        0.940  l'explication de l'eau qui répète « éliminer les toxines du corps »
        0.286  la boucle dure (« Le chat dort sur le canapé. » × 4)
    D'où le seuil à 0.95 et la pénalité pleine à 0.45 : un défaut mineur coûte
    quelques centièmes, une boucle coûte tout."""
    mots = re.findall(r"\w+", bloc.lower())
    if len(mots) < 20:              # trop court pour qu'une reprise soit significative
        return 0.0
    grams = [tuple(mots[i:i + n]) for i in range(len(mots) - n + 1)]
    if not grams:
        return 0.0
    uniq = len(set(grams)) / len(grams)
    return min(1.0, max(0.0, (0.95 - uniq) / 0.50))


def penalite_radotage(txt: str, n: int = 4) -> float:
    """Radotage du rollout : le pire des deux blocs, brouillon et réponse.

    Séparer les deux est indispensable. Le modèle produit très souvent un think qui
    RECOPIE la réponse finale (« <think>Un triangle a trois côtés.</think>Un triangle
    a trois côtés. ») : mesurée d'un bloc, cette réponse juste et concise affichait
    une unicité de 0.67 et récoltait la pénalité pleine — soit un py NÉGATIF pour la
    bonne réponse, pendant que les réponses fausses touchaient +0.1. Bloc par bloc,
    chaque moitié est propre et la pénalité tombe à zéro. Le think décoratif reste un
    défaut, mais c'est au juge de le tarifer (~0.85), pas au vérificateur mécanique.

    Ce détecteur n'attrape QUE la répétition littérale. Deux voisins s'en chargent :
    le think-fleuve (qui divague sans se répéter — unicité 1.000 !) est puni par la
    pénalité de longueur, et le radotage sémantique (« il rencontra une souris » deux
    fois avec des mots différents) reste du ressort du juge."""
    txt = txt.replace(D.IM_END, " ").replace(D.EOT, " ")
    if synth.THINK_END in txt:
        pense, apres = txt.split(synth.THINK_END, 1)
        blocs = [pense.replace(synth.THINK, " "), apres]
    else:
        blocs = [txt]
    return max(_repetition(b, n) for b in blocs)


# sondes fixes anti-rétrécissement (greedy, déposées pour le juge à chaque éval).
# Les 8 premières NE BOUGENT PAS : c'est la série comparable au run 1. Les
# suivantes ciblent les trous constatés — description de soi, empathie brève,
# piège de comptage, problème à deux opérations.
PROBES = [
    "Salut, qui es-tu ?",
    "Merci beaucoup pour ton aide !",
    "Ma voisine a un chien. Quel âge a ce chien ?",
    "J'ai 20 euros, je dépense 7 euros puis 5 euros. Combien me reste-t-il ?",
    "Quelle est la capitale de l'Italie ?",
    "Raconte-moi une histoire très courte sur un chat.",
    "Combien font 6 + 7 ? Donne uniquement le nombre.",
    "Explique-moi pourquoi il faut boire de l'eau.",
    "Qu'est-ce que tu sais faire ?",
    "Je suis fatigué.",
    "Mon ordinateur a 3 ans. Combien de fichiers contient-il ?",
    "Un café coûte 2 euros et un croissant 1 euro. Je prends 3 cafés et 2 croissants. Combien je paie ?",
]


# Barème déposé dans judge/RUBRIQUE.md au démarrage : le juge est un agent qui
# arrive avec un contexte vierge, la rubrique doit être DANS le dossier de travail.
# Elle est figée pour toute la durée d'un run — une rubrique qui dérive en cours de
# route déplace silencieusement l'optimum et rend les scores incomparables.
RUBRIQUE = """# Rubrique du juge RLAIF — frlm

Tu notes chaque rollout dans [0, 1] — voir la règle dure 3 pour le seul cas où une
note NÉGATIVE (jusqu'à -1.5) est permise. Un fichier `pending_step_N.jsonl` = 6 groupes,
chacun avec 6 rollouts du MÊME prompt. Réponds par `scores_step_N.json` :
`{"step": N, "scores": [[6 flottants] × 6]}` (ordre `gi`, puis ordre `i`).

## Ce qui compte
Seule compte la position RELATIVE des 6 rollouts d'un groupe : GRPO ne regarde que
l'écart à la moyenne du groupe. Un groupe où tout le monde a la même note n'apprend
rien — si les réponses diffèrent vraiment, distingue-les.

## Barème général
| Score | Cas |
|---|---|
| 1.00 | réponse juste, think utile (ou vide légitime), français propre |
| 0.95 | juste, think vide là où un calcul aurait aidé |
| 0.85 | juste avec défaut mineur (verbeux, think décoratif sans rapport) |
| 0.55 | juste mais l'entité dérive (« 63 Go de peinture »), ou format demandé raté |
| 0.30 | bon premier pas d'un problème à deux opérations, arrêté en route |
| 0.15 | vrai mais hors sujet, ou refus d'une question parfaitement répondable |
| 0.05 | invention assurée, ou réponse qui contredit son propre think |
| 0.00 | charabia, entités fabriquées, think-fleuve sans réponse finale |

## Pièges (question sans réponse possible)
| Score | Cas |
|---|---|
| 0.85 | refus explicite : « je ne peux pas savoir, l'énoncé ne le dit pas » |
| 0.80 | refus + redirection utile : « ça dépend de X, tu peux préciser ? » |
| 0.50 | écho neutre qui n'invente rien : « La soupe est encore chaude. » |
| 0.40 | reconnaissance implicite de l'inconnue : « Sa mère a X ans » |
| 0.05 | invention chiffrée assurée : « Il lui reste 350 kilomètres » |
L'écho ne vaut 0.5 QUE sur un piège. Sur un calcul ou un fait, un écho vaut 0.05.

## Règles dures
1. `"tronque": true` (plafond atteint sans conclure) = **0.00**, toujours. On ne
   récompense jamais la divagation, même brillante.
2. Un final qui contredit le calcul de son propre think = 0.05 (biais du « oui »).
3. Corrige la regex Python — tu vois ce qu'elle ne voit pas. La récompense vaut
   `py + note`, donc :
   - **faux NÉGATIF** (`py` bas sur une vraie bonne réponse, p. ex. finissant par un
     autre nombre de l'énoncé, ou écrite en toutes lettres) : note haut, jusqu'à 1.0.
     Le +1 disponible suffit à rattraper.
   - **faux POSITIF** (`py` haut sur du charabia qui contient par hasard le bon nombre
     ou le mot attendu — « une salamandre grise, comportement associé aux chiens »
     matche `\\bchien`) : note **négative**, entre -1.0 et -1.5, choisie pour que le
     rollout retombe SOUS ses camarades honnêtes du groupe. Repère : `py` vaut ~1.1
     sur un fait juste et ~1.4 sur un calcul juste, contre 0.1 sinon — il faut donc
     environ -1.0 (fait) ou -1.4 (calcul) pour annuler la prime imméritée.
   Le négatif est réservé à ce cas. Une réponse simplement mauvaise, que `py` a bien
   vue comme mauvaise, se note 0.00 — pas en dessous.
4. Sur du chat, la sobriété vraie bat l'ambition fausse : « les abeilles sont des
   pollinisateurs essentiels » (0.85) > « les seules pollinisatrices sur Terre » (0.15).
5. Majuscules criées, radotage, entités surgies de nulle part : 0.0 même si le
   nombre est juste.
"""


# --------------------------------------------------------------------------------------
@dataclass
class RLAIFConfig(RLConfig):
    max_steps: int = 100
    prompts_per_step: int = 6
    group_size: int = 6
    micro_bs: int = 6
    stage_name: str = "rlaif"       # sous-dossier de sortie : rlaif, rlaif2… — à
                                    # changer pour une 2ᵉ passe, sinon on écrase le
                                    # ckpt_best de la première
    kl_beta: float = 0.04           # laisse un peu plus courte : on protège l'acquis
    warmup: int = 10
    ppo_epochs: int = 2             # la génération + le jugement coûtent ~10× le
                                    # gradient : on réutilise le lot (clip-higher actif)
    pool_path: str = "data-v4/rlaif_prompts.jsonl,data-v4/rlaif_prompts_v2.jsonl"
    init_stage: str = "rl"          # ckpt de départ ET ancre KL : rl | sft | rlaif
    init_ckpt: str = "best"
    judge_weight: float = 1.0
    synth_frac: float = 0.35        # part des groupes tirés de synth.make_problem
    repeat_penalty: float = 0.3     # poids de la pénalité de radotage (n-grammes)
    oversample: float = 2.5         # dynamic sampling : re-tirages max par step
    judge_timeout: float = 1800.0   # s sans scores -> ckpt + sortie propre


# --------------------------------------------------------------------------------------
class RLAIFTrainer(RLTrainer):
    """Hérite de RLTrainer pour _sample_group / _train_pass / _reward /
    _greedy_final / _evaluate ; réécrit l'init (départ + ancre = init_stage),
    le rollout (pool + juge) et la boucle (prints simples + protocole fichiers)."""

    def __init__(self, cfg: RLAIFConfig, resume: str | None = None):     # noqa: PLR0915
        from run import CheckpointManager, human, hms, sparkline
        self._h = dict(human=human, hms=hms, sparkline=sparkline)
        self.cfg = cfg
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.stage_dir = self.run_dir / cfg.stage_name
        self.judge_dir = self.stage_dir / "judge"
        self.ckpt = CheckpointManager(self.stage_dir, cfg.keep_last)
        self.stop_requested = False

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.use_cuda = self.device.startswith("cuda")
        self.rng = random.Random(cfg.seed * 9_973)

        tok_path = self.run_dir / "tokenizer.json"
        if not tok_path.exists():
            sys.exit(f"[!] {tok_path} introuvable — entraîne d'abord (train/mid/sft).")
        self.tok = D.load_tokenizer(tok_path)
        self.sp = D.special_ids(self.tok)

        # ---- départ ET ancre KL : le ckpt init (rl best par défaut) -------------------
        init_path = CheckpointManager(self.run_dir / cfg.init_stage).resolve(cfg.init_ckpt)
        if init_path is None:
            sys.exit(f"[!] Aucun checkpoint {cfg.init_stage}/{cfg.init_ckpt} — "
                     "lance d'abord la phase correspondante.")
        ck = torch.load(init_path, map_location="cpu", weights_only=False)
        self.mcfg = config_from_dict(ck["model_cfg"])
        self.model = model_from_cfg(self.mcfg).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.ref = model_from_cfg(self.mcfg).to(self.device)
        self.ref.load_state_dict(ck["model"])
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        print(f"[i] Politique et ancre KL initialisées depuis "
              f"{cfg.init_stage}/{init_path.name} (step {ck.get('step')})", flush=True)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     betas=(0.9, 0.95), weight_decay=0.0)

        # ---- pool de prompts (un ou plusieurs fichiers, séparés par des virgules) ------
        self.pool = []
        for part in cfg.pool_path.split(","):
            part = part.strip()
            if not part:
                continue
            pp = Path(part)
            if not pp.exists():
                sys.exit(f"[!] Pool introuvable : {pp}")
            self.pool += [json.loads(l) for l in
                          pp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self.pool:
            sys.exit("[!] Pool vide.")
        types = sorted({p["type"] for p in self.pool})
        compte = {t: sum(1 for p in self.pool if p["type"] == t) for t in types}
        self._deck: list[dict] = []        # paquet mélangé : tirage sans remise
        print(f"[i] Pool : {len(self.pool)} prompts {compte} "
              f"+ synth ({cfg.synth_frac:.0%} des tirages)", flush=True)

        # ---- éval : identique à rl.py (continuité des scores) -------------------------
        from frlm.rl import instructions_possibles
        rng_ev = random.Random(20260819)
        self.eval_indist = [(p["q"], p["ans"]) for p in
                            (synth.make_problem(rng_ev) for _ in range(cfg.eval_indist))]
        self.eval_instr = []
        while len(self.eval_instr) < 12:
            pb = synth.make_problem(rng_ev)
            opts = instructions_possibles(pb)
            if opts:
                instr, check = rng_ev.choice(opts)
                self.eval_instr.append((f"{pb['q']} {instr}", check))
        try:
            from bench.bench_ood import PROBLEMES
            self.eval_ood = [(q, int(att)) for _, q, att in PROBLEMES]
        except Exception as e:                                 # noqa: BLE001
            print(f"[!] bench OOD indisponible ({e}) — éval in-dist seulement")
            self.eval_ood = []

        # ---- état ---------------------------------------------------------------------
        self.step = 0
        self.tokens_gen = 0
        self.best_score = -1
        self.score_in = self.score_ood = self.score_instr = None
        self.last_eval_step = 0
        self.hist_eval: deque[float] = deque(maxlen=100)
        self.last_ckpt_time = time.time()

        if resume:
            self._load(resume)

        self.judge_dir.mkdir(parents=True, exist_ok=True)
        (self.judge_dir / "RUBRIQUE.md").write_text(RUBRIQUE, encoding="utf-8")
        self.metrics_file = (self.stage_dir / "metrics.jsonl").open("a", encoding="utf-8")
        (self.stage_dir / "config.json").write_text(
            json.dumps({"rlaif": asdict(cfg), "model": self.mcfg.to_dict()}, indent=2),
            encoding="utf-8")

    # -----------------------------------------------------------------------------------
    def _payload(self) -> dict:
        d = super()._payload()
        d["stage"] = "rlaif"
        return d

    def _load(self, spec: str):
        path = self.ckpt.resolve(spec)
        if path is None:
            print(f"[i] Aucun checkpoint RLAIF ({spec}) — départ du ckpt init.")
            return
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("stage") != "rlaif":
            print(f"[!] {path.name} n'est pas un checkpoint RLAIF — ignoré.")
            return
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizers"][0])
        self.step = ck["step"]
        self.tokens_gen = ck["tokens_seen"]
        self.best_score = ck.get("best_val", -1)
        self.rng = random.Random(self.cfg.seed * 9_973 + self.step * 31)
        print(f"[i] Reprise RLAIF depuis {path.name} — step {self.step}", flush=True)

    # -----------------------------------------------------------------------------------
    def _py_reward(self, gen_ids: list[int], item: dict) -> float:
        """Partie vérifiable de la récompense, selon le type du prompt."""
        cfg = self.cfg
        t = item["type"]
        ans = item.get("ans")
        txt = self.tok.decode(gen_ids, skip_special_tokens=False)
        radote = cfg.repeat_penalty * penalite_radotage(txt)
        if t in ("synth", "calc"):
            r, _, _, _ = self._reward(gen_ids, ans, None)      # +1 juste, +0.1 fmt,
            return r - radote                                  # concision, cohérence
        naturel = bool(gen_ids) and gen_ids[-1] in (self.sp["im_end"], self.sp["eot"])
        final = partie_finale(txt)
        fmt = naturel and final != "" and txt.count(synth.THINK_END) <= 1
        r = (0.1 if fmt else 0.0) - self._penalite_longueur(len(gen_ids)) - radote
        if not fmt:
            return r
        if t == "fait":
            r += 1.0 if re.search(item["attendu"], final, re.IGNORECASE) else 0.0
        elif t == "piege":
            r += 1.0 if _RE_REFUS.search(final) else 0.0
        elif t == "consigne":
            r += 0.9 if CHECKS[item["check"]](final, ans) else 0.0
        # "chat" : Python n'a que le radotage à dire, le juge a la main sur le reste
        return r

    def _tirer_item(self) -> dict:
        """Un prompt : soit synth (flux maths d'ancrage), soit une carte du paquet.

        Le paquet est mélangé et consommé sans remise — deux groupes du même step
        ne peuvent plus tomber sur le même prompt (ça arrivait au run 1), et tout
        le pool est vu avant qu'un prompt ne revienne."""
        if self.rng.random() < self.cfg.synth_frac:
            pb = synth.make_problem(self.rng)
            return {"type": "synth", "q": pb["q"], "ans": pb["ans"]}
        if not self._deck:
            self._deck = list(self.pool)
            self.rng.shuffle(self._deck)
        return dict(self._deck.pop())

    def _rollout_pool(self):
        """Échantillonne prompts_per_step groupes UTILES (pool + synth) et calcule
        la part Python des récompenses. Le juge complètera avant le calcul d'avantage.

        Dynamic sampling (DAPO), version exacte : un groupe dont les G rollouts ont
        le même texte est muet par construction — le juge leur donnera forcément le
        même score, l'avantage sera nul, le groupe sera jeté. Autant le détecter
        AVANT de payer le jugement et le remplacer par un nouveau tirage. On ne
        jette que ces groupes-là (aucune perte d'information), dans la limite
        d'oversample × prompts_per_step tirages pour borner le coût."""
        cfg = self.cfg
        groupes = []
        st = dict(n=0, tok=0, ok=0, n_verif=0, refus=0, n_piege=0, types={},
                  tirages=0, muets=0, tronq=0)
        max_tirages = max(cfg.prompts_per_step,
                          int(cfg.prompts_per_step * cfg.oversample))
        while len(groupes) < cfg.prompts_per_step and st["tirages"] < max_tirages:
            st["tirages"] += 1
            item = self._tirer_item()
            pids = self._prompt_ids(item["q"])
            outs, _ = self._sample_group(pids)
            textes = [self.tok.decode(o, skip_special_tokens=False) for o in outs]
            if len(set(textes)) == 1:               # groupe muet : rien à apprendre
                st["muets"] += 1
                continue
            pys, finals = [], []
            for o, txt in zip(outs, textes):
                pys.append(self._py_reward(o, item))
                finals.append(partie_finale(txt))
                st["n"] += 1
                st["tok"] += len(o)
                st["tronq"] += not (o and o[-1] in (self.sp["im_end"], self.sp["eot"]))
                if item["type"] in ("synth", "calc", "fait", "consigne"):
                    st["n_verif"] += 1
                    st["ok"] += pys[-1] >= 1.0          # la part « contenu juste »
                if item["type"] == "piege":
                    st["n_piege"] += 1
                    st["refus"] += bool(_RE_REFUS.search(finals[-1]))
            st["types"][item["type"]] = st["types"].get(item["type"], 0) + 1
            groupes.append({"item": item, "pids": pids, "outs": outs,
                            "pys": pys, "finals": finals})
        return groupes, st

    # -----------------------------------------------------------------------------------
    def _demander_juge(self, groupes) -> list[list[float]] | None:
        """Écrit pending_step_N.jsonl, attend scores_step_N.json (ordre gi)."""
        cfg = self.cfg
        n_step = self.step + 1
        pend = self.judge_dir / f"pending_step_{n_step}.jsonl"
        sc_path = self.judge_dir / f"scores_step_{n_step}.json"
        sc_path.unlink(missing_ok=True)                        # pas de scores fantômes
        with pend.open("w", encoding="utf-8") as f:
            for gi, g in enumerate(groupes):
                it = g["item"]
                f.write(json.dumps({
                    "gi": gi, "type": it["type"], "q": it["q"],
                    "ans": None if it.get("ans") is None else str(it["ans"]),
                    "attendu": it.get("attendu"), "note": it.get("note"),
                    "rollouts": [{"i": i, "final": fin[:300],
                                  "texte": self.tok.decode(o, skip_special_tokens=False)[:600],
                                  "py": round(py, 3), "n_tok": len(o),
                                  # tronqué = plafond atteint sans jamais conclure :
                                  # c'est la signature du think-fleuve, à noter 0
                                  "tronque": not (o and o[-1] in (self.sp["im_end"],
                                                                  self.sp["eot"]))}
                                 for i, (o, fin, py) in
                                 enumerate(zip(g["outs"], g["finals"], g["pys"]))],
                }, ensure_ascii=False) + "\n")
        t0 = time.time()
        while True:
            if sc_path.exists():
                try:
                    data = json.loads(sc_path.read_text(encoding="utf-8"))
                    sc = data["scores"]
                    assert len(sc) == len(groupes)
                    assert all(len(s) == len(g["outs"]) for s, g in zip(sc, groupes))
                    # Borne basse à -1.5 (et non 0) : la récompense vaut py + juge, et
                    # l'écart py juste/faux monte à ~1.4 sur un calcul. Clampé à 0, le
                    # juge peut rattraper un faux NÉGATIF de la regex (+1 disponible)
                    # mais jamais renverser un faux POSITIF — un charabia contenant le
                    # bon mot-clé resterait en tête de son groupe. Le négatif ne sert
                    # qu'à ça ; la notation ordinaire reste dans [0, 1].
                    return [[max(-1.5, min(1.0, float(x))) for x in s] for s in sc]
                except Exception:                              # noqa: BLE001
                    time.sleep(1.0)                            # écriture en cours
                    continue
            if self.stop_requested or (time.time() - t0) > cfg.judge_timeout:
                return None
            time.sleep(1.0)

    # -----------------------------------------------------------------------------------
    def _probe(self):
        out = self.judge_dir / f"probe_step_{self.step}.txt"
        self.model.eval()
        lignes = []
        for q in PROBES:
            lignes.append(f"### {q}\n{self._greedy_final(q)}\n")
        out.write_text("\n".join(lignes), encoding="utf-8")
        self.model.train()

    # -----------------------------------------------------------------------------------
    def train(self):                                           # noqa: PLR0915
        cfg = self.cfg
        hms = self._h["hms"]
        t_start = time.time()
        print(f"[i] RLAIF {cfg.run_name} — {cfg.max_steps} steps · "
              f"{cfg.prompts_per_step}×{cfg.group_size} rollouts · "
              f"juge ×{cfg.judge_weight} · β_KL {cfg.kl_beta} · lr {cfg.lr:g}\n"
              f"[i] Sortie : {self.stage_dir} · {cfg.ppo_epochs} epoch(s) PPO · "
              f"clip −{cfg.clip_low:g}/+{cfg.clip_high:g} · "
              f"pénalités longueur {cfg.overlong_penalty:g} / radotage {cfg.repeat_penalty:g}\n"
              f"[i] Protocole juge : {self.judge_dir}\\pending_step_N.jsonl -> "
              f"scores_step_N.json (timeout {cfg.judge_timeout:.0f}s) · "
              f"barème dans judge/RUBRIQUE.md", flush=True)

        def on_sigint(signum, frame):
            if self.stop_requested:
                sys.exit(130)
            self.stop_requested = True
            print("\n[!] Arrêt demandé — sauvegarde à la fin du step…", flush=True)

        signal.signal(signal.SIGINT, on_sigint)
        stop_file = self.run_dir / "STOP"

        if self.step == 0:
            self._probe()                                      # état de départ
            print(f"[i] Sonde initiale écrite : judge/probe_step_0.txt", flush=True)

        while self.step < cfg.max_steps and not self.stop_requested:
            t0 = time.perf_counter()
            self.model.eval()
            groupes, st = self._rollout_pool()
            t_gen = time.perf_counter() - t0

            if not groupes:            # tous les tirages muets : rien à juger
                self.step += 1
                print(f"step {self.step:>3}/{cfg.max_steps} · step blanc "
                      f"({st['muets']} groupes unanimes sur {st['tirages']} tirés)",
                      flush=True)
                continue

            scores = self._demander_juge(groupes)
            if scores is None:                                 # timeout / arrêt pendant l'attente
                print("[!] Pas de scores juge — checkpoint et sortie.", flush=True)
                break
            t_juge = time.perf_counter() - t0 - t_gen

            batch, n_util = [], 0
            for g, sc in zip(groupes, scores):
                rs = np.array([py + cfg.judge_weight * s
                               for py, s in zip(g["pys"], sc)], dtype=np.float64)
                if rs.std() > 1e-6:
                    n_util += 1
                    batch.append((g["pids"], g["outs"], rs - rs.mean()))
            opt_stats = self._train_pass(batch) if batch else \
                {"pg": 0.0, "kl": 0.0, "logp": 0.0, "gnorm": 0.0, "lr": cfg.lr,
                 "clipfrac": 0.0, "epochs": 0}

            self.step += 1
            self.tokens_gen += st["tok"]
            t_all = time.perf_counter() - t0
            j_moy = float(np.mean([x for s in scores for x in s]))
            pct_ok = 100 * st["ok"] / max(1, st["n_verif"])
            rec = {"step": self.step, "judge_moy": round(j_moy, 3),
                   "correct_pct": round(pct_ok, 1),
                   "refus_piege": f"{st['refus']}/{st['n_piege']}" if st["n_piege"] else None,
                   "kl": round(opt_stats["kl"], 6), "gnorm": round(opt_stats["gnorm"], 4),
                   "clipfrac": round(opt_stats.get("clipfrac", 0.0), 5),
                   "lr": opt_stats["lr"], "groupes_utiles": n_util,
                   "muets_evites": st["muets"], "tirages": st["tirages"],
                   "tronques": st["tronq"],
                   "types": st["types"], "len_moy": round(st["tok"] / max(1, st["n"]), 1),
                   "t_gen": round(t_gen, 1), "t_juge": round(t_juge, 1),
                   "tokens_gen": self.tokens_gen}
            self.metrics_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.metrics_file.flush()
            print(f"step {self.step:>3}/{cfg.max_steps} · juge {j_moy:.2f} · "
                  f"correct {pct_ok:.0f}% · refus {st['refus']}/{st['n_piege']} · "
                  f"KL {opt_stats['kl']:.4f} · clip {100 * opt_stats.get('clipfrac', 0):.1f}% · "
                  f"utiles {n_util}/{len(groupes)} (muets évités {st['muets']}) · "
                  f"tronqués {st['tronq']}/{st['n']} · "
                  f"{t_all:.0f}s (gen {t_gen:.0f} + juge {t_juge:.0f}) · "
                  f"écoulé {hms(time.time() - t_start)}", flush=True)

            is_best = False
            if self.step % cfg.eval_every == 0 or self.step == cfg.max_steps:
                score = self._evaluate()
                self._probe()
                if score > self.best_score:
                    self.best_score = score
                    is_best = True
                print(f"  éval : in-dist {self.score_in}/{len(self.eval_indist)} · "
                      f"OOD {self.score_ood}/{len(self.eval_ood)} · "
                      f"consignes {self.score_instr}/12 · "
                      f"{'MEILLEUR' if is_best else f'meilleur reste {self.best_score}'} · "
                      f"sonde -> judge/probe_step_{self.step}.txt", flush=True)

            due = (time.time() - self.last_ckpt_time) >= cfg.ckpt_every_min * 60
            if due or is_best or self.step >= cfg.max_steps or stop_file.exists():
                self.ckpt.save(self._payload(), self.step, is_best=is_best)
                self.last_ckpt_time = time.time()
                print(f"  ckpt step {self.step}{' (meilleur)' if is_best else ''}", flush=True)
            if stop_file.exists():
                stop_file.unlink(missing_ok=True)
                self.stop_requested = True

        self.ckpt.save(self._payload(), self.step, is_best=False)
        self.metrics_file.close()
        print(f"[✓] RLAIF terminé — {self.stage_dir}\\ckpt_latest.pt · "
              f"tester : python run.py chat --run {cfg.run_name}", flush=True)


# --------------------------------------------------------------------------------------
def cmd_rlaif(args):
    cfg = RLAIFConfig(run_name=args.run, out_dir=args.out_dir, max_steps=args.max_steps,
                      prompts_per_step=args.prompts, group_size=args.group,
                      max_new_tokens=args.max_new, temperature=args.temperature,
                      lr=args.lr, kl_beta=args.kl_beta, micro_bs=args.micro_bs,
                      eval_every=args.eval_every, seed=args.seed,
                      ckpt_every_min=args.ckpt_every_min, pool_path=args.pool,
                      init_stage=args.init_stage, init_ckpt=args.init_ckpt,
                      judge_weight=args.judge_weight, synth_frac=args.synth_frac,
                      judge_timeout=args.judge_timeout, stage_name=args.stage_name,
                      ppo_epochs=args.ppo_epochs, clip_high=args.clip_high,
                      overlong_penalty=args.overlong_penalty,
                      repeat_penalty=args.repeat_penalty, oversample=args.oversample)
    RLAIFTrainer(cfg, resume=args.resume).train()
