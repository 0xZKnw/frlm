"""rl.py — GRPO à récompenses vérifiables : le dernier étage de la fusée.

Pourquoi du RL ici ?
  Le bench OOD et les tests manuels ont montré trois pannes que le SFT seul ne
  répare pas :
    1. la réponse finale TRAHIT le brouillon (calcul juste dans <think>, réponse
       fausse après) ;
    2. hors de ses gabarits, le modèle part en radotage type GSM8K traduit
       (« Attends, laisse-moi vérifier… » × 40) et invente des entités ;
    3. les formulations elliptiques (« Le double de 16 ? ») déclenchent une
       opération au hasard.
  Le RL attaque les trois à la racine : on ne récompense QUE la réponse finale
  juste (vérifiée par Python via synth.ans), avec un bonus de brièveté qui tue le
  radotage, et une laisse KL vers le SFT qui préserve le français.

L'algorithme (GRPO, façon Dr. GRPO + emprunts à DAPO) :
  - pour chaque problème, on échantillonne G réponses complètes (température 1) ;
  - récompense r_i par réponse : +1 si la réponse finale est juste, +0.1 si le
    format est propre (un seul <think>, arrêt naturel), +0.25 × (1 − len/cible)
    de bonus de concision quand c'est juste, − pénalité de longueur excessive ;
  - avantage A_i = r_i − moyenne(groupe), SANS division par l'écart-type : c'est
    le correctif Dr. GRPO (diviser surpondère les groupes quasi unanimes). Pas de
    critic, le groupe EST la baseline ; un groupe unanime n'apprend rien -> écarté ;
  - perte = −Σ_tokens min(ρ·A, clip(ρ)·A) / Σ_tokens + β · KL(π ‖ π_SFT), la
    normalisation étant faite sur le TOTAL des tokens du lot (token-level loss de
    DAPO : une réponse longue pèse proportionnellement, pas à égalité) ;
  - ρ = ratio d'importance π/π_old. À ppo_epochs=1 il vaut exactement 1 et le clip
    est inerte ; au-delà, on réutilise le même lot de rollouts (coûteux à générer)
    avec **clip-higher** — borne haute 1+0.28 plus lâche que la basse 1−0.2, ce
    qui laisse remonter les tokens rares mais bons et freine l'effondrement
    d'entropie observé en phase 1 (0.377 -> 0.077 nats) ;
  - KL estimée par token avec l'estimateur k3 (exp(d) − d − 1, toujours ≥ 0).

Overlong reward shaping (DAPO) : au-delà de max_new − overlong_cache tokens, une
pénalité linéaire monte jusqu'à overlong_penalty (pleine quand la génération est
coupée sans jamais conclure). C'est le remède direct au « think-fleuve » : les
divagations qui saturent la fenêtre sans produire de réponse deviennent le pire
choix du groupe, pas seulement un choix neutre.

Les prompts d'entraînement sortent de synth.make_problem : 17 familles, plusieurs
formulations par concept (v2.2), symboles machine, litres, parenthèses… La graine
d'éval in-dist (20260819) et les 40 problèmes OOD écrits à la main ne sont JAMAIS
échantillonnés à l'entraînement.

Usage :  python run.py rl --run fr-v2          (reprendre : --resume)
"""

from __future__ import annotations

import json
import math
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
import torch.nn.functional as F

from frlm import data as D
from frlm import synth
from frlm.model import ModelConfig, build_model


# --------------------------------------------------------------------------------------
# Vérification des réponses (la « récompense vérifiable »)
# --------------------------------------------------------------------------------------
_RE_NUM = re.compile(r"\d(?:[\d ]*\d)?")


def dernier_nombre(txt: str) -> str | None:
    m = _RE_NUM.findall(txt)
    return m[-1].replace(" ", "") if m else None


def partie_finale(txt: str) -> str:
    """Ce qui suit le dernier </think> : la réponse que l'utilisateur lit."""
    if synth.THINK_END in txt:
        txt = txt.split(synth.THINK_END)[-1]
    return txt.split(D.IM_END)[0].split(D.EOT)[0].strip()


def verifier(ans, final: str) -> bool:
    """Compare la réponse canonique de synth (int ou str) au texte du modèle."""
    if final == "":
        return False
    if isinstance(ans, (int, float)):
        return dernier_nombre(final) == str(int(ans))
    a, f = str(ans).lower(), final.lower()
    if a in ("pair", "impair"):
        pred = "impair" if "impair" in f else ("pair" if re.search(r"\bpair", f) else None)
        return pred == a
    if a in ("oui", "non"):
        return re.search(rf"\b{a}\b", f) is not None
    return a in f       # prénom (transitivité)


# habillages neutres : la réponse ne change pas, la surface oui
_DECORS = (("", ""), ("", ""), ("", ""), ("", ""),
           ("Petite question : ", ""), ("Dis-moi : ", ""), ("", " Tu peux m'aider ?"))


# --------------------------------------------------------------------------------------
# Instructions vérifiables (IF-RLVR, façon Tülu 3) : la contrainte se vérifie en
# Python aussi mécaniquement que le calcul. On greffe la consigne sur un problème
# dont on connaît déjà la réponse -> double récompense contenu + conformité.
# --------------------------------------------------------------------------------------
def _mots(final: str) -> list[str]:
    return final.strip().strip(".!?…»«\"' ").split()


def instructions_possibles(pb: dict) -> list[tuple[str, "callable"]]:
    """Consignes applicables à ce problème, chacune avec son vérificateur."""
    outs = []
    ans = pb["ans"]
    if isinstance(ans, (int, float)):
        cible = str(int(ans))
        outs.append(("Donne uniquement le nombre, rien d'autre.",
                     lambda f, c=cible: len(_mots(f)) == 1 and dernier_nombre(f) == c))
        outs.append(("Réponds en 8 mots maximum.",
                     lambda f, a=ans: len(_mots(f)) <= 8 and verifier(a, f)))
    if ans in ("oui", "non"):
        outs.append(("Réponds par oui ou non, rien d'autre.",
                     lambda f, a=ans: " ".join(_mots(f)).lower() == str(a)))
    if isinstance(ans, str) and ans not in ("oui", "non"):    # pair/impair, prénom
        outs.append(("Réponds en un seul mot.",
                     lambda f, a=ans: len(_mots(f)) == 1 and verifier(a, f)))
    outs.append(("Réponds en majuscules.",
                 lambda f, a=ans: f == f.upper() and any(ch.isalpha() for ch in f)
                 and verifier(a, f)))
    return outs


# --------------------------------------------------------------------------------------
@dataclass
class RLConfig:
    run_name: str = "fr-v2"
    out_dir: str = "runs"
    max_steps: int = 500
    prompts_per_step: int = 8       # groupes GRPO par step
    group_size: int = 8             # le G : réponses échantillonnées par problème
    max_new_tokens: int = 220
    temperature: float = 1.0        # échantillonnage pur = gradients non biaisés
    top_p: float = 1.0
    lr: float = 1e-5                # AdamW doux : on sculpte, on ne laboure pas
    warmup: int = 20
    kl_beta: float = 0.03           # la laisse vers le SFT (français préservé)
    grad_clip: float = 1.0
    micro_bs: int = 8               # séquences par micro-batch du passage de gradient
    ppo_epochs: int = 1             # réutilisations du même lot (1 = on-policy pur)
    clip_low: float = 0.2           # borne basse du ratio d'importance
    clip_high: float = 0.28         # clip-higher (DAPO) : borne haute plus lâche
    overlong_cache: int = 60        # fenêtre (tokens) où la pénalité de longueur monte
    overlong_penalty: float = 0.3   # pénalité pleine si la génération est coupée
    brevity_target: int = 160       # tokens : au-delà, plus de bonus de concision
    instr_frac: float = 0.3         # part des prompts avec consigne vérifiable (IF-RLVR)
    oversample: float = 2.0         # dynamic sampling (DAPO) : re-tirages max pour
                                    # remplir le batch de groupes UTILES
    eval_every: int = 25
    eval_indist: int = 30
    ckpt_every_min: float = 5.0
    keep_last: int = 2
    seed: int = 42
    device: str = "cuda"


# --------------------------------------------------------------------------------------
class RLTrainer:
    def __init__(self, cfg: RLConfig, resume: str | None = None):
        # import tardif : run.py importe frlm.rl, on évite le cycle au chargement
        from run import CheckpointManager, GpuMon, human, hms, sparkline
        self._h = dict(human=human, hms=hms, sparkline=sparkline)
        self.cfg = cfg
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.stage_dir = self.run_dir / "rl"
        self.ckpt = CheckpointManager(self.stage_dir, cfg.keep_last)
        self.gpu = GpuMon()
        self.stop_requested = False

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.use_cuda = self.device.startswith("cuda")
        self.rng = random.Random(cfg.seed * 6_007)

        tok_path = self.run_dir / "tokenizer.json"
        if not tok_path.exists():
            sys.exit(f"[!] {tok_path} introuvable — entraîne d'abord (train/mid/sft).")
        self.tok = D.load_tokenizer(tok_path)
        self.sp = D.special_ids(self.tok)

        # ---- point de départ : le SFT (poids ET ancre KL) -----------------------------
        sft_path = None
        for spec in ("best", "latest"):
            sft_path = CheckpointManager(self.run_dir / "sft").resolve(spec)
            if sft_path:
                break
        if sft_path is None:
            sys.exit("[!] Aucun checkpoint SFT — le RL part du SFT, lance-le d'abord.")
        ck = torch.load(sft_path, map_location="cpu", weights_only=False)
        from frlm import config_from_dict, model_from_cfg
        self.mcfg = config_from_dict(ck["model_cfg"])

        self.model = model_from_cfg(self.mcfg).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.ref = model_from_cfg(self.mcfg).to(self.device)   # π_SFT gelé : l'ancre KL
        self.ref.load_state_dict(ck["model"])
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        print(f"[i] Politique et référence initialisées depuis {sft_path.name} "
              f"(step {ck['step']}, val {ck.get('val_loss')})")

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     betas=(0.9, 0.95), weight_decay=0.0)

        # ---- éval : 30 problèmes synth frais + les 40 OOD écrits à la main ------------
        rng_ev = random.Random(20260819)                       # graine JAMAIS entraînée
        self.eval_indist = [(p["q"], p["ans"]) for p in
                            (synth.make_problem(rng_ev) for _ in range(cfg.eval_indist))]
        self.eval_instr = []                                   # 12 consignes vérifiables
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

        # ---- état ----------------------------------------------------------------------
        self.step = 0
        self.tokens_gen = 0
        self.best_score = -1
        self.score_in = self.score_ood = self.score_instr = None
        self.last_eval_step = 0
        self.hist_correct: deque[float] = deque(maxlen=400)
        self.hist_reward: deque[float] = deque(maxlen=400)
        self.hist_eval: deque[float] = deque(maxlen=100)
        self.reward_ema = None
        self.correct_ema = None
        self.last_sample = ""
        self.last_ckpt_msg = "—"
        self.last_ckpt_time = time.time()

        if resume:
            self._load(resume)

        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = (self.stage_dir / "metrics.jsonl").open("a", encoding="utf-8")
        (self.stage_dir / "config.json").write_text(
            json.dumps({"rl": asdict(cfg), "model": self.mcfg.to_dict()}, indent=2),
            encoding="utf-8")

    # -----------------------------------------------------------------------------------
    def _load(self, spec: str):
        path = self.ckpt.resolve(spec)
        if path is None:
            print(f"[i] Aucun checkpoint RL ({spec}) — départ du SFT.")
            return
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get("stage") != "rl":
            print(f"[!] {path.name} n'est pas un checkpoint RL — ignoré.")
            return
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizers"][0])
        self.step = ck["step"]
        self.tokens_gen = ck["tokens_seen"]
        self.best_score = ck.get("best_val", -1)
        self.rng = random.Random(self.cfg.seed * 6_007 + self.step * 31)
        print(f"[i] Reprise RL depuis {path.name} — step {self.step}")

    def _payload(self) -> dict:
        return {"model": self.model.state_dict(),
                "optimizers": [self.opt.state_dict()],
                "model_cfg": self.mcfg.to_dict(), "rl_cfg": asdict(self.cfg),
                "step": self.step, "tokens_seen": self.tokens_gen,
                "best_val": self.best_score, "val_loss": float("nan"),
                "scores": {"indist": self.score_in, "ood": self.score_ood,
                           "instr": self.score_instr},
                "stage": "rl"}

    # -----------------------------------------------------------------------------------
    def _prompt_ids(self, question: str) -> list[int]:
        txt = f"{D.IM_START}user\n{question}{D.IM_END}\n{D.IM_START}assistant\n"
        return self.tok.encode(txt).ids

    @torch.inference_mode()
    def _sample_group(self, prompt_ids: list[int]) -> tuple[list[list[int]], float]:
        """G complétions en parallèle pour UN prompt (même longueur -> pas de padding)."""
        cfg, model = self.cfg, self.model
        G = cfg.group_size
        ids = torch.tensor([prompt_ids] * G, device=self.device)
        T = ids.shape[1]
        max_len = min(self.mcfg.max_seq_len, T + cfg.max_new_tokens)
        dtype = next(model.parameters()).dtype
        caches = model._alloc_caches(G, max_len, self.device, dtype)
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda)
        with amp:
            logits = model._forward_cached(ids, caches, 0)
        pos = T
        stop = (self.sp["im_end"], self.sp["eot"])
        fini = [False] * G
        outs: list[list[int]] = [[] for _ in range(G)]
        ent_sum, ent_n = 0.0, 0

        for _ in range(cfg.max_new_tokens):
            lg = logits[:, -1, :].float()
            if cfg.temperature > 0:
                lg = lg / cfg.temperature
                if cfg.top_p < 1.0:
                    srt, idx = torch.sort(lg, descending=True, dim=-1)
                    cum = torch.softmax(srt, dim=-1).cumsum(-1)
                    rm = cum - torch.softmax(srt, dim=-1) > cfg.top_p
                    srt = srt.masked_fill(rm, float("-inf"))
                    lg = torch.full_like(lg, float("-inf")).scatter(-1, idx, srt)
                probs = torch.softmax(lg, dim=-1)
                actifs = [i for i in range(G) if not fini[i]]
                ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
                ent_sum += float(ent[actifs].mean())
                ent_n += 1
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = lg.argmax(-1, keepdim=True)
            for i, t in enumerate(nxt.squeeze(1).tolist()):
                if not fini[i]:
                    outs[i].append(t)
                    if t in stop:
                        fini[i] = True
            if all(fini) or pos + 1 >= max_len:
                break
            with amp:
                logits = model._forward_cached(nxt, caches, pos)
            pos += 1
        return outs, ent_sum / max(1, ent_n)

    # -----------------------------------------------------------------------------------
    def _penalite_longueur(self, n_tok: int) -> float:
        """Overlong reward shaping (DAPO) : 0 tant qu'on reste sous le seuil, puis
        montée linéaire sur les overlong_cache derniers tokens. Une génération
        coupée par le plafond prend la pénalité pleine — c'est ce qui rend le
        « think-fleuve » (divagation qui sature la fenêtre sans conclure) STRICTEMENT
        pire qu'une réponse fausse mais courte, au lieu de simplement neutre."""
        cfg = self.cfg
        if cfg.overlong_penalty <= 0:
            return 0.0
        seuil = cfg.max_new_tokens - cfg.overlong_cache
        if n_tok <= seuil:
            return 0.0
        return cfg.overlong_penalty * min(1.0, (n_tok - seuil) / max(1, cfg.overlong_cache))

    def _reward(self, gen_ids: list[int], ans, check=None) -> tuple[float, bool, bool, bool]:
        naturel = bool(gen_ids) and gen_ids[-1] in (self.sp["im_end"], self.sp["eot"])
        txt = self.tok.decode(gen_ids, skip_special_tokens=False)
        final = partie_finale(txt)
        fmt = naturel and final != "" and txt.count(synth.THINK_END) <= 1
        correct = fmt and verifier(ans, final)
        r = (0.1 if fmt else 0.0) - self._penalite_longueur(len(gen_ids))
        if check is not None:
            # prompt à consigne : contenu et conformité récompensés séparément —
            # l'avantage étant relatif AU GROUPE, l'échelle propre ne gêne pas
            conforme = fmt and check(final)
            r += 0.6 * correct + 0.6 * conforme
            return r, correct, fmt, conforme
        if correct:
            # bonus de concision : pousse vers le brouillon court façon synth,
            # contre le radotage GSM8K (« Attends, laisse-moi vérifier… » × 40)
            r += 1.0 + 0.25 * max(0.0, 1.0 - len(gen_ids) / self.cfg.brevity_target)
            # bonus de cohérence : le brouillon doit CONCLURE sur la réponse —
            # cible directe de la « trahison » (think juste, réponse fausse)
            if isinstance(ans, (int, float)) and synth.THINK in txt and synth.THINK_END in txt:
                pense = txt.split(synth.THINK, 1)[-1].split(synth.THINK_END, 1)[0]
                if dernier_nombre(pense) == str(int(ans)):
                    r += 0.15
        return r, correct, fmt, False

    def _rollout(self):
        """Dynamic sampling (DAPO) : on re-tire des problèmes jusqu'à obtenir
        prompts_per_step groupes UTILES (variance non nulle), dans la limite
        d'oversample× tirages — le batch de gradient reste plein même quand le
        modèle devient bon et que les groupes tout-justes se multiplient."""
        cfg = self.cfg
        groupes = []
        st = dict(n=0, ok=0, fmt=0, tok=0, ent=0.0, util=0, len_ok=[],
                  n_instr=0, ok_instr=0, tirages=0)
        max_tirages = max(cfg.prompts_per_step, int(cfg.prompts_per_step * cfg.oversample))
        while len(groupes) < cfg.prompts_per_step and st["tirages"] < max_tirages:
            st["tirages"] += 1
            pb = synth.make_problem(self.rng)
            check = None
            if self.rng.random() < cfg.instr_frac:
                opts = instructions_possibles(pb)
                if opts:
                    instr, check = self.rng.choice(opts)
                    q = f"{pb['q']} {instr}"
            if check is None:
                pre, suf = self.rng.choice(_DECORS)
                q = f"{pre}{pb['q']}{suf}"
            pids = self._prompt_ids(q)
            outs, ent = self._sample_group(pids)
            rs = []
            for o in outs:
                r, ok, fmt, conforme = self._reward(o, pb["ans"], check)
                rs.append(r)
                st["n"] += 1
                st["ok"] += ok
                st["fmt"] += fmt
                st["tok"] += len(o)
                if check is not None:
                    st["n_instr"] += 1
                    st["ok_instr"] += conforme
                if ok:
                    st["len_ok"].append(len(o))
            st["r_sum"] = st.get("r_sum", 0.0) + float(sum(rs))
            st["ent"] += ent
            arr = np.array(rs, dtype=np.float64)
            if arr.std() > 1e-6:                    # groupe tout bon/tout mauvais = muet
                st["util"] += 1
                groupes.append((pids, outs, arr - arr.mean()))
            if st["tirages"] == 1:                  # vitrine pour le dashboard
                i_best = int(np.argmax(rs))
                gen = self.tok.decode(outs[i_best], skip_special_tokens=False)
                self.last_sample = (f"[{rs[i_best]:.2f}] {q}\n→ " + gen.strip())[:600]
        st["ent"] /= max(1, st["tirages"])
        return groupes, st

    def _train_pass(self, groupes) -> dict:
        """Mise à jour de politique sur tous les groupes gardés.

        ppo_epochs passages sur le MÊME lot : la génération coûte ~10× le gradient,
        autant s'en servir plusieurs fois. Au premier passage le ratio d'importance
        vaut 1 par construction (π = π_old) donc le clip est inerte et le gradient
        est identique à du GRPO on-policy ; aux suivants, le clip asymétrique de
        DAPO borne la dérive. Les log-probs de référence, elles, ne dépendent pas
        de θ : calculées une fois, mises en cache pour les epochs suivants."""
        cfg = self.cfg
        seqs, pls, advs = [], [], []
        for pids, outs, adv in groupes:
            for o, a in zip(outs, adv):
                seqs.append(pids + o)
                pls.append(len(pids))
                advs.append(float(a))
        total_ct = sum(len(s) - pl for s, pl in zip(seqs, pls))
        self.model.train()

        # ---- micro-batches figés (mêmes découpes à chaque epoch) ----------------------
        micro = []
        for i0 in range(0, len(seqs), cfg.micro_bs):
            ch, cpl = seqs[i0:i0 + cfg.micro_bs], pls[i0:i0 + cfg.micro_bs]
            cad = advs[i0:i0 + cfg.micro_bs]
            Tm = max(len(s) for s in ch)
            x = torch.full((len(ch), Tm), self.sp["eot"], dtype=torch.long)
            m = torch.zeros((len(ch), Tm), dtype=torch.bool)
            for j, (s, pl) in enumerate(zip(ch, cpl)):
                x[j, :len(s)] = torch.tensor(s, dtype=torch.long)
                m[j, pl - 1:len(s) - 1] = True      # le logit t prédit le token t+1
            micro.append({"x": x.to(self.device), "m": m.to(self.device)[:, :Tm - 1],
                          "adv": torch.tensor(cad, device=self.device).unsqueeze(1),
                          "lp_old": None, "rlp": None})

        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda)
        n_ep = max(1, cfg.ppo_epochs)
        mult = min(1.0, (self.step + 1) / max(1, cfg.warmup))
        for grp in self.opt.param_groups:
            grp["lr"] = cfg.lr * mult
        pg_sum = kl_sum = lp_sum = clip_sum = 0.0
        gnorm = 0.0

        for ep in range(n_ep):
            pg_sum = kl_sum = lp_sum = clip_sum = 0.0   # on ne garde que le dernier epoch
            self.opt.zero_grad(set_to_none=True)
            for mb in micro:
                x, m, adv_t = mb["x"], mb["m"], mb["adv"]
                tgt = x[:, 1:]
                with amp:
                    logits, _, _ = self.model(x)
                # cross_entropy(reduction none) = −log p par token, sans matérialiser
                # le log_softmax fp32 complet (16k de vocab × T positions)
                V = logits.size(-1)
                lp = -F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                                      tgt.reshape(-1), reduction="none").view_as(tgt)
                if mb["rlp"] is None:               # π_ref est gelée : un seul calcul
                    with torch.no_grad(), amp:
                        rlogits, _, _ = self.ref(x)
                    mb["rlp"] = (-F.cross_entropy(rlogits[:, :-1].reshape(-1, V).float(),
                                                  tgt.reshape(-1),
                                                  reduction="none").view_as(tgt)).detach()
                if mb["lp_old"] is None:            # epoch 0 : π_old = π -> ratio ≡ 1
                    mb["lp_old"] = lp.detach()
                ratio = (lp - mb["lp_old"]).exp()
                surr = torch.min(ratio * adv_t,
                                 ratio.clamp(1.0 - cfg.clip_low, 1.0 + cfg.clip_high) * adv_t)
                diff = (mb["rlp"] - lp)             # estimateur k3 : ≥ 0, faible variance
                kl_tok = diff.exp() - diff - 1
                pg = -surr[m].sum() / total_ct
                kl = kl_tok[m].sum() / total_ct
                (pg + cfg.kl_beta * kl).backward()
                pg_sum += float(pg.detach())
                kl_sum += float(kl.detach())
                lp_sum += float(lp.detach()[m].sum())
                with torch.no_grad():
                    hors = ((ratio < 1.0 - cfg.clip_low) | (ratio > 1.0 + cfg.clip_high))
                    clip_sum += float(hors[m].sum())
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                         cfg.grad_clip))
            self.opt.step()
        return {"pg": pg_sum, "kl": kl_sum, "logp": lp_sum / max(1, total_ct),
                "gnorm": gnorm, "lr": cfg.lr * mult,
                "clipfrac": clip_sum / max(1, total_ct), "epochs": n_ep}

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def _greedy_final(self, question: str) -> str:
        ids = torch.tensor([self._prompt_ids(question)], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda):
            out = self.model.generate(ids, max_new_tokens=self.cfg.max_new_tokens,
                                      temperature=0.0, repetition_penalty=1.0,
                                      stop_ids=(self.sp["im_end"], self.sp["eot"]))
        txt = self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=False)
        return partie_finale(txt)

    def _evaluate(self):
        self.model.eval()
        ok_in = sum(verifier(a, self._greedy_final(q)) for q, a in self.eval_indist)
        ok_ood = sum(verifier(a, self._greedy_final(q)) for q, a in self.eval_ood)
        self.score_instr = sum(bool(chk(self._greedy_final(q))) for q, chk in self.eval_instr)
        self.model.train()
        self.score_in, self.score_ood = ok_in, ok_ood
        self.last_eval_step = self.step
        self.hist_eval.append(100 * (ok_in + ok_ood) / max(1, len(self.eval_indist) + len(self.eval_ood)))
        return ok_in + ok_ood

    # -----------------------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.columns import Columns
        from rich.text import Text

        human, hms = self._h["human"], self._h["hms"]
        console = Console()
        self.t_start = time.time()
        n_params = self.model.num_params()

        console.print(Panel.fit(
            f"[bold]{cfg.run_name}[/] · phase [cyan]rl (GRPO)[/] · "
            f"{human(n_params)} params · ancre KL = SFT gelé (β={cfg.kl_beta})\n"
            f"groupes : {cfg.prompts_per_step} problèmes × {cfg.group_size} réponses "
            f"= {cfg.prompts_per_step * cfg.group_size} rollouts/step · "
            f"T={cfg.temperature} · ≤{cfg.max_new_tokens} tokens\n"
            f"récompense : réponse juste (+1) · format (+0.1) · concision (+0.25 max) — "
            f"le tout vérifié par Python\n"
            f"éval : {len(self.eval_indist)} in-dist (graine vierge) + "
            f"{len(self.eval_ood)} OOD écrits main, greedy, toutes les {cfg.eval_every} steps",
            title="[bold green]Renforcement[/]", border_style="green"))
        console.print("[dim]Ctrl+C = arrêt propre (checkpoint sauvegardé). "
                      "Fichier STOP dans le dossier du run = pareil.[/]\n")

        def on_sigint(signum, frame):
            if self.stop_requested:
                console.print("\n[red]Second Ctrl+C : sortie immédiate.[/]")
                sys.exit(130)
            self.stop_requested = True
            console.print("\n[yellow]Arrêt demandé — sauvegarde à la fin du step…[/]")

        signal.signal(signal.SIGINT, on_sigint)
        stop_file = self.run_dir / "STOP"
        step_times = deque(maxlen=30)

        with Live(console=console, refresh_per_second=2, transient=False) as live:
            while self.step < cfg.max_steps and not self.stop_requested:
                t0 = time.perf_counter()
                self.model.eval()
                groupes, st = self._rollout()
                t_gen = time.perf_counter() - t0

                if groupes:
                    opt_stats = self._train_pass(groupes)
                else:                               # aucun signal : step blanc
                    opt_stats = {"pg": 0.0, "kl": 0.0, "logp": 0.0, "gnorm": 0.0,
                                 "clipfrac": 0.0, "epochs": 0,
                                 "lr": cfg.lr * min(1.0, (self.step + 1) / max(1, cfg.warmup))}
                t_all = time.perf_counter() - t0

                self.step += 1
                self.tokens_gen += st["tok"]
                step_times.append(t_all)
                pct_ok = 100 * st["ok"] / max(1, st["n"])
                self.hist_correct.append(pct_ok)
                self.correct_ema = pct_ok if self.correct_ema is None else \
                    0.95 * self.correct_ema + 0.05 * pct_ok

                r_step = st.get("r_sum", 0.0) / max(1, st["n"])
                self.hist_reward.append(r_step)
                self.reward_ema = r_step if self.reward_ema is None else \
                    0.95 * self.reward_ema + 0.05 * r_step

                rec = {"step": self.step, "reward": round(r_step, 4),
                       "correct_pct": round(pct_ok, 2),
                       "fmt_pct": round(100 * st["fmt"] / max(1, st["n"]), 2),
                       "kl": round(opt_stats["kl"], 6), "logp": round(opt_stats["logp"], 4),
                       "clipfrac": round(opt_stats.get("clipfrac", 0.0), 5),
                       "entropie": round(st["ent"], 4), "lr": opt_stats["lr"],
                       "gnorm": round(opt_stats["gnorm"], 4),
                       "groupes_utiles": st["util"], "tirages": st["tirages"],
                       "instr_pct": round(100 * st["ok_instr"] / max(1, st["n_instr"]), 1)
                       if st["n_instr"] else None,
                       "tokens_gen": self.tokens_gen,
                       "len_moy": round(st["tok"] / max(1, st["n"]), 1),
                       "tok_s_gen": round(st["tok"] / max(1e-6, t_gen))}
                self.metrics_file.write(json.dumps(rec) + "\n")
                self.metrics_file.flush()

                # ---- éval périodique + meilleur checkpoint -----------------------------
                is_best = False
                if self.step % cfg.eval_every == 0 or self.step == cfg.max_steps:
                    score = self._evaluate()
                    if score > self.best_score:
                        self.best_score = score
                        is_best = True

                due = (time.time() - self.last_ckpt_time) >= cfg.ckpt_every_min * 60
                if due or is_best or self.step >= cfg.max_steps or stop_file.exists():
                    self.ckpt.save(self._payload(), self.step, is_best=is_best)
                    self.last_ckpt_time = time.time()
                    tag = " [green](meilleur)[/]" if is_best else ""
                    self.last_ckpt_msg = f"step {self.step} · {time.strftime('%H:%M:%S')}{tag}"

                if stop_file.exists():
                    stop_file.unlink(missing_ok=True)
                    self.stop_requested = True

                live.update(self._dashboard(Group, Panel, Table, Columns, Text,
                                            st, opt_stats, t_gen, t_all, step_times))

        self.ckpt.save(self._payload(), self.step, is_best=False)
        self.metrics_file.close()
        console.print(f"\n[bold green]✓[/] Checkpoint final : {self.stage_dir / 'ckpt_latest.pt'}")
        console.print(f"  tester :  [cyan]python run.py chat --run {cfg.run_name}[/]  "
                      "(le chat charge la phase rl en priorité)")

    # -----------------------------------------------------------------------------------
    def _dashboard(self, Group, Panel, Table, Columns, Text, st, opt, t_gen, t_all, times):
        cfg = self.cfg
        human, hms, sparkline = self._h["human"], self._h["hms"], self._h["sparkline"]
        avg = float(np.mean(times)) if times else 0.0
        eta = (cfg.max_steps - self.step) * avg
        pct = self.step / max(1, cfg.max_steps)

        def kv(t, k, v, style=""):
            t.add_row(k, f"[{style}]{v}[/]" if style else str(v))

        t1 = Table.grid(padding=(0, 2))
        t1.add_column(style="dim", justify="right", min_width=16)
        t1.add_column(min_width=14)
        kv(t1, "récompense", f"{self.hist_reward[-1]:.3f}" if self.hist_reward else "—")
        kv(t1, "récompense EMA", f"{self.reward_ema:.3f}" if self.reward_ema else "—", "bold cyan")
        kv(t1, "% correct (step)", f"{self.hist_correct[-1]:.0f} %" if self.hist_correct else "—")
        kv(t1, "% correct EMA", f"{self.correct_ema:.1f} %" if self.correct_ema is not None else "—", "bold green")
        kv(t1, "% format propre", f"{100 * st['fmt'] / max(1, st['n']):.0f} %")
        kv(t1, "groupes utiles", f"{st['util']} sur {st['tirages']} tirés")
        if st["n_instr"]:
            kv(t1, "consignes suivies", f"{100 * st['ok_instr'] / max(1, st['n_instr']):.0f} %", "magenta")
        kv(t1, "long. moyenne", f"{st['tok'] / max(1, st['n']):.0f} tok")
        if st["len_ok"]:
            kv(t1, "long. si juste", f"{np.mean(st['len_ok']):.0f} tok", "green")

        t2 = Table.grid(padding=(0, 2))
        t2.add_column(style="dim", justify="right", min_width=16)
        t2.add_column(min_width=14)
        kv(t2, "KL vs SFT /tok", f"{opt['kl']:.5f}", "yellow")
        kv(t2, "log p /tok", f"{opt['logp']:.3f}")
        kv(t2, "entropie échant.", f"{st['ent']:.3f} nats")
        kv(t2, "learning rate", f"{opt['lr']:.2e}")
        kv(t2, "norme du grad", f"{opt['gnorm']:.3f}")
        kv(t2, "β KL", f"{cfg.kl_beta:g}")
        if cfg.ppo_epochs > 1:
            kv(t2, "tokens clippés", f"{100 * opt.get('clipfrac', 0.0):.2f} %",
               "red" if opt.get("clipfrac", 0.0) > 0.15 else "")
            kv(t2, "epochs PPO", f"{cfg.ppo_epochs} · clip −{cfg.clip_low:g}/+{cfg.clip_high:g}")
        kv(t2, "température", f"{cfg.temperature:g}")

        t3 = Table.grid(padding=(0, 2))
        t3.add_column(style="dim", justify="right", min_width=16)
        t3.add_column(min_width=14)
        n_in, n_ood = len(self.eval_indist), len(self.eval_ood)
        kv(t3, "éval in-dist", f"{self.score_in}/{n_in}" if self.score_in is not None else "—", "bold")
        kv(t3, "éval OOD", f"{self.score_ood}/{n_ood}" if self.score_ood is not None else "—", "bold magenta")
        kv(t3, "éval consignes", f"{self.score_instr}/{len(self.eval_instr)}"
           if self.score_instr is not None else "—")
        kv(t3, "meilleur total", f"{self.best_score}/{n_in + n_ood}" if self.best_score >= 0 else "—", "green")
        kv(t3, "dernière éval", f"step {self.last_eval_step}" if self.last_eval_step else "—")
        kv(t3, "prochaine éval", f"step {((self.step // cfg.eval_every) + 1) * cfg.eval_every}")

        t4 = Table.grid(padding=(0, 2))
        t4.add_column(style="dim", justify="right", min_width=16)
        t4.add_column(min_width=14)
        kv(t4, "step", f"{self.step}/{cfg.max_steps}  ({100 * pct:.1f} %)", "bold")
        kv(t4, "tokens générés", human(self.tokens_gen))
        kv(t4, "génération", f"{t_gen:.1f}s ({st['tok'] / max(1e-6, t_gen):.0f} tok/s)")
        kv(t4, "step complet", f"{t_all:.1f}s")
        kv(t4, "écoulé", hms(time.time() - self.t_start))
        kv(t4, "ETA", hms(eta), "cyan")
        if torch.cuda.is_available():
            kv(t4, "VRAM pic", f"{torch.cuda.max_memory_allocated() / 2**30:.2f} Go", "red")
        g = self.gpu.read()
        if g:
            kv(t4, "GPU", f"{g['util']} % · {g['temp']} °C · {g['power']:.0f} W")
        kv(t4, "dernier ckpt", self.last_ckpt_msg)

        bar_w = 58
        filled = int(bar_w * pct)
        bar = "[green]" + "━" * filled + "[/][dim]" + "━" * (bar_w - filled) + "[/]"

        lines = [f"[dim]%corr[/]  {sparkline(self.hist_correct)}  "
                 f"[green]{self.correct_ema:.1f} %[/]" if self.correct_ema is not None else "[dim]%corr[/]"]
        if self.hist_eval:
            lines.append(f"[dim]éval [/]  {sparkline(self.hist_eval)}  "
                         f"[magenta]{self.hist_eval[-1]:.1f} %[/]")
        spark = Panel(Text.from_markup("\n".join(lines)),
                      title="courbes (% correct train / éval)", border_style="dim", padding=(0, 1))
        sample = Panel(Text(self.last_sample or "(premier rollout en cours…)"),
                       title="dernier rollout (meilleure réponse du groupe)",
                       border_style="blue", padding=(0, 1))
        return Group(
            Columns([Panel(t1, title="[cyan]récompense[/]", border_style="cyan", padding=(0, 1)),
                     Panel(t2, title="[yellow]politique[/]", border_style="yellow", padding=(0, 1)),
                     Panel(t3, title="[magenta]éval[/]", border_style="magenta", padding=(0, 1)),
                     Panel(t4, title="[white]avancement[/]", border_style="white", padding=(0, 1))],
                    equal=False, expand=False),
            Text.from_markup(bar), spark, sample)


# --------------------------------------------------------------------------------------
def cmd_rl(args):
    cfg = RLConfig(run_name=args.run, out_dir=args.out_dir, max_steps=args.max_steps,
                   prompts_per_step=args.prompts, group_size=args.group,
                   max_new_tokens=args.max_new, temperature=args.temperature,
                   top_p=args.top_p, lr=args.lr, kl_beta=args.kl_beta,
                   micro_bs=args.micro_bs, eval_every=args.eval_every, seed=args.seed,
                   ckpt_every_min=args.ckpt_every_min,
                   instr_frac=args.instr_frac, oversample=args.oversample)
    RLTrainer(cfg, resume=args.resume).train()
