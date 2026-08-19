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

L'algorithme (GRPO, façon Dr. GRPO) :
  - pour chaque problème, on échantillonne G réponses complètes (température 1) ;
  - récompense r_i par réponse : +1 si la réponse finale est juste, +0.1 si le
    format est propre (un seul <think>, arrêt naturel), +0.25 × (1 − len/cible)
    de bonus de concision quand c'est juste ;
  - avantage A_i = r_i − moyenne(groupe) : pas de critic, le groupe EST la
    baseline. Un groupe tout juste ou tout faux n'apprend rien -> écarté ;
  - perte = −moyenne_tokens(A_i · log p(token)) + β · KL(π ‖ π_SFT), une seule
    mise à jour par lot (le ratio d'importance vaut 1, pas besoin de clip) ;
  - KL estimée par token avec l'estimateur k3 (exp(d) − d − 1, toujours ≥ 0).

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
    brevity_target: int = 160       # tokens : au-delà, plus de bonus de concision
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
        self.mcfg = ModelConfig.from_dict(ck["model_cfg"])

        self.model = build_model(self.mcfg).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.ref = build_model(self.mcfg).to(self.device)      # π_SFT gelé : l'ancre KL
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
        self.score_in = self.score_ood = None
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
                "scores": {"indist": self.score_in, "ood": self.score_ood},
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
    def _reward(self, gen_ids: list[int], ans) -> tuple[float, bool, bool]:
        naturel = bool(gen_ids) and gen_ids[-1] in (self.sp["im_end"], self.sp["eot"])
        txt = self.tok.decode(gen_ids, skip_special_tokens=False)
        final = partie_finale(txt)
        fmt = naturel and final != "" and txt.count(synth.THINK_END) <= 1
        correct = fmt and verifier(ans, final)
        r = 0.1 if fmt else 0.0
        if correct:
            # bonus de concision : pousse vers le brouillon court façon synth,
            # contre le radotage GSM8K (« Attends, laisse-moi vérifier… » × 40)
            r += 1.0 + 0.25 * max(0.0, 1.0 - len(gen_ids) / self.cfg.brevity_target)
        return r, correct, fmt

    def _rollout(self):
        """prompts_per_step groupes ; ne garde que ceux qui portent un gradient."""
        cfg = self.cfg
        groupes, st = [], dict(n=0, ok=0, fmt=0, tok=0, ent=0.0, util=0, len_ok=[])
        for g in range(cfg.prompts_per_step):
            pb = synth.make_problem(self.rng)
            pre, suf = self.rng.choice(_DECORS)
            q = f"{pre}{pb['q']}{suf}"
            pids = self._prompt_ids(q)
            outs, ent = self._sample_group(pids)
            st["ent"] += ent / cfg.prompts_per_step
            rs = []
            for o in outs:
                r, ok, fmt = self._reward(o, pb["ans"])
                rs.append(r)
                st["n"] += 1
                st["ok"] += ok
                st["fmt"] += fmt
                st["tok"] += len(o)
                if ok:
                    st["len_ok"].append(len(o))
            arr = np.array(rs, dtype=np.float64)
            if arr.std() > 1e-6:                    # groupe tout bon/tout mauvais = muet
                st["util"] += 1
                groupes.append((pids, outs, arr - arr.mean()))
            if g == 0:                              # vitrine pour le dashboard
                i_best = int(np.argmax(rs))
                gen = self.tok.decode(outs[i_best], skip_special_tokens=False)
                self.last_sample = (f"[{rs[i_best]:.2f}] {q}\n→ " + gen.strip())[:600]
        return groupes, st

    def _train_pass(self, groupes) -> dict:
        """Une mise à jour de politique sur tous les groupes gardés."""
        cfg = self.cfg
        seqs, pls, advs = [], [], []
        for pids, outs, adv in groupes:
            for o, a in zip(outs, adv):
                seqs.append(pids + o)
                pls.append(len(pids))
                advs.append(float(a))
        total_ct = sum(len(s) - pl for s, pl in zip(seqs, pls))
        pg_sum = kl_sum = lp_sum = 0.0
        self.opt.zero_grad(set_to_none=True)
        self.model.train()
        for i0 in range(0, len(seqs), cfg.micro_bs):
            ch, cpl = seqs[i0:i0 + cfg.micro_bs], pls[i0:i0 + cfg.micro_bs]
            cad = advs[i0:i0 + cfg.micro_bs]
            Tm = max(len(s) for s in ch)
            x = torch.full((len(ch), Tm), self.sp["eot"], dtype=torch.long)
            m = torch.zeros((len(ch), Tm), dtype=torch.bool)
            for j, (s, pl) in enumerate(zip(ch, cpl)):
                x[j, :len(s)] = torch.tensor(s, dtype=torch.long)
                m[j, pl - 1:len(s) - 1] = True      # le logit t prédit le token t+1
            x, m = x.to(self.device), m.to(self.device)
            m = m[:, :Tm - 1]
            tgt = x[:, 1:]
            adv_t = torch.tensor(cad, device=self.device).unsqueeze(1)

            amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda)
            with amp:
                logits, _, _ = self.model(x)
            # cross_entropy(reduction none) = −log p par token, sans matérialiser
            # le log_softmax fp32 complet (16k de vocab × T positions)
            V = logits.size(-1)
            lp = -F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                                  tgt.reshape(-1), reduction="none").view_as(tgt)
            with torch.no_grad(), amp:
                rlogits, _, _ = self.ref(x)
            rlp = -F.cross_entropy(rlogits[:, :-1].reshape(-1, V).float(),
                                   tgt.reshape(-1), reduction="none").view_as(tgt)
            diff = (rlp - lp)                       # estimateur k3 : ≥ 0, faible variance
            kl_tok = diff.exp() - diff - 1
            pg = -(adv_t * lp)[m].sum() / total_ct
            kl = kl_tok[m].sum() / total_ct
            (pg + cfg.kl_beta * kl).backward()
            pg_sum += float(pg.detach())
            kl_sum += float(kl.detach())
            lp_sum += float(lp.detach()[m].sum())
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        mult = min(1.0, (self.step + 1) / max(1, cfg.warmup))
        for grp in self.opt.param_groups:
            grp["lr"] = cfg.lr * mult
        self.opt.step()
        return {"pg": pg_sum, "kl": kl_sum, "logp": lp_sum / max(1, total_ct),
                "gnorm": float(gnorm), "lr": cfg.lr * mult}

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def _answer_ok(self, question: str, ans) -> bool:
        ids = torch.tensor([self._prompt_ids(question)], device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda):
            out = self.model.generate(ids, max_new_tokens=self.cfg.max_new_tokens,
                                      temperature=0.0, repetition_penalty=1.0,
                                      stop_ids=(self.sp["im_end"], self.sp["eot"]))
        txt = self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=False)
        return verifier(ans, partie_finale(txt))

    def _evaluate(self):
        self.model.eval()
        ok_in = sum(self._answer_ok(q, a) for q, a in self.eval_indist)
        ok_ood = sum(self._answer_ok(q, a) for q, a in self.eval_ood)
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
                                 "lr": cfg.lr * min(1.0, (self.step + 1) / max(1, cfg.warmup))}
                t_all = time.perf_counter() - t0

                self.step += 1
                self.tokens_gen += st["tok"]
                step_times.append(t_all)
                pct_ok = 100 * st["ok"] / max(1, st["n"])
                self.hist_correct.append(pct_ok)
                self.correct_ema = pct_ok if self.correct_ema is None else \
                    0.95 * self.correct_ema + 0.05 * pct_ok

                # récompense moyenne réelle du step (reconstituée des compteurs)
                # (on la retrace exactement : 0.1·fmt + 1·ok + bonus concision moyen)
                bonus = sum(0.25 * max(0.0, 1.0 - L / cfg.brevity_target)
                            for L in st["len_ok"])
                r_step = (0.1 * st["fmt"] + 1.0 * st["ok"] + bonus) / max(1, st["n"])
                self.hist_reward.append(r_step)
                self.reward_ema = r_step if self.reward_ema is None else \
                    0.95 * self.reward_ema + 0.05 * r_step

                rec = {"step": self.step, "reward": round(r_step, 4),
                       "correct_pct": round(pct_ok, 2),
                       "fmt_pct": round(100 * st["fmt"] / max(1, st["n"]), 2),
                       "kl": round(opt_stats["kl"], 6), "logp": round(opt_stats["logp"], 4),
                       "entropie": round(st["ent"], 4), "lr": opt_stats["lr"],
                       "gnorm": round(opt_stats["gnorm"], 4),
                       "groupes_utiles": st["util"], "tokens_gen": self.tokens_gen,
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
        kv(t1, "groupes utiles", f"{st['util']}/{cfg.prompts_per_step}")
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
        kv(t2, "température", f"{cfg.temperature:g}")

        t3 = Table.grid(padding=(0, 2))
        t3.add_column(style="dim", justify="right", min_width=16)
        t3.add_column(min_width=14)
        n_in, n_ood = len(self.eval_indist), len(self.eval_ood)
        kv(t3, "éval in-dist", f"{self.score_in}/{n_in}" if self.score_in is not None else "—", "bold")
        kv(t3, "éval OOD", f"{self.score_ood}/{n_ood}" if self.score_ood is not None else "—", "bold magenta")
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
                   ckpt_every_min=args.ckpt_every_min)
    RLTrainer(cfg, resume=args.resume).train()
