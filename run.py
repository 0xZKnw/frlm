#!/usr/bin/env python
"""
run.py — Point d'entrée unique.

    python run.py prepare            # télécharge le français, entraîne le tokenizer, binarise
    python run.py train              # pré-entraînement (Ctrl+C = arrêt propre avec checkpoint)
    python run.py train --resume     # reprend au dernier checkpoint
    python run.py chat               # discute avec le dernier checkpoint
    python run.py sft                # affine le modèle sur les dialogues
    python run.py info               # inspecte un checkpoint / le corpus

Tout est checkpointé toutes les N minutes (5 par défaut) : on peut couper
l'entraînement à tout moment, tester le modèle, puis reprendre exactement où on
en était (step, optimiseur, RNG, ordre des batchs).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from frlm import data as D
from frlm import config_from_dict, model_from_cfg
from frlm.model import PRESETS, ModelConfig, build_model
from frlm.model_v3 import PRESETS_V3, ModelConfigV3
from frlm.optim import build_optimizers, lr_multiplier

ROOT = Path(__file__).resolve().parent

# Windows : la console est en cp1252 par défaut, ce qui casse tout affichage
# d'accents / de caractères de dessin. On force l'UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ======================================================================================
# Utilitaires d'affichage
# ======================================================================================
BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(vals, width: int = 58) -> str:
    if not vals:
        return ""
    v = list(vals)[-width:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-9:
        return BLOCKS[0] * len(v)
    return "".join(BLOCKS[min(7, int((x - lo) / (hi - lo) * 7.999))] for x in v)


def human(n: float, unit: str = "") -> str:
    for suf, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n/div:.2f}{suf}{unit}"
    return f"{n:.0f}{unit}"


def hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class GpuMon:
    """Télémétrie GPU via NVML (optionnelle)."""

    def __init__(self):
        self.h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None

    def read(self) -> dict:
        if self.h is None:
            return {}
        n = self.nvml
        try:
            u = n.nvmlDeviceGetUtilizationRates(self.h)
            return {
                "util": u.gpu,
                "mem_util": u.memory,
                "temp": n.nvmlDeviceGetTemperature(self.h, n.NVML_TEMPERATURE_GPU),
                "power": n.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                "power_cap": n.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000.0,
                "clock": n.nvmlDeviceGetClockInfo(self.h, n.NVML_CLOCK_SM),
            }
        except Exception:
            return {}


# ======================================================================================
# Configuration d'entraînement
# ======================================================================================
@dataclass
class TrainConfig:
    run_name: str = "fr-micro"
    data_dir: str = "data"
    out_dir: str = "runs"
    stage: str = "pretrain"           # "pretrain" | "mid" | "sft"
    preset: str = "micro"
    hybrid: bool = False              # archi Qwen3.5 complete (DeltaNet 3:1), ~3,5x plus lent

    # batch (calibré RTX 4060 + torch.compile : 58k tok/s, 4,8 Go de pic)
    batch_size: int = 16
    grad_accum: int = 2
    seq_len: int = 1024

    # optimisation
    optimizer: str = "muon"           # "muon" | "adamw"
    lr: float = 0.02                  # LR Muon (matrices). Pour adamw : 6e-4
    adam_lr: float = 1.5e-3           # LR AdamW (embeddings + normes)
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    schedule: str = "wsd"             # "wsd" | "cosine"
    warmup: int = 300
    min_lr_frac: float = 0.02
    decay_frac: float = 0.2
    z_loss: float = 1e-4
    max_steps: int = 20000
    replay_frac: float = 0.0          # part de batchs mid rejoués pendant le SFT
    replay_mix: str = ""              # chemins=poids, ex. train.bin=0.7,mid...=0.3
    replay_val: str = ""              # validation non masquée associée au replay
    mid_curriculum: str = ""          # "v4.3" -> deux bins 80/20 spécialisés
    sft_recipe: str = ""              # "v4.4" -> sampling explicite par capacité

    # évaluation / logs
    eval_every: int = 500
    eval_iters: int = 40
    log_every: int = 10
    profile_every: int = 50
    sample_every: int = 1000
    sample_tokens: int = 120

    # checkpoints
    ckpt_every_min: float = 5.0
    save_every_steps: int = 0          # 0 = seulement minuterie/best/fin
    keep_last: int = 3

    # système
    seed: int = 1337
    compile: bool = True                # +94% de débit mesuré (triton-windows requis)
    dtype: str = "bfloat16"
    device: str = "cuda"
    # Pic bf16 DENSE reel d'une RTX 4060 : ~30 TFLOPS. Le "121 TFLOPS" du
    # marketing NVIDIA compte le FP8 avec sparsite, ce qui ne s'applique pas ici.
    gpu_peak_tflops: float = 30.0

    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.seq_len


# ======================================================================================
# Checkpoints
# ======================================================================================
class CheckpointManager:
    def __init__(self, run_dir: Path, keep_last: int = 3):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def _atomic_save(self, payload: dict, path: Path):
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def save(self, payload: dict, step: int, is_best: bool = False) -> Path:
        latest = self.dir / "ckpt_latest.pt"
        self._atomic_save(payload, latest)
        rolling = self.dir / f"ckpt_step{step:07d}.pt"
        shutil.copyfile(latest, rolling)
        if is_best:
            shutil.copyfile(latest, self.dir / "ckpt_best.pt")
        # rotation
        olds = sorted(self.dir.glob("ckpt_step*.pt"))
        for p in olds[: max(0, len(olds) - self.keep_last)]:
            p.unlink(missing_ok=True)
        return latest

    def resolve(self, spec: str) -> Path | None:
        if spec in ("latest", "auto", ""):
            p = self.dir / "ckpt_latest.pt"
            return p if p.exists() else None
        if spec == "best":
            p = self.dir / "ckpt_best.pt"
            return p if p.exists() else None
        p = Path(spec)
        return p if p.exists() else None


# ======================================================================================
# Entraîneur
# ======================================================================================
class Trainer:
    def __init__(self, cfg: TrainConfig, resume: str | None = None):
        self.cfg = cfg
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        # Un dossier de checkpoints PAR PHASE : un SFT ne doit jamais écraser le
        # pré-entraînement, sinon on ne peut plus reprendre ce dernier.
        self.stage_dir = self.run_dir / cfg.stage
        self.ckpt = CheckpointManager(self.stage_dir, cfg.keep_last)
        self.gpu = GpuMon()
        self.stop_requested = False

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                          "float32": torch.float32}[cfg.dtype]

        # ---- tokenizer & corpus ------------------------------------------------------
        data_dir = Path(cfg.data_dir)
        tok_path = data_dir / "tokenizer.json"
        if not tok_path.exists():
            sys.exit(f"[!] Tokenizer introuvable ({tok_path}). Lance d'abord :  python run.py prepare")
        self.tok = D.load_tokenizer(tok_path)
        self.sp = D.special_ids(self.tok)

        masked = cfg.stage == "sft"
        self.val_sources: dict[str, D.BinCorpus] = {}
        self.mid_curriculum: list[tuple[float, str, D.BinCorpus]] = []
        sft_recipe = cfg.sft_recipe.casefold().replace("v", "").replace(".", "")
        if masked and sft_recipe:
            if sft_recipe not in ("44", "45"):
                sys.exit(f"[!] Recette SFT inconnue : {cfg.sft_recipe}")
            try:
                key = f"sft_v{sft_recipe}"
                expected_recipe = {
                    "44": "v4.4-balanced-capabilities-18m",
                    "45": "v4.5-audited-isolated-24m",
                }[sft_recipe]
                section = json.loads((data_dir / "meta.json").read_text(
                    encoding="utf-8"))[key]
                if section["recipe"] != expected_recipe:
                    raise ValueError(f"recette inattendue : {section['recipe']}")
                corpus_cls = D.ConversationCorpus if sft_recipe == "45" else D.BinCorpus
                corpora = []
                for name, capability in section["capabilities"].items():
                    if int(capability["actual_supervised"]) <= 0:
                        continue
                    if sft_recipe == "45":
                        train = corpus_cls(data_dir / capability["train_path"], cfg.seq_len)
                    else:
                        train = corpus_cls(data_dir / capability["train_path"],
                                           cfg.seq_len, with_mask=True)
                    sampling_weight = (capability["train_conversations"]
                                       if sft_recipe == "45"
                                       else capability["actual_supervised"])
                    corpora.append((name, train, float(sampling_weight)))
                    if sft_recipe == "45":
                        self.val_sources[name] = corpus_cls(
                            data_dir / capability["val_path"], cfg.seq_len
                        )
                    else:
                        self.val_sources[name] = corpus_cls(
                            data_dir / capability["val_path"], cfg.seq_len, with_mask=True
                        )
                self.train_data = D.SourceMixtureCorpus(corpora)
                if sft_recipe == "45":
                    self.val_data = D.ConversationCorpus(
                        data_dir / section["val_path"], cfg.seq_len
                    )
                else:
                    self.val_data = D.BinCorpus(data_dir / section["val_path"],
                                                cfg.seq_len, with_mask=True)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                sys.exit(f"[!] Données SFT v4.{sft_recipe[-1]} absentes ou incohérentes ({exc}). "
                         f"Lance : python run.py prepare-sft-v{sft_recipe} --data-dir data-v4")
            total_weight = sum(weight for _, _, weight in corpora)
            detail = ", ".join(
                f"{name}={100 * weight / total_weight:.0f}%"
                for name, _, weight in corpora
            )
            print(f"[i] SFT v4.{sft_recipe[-1]} par capacités actif : {detail}")
        elif cfg.stage == "mid" and cfg.mid_curriculum:
            recipe = cfg.mid_curriculum.casefold().replace("v", "").replace(".", "")
            if recipe != "43":
                sys.exit(f"[!] Curriculum mid inconnu : {cfg.mid_curriculum}")
            meta_path = data_dir / "meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))["midtrain_v43"]
                for stage_meta in meta["stages"]:
                    stage_path = data_dir / stage_meta["path"]
                    corpus = D.BinCorpus(stage_path, cfg.seq_len)
                    self.mid_curriculum.append((float(stage_meta["end_fraction"]),
                                                stage_meta["name"], corpus))
                val_bin = data_dir / meta["validation"]["path"]
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                sys.exit(f"[!] Données du curriculum v4.3 absentes ou incohérentes ({exc}). "
                         "Lance : python run.py prepare-mid-v43 --data-dir data-v4")
            self.train_data = self.mid_curriculum[0][2]
            self.val_data = D.BinCorpus(val_bin, cfg.seq_len)
            detail = " → ".join(f"{name} jusqu'à {end:.0%}"
                                for end, name, _ in self.mid_curriculum)
            print(f"[i] curriculum mid v4.3 actif : {detail}")
        else:
            prefix = {"sft": "sft_", "mid": "mid_"}.get(cfg.stage, "")
            train_bin = data_dir / f"{prefix}train.bin"
            val_bin = data_dir / f"{prefix}val.bin"
            if not train_bin.exists():
                sys.exit(f"[!] {train_bin} introuvable. Lance :  python run.py prepare")
            self.train_data = D.BinCorpus(train_bin, cfg.seq_len, with_mask=masked)
            self.val_data = D.BinCorpus(val_bin, cfg.seq_len, with_mask=masked)
        self.replay_train = self.replay_val = None
        if masked:
            if not self.val_sources:
                for source_path in sorted(data_dir.glob("sft_val_*.bin")):
                    source = source_path.stem.removeprefix("sft_val_")
                    try:
                        self.val_sources[source] = D.BinCorpus(source_path, cfg.seq_len,
                                                               with_mask=True)
                    except ValueError as exc:
                        print(f"[!] validation {source} ignorée ({exc})")
            mid_train, mid_val = data_dir / "mid_train.bin", data_dir / "mid_val.bin"
            if cfg.replay_frac > 0 and cfg.replay_mix:
                replay_corpora = []
                try:
                    for index, item in enumerate(cfg.replay_mix.split(",")):
                        path_text, weight_text = item.rsplit("=", 1)
                        path = Path(path_text.strip())
                        path = path if path.is_absolute() else data_dir / path
                        replay_corpora.append((path.stem, D.BinCorpus(path, cfg.seq_len),
                                               float(weight_text)))
                except (OSError, ValueError) as exc:
                    sys.exit(f"[!] --replay-mix invalide ({exc})")
                self.replay_train = D.SourceMixtureCorpus(replay_corpora)
                base_val = Path(cfg.replay_val) if cfg.replay_val else data_dir / "val.bin"
                if not base_val.is_absolute():
                    base_val = data_dir / base_val
                if not base_val.exists():
                    sys.exit(f"[!] validation replay introuvable : {base_val}")
                self.replay_val = D.BinCorpus(base_val, cfg.seq_len)
                print(f"[i] replay anti-oubli actif : {100*cfg.replay_frac:.0f} % "
                      f"({cfg.replay_mix})")
            elif cfg.replay_frac > 0 and mid_train.exists() and mid_val.exists():
                self.replay_train = D.BinCorpus(mid_train, cfg.seq_len)
                self.replay_val = D.BinCorpus(mid_val, cfg.seq_len)
                print(f"[i] replay anti-oubli actif : {100*cfg.replay_frac:.0f} % de batchs mid")
            elif cfg.replay_frac > 0:
                sys.exit("[!] --replay-frac exige mid_train.bin et mid_val.bin dans --data-dir")

        # ---- modèle ------------------------------------------------------------------
        # le nom du preset choisit l'architecture : "v3-*" -> model_v3 (speedrun)
        if cfg.preset in PRESETS_V3:
            mcfg = ModelConfigV3(**PRESETS_V3[cfg.preset])
        else:
            mcfg = ModelConfig(**PRESETS[cfg.preset])
            mcfg.hybrid = cfg.hybrid
        mcfg.vocab_size = self.tok.get_vocab_size()
        mcfg.max_seq_len = max(mcfg.max_seq_len, cfg.seq_len)
        mcfg.eos_id = self.sp["eot"]
        mcfg.bos_id = self.sp["eot"]
        self.mcfg = mcfg

        self.model = model_from_cfg(mcfg).to(self.device)
        self.raw_model = self.model
        self.opts, self.opt_info = build_optimizers(self.model, cfg)

        # ---- état --------------------------------------------------------------------
        self.step = 0
        self.tokens_seen = 0
        self.best_val = float("inf")
        self.elapsed_prev = 0.0
        self.val_loss = float("nan")
        self.val_breakdown: dict[str, float] = {}
        self.replay_val_loss = float("nan")
        self.last_sample = ""
        self.last_sample_step = 0

        if resume:
            self.load_checkpoint(resume)

        if cfg.compile:
            try:
                import triton  # noqa: F401  (vérifie que le backend inductor a son compilateur)
                self.model = torch.compile(self.model)
                print("[i] torch.compile actif — le premier step compile (~30-60 s), c'est normal")
            except ImportError:
                print("[!] triton absent — entraînement non compilé (~2x plus lent). "
                      "Installe :  pip install \"triton-windows<3.3\"")
            except Exception as e:
                print(f"[!] torch.compile indisponible ({e}) — on continue sans")

        self.metrics_file = (self.stage_dir / "metrics.jsonl").open("a", encoding="utf-8")
        (self.stage_dir / "config.json").write_text(
            json.dumps({"train": asdict(cfg), "model": mcfg.to_dict()}, indent=2), encoding="utf-8")
        shutil.copyfile(tok_path, self.run_dir / "tokenizer.json")

        # ---- historique pour le dashboard --------------------------------------------
        self.hist_loss: deque[float] = deque(maxlen=400)
        self.hist_val: deque[float] = deque(maxlen=100)
        self.loss_ema = None
        self.tps_ema = None
        self.clip_hits = deque(maxlen=100)
        self.breakdown = {"data": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}
        self.last_ckpt_msg = "—"
        self.last_ckpt_time = time.time()

    # ---------------------------------------------------------------------------------
    def state_payload(self) -> dict:
        return {
            "model": self.raw_model.state_dict(),
            "optimizers": [o.state_dict() for o in self.opts],
            "model_cfg": self.mcfg.to_dict(),
            "train_cfg": asdict(self.cfg),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val": self.best_val,
            "val_loss": self.val_loss,
            "elapsed": self.elapsed_prev + (time.time() - self.t_start),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
            },
            "stage": self.cfg.stage,
        }

    def training_corpus(self, step: int):
        """Corpus actif du curriculum ; une reprise retombe sur la bonne étape."""
        if not self.mid_curriculum:
            return self.train_data
        fraction = step / max(1, self.cfg.max_steps)
        for end_fraction, _, corpus in self.mid_curriculum:
            if fraction < end_fraction:
                return corpus
        return self.mid_curriculum[-1][2]

    def load_checkpoint(self, spec: str):
        path = self.ckpt.resolve(spec)
        if path is None:
            # Un Volume peut ne contenir que latest alors que la copie locale ne
            # contient que best (ou inversement). Pour un chemin explicite, essayer
            # le checkpoint frère de la même phase avant tout autre repli.
            requested = Path(spec)
            sibling = {"ckpt_best.pt": "ckpt_latest.pt",
                       "ckpt_latest.pt": "ckpt_best.pt"}.get(requested.name)
            if sibling:
                alternate = requested.with_name(sibling)
                if alternate.exists():
                    path = alternate
                    print(f"[i] {requested.name} absent, repli sur {alternate}")
        if path is None and spec in ("latest", "auto", ""):
            # Un téléchargement depuis Modal garde souvent seulement ckpt_best.pt.
            # Retomber dessus vaut infiniment mieux qu'un démarrage silencieux à zéro.
            path = self.ckpt.resolve("best")
            if path:
                print(f"[i] {self.cfg.stage} : latest absent, repli sur {path.name}")
        if path is None and self.cfg.stage == "sft":
            # pas encore de checkpoint SFT -> on part du midtrain s'il existe, sinon
            # du pré-entraînement
            for phase in ("mid", "pretrain"):
                manager = CheckpointManager(self.run_dir / phase)
                for candidate in ("latest", "best"):
                    path = manager.resolve(candidate)
                    if path:
                        print(f"[i] Démarrage du SFT depuis la phase '{phase}' : {path}")
                        break
                if path:
                    break
        if path is None and self.cfg.stage == "mid":
            manager = CheckpointManager(self.run_dir / "pretrain")
            for candidate in ("latest", "best"):
                path = manager.resolve(candidate)
                if path:
                    print(f"[i] Démarrage du midtrain depuis le pré-entraînement : {path}")
                    break
        if path is None:
            sys.exit(f"[!] Checkpoint demandé introuvable ({spec}). Refus de démarrer "
                     f"la phase '{self.cfg.stage}' à zéro.")
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ck["model"])
        # au passage pretrain -> sft on repart des poids mais pas de l'état optimiseur
        same_stage = ck.get("stage", "pretrain") == self.cfg.stage
        if same_stage and len(ck.get("optimizers", [])) == len(self.opts):
            for o, sd in zip(self.opts, ck["optimizers"]):
                o.load_state_dict(sd)
            self.step = ck["step"]
            self.tokens_seen = ck["tokens_seen"]
            self.best_val = ck.get("best_val", float("inf"))
            self.elapsed_prev = ck.get("elapsed", 0.0)
            try:
                torch.set_rng_state(ck["rng"]["torch"].cpu() if hasattr(ck["rng"]["torch"], "cpu") else ck["rng"]["torch"])
                if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng"]["cuda"]])
                np.random.set_state(ck["rng"]["numpy"])
            except Exception:
                pass
            print(f"[i] Reprise depuis {path.name} — step {self.step}, {human(self.tokens_seen)} tokens vus")
        else:
            print(f"[i] Poids chargés depuis {path.name} (nouvelle phase '{self.cfg.stage}' : optimiseur réinitialisé)")

    # ---------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        def corpus_loss(corpus, n_iters: int, seed: int) -> float:
            loss_sum = 0.0
            token_count = 0
            for i in range(n_iters):
                x, y, m = corpus.get_batch(i, self.cfg.batch_size, seed=seed, device=self.device)
                with torch.autocast("cuda", dtype=self.amp_dtype,
                                    enabled=self.device.startswith("cuda")):
                    reduction = "sum" if m is not None else "mean"
                    _, loss, _ = self.model(
                        x, y, m, z_loss=0.0, diagnostics=False,
                        loss_reduction=reduction,
                    )
                if m is None:
                    loss_sum += loss.item()
                    token_count += 1
                else:
                    loss_sum += loss.item()
                    token_count += int(m.sum().item())
            return loss_sum / max(1, token_count)

        if self.val_sources:
            per_source_iters = max(2, self.cfg.eval_iters // len(self.val_sources))
            self.val_breakdown = {
                source: corpus_loss(corpus, per_source_iters, seed=999 + i * 17)
                for i, (source, corpus) in enumerate(self.val_sources.items())
            }
            val = float(np.mean(list(self.val_breakdown.values())))
        else:
            self.val_breakdown = {}
            val = corpus_loss(self.val_data, self.cfg.eval_iters, seed=999)
        if self.replay_val is not None:
            self.replay_val_loss = corpus_loss(self.replay_val, min(6, self.cfg.eval_iters), seed=1777)
        self.model.train()
        return val

    @torch.no_grad()
    def sample(self, prompt: str | None = None) -> str:
        self.model.eval()
        if self.cfg.stage == "sft":
            q = prompt or "Salut ! Tu peux te présenter en deux phrases ?"
            text = f"{D.IM_START}user\n{q}{D.IM_END}\n{D.IM_START}assistant\n"
        else:
            text = prompt or "La capitale de la France est"
        ids = torch.tensor([self.tok.encode(text).ids], device=self.device)
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.device.startswith("cuda")):
            out = self.raw_model.generate(
                ids, max_new_tokens=self.cfg.sample_tokens, temperature=0.8, top_k=50, top_p=0.95,
                stop_ids=(self.sp["im_end"], self.sp["eot"]),
            )
        gen = self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=False)
        self.model.train()
        with (self.stage_dir / "samples.txt").open("a", encoding="utf-8") as f:
            f.write(f"\n===== step {self.step} =====\n{text}{gen}\n")
        return gen.strip()

    # ---------------------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.columns import Columns
        from rich.text import Text

        console = Console()
        self.t_start = time.time()
        self.model.train()

        n_params = self.raw_model.num_params()
        n_ne = self.raw_model.num_params(non_embedding=True)
        fpt = self.raw_model.flops_per_token()
        tps_step = cfg.tokens_per_step()
        if cfg.optimizer == "muon":
            optim_detail = (
                f"{human(self.opt_info['muon_params'])} via Muon, "
                f"{human(self.opt_info['adam_params'])} via AdamW"
            )
        else:
            total_optim = self.opt_info["muon_params"] + self.opt_info["adam_params"]
            optim_detail = f"{human(total_optim)} via AdamW"

        console.print(Panel.fit(
            f"[bold]{cfg.run_name}[/] · phase [cyan]{cfg.stage}[/] · preset [cyan]{cfg.preset}[/]\n"
            f"params : [bold]{human(n_params)}[/] (dont {human(n_ne)} hors embeddings) · vocab {self.mcfg.vocab_size}\n"
            f"archi  : {self.mcfg.n_layer}L [{self.raw_model.describe()}] · d={self.mcfg.d_model} · "
            f"{self.mcfg.n_head}Q/{self.mcfg.n_kv_head}KV (GQA {self.mcfg.n_head//self.mcfg.n_kv_head}:1) · "
            f"gate+QK-Norm zc · RoPE {self.mcfg.rope_dims}/{self.mcfg.head_dim} · "
            f"ffn={self.mcfg.d_ff} · ctx={cfg.seq_len}\n"
            f"batch  : {cfg.batch_size} × {cfg.grad_accum} accum × {cfg.seq_len} = [bold]{human(tps_step)}[/] tokens/step\n"
            f"optim  : {cfg.optimizer} ({optim_detail}) · schedule {cfg.schedule}\n"
            f"corpus : {human(len(self.train_data))} tokens train / {human(len(self.val_data))} val\n"
            f"cible  : {cfg.max_steps} steps = {human(cfg.max_steps*tps_step)} tokens "
            f"({cfg.max_steps*tps_step/max(1,n_ne):.1f} tokens/param hors emb.)",
            title="[bold green]Entraînement[/]", border_style="green"))
        console.print("[dim]Ctrl+C = arrêt propre (checkpoint sauvegardé). "
                      f"Checkpoint auto toutes les {cfg.ckpt_every_min:g} min.[/]\n")

        def on_sigint(signum, frame):
            if self.stop_requested:
                console.print("\n[red]Second Ctrl+C : sortie immédiate.[/]")
                sys.exit(130)
            self.stop_requested = True
            console.print("\n[yellow]Arrêt demandé — sauvegarde à la fin du step en cours…[/]")

        signal.signal(signal.SIGINT, on_sigint)
        stop_file = self.run_dir / "STOP"

        use_cuda = self.device.startswith("cuda")
        step_times = deque(maxlen=50)

        # sans TTY (logs Modal, nohup, CI) le tableau Live n'est jamais rafraîchi :
        # on bascule sur des lignes de log classiques
        interactive = console.is_terminal
        print_every = max(cfg.log_every, 100)
        if not interactive:
            console.print(f"[i] sortie non-interactive — progression toutes les {print_every} steps, "
                          f"eval toutes les {cfg.eval_every}")

        live_ctx = (Live(console=console, refresh_per_second=4, transient=False)
                    if interactive else contextlib.nullcontext())
        with live_ctx as live:
            while self.step < cfg.max_steps and not self.stop_requested:
                t_step = time.perf_counter()
                profile = (self.step % cfg.profile_every == 0) and use_cuda

                # --- learning rate -------------------------------------------------
                mult = lr_multiplier(self.step, cfg.max_steps, cfg.warmup, cfg.schedule,
                                     cfg.min_lr_frac, cfg.decay_frac)
                base_lrs = [cfg.lr, cfg.adam_lr] if cfg.optimizer == "muon" else [cfg.lr]
                for o, base in zip(self.opts, base_lrs):
                    for g in o.param_groups:
                        g["lr"] = base * mult
                cur_lr = base_lrs[0] * mult

                # --- accumulation ---------------------------------------------------
                t_data = t_fwd = t_bwd = 0.0
                loss_sum = 0.0
                stats_acc = {}
                batch_plan = []
                if cfg.stage == "sft":
                    # On connaît les deux dénominateurs avant le backward : la loss
                    # assistant est ainsi normalisée par le nombre GLOBAL de tokens
                    # supervisés, et le replay garde un poids explicite au lieu de
                    # dominer parce que ses séquences sont beaucoup plus denses.
                    replay_micros = (min(cfg.grad_accum - 1,
                                         max(1, round(cfg.grad_accum * cfg.replay_frac)))
                                     if self.replay_train is not None and cfg.grad_accum > 1
                                     else 0)
                    for micro in range(cfg.grad_accum):
                        idx = self.step * cfg.grad_accum + micro
                        replay = replay_micros > 0 and micro >= cfg.grad_accum - replay_micros
                        corpus = self.replay_train if replay else self.training_corpus(self.step)
                        batch_seed = cfg.seed + 101 if replay else cfg.seed
                        # Préchargement CPU borné (quelques dizaines de Mo) pour
                        # calculer le dénominateur exact sans garder plusieurs graphes GPU.
                        x, y, m = corpus.get_batch(idx, cfg.batch_size, batch_seed, "cpu")
                        count = int(m.sum().item()) if m is not None else int(y.numel())
                        batch_plan.append((idx, replay, x, y, m, count))
                    sft_count = sum(item[5] for item in batch_plan if not item[1])
                    replay_count = sum(item[5] for item in batch_plan if item[1])
                    if sft_count <= 0:
                        raise RuntimeError("batch SFT sans aucun token assistant supervisé")
                else:
                    batch_plan = [(self.step * cfg.grad_accum + micro, False,
                                   None, None, None, 0)
                                  for micro in range(cfg.grad_accum)]

                for micro, (idx, replay, x, y, m, count) in enumerate(batch_plan):
                    idx = self.step * cfg.grad_accum + micro
                    if profile:
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                    if cfg.stage == "sft":
                        if self.device.startswith("cuda"):
                            x = x.pin_memory().to(self.device, non_blocking=True)
                            y = y.pin_memory().to(self.device, non_blocking=True)
                            m = m.pin_memory().to(self.device, non_blocking=True) if m is not None else None
                        else:
                            x, y = x.to(self.device), y.to(self.device)
                            m = m.to(self.device) if m is not None else None
                    else:
                        corpus = self.training_corpus(self.step)
                        x, y, m = corpus.get_batch(idx, cfg.batch_size, cfg.seed, self.device)
                    if profile:
                        torch.cuda.synchronize(); t1 = time.perf_counter(); t_data += t1 - t0

                    last = micro == cfg.grad_accum - 1
                    with torch.autocast("cuda", dtype=self.amp_dtype, enabled=use_cuda):
                        reduction = "sum" if cfg.stage == "sft" else "mean"
                        _, loss, st = self.model(
                            x, y, m, z_loss=cfg.z_loss, diagnostics=last,
                            loss_reduction=reduction,
                        )
                    if profile:
                        torch.cuda.synchronize(); t2 = time.perf_counter(); t_fwd += t2 - t1
                    if cfg.stage == "sft":
                        if replay:
                            weight = cfg.replay_frac / max(1, replay_count)
                        else:
                            weight = (1.0 - cfg.replay_frac) / sft_count
                        weighted_loss = loss * weight
                    else:
                        weighted_loss = loss / cfg.grad_accum
                    weighted_loss.backward()
                    if profile:
                        torch.cuda.synchronize(); t_bwd += time.perf_counter() - t2
                    loss_sum += weighted_loss.detach()
                    if last:
                        stats_acc = st

                if cfg.stage == "sft":
                    stats_acc["assistant_tokens_update"] = float(sft_count)
                    stats_acc["replay_tokens_update"] = float(replay_count)

                # --- clipping + step -------------------------------------------------
                if profile:
                    torch.cuda.synchronize(); t3 = time.perf_counter()
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                for o in self.opts:
                    o.step()
                    o.zero_grad(set_to_none=True)
                if profile:
                    torch.cuda.synchronize(); self.breakdown = {
                        "data": t_data * 1000 / cfg.grad_accum, "fwd": t_fwd * 1000 / cfg.grad_accum,
                        "bwd": t_bwd * 1000 / cfg.grad_accum, "opt": (time.perf_counter() - t3) * 1000}

                # --- métriques --------------------------------------------------------
                loss_val = loss_sum.item()
                gn = gnorm.item()
                self.step += 1
                self.tokens_seen += tps_step
                dt = time.perf_counter() - t_step
                step_times.append(dt)
                self.hist_loss.append(loss_val)
                self.loss_ema = loss_val if self.loss_ema is None else 0.95 * self.loss_ema + 0.05 * loss_val
                tps = tps_step / dt
                self.tps_ema = tps if self.tps_ema is None else 0.9 * self.tps_ema + 0.1 * tps
                self.clip_hits.append(1.0 if gn > cfg.grad_clip else 0.0)

                mfu = fpt * self.tps_ema / (cfg.gpu_peak_tflops * 1e12)

                if self.step % cfg.log_every == 0:
                    rec = {"step": self.step, "loss": round(loss_val, 4), "lr": cur_lr,
                           "grad_norm": round(gn, 4), "tokens": self.tokens_seen,
                           "tok_s": round(self.tps_ema), "mfu": round(mfu, 4),
                           **{k: round(float(v), 4) for k, v in stats_acc.items()}}
                    self.metrics_file.write(json.dumps(rec) + "\n")
                    self.metrics_file.flush()

                # --- évaluation --------------------------------------------------------
                is_best = False
                if self.step % cfg.eval_every == 0 or self.step == cfg.max_steps:
                    self.val_loss = self.evaluate()
                    self.hist_val.append(self.val_loss)
                    if self.val_loss < self.best_val:
                        self.best_val = self.val_loss
                        is_best = True
                    if not interactive:
                        detail = "".join(f" · {name} {value:.3f}"
                                         for name, value in self.val_breakdown.items())
                        replay_detail = (f" · mid {self.replay_val_loss:.3f}"
                                         if self.replay_val_loss == self.replay_val_loss else "")
                        console.print(f"eval  step {self.step} · val {self.val_loss:.4f} · "
                                      f"ppl {math.exp(min(20, self.val_loss)):.1f}"
                                      + detail + replay_detail
                                      + (" · meilleur" if is_best else ""))
                    self.metrics_file.write(json.dumps({
                        "step": self.step, "val": round(self.val_loss, 5),
                        "val_sources": {k: round(v, 5) for k, v in self.val_breakdown.items()},
                        "mid_val": (round(self.replay_val_loss, 5)
                                    if self.replay_val_loss == self.replay_val_loss else None),
                        "best": is_best,
                    }) + "\n")
                    self.metrics_file.flush()

                if cfg.sample_every and self.step % cfg.sample_every == 0:
                    self.last_sample = self.sample()
                    self.last_sample_step = self.step
                    if not interactive:
                        console.print(f"éch.  step {self.step} · {self.last_sample[:200]}",
                                      markup=False, highlight=False)

                # --- checkpoint temporel ------------------------------------------------
                due_time = (time.time() - self.last_ckpt_time) >= cfg.ckpt_every_min * 60
                due_step = (cfg.save_every_steps > 0
                            and self.step % cfg.save_every_steps == 0)
                if due_time or due_step or is_best or self.step >= cfg.max_steps or stop_file.exists():
                    self.ckpt.save(self.state_payload(), self.step, is_best=is_best)
                    self.last_ckpt_time = time.time()
                    tag = " [green](meilleur)[/]" if is_best else ""
                    self.last_ckpt_msg = f"step {self.step} · {time.strftime('%H:%M:%S')}{tag}"
                    if not interactive:
                        console.print(f"ckpt  step {self.step} sauvegardé"
                                      + (" (meilleur)" if is_best else ""))

                if stop_file.exists():
                    stop_file.unlink(missing_ok=True)
                    self.stop_requested = True

                if interactive:
                    live.update(self._dashboard(
                        Group, Panel, Table, Columns, Text, cur_lr, gn, mfu, stats_acc,
                        n_params, n_ne, tps_step, step_times))
                elif self.step % print_every == 0 or self.step == cfg.max_steps:
                    eta = (cfg.max_steps - self.step) * float(np.mean(step_times))
                    console.print(f"step  {self.step}/{cfg.max_steps} "
                                  f"({100*self.step/cfg.max_steps:.1f}%) · "
                                  f"loss {loss_val:.4f} · EMA {self.loss_ema:.4f} · "
                                  f"lr {cur_lr:.2e} · {self.tps_ema/1e3:.0f}k tok/s · "
                                  f"MFU {100*mfu:.1f}% · ETA {int(eta//3600)}h{int(eta%3600//60):02d}")

        # ---- sortie ---------------------------------------------------------------
        self.ckpt.save(self.state_payload(), self.step, is_best=False)
        self.metrics_file.close()
        console.print(f"\n[bold green]✓[/] Checkpoint final : {self.stage_dir/'ckpt_latest.pt'}  "
                      f"(step {self.step}, {human(self.tokens_seen)} tokens)")
        cmd = {"pretrain": "train", "mid": "mid", "sft": "sft"}[self.cfg.stage]
        console.print(f"  reprendre :  [cyan]python run.py {cmd} --resume --run {self.cfg.run_name}[/]")
        console.print(f"  tester    :  [cyan]python run.py chat --run {self.cfg.run_name}[/]")

    # ---------------------------------------------------------------------------------
    def _dashboard(self, Group, Panel, Table, Columns, Text, lr, gnorm, mfu, stats,
                   n_params, n_ne, tps_step, step_times):
        cfg = self.cfg
        elapsed = self.elapsed_prev + (time.time() - self.t_start)
        avg_dt = float(np.mean(step_times)) if step_times else 0.0
        eta = (cfg.max_steps - self.step) * avg_dt
        pct = self.step / max(1, cfg.max_steps)

        def kv(t, k, v, style=""):
            t.add_row(k, f"[{style}]{v}[/]" if style else str(v))

        # --- optimisation ---
        t1 = Table.grid(padding=(0, 2))
        t1.add_column(style="dim", justify="right", min_width=16)
        t1.add_column(min_width=14)
        loss = self.hist_loss[-1] if self.hist_loss else float("nan")
        kv(t1, "loss (brute)", f"{loss:.4f}")
        kv(t1, "loss (EMA)", f"{self.loss_ema:.4f}" if self.loss_ema else "—", "bold cyan")
        kv(t1, "perplexité", f"{math.exp(min(20, self.loss_ema)):.1f}" if self.loss_ema else "—")
        kv(t1, "bits/token", f"{(self.loss_ema/math.log(2)):.3f}" if self.loss_ema else "—")
        kv(t1, "val loss", f"{self.val_loss:.4f}" if self.val_loss == self.val_loss else "—", "magenta")
        kv(t1, "val ppl", f"{math.exp(min(20,self.val_loss)):.1f}" if self.val_loss == self.val_loss else "—")
        kv(t1, "meilleure val", f"{self.best_val:.4f}" if self.best_val < 1e9 else "—", "green")
        kv(t1, "top-1 acc", f"{100*float(stats.get('acc_top1', float('nan'))):.2f} %" if stats else "—")
        kv(t1, "entropie préd.", f"{float(stats.get('entropy', float('nan'))):.3f} nats" if stats else "—")
        kv(t1, "RMS des logits", f"{float(stats.get('logit_rms', float('nan'))):.2f}" if stats else "—")

        # --- dynamique de l'optimiseur ---
        t2 = Table.grid(padding=(0, 2))
        t2.add_column(style="dim", justify="right", min_width=16)
        t2.add_column(min_width=14)
        kv(t2, "learning rate", f"{lr:.2e}", "yellow")
        kv(t2, "% du LR max", f"{100*lr/max(1e-12,cfg.lr):.1f} %")
        kv(t2, "norme du grad", f"{gnorm:.3f}")
        kv(t2, "clip (100 steps)", f"{100*np.mean(self.clip_hits):.0f} %")
        if self.step % 20 == 0:                      # coûteux -> pas à chaque step
            with torch.no_grad():
                sq = torch.zeros((), device=next(self.raw_model.parameters()).device)
                for p in self.raw_model.parameters():
                    if p.ndim >= 2:
                        sq += p.detach().float().pow(2).sum()
                self._pnorm = float(sq.sqrt())
        kv(t2, "norme des poids", f"{getattr(self, '_pnorm', 0):.1f}")
        kv(t2, "Δp/p estimé", f"{lr*gnorm/max(1e-9,getattr(self,'_pnorm',1)):.2e}")
        kv(t2, "z-loss", f"{cfg.z_loss:g}")
        kv(t2, "optimiseur", cfg.optimizer)
        kv(t2, "schedule", cfg.schedule)
        kv(t2, "warmup", f"{cfg.warmup} steps")

        # --- débit / matériel ---
        t3 = Table.grid(padding=(0, 2))
        t3.add_column(style="dim", justify="right", min_width=16)
        t3.add_column(min_width=14)
        kv(t3, "tokens/s", human(self.tps_ema or 0), "bold green")
        kv(t3, "temps/step", f"{avg_dt*1000:.0f} ms")
        kv(t3, "MFU", f"{100*mfu:.1f} %", "bold")
        b = self.breakdown
        kv(t3, "data/fwd/bwd/opt", f"{b['data']:.0f}/{b['fwd']:.0f}/{b['bwd']:.0f}/{b['opt']:.0f} ms")
        if torch.cuda.is_available():
            kv(t3, "VRAM allouée", f"{torch.cuda.memory_allocated()/2**30:.2f} Go")
            kv(t3, "VRAM réservée", f"{torch.cuda.memory_reserved()/2**30:.2f} Go")
            kv(t3, "VRAM pic", f"{torch.cuda.max_memory_allocated()/2**30:.2f} Go", "red")
        g = self.gpu.read()
        if g:
            kv(t3, "GPU util", f"{g['util']} %")
            kv(t3, "température", f"{g['temp']} °C")
            kv(t3, "puissance", f"{g['power']:.0f}/{g['power_cap']:.0f} W")
            kv(t3, "horloge SM", f"{g['clock']} MHz")
        else:
            kv(t3, "télémétrie GPU", "pip install nvidia-ml-py")

        # --- avancement ---
        t4 = Table.grid(padding=(0, 2))
        t4.add_column(style="dim", justify="right", min_width=16)
        t4.add_column(min_width=14)
        kv(t4, "step", f"{self.step}/{cfg.max_steps}  ({100*pct:.1f} %)", "bold")
        kv(t4, "tokens vus", human(self.tokens_seen), "bold")
        kv(t4, "tokens/param", f"{self.tokens_seen/max(1,n_ne):.1f} (hors emb.)")
        kv(t4, "époques corpus", f"{self.tokens_seen/max(1,len(self.train_data)):.2f}")
        kv(t4, "écoulé", hms(elapsed))
        kv(t4, "ETA", hms(eta), "cyan")
        kv(t4, "fin prévue", time.strftime("%H:%M", time.localtime(time.time() + eta)))
        kv(t4, "dernier ckpt", self.last_ckpt_msg)
        nxt = cfg.ckpt_every_min * 60 - (time.time() - self.last_ckpt_time)
        kv(t4, "prochain ckpt", hms(nxt))
        kv(t4, "params", f"{human(n_params)} ({human(n_ne)} hors emb.)")

        bar_w = 58
        filled = int(bar_w * pct)
        bar = "[green]" + "━" * filled + "[/][dim]" + "━" * (bar_w - filled) + "[/]"

        lines = [f"[dim]train[/]  {sparkline(self.hist_loss)}  "
                 f"[cyan]{self.loss_ema:.3f}[/]" if self.loss_ema else "[dim]train[/]"]
        if self.hist_val:
            lines.append(f"[dim]val  [/]  {sparkline(self.hist_val)}  [magenta]{self.val_loss:.3f}[/]")
        spark = Panel(Text.from_markup("\n".join(lines)),
                      title="courbes (loss train / val)", border_style="dim", padding=(0, 1))

        sample_panel = Panel(
            Text((self.last_sample or "(échantillon au prochain palier…)")[:600]),
            title=f"génération du modèle · step {self.last_sample_step}", border_style="blue", padding=(0, 1))

        return Group(
            Columns([
                Panel(t1, title="[cyan]qualité[/]", border_style="cyan", padding=(0, 1)),
                Panel(t2, title="[yellow]optimisation[/]", border_style="yellow", padding=(0, 1)),
                Panel(t3, title="[green]débit & matériel[/]", border_style="green", padding=(0, 1)),
                Panel(t4, title="[white]avancement[/]", border_style="white", padding=(0, 1)),
            ], equal=False, expand=False),
            Text.from_markup(bar),
            spark,
            sample_panel,
        )


# ======================================================================================
# Mode chat
# ======================================================================================
def cmd_chat(args):
    from rich.console import Console

    console = Console()
    run_dir = Path(args.out_dir) / args.run
    # On préfère la phase RL (la plus affûtée), puis SFT, et on retombe sur le
    # pré-entraînement si rien d'autre n'a tourné.
    # Les phases RLAIF successives vivent dans rlaif, rlaif2, rlaif3… : on prend la
    # plus récente d'abord (tri décroissant), puis on redescend la chaîne.
    rlaifs = sorted((d.name for d in run_dir.glob("rlaif*") if d.is_dir()), reverse=True)
    stages = [args.stage] if args.stage else [*rlaifs, "rl", "sft", "mid", "pretrain"]
    path = None
    for st in stages:
        d = run_dir / st
        if d.exists():
            path = CheckpointManager(d).resolve(args.ckpt)
            if path is not None:
                break
    if path is None and Path(args.ckpt).exists():
        path = Path(args.ckpt)
    if path is None:
        sys.exit(f"[!] Aucun checkpoint dans {run_dir}. Entraîne d'abord : python run.py train")

    ck = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = config_from_dict(ck["model_cfg"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_from_cfg(mcfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if device == "cuda":
        model = model.to(torch.bfloat16)

    tok = D.load_tokenizer(run_dir / "tokenizer.json")
    sp = D.special_ids(tok)

    console.print(f"[bold green]Modèle chargé[/] : {path.name} · step {ck['step']} · "
                  f"{human(ck['tokens_seen'])} tokens vus · val loss "
                  f"{ck.get('val_loss', float('nan')):.4f} · phase {ck.get('stage','?')}")
    console.print(f"[dim]{human(model.num_params())} params · {mcfg.n_layer}L · d={mcfg.d_model}[/]")
    console.print("[dim]Commandes : /reset  /think auto|on|off  /temp 0.8  /topp 0.95  /topk 50  "
                  "/max 200  /raw <texte>  /stats  /quit[/]\n")

    gen_cfg = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                   max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty)
    history: list[dict] = []
    think_mode = "auto"   # auto = le modèle décide · on = <think> forcé · off = bloc vide préinséré

    def complete(prompt_text: str, stop_ids):
        ids = torch.tensor([tok.encode(prompt_text).ids], device=device)
        buf, printed = [], 0
        t0 = time.perf_counter()

        def on_token(t):
            nonlocal printed
            if t in stop_ids:
                return
            buf.append(t)
            # skip_special_tokens=False : on veut VOIR les balises <think>…</think>
            txt = tok.decode(buf, skip_special_tokens=False)
            if len(txt) > printed:
                sys.stdout.write(txt[printed:])
                sys.stdout.flush()
                printed = len(txt)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            model.generate(ids, stop_ids=stop_ids, on_token=on_token, **gen_cfg)
        dt = time.perf_counter() - t0
        print()
        console.print(f"[dim]{len(buf)} tokens · {len(buf)/max(dt,1e-6):.1f} tok/s · {dt:.2f}s[/]")
        return tok.decode(buf, skip_special_tokens=False)

    while True:
        try:
            user = console.input("[bold cyan]toi ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/reset":
            history.clear()
            console.print("[dim]historique effacé[/]")
            continue
        if user == "/stats":
            console.print(json.dumps({k: v for k, v in gen_cfg.items()}, indent=2))
            console.print(f"[dim]tours en mémoire : {len(history)}[/]")
            continue
        if user.startswith("/raw "):
            console.print("[bold magenta]modèle ›[/] ", end="")
            complete(user[5:], (sp["eot"],))
            continue
        if user.startswith("/think"):
            arg = (user.split() + ["auto"])[1]
            if arg in ("auto", "on", "off"):
                think_mode = arg
                console.print(f"[dim]réflexion : {think_mode}[/]")
            else:
                console.print("[dim]usage : /think auto|on|off[/]")
            continue
        for key, cast in (("/temp", float), ("/topp", float), ("/topk", int), ("/max", int)):
            if user.startswith(key + " "):
                name = {"/temp": "temperature", "/topp": "top_p", "/topk": "top_k",
                        "/max": "max_new_tokens"}[key]
                gen_cfg[name] = cast(user.split()[1])
                console.print(f"[dim]{name} = {gen_cfg[name]}[/]")
                break
        else:
            history.append({"role": "user", "text": user})
            # on = on force l'ouverture du bloc de réflexion ;
            # off = on préinsère un bloc vide (le modèle passe direct à la réponse)
            prefill = {"auto": "", "on": f"{D.THINK}\n",
                       "off": f"{D.THINK}\n\n{D.THINK_END}\n"}[think_mode]

            def build_prompt():
                return D.render_chat(history) + f"{D.IM_START}assistant\n" + prefill

            prompt = build_prompt()
            # on tronque l'historique si le prompt dépasse le contexte
            while len(tok.encode(prompt).ids) > mcfg.max_seq_len - gen_cfg["max_new_tokens"] and len(history) > 1:
                history.pop(0)
                prompt = build_prompt()
            console.print("[bold magenta]modèle ›[/] ", end="")
            if prefill:
                sys.stdout.write(prefill)
            reply = prefill + complete(prompt, (sp["im_end"], sp["eot"]))
            # l'historique ne garde que la réponse finale : les traces de réflexion
            # rempliraient le contexte pour rien au tour suivant
            import re as _re
            clean = _re.sub(r"<think>.*?</think>\s*", "", reply, flags=_re.S).strip()
            history.append({"role": "assistant", "text": clean or reply.strip()})


# ======================================================================================
# Infos
# ======================================================================================
def cmd_info(args):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    data_dir = Path(args.data_dir)
    meta = data_dir / "meta.json"
    if meta.exists():
        console.print("[bold]Corpus[/]")
        console.print_json(meta.read_text(encoding="utf-8"))
    for b in sorted(data_dir.glob("*.bin")):
        n = b.stat().st_size // 2
        console.print(f"  {b.name:16s} {human(n)} tokens  ({b.stat().st_size/2**30:.2f} Go)")

    run_dir = Path(args.out_dir) / args.run
    for stage in ("pretrain", "mid", "sft"):
        sdir = run_dir / stage
        if not sdir.exists():
            continue
        console.print(f"\n[bold]Checkpoints — phase {stage}[/]  [dim]{sdir}[/]")
        t = Table("fichier", "step", "tokens", "val loss", "taille", "date")
        for p in sorted(sdir.glob("ckpt_*.pt")):
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                t.add_row(p.name, str(ck["step"]), human(ck["tokens_seen"]),
                          f"{ck.get('val_loss', float('nan')):.4f}",
                          f"{p.stat().st_size/2**20:.0f} Mo",
                          time.strftime("%d/%m %H:%M", time.localtime(p.stat().st_mtime)))
            except Exception as e:
                t.add_row(p.name, "?", "?", f"illisible ({e})", "", "")
        console.print(t)

        mfile = sdir / "metrics.jsonl"
        if mfile.exists():
            rows = [json.loads(l) for l in mfile.read_text(encoding="utf-8").splitlines() if l.strip()]
            if rows:
                losses = [r["loss"] for r in rows]
                console.print(f"  loss  {sparkline(losses, 80)}  "
                              f"{losses[0]:.3f} → {losses[-1]:.3f}   "
                              f"({len(rows)} logs · {human(rows[-1]['tokens'])} tokens · "
                              f"{rows[-1]['tok_s']:.0f} tok/s)")


# ======================================================================================
# CLI
# ======================================================================================
def add_train_args(p):
    p.add_argument("--run", default="fr-micro", help="nom du run (dossier dans runs/)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--preset", default="micro", choices=list(PRESETS) + list(PRESETS_V3),
                   help="presets v3-* = architecture speedrun (model_v3.py)")
    p.add_argument("--hybrid", action="store_true",
                   help="archi Qwen3.5 complète : couches Gated DeltaNet + attention en 3:1 "
                        "(exacte mais ~3,5x plus lente sans kernels Triton)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr", type=float, default=None, help="LR principal (0.02 pour muon, 6e-4 pour adamw)")
    p.add_argument("--adam-lr", type=float, default=1.5e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--schedule", default="wsd", choices=["wsd", "cosine"])
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--min-lr-frac", type=float, default=0.02,
                   help="multiplicateur LR minimal en fin de schedule")
    p.add_argument("--decay-frac", type=float, default=0.2,
                   help="fraction finale de décroissance du WSD")
    p.add_argument("--z-loss", type=float, default=1e-4)
    p.add_argument("--replay-frac", type=float, default=0.0,
                   help="fraction de batchs mid rejoués pendant le SFT (anti-oubli)")
    p.add_argument("--replay-mix", default="",
                   help="bins non masqués et poids, ex. train.bin=0.7,mid.bin=0.3")
    p.add_argument("--replay-val", default="",
                   help="bin de validation non masqué associé à --replay-mix")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--sample-every", type=int, default=1000)
    p.add_argument("--ckpt-every-min", type=float, default=5.0)
    p.add_argument("--save-every", type=int, default=0,
                   help="checkpoint roulant tous les N steps (0 = désactivé)")
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--mid-curriculum", default="",
                   help="pour la phase mid : v4.3 active les bins curriculum 80/20")
    p.add_argument("--sft-recipe", default="",
                   help="pour la phase SFT : v4.4/v4.5 activent le sampling par capacités")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="désactive torch.compile (actif par défaut : +94%% de débit ; "
                        "retombe tout seul en mode non compilé si triton manque)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gpu-peak-tflops", type=float, default=30.0,
                   help="pic bf16 dense du GPU en TFLOPS, pour le calcul du MFU (4060 ~= 30)")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="reprend au checkpoint (latest | best | chemin)")


def cfg_from_args(args, stage: str) -> TrainConfig:
    if args.lr is not None:
        lr = args.lr
    elif stage == "sft":
        # on affine des poids déjà bons : un LR de pré-entraînement les détruirait
        lr = 0.004 if args.optimizer == "muon" else 1e-4
    elif stage == "mid":
        # midtrain = recuit : on repart à mi-hauteur du plateau et on décroît sur
        # toute la phase (schedule cosine par défaut, voir le parser "mid")
        lr = 0.01 if args.optimizer == "muon" else 3e-4
    else:
        lr = 0.02 if args.optimizer == "muon" else 6e-4
    if stage == "sft":
        args.adam_lr = min(args.adam_lr, 2e-4)
    return TrainConfig(
        run_name=args.run, data_dir=args.data_dir, out_dir=args.out_dir, stage=stage,
        preset=args.preset, hybrid=args.hybrid, batch_size=args.batch_size, grad_accum=args.grad_accum,
        seq_len=args.seq_len, optimizer=args.optimizer, lr=lr, adam_lr=args.adam_lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, schedule=args.schedule,
        warmup=args.warmup, min_lr_frac=args.min_lr_frac, decay_frac=args.decay_frac,
        z_loss=args.z_loss, max_steps=args.max_steps,
        replay_frac=args.replay_frac, replay_mix=args.replay_mix,
        replay_val=args.replay_val,
        mid_curriculum=args.mid_curriculum, sft_recipe=args.sft_recipe,
        eval_every=args.eval_every, eval_iters=args.eval_iters, sample_every=args.sample_every,
        ckpt_every_min=args.ckpt_every_min, save_every_steps=args.save_every,
        keep_last=args.keep_last, compile=args.compile,
        dtype=args.dtype, seed=args.seed, gpu_peak_tflops=args.gpu_peak_tflops,
    )


def main():
    ap = argparse.ArgumentParser(description="Entraînement d'un petit LLM français sur RTX 4060")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="télécharge le corpus FR, entraîne le tokenizer, binarise")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--target-tokens", type=float, default=300e6,
                   help="taille visée du corpus de pré-entraînement (défaut 300M)")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--mix", default="fineweb:0.40,wiki:0.20,chat:0.15,maths:0.15,books:0.05,oral:0.05",
                   help="mélange source:poids séparés par des virgules")
    p.add_argument("--mid-frac", type=float, default=0.2,
                   help="taille du corpus midtrain en fraction du pré-entraînement (0 = pas de midtrain)")
    p.add_argument("--sft-target-supervised", type=float, default=D.SFT_TARGET_SUPERVISED,
                   help="tokens assistant visés dans le mix SFT pondéré (défaut 50M)")
    p.add_argument("--no-sft", action="store_true", help="ne pas préparer le jeu de dialogue")
    p.add_argument("--skip-download", action="store_true", help="réutilise les .jsonl déjà téléchargés")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--rebin", action="store_true",
                   help="re-binarise SEULEMENT mid+SFT avec le tokenizer existant "
                        "(après un changement de recette : distill, sur-échantillonnage…)")
    p.add_argument("--sft-only", action="store_true",
                   help="avec --rebin, reconstruit seulement le SFT et préserve les bins mid")

    p = sub.add_parser("prepare-mid-v43",
                       help="construit le curriculum mid v4.3 sans toucher au tokenizer/pretrain")
    p.add_argument("--data-dir", default="data-v4")
    p.add_argument("--target-tokens", type=float, default=1.5e9,
                   help="budget traité maximal de la phase (plafond 1,5 milliard)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--skip-download", action="store_true",
                   help="réutilise les JSONL locaux et les sources QA déjà filtrées")

    p = sub.add_parser("prepare-sft-v44",
                       help="construit le SFT v4.4 équilibré sans toucher au mid/pretrain")
    p.add_argument("--data-dir", default="data-v4")
    p.add_argument("--target-supervised", type=float, default=18e6,
                   help="plafond de tokens assistant (les shortfalls ne sont pas redistribués)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--skip-download", action="store_true",
                   help="réutilise les JSONL v4.4 déjà présents")

    p = sub.add_parser("prepare-sft-v45",
                       help="construit le SFT v4.5 audité sans toucher au mid/pretrain")
    p.add_argument("--data-dir", default="data-v4")
    p.add_argument("--target-supervised", type=float, default=24e6,
                   help="tokens assistant uniques visés (défaut 24 millions)")
    p.add_argument("--seq-len", type=int, default=512,
                   help="longueur maximale d'une conversation isolée")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--skip-download", action="store_true",
                   help="réutilise toutes les sources JSONL déjà présentes")

    p = sub.add_parser("train", help="pré-entraînement")
    add_train_args(p)

    p = sub.add_parser("mid", help="midtrain : recuit sur un mix dense en raisonnement (mid_train.bin)")
    add_train_args(p)
    # v4.1 : recuit doux. L'ancien 0.01/1.5e-3 redémarrait très au-dessus de la fin
    # du prétrain et pouvait effacer les acquis avant même la décroissance.
    # À bs=16, accumulation=2 et seq=2048, 6000 steps font ~393 M tokens :
    # quasiment une passe sur le corpus v4.1 mesuré à 405 M tokens.
    p.set_defaults(max_steps=6000, schedule="cosine", warmup=50, lr=0.004,
                   adam_lr=2e-4, eval_every=100, sample_every=500)

    p = sub.add_parser("sft", help="fine-tuning dialogue (masque de loss sur les réponses)")
    add_train_args(p)
    # v4.2 : ~1 token traité par paramètre sur un corpus sélectionné, LR bas,
    # batch effectif 64 et 15 % de replay mid pour limiter l'oubli catastrophique.
    p.set_defaults(max_steps=1800, schedule="cosine", warmup=50, optimizer="adamw",
                   lr=1e-5, weight_decay=0.01, grad_accum=4, replay_frac=0.15,
                   eval_every=100, eval_iters=42, sample_every=100)

    p = sub.add_parser("rl", help="GRPO : renforcement à récompenses vérifiables (part du SFT)")
    p.add_argument("--run", default="fr-micro")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--prompts", type=int, default=8, help="problèmes (groupes GRPO) par step")
    p.add_argument("--group", type=int, default=8, help="réponses échantillonnées par problème")
    p.add_argument("--max-new", type=int, default=220)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.03)
    p.add_argument("--micro-bs", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--instr-frac", type=float, default=0.3,
                   help="part des prompts avec consigne vérifiable (IF-RLVR, façon Tülu 3)")
    p.add_argument("--oversample", type=float, default=2.0,
                   help="dynamic sampling (DAPO) : re-tirages max pour remplir le batch de groupes utiles")
    p.add_argument("--ckpt-every-min", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", nargs="?", const="latest", default=None)

    p = sub.add_parser("rlaif", help="GRPO à juge LLM : pool quotidien + protocole fichiers (voir frlm/rlaif.py)")
    p.add_argument("--run", default="fr-v4")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--prompts", type=int, default=6, help="groupes GRPO par step")
    p.add_argument("--group", type=int, default=6, help="réponses échantillonnées par problème")
    p.add_argument("--max-new", type=int, default=220)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--micro-bs", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--pool", default="data-v4/rlaif_prompts.jsonl,data-v4/rlaif_prompts_v2.jsonl",
                   help="un ou plusieurs fichiers de prompts, séparés par des virgules")
    p.add_argument("--stage-name", default="rlaif",
                   help="sous-dossier de sortie (rlaif2 pour une 2e passe : sinon on "
                        "écrase le ckpt_best de la première)")
    p.add_argument("--init-stage", default="rl", choices=["rl", "sft", "rlaif", "rlaif2"],
                   help="ckpt de départ ET ancre KL")
    p.add_argument("--init-ckpt", default="best")
    p.add_argument("--judge-weight", type=float, default=1.0)
    p.add_argument("--synth-frac", type=float, default=0.35)
    p.add_argument("--ppo-epochs", type=int, default=2,
                   help="réutilisations du lot de rollouts (clip-higher actif au-delà de 1)")
    p.add_argument("--clip-high", type=float, default=0.28,
                   help="borne haute du ratio (DAPO clip-higher, anti-collapse d'entropie)")
    p.add_argument("--overlong-penalty", type=float, default=0.3,
                   help="pénalité de génération coupée sans conclusion (anti think-fleuve)")
    p.add_argument("--repeat-penalty", type=float, default=0.3,
                   help="pénalité de radotage (n-grammes répétés)")
    p.add_argument("--oversample", type=float, default=2.5,
                   help="dynamic sampling : re-tirages max pour éviter les groupes muets")
    p.add_argument("--judge-timeout", type=float, default=1800.0)
    p.add_argument("--ckpt-every-min", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", nargs="?", const="latest", default=None)

    p = sub.add_parser("chat", help="discute avec un checkpoint")
    p.add_argument("--run", default="fr-micro")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--ckpt", default="latest", help="latest | best | chemin vers un .pt")
    p.add_argument("--stage", default=None, choices=["pretrain", "mid", "sft", "rl", "rlaif"],
                   help="force la phase à charger (défaut : rlaif, puis rl, sft, mid, pretrain)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--max-new-tokens", type=int, default=200)

    p = sub.add_parser("info", help="stats corpus + checkpoints")
    p.add_argument("--run", default="fr-micro")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")

    args = ap.parse_args()

    if args.cmd == "prepare-mid-v43":
        from frlm.mid_v43 import prepare as prepare_mid_v43

        rep = prepare_mid_v43(Path(args.data_dir), target_tokens=int(args.target_tokens),
                              seed=args.seed, skip_download=args.skip_download)
        print("\n=== Mid v4.3 prêt ===")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("\nÉtape suivante : python run.py mid --mid-curriculum v4.3 "
              "--max-steps 11444 --batch-size 16 --grad-accum 4 --seq-len 2048")

    elif args.cmd == "prepare-sft-v44":
        from frlm.sft_v44 import prepare as prepare_sft_v44

        rep = prepare_sft_v44(
            Path(args.data_dir), target_supervised=int(args.target_supervised),
            max_seq_len=args.seq_len, seed=args.seed, skip_download=args.skip_download
        )
        print("\n=== SFT v4.4 prêt ===")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("\nÉtape suivante (après benchmark du mid) : python run.py sft "
              "--sft-recipe v4.4 --max-steps 350 --lr 1e-4")

    elif args.cmd == "prepare-sft-v45":
        from frlm.sft_v45 import prepare as prepare_sft_v45

        rep = prepare_sft_v45(
            Path(args.data_dir), target_supervised=int(args.target_supervised),
            max_seq_len=args.seq_len, seed=args.seed,
            skip_download=args.skip_download,
        )
        print("\n=== SFT v4.5 prêt ===")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("\nÉtape suivante : python run.py sft --sft-recipe v4.5 "
              "--seq-len 512 --batch-size 128 --grad-accum 12 --max-steps 736 "
              "--optimizer adamw --lr 2e-5 --replay-frac 0.12 "
              "--resume runs/fr-v4-v43/mid/ckpt_latest.pt")

    elif args.cmd == "prepare" and args.rebin:
        rep = D.rebin_mid_sft(Path(args.data_dir), mid_frac=args.mid_frac,
                              max_seq_len=args.seq_len,
                              sft_target_supervised=int(args.sft_target_supervised),
                              sft_only=args.sft_only)
        print("\n=== Récapitulatif ===")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("\nÉtape suivante :  python run.py sft --resume <checkpoint-mid>")

    elif args.cmd == "prepare":
        mix = {}
        for part in args.mix.split(","):
            k, v = part.split(":")
            mix[k.strip()] = float(v)
        rep = D.prepare_all(Path(args.data_dir), int(args.target_tokens), args.vocab_size, mix,
                            sft=not args.no_sft, max_seq_len=args.seq_len,
                            skip_download=args.skip_download, mid_frac=args.mid_frac,
                            sft_target_supervised=int(args.sft_target_supervised))
        print("\n=== Récapitulatif ===")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("\nÉtape suivante :  python run.py train")

    elif args.cmd in ("train", "mid", "sft"):
        stage = {"train": "pretrain", "mid": "mid", "sft": "sft"}[args.cmd]
        cfg = cfg_from_args(args, stage=stage)
        if args.cmd in ("mid", "sft") and args.resume is None:
            args.resume = "latest"   # mid/SFT partent forcément de poids déjà entraînés
        Trainer(cfg, resume=args.resume).train()

    elif args.cmd == "rl":
        from frlm.rl import cmd_rl
        cmd_rl(args)

    elif args.cmd == "rlaif":
        from frlm.rlaif import cmd_rlaif
        cmd_rlaif(args)

    elif args.cmd == "chat":
        cmd_chat(args)

    elif args.cmd == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
