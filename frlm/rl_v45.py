"""RLVR local, auditable et économe en VRAM pour FRLM v4.5.

Cette implémentation n'utilise jamais le benchmark OOD pendant l'entraînement.
Les groupes sont retenus uniquement s'ils contiennent à la fois des réussites et
des échecs primaires. Les avantages DrGRPO sont centrés sans division par std.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import signal
import time
from copy import deepcopy
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from frlm import data as D
from frlm.rl_engine_v45 import (
    RolloutEngine, clone_reference, load_policy, resolve_checkpoint, resolve_tokenizer,
)
from frlm.rl_tasks_v45 import CAPABILITY_WEIGHTS, SCHEMAS_BY_CAPABILITY, TaskSpec, make_task
from frlm.verifiers_v45 import VERIFIER_VERSION, final_text, verify


DIFFICULTY_AWARE_CAPABILITIES = {"reasoning_program"}


@dataclass
class RLVRConfig:
    run_name: str = "fr-v4-v45-sft"
    data_dir: str = "data-v4"
    out_dir: str = "runs"
    stage_name: str = "rlvr-v45"
    init_stage: str = "sft"
    init_ckpt: str = "best"
    ref_stage: str = "sft"
    ref_ckpt: str = "best"
    profile_name: str = "profile.json"
    anchor_resume_reference: bool = True
    accepted_updates: int = 200
    prompts_per_update: int = 3
    group_size: int = 6
    max_new_tokens: int = 112
    micro_bs: int = 2
    lr: float = 5e-7
    warmup: int = 5
    kl_beta: float = 0.10
    kl_target: float = 0.012
    kl_beta_max: float = 1.0
    kl_soft_max: float = 0.05
    kl_hard_max: float = 0.12
    kl_excursion_patience: int = 3
    min_lr_scale: float = 0.125
    replay_weight: float = 0.15
    grad_clip: float = 1.0
    oversample: float = 4.0
    eval_every: int = 5
    eval_tasks: int = 60
    save_every: int = 10
    keep_last: int = 2
    max_empty_batches: int = 20
    max_kl_rejections: int = 12
    max_dev_drop: float = 0.05
    max_capability_drop: float = 0.10
    retention_kl_weight: float = 0.05
    max_auto_recoveries: int = 6
    seed: int = 455_100
    device: str = "cuda"
    require_profile: bool = True
    reset_optimizer_on_resume: bool = False


_CURATED_REPLAY = (
    ("Salut, ça va ?", "Salut ! Oui, merci. Comment puis-je t'aider ?"),
    ("Réponds en deux phrases : présente-toi sans inventer d'identité humaine.",
     "Je suis FRLM, un modèle de langue. Je peux aider à expliquer, rédiger et raisonner."),
    ("Je n'ai donné aucune date. En quelle année est-ce arrivé ?",
     "Les informations fournies ne permettent pas de déterminer l'année."),
    ("Léa a 3 pommes, elle en mange 3 puis en récupère 1. Combien en a-t-elle ?", "1"),
    ("Quelle est la capitale de la France ?", "Paris."),
    ("Donne uniquement le résultat de 17 - 9.", "8"),
    ("Si tu n'es pas certain d'un fait, que dois-tu faire ?",
     "Je dois signaler l'incertitude plutôt que d'inventer une réponse."),
    ("Résume : le train part à huit heures et arrive à dix heures.",
     "Le trajet en train dure de huit heures à dix heures."),
)


def _clone_to_cpu(value):
    """Clone récursivement un état PyTorch sans conserver de stockage CUDA partagé."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return deepcopy(value)


def _atomic_torch_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _atomic_link(source: Path, destination: Path):
    """Alias atomique sans dupliquer plusieurs gigaoctets de checkpoint."""
    tmp = destination.with_suffix(destination.suffix + ".link.tmp")
    tmp.unlink(missing_ok=True)
    os.link(source, tmp)
    tmp.replace(destination)


def _repeat_ratio(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[index:index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _canonical_answer(task: TaskSpec) -> str:
    if task.answer.kind == "json":
        return json.dumps(task.answer.value, ensure_ascii=False, separators=(",", ":"))
    if task.answer.kind == "abstain":
        return "Le contexte ne permet pas de le déterminer."
    if task.answer.kind == "code":
        name = task.answer.function_name
        if task.schema_id == "code_0":
            return f"def {name}(n):\n    return 2 * n"
        if task.schema_id == "code_1":
            return f"def {name}(n, minimum, maximum):\n    return max(minimum, min(n, maximum))"
        return f"def {name}(valeurs):\n    return sum(x * x for x in valeurs)"
    return str(task.answer.value)


class RLVRTrainer:
    def __init__(self, cfg: RLVRConfig, resume: str | None = None):
        self.cfg = cfg
        if cfg.group_size < 2 or cfg.prompts_per_update < 1:
            raise ValueError("group_size >= 2 et prompts_per_update >= 1 requis")
        if cfg.micro_bs > 2:
            raise ValueError("micro_bs > 2 est refusé par le profil mémoire RTX 4060")
        if not (0 < cfg.kl_target < cfg.kl_soft_max < cfg.kl_hard_max):
            raise ValueError("il faut kl_target < kl_soft_max < kl_hard_max")
        if cfg.kl_excursion_patience < 1 or cfg.max_kl_rejections < 1:
            raise ValueError("les patiences KL doivent être positives")
        if not (0 <= cfg.max_capability_drop <= 1) or cfg.max_auto_recoveries < 1:
            raise ValueError("gates de capacité ou reprises automatiques invalides")
        if cfg.retention_kl_weight < 0:
            raise ValueError("retention_kl_weight doit être positif")
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible")
        self.device = cfg.device
        self.use_cuda = self.device.startswith("cuda")
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.stage_dir = self.run_dir / cfg.stage_name
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self.stage_dir / cfg.profile_name
        if cfg.require_profile and not profile_path.is_file():
            raise RuntimeError(
                f"profil pass@k absent : {profile_path}. Lance d'abord `python run.py rl-profile-v45`."
            )
        self.profile = (json.loads(profile_path.read_text(encoding="utf-8"))
                        if profile_path.is_file() else None)
        self.profile_sha256 = (hashlib.sha256(profile_path.read_bytes()).hexdigest()
                               if profile_path.is_file() else None)
        self.frontier_specs, self.bridge_specs = self._profile_curriculum()
        self.tok = D.load_tokenizer(resolve_tokenizer(self.run_dir, Path(cfg.data_dir)))
        resume_path = (resolve_checkpoint(self.run_dir, cfg.stage_name, resume)
                       if resume else None)
        init_path = (resume_path if resume_path is not None and cfg.anchor_resume_reference
                     else resolve_checkpoint(self.run_dir, cfg.init_stage, cfg.init_ckpt))
        self.model, self.mcfg, self.init_meta = load_policy(init_path, self.device, torch.float32)
        if resume_path is not None and cfg.anchor_resume_reference:
            self.ref = clone_reference(self.model, self.mcfg, self.device)
            self.ref_meta = dict(self.init_meta)
            self.ref_meta["anchor"] = "resume"
        elif cfg.ref_stage == cfg.init_stage and cfg.ref_ckpt == cfg.init_ckpt:
            self.ref = clone_reference(self.model, self.mcfg, self.device)
            self.ref_meta = dict(self.init_meta)
        else:
            ref_path = resolve_checkpoint(self.run_dir, cfg.ref_stage, cfg.ref_ckpt)
            self.ref, ref_cfg, self.ref_meta = load_policy(ref_path, self.device, torch.float32)
            if ref_cfg.to_dict() != self.mcfg.to_dict():
                raise ValueError("l'ancre KL et la politique n'ont pas la même architecture")
            self.ref.eval()
            for parameter in self.ref.parameters():
                parameter.requires_grad_(False)
        self.engine = RolloutEngine(self.model, self.tok, self.device, cfg.max_new_tokens)
        self.sp = D.special_ids(self.tok)
        self.optimizer = self._make_optimizer()
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(cfg.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
        self.update = 0
        self.rollout_index = 0
        self.tokens_generated = 0
        self.kl_beta = cfg.kl_beta
        self.lr_scale = 1.0
        self.kl_excursions = 0
        self.optimizer_start_update = 0
        self.replay_index = 0
        self.replay_scale = 1.0
        self.best_score = -1.0
        self.baseline_score = None
        self.best_per_capability: dict[str, float] = {}
        self.capability_peaks: dict[str, float] = {}
        self.recovery_count = 0
        self._resume_revalidate = False
        # Seul reasoning_program possède une difficulté sémantique graduelle.
        # Les autres familles adaptent directement le poids de leurs frontières.
        self.difficulty = {capability: 0.25
                           for capability in DIFFICULTY_AWARE_CAPABILITIES}
        self.cap_history = {capability: deque(maxlen=40)
                            for capability in DIFFICULTY_AWARE_CAPABILITIES}
        self.frontier_scales = {row["key"]: 1.0 for row in self.frontier_specs}
        self.frontier_history = {row["key"]: deque(maxlen=24)
                                 for row in self.frontier_specs}
        self.frontier_base_weights = {row["key"]: float(row["weight"])
                                      for row in self.frontier_specs}
        self._last_frontier_key: str | None = None
        self.stop_requested = False
        self.metrics_path = self.stage_dir / "metrics.jsonl"
        if resume:
            self._resume(resume)
        (self.stage_dir / "config.json").write_text(
            json.dumps({"rlvr_v45": asdict(cfg), "model": self.mcfg.to_dict(),
                        "init": self.init_meta, "reference": self.ref_meta,
                        "profile_name": cfg.profile_name,
                        "profile_sha256": self.profile_sha256,
                        "frontier_schemas": sorted({row["schema_id"]
                                                    for row in self.frontier_specs}),
                        "bridge_schemas": sorted({row["schema_id"]
                                                  for row in self.bridge_specs}),
                        "adaptive_frontiers": len(self.frontier_specs),
                        "semantic_difficulty": sorted(DIFFICULTY_AWARE_CAPABILITIES),
                        "verifier_version": VERIFIER_VERSION},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _profile_curriculum(self):
        if self.profile is None:
            return [], []
        k = int(self.profile["config"]["k"])
        frontier_k = int(self.profile["config"].get("frontier_k", k))
        rows = self.profile["rows"]
        incomplete = [row["task_id"] for row in rows
                      if int(row.get("initial_successes", 0)) < k
                      and int(row.get("k", k)) < frontier_k]
        if self.cfg.require_profile and incomplete:
            raise RuntimeError(
                f"profil pass@{frontier_k} incomplet pour {len(incomplete)} tâches. "
                "Relance `rl-profile-v45 --refine-from profile.json --output profile.json`."
            )
        frontier = []
        schema_has_success = defaultdict(bool)
        schema_rows = defaultdict(list)
        for row in rows:
            schema_rows[(row["capability"], row["schema_id"])].append(row)
            successes = int(row.get("successes", row["initial_successes"]))
            measured_k = int(row.get("k", k))
            schema_has_success[(row["capability"], row["schema_id"])] |= successes > 0
            if 0 < successes < measured_k:
                probability = successes / measured_k
                frontier.append({
                    "key": row["task_id"],
                    "capability": row["capability"], "schema_id": row["schema_id"],
                    "difficulty": float(row["difficulty"]),
                    "weight": max(0.02, probability * (1.0 - probability)),
                })
        bridge = []
        all_schemas = {(capability, schema_id)
                       for capability, schemas in SCHEMAS_BY_CAPABILITY.items()
                       for schema_id in schemas}
        for key in sorted(all_schemas):
            grouped = schema_rows.get(key, [])
            if schema_has_success[key]:
                continue
            capability, schema_id = key
            bridge.append({"capability": capability, "schema_id": schema_id,
                           "difficulty": (min(float(row["difficulty"]) for row in grouped)
                                          if grouped else 0.20),
                           "weight": 1.0})
        if self.cfg.require_profile and not frontier:
            raise RuntimeError("le profil ne contient aucun groupe dynamique : bridge SFT requis avant RL")
        return frontier, bridge

    def _make_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.0, foreach=False,
        )

    def _checkpoint_payload(self) -> dict:
        return {
            "model": self.model.state_dict(), "optimizers": [self.optimizer.state_dict()],
            "model_cfg": self.mcfg.to_dict(), "rlvr_cfg": asdict(self.cfg),
            "stage": self.cfg.stage_name, "step": self.update,
            "accepted_updates": self.update, "rollout_index": self.rollout_index,
            "tokens_seen": self.init_meta.get("tokens_seen", 0) + self.tokens_generated,
            "tokens_generated": self.tokens_generated, "kl_beta": self.kl_beta,
            "lr_scale": self.lr_scale, "kl_excursions": self.kl_excursions,
            "optimizer_start_update": self.optimizer_start_update,
            "replay_index": self.replay_index,
            "replay_scale": self.replay_scale,
            "best_score": self.best_score, "baseline_score": self.baseline_score,
            "best_per_capability": self.best_per_capability,
            "capability_peaks": self.capability_peaks,
            "recovery_count": self.recovery_count,
            "difficulty": self.difficulty, "profile_sha256": self.profile_sha256,
            "verifier_version": VERIFIER_VERSION,
            "frontier_scales": self.frontier_scales,
            "frontier_history": {key: list(history)
                                 for key, history in self.frontier_history.items()},
            "cap_history": {key: list(history) for key, history in self.cap_history.items()},
            "rng": {"python": self.rng.getstate(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if self.use_cuda else None,
                    "numpy": np.random.get_state()},
        }

    def _save(self, best: bool = False):
        payload = self._checkpoint_payload()
        numbered = self.stage_dir / f"ckpt_{self.update:06d}.pt"
        _atomic_torch_save(payload, numbered)
        _atomic_link(numbered, self.stage_dir / "ckpt_latest.pt")
        if best:
            _atomic_link(numbered, self.stage_dir / "ckpt_best.pt")
        numbered_files = sorted(self.stage_dir.glob("ckpt_[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
        for old in numbered_files[:-self.cfg.keep_last]:
            old.unlink(missing_ok=True)

    def _resume(self, spec: str):
        path = resolve_checkpoint(self.run_dir, self.cfg.stage_name, spec)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != self.cfg.stage_name:
            raise ValueError(f"mauvaise phase dans {path}")
        profile_changed = checkpoint.get("profile_sha256") not in (None, self.profile_sha256)
        if profile_changed and not self._profile_matches_checkpoint(checkpoint):
            raise ValueError("le profil pass@k a changé et ne correspond pas au checkpoint repris")
        if profile_changed:
            print(f"[i] Profil phase 2 lié au checkpoint {checkpoint.get('accepted_updates')} accepté.")
        self.model.load_state_dict(checkpoint["model"])
        self.update = int(checkpoint["accepted_updates"])
        if self.cfg.reset_optimizer_on_resume:
            self.optimizer_start_update = self.update
            print("[i] AdamW réinitialisé pour la branche stabilisée.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizers"][0])
            self.optimizer_start_update = int(checkpoint.get("optimizer_start_update", 0))
        self.rollout_index = int(checkpoint.get("rollout_index", 0))
        self.tokens_generated = int(checkpoint.get("tokens_generated", 0))
        self.kl_beta = (self.cfg.kl_beta if self.cfg.reset_optimizer_on_resume
                        else float(checkpoint.get("kl_beta", self.cfg.kl_beta)))
        self.lr_scale = (1.0 if self.cfg.reset_optimizer_on_resume
                         else float(checkpoint.get("lr_scale", 1.0)))
        self.kl_excursions = (0 if self.cfg.reset_optimizer_on_resume
                              else int(checkpoint.get("kl_excursions", 0)))
        self.replay_index = (0 if profile_changed
                             else int(checkpoint.get("replay_index", 0)))
        self.replay_scale = (1.0 if self.cfg.reset_optimizer_on_resume
                             else float(checkpoint.get("replay_scale", 1.0)))
        self.best_score = float(checkpoint.get("best_score", -1.0))
        self.baseline_score = checkpoint.get("baseline_score")
        self.best_per_capability = {
            key: float(value) for key, value in checkpoint.get("best_per_capability", {}).items()
        }
        self.capability_peaks = {
            key: float(value) for key, value in checkpoint.get("capability_peaks", {}).items()
        }
        self.recovery_count = int(checkpoint.get("recovery_count", 0))
        self._resume_revalidate = (
            checkpoint.get("verifier_version") != VERIFIER_VERSION
            or profile_changed or not self.best_per_capability
        )
        if not profile_changed:
            saved_difficulty = checkpoint.get("difficulty", {})
            for capability in DIFFICULTY_AWARE_CAPABILITIES:
                if capability in saved_difficulty:
                    self.difficulty[capability] = float(saved_difficulty[capability])
            for key, value in checkpoint.get("frontier_scales", {}).items():
                if key in self.frontier_scales:
                    self.frontier_scales[key] = float(value)
            for key, values in checkpoint.get("frontier_history", {}).items():
                if key in self.frontier_history:
                    self.frontier_history[key].extend(float(value) for value in values)
            for key, values in checkpoint.get("cap_history", {}).items():
                if key in self.cap_history:
                    self.cap_history[key].extend(float(value) for value in values)
        rng = checkpoint.get("rng", {})
        if rng.get("python") is not None:
            self.rng.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if self.use_cuda and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        del checkpoint
        gc.collect()
        self._rewind_future_metrics()
        latest = self.stage_dir / "ckpt_latest.pt"
        if path.is_file() and path.resolve() != latest.resolve():
            _atomic_link(path, latest)
        print(f"[i] Reprise RLVR v4.5 : {path.name}, update {self.update}")
        if self._resume_revalidate:
            print(f"[i] Évaluation de reprise requise ({VERIFIER_VERSION}) : "
                  "score et floors par capacité recalculés avant la reprise.")

    def _profile_matches_checkpoint(self, checkpoint: dict) -> bool:
        if self.profile is None or not self.cfg.anchor_resume_reference:
            return False
        profiled = self.profile.get("checkpoint", {})
        return (profiled.get("stage") == checkpoint.get("stage")
                and int(profiled.get("step", -1))
                == int(checkpoint.get("accepted_updates", checkpoint.get("step", -2))))

    def _rewind_future_metrics(self):
        """Isole les métriques d'une branche abandonnée lors d'une reprise ancienne."""
        if not self.metrics_path.is_file():
            return
        kept, abandoned = [], []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines(keepends=True):
            if not line.strip():
                continue
            row = json.loads(line)
            (kept if int(row.get("update", -1)) <= self.update else abandoned).append(line)
        if not abandoned:
            return
        index = 1
        while True:
            archive = self.stage_dir / (
                f"metrics_abandoned_after_{self.update:06d}_{index:02d}.jsonl"
            )
            if not archive.exists():
                break
            index += 1
        archive.write_text("".join(abandoned), encoding="utf-8")
        tmp = self.metrics_path.with_suffix(self.metrics_path.suffix + ".rewind.tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        tmp.replace(self.metrics_path)
        print(f"[i] Reprise historique : {len(abandoned)} métriques postérieures isolées "
              f"dans {archive.name}")

    def _next_task(self) -> TaskSpec:
        self._last_frontier_key = None
        if self.frontier_specs:
            spec = self.rng.choices(
                self.frontier_specs,
                weights=[row["weight"] * self.frontier_scales.get(row["key"], 1.0)
                         * self._capability_focus(row["capability"])
                         for row in self.frontier_specs], k=1,
            )[0]
            self._last_frontier_key = spec["key"]
        else:
            raise RuntimeError("aucune frontière pass@32 : les schémas 0/32 doivent rester en bridge SFT")
        capability = spec["capability"]
        seed = self.cfg.seed + 1_000_000 + self.rollout_index * 31
        self.rollout_index += 1
        offset = (self.difficulty[capability] - 0.25
                  if capability in DIFFICULTY_AWARE_CAPABILITIES else 0.0)
        difficulty = max(0.05, min(0.95, float(spec["difficulty"]) + offset))
        return make_task(seed, "train", difficulty, capability, spec["schema_id"])

    def _reward(self, task: TaskSpec, sample) -> tuple[float, bool, str | None]:
        result = verify(task.answer, sample.text)
        visible = final_text(sample.text)
        clean_format = sample.stopped and bool(visible)
        reward = result.primary_score
        if result.primary_success:
            reward += 0.03 * float(clean_format) + 0.02 * result.format_score
        if not sample.stopped:
            reward -= 0.20
        if _repeat_ratio(visible) > 0.22:
            reward -= 0.05
        return max(-0.25, min(1.05, reward)), result.primary_success, result.failure_code

    def _rollouts(self):
        groups = []
        stats = defaultdict(float)
        max_attempts = max(self.cfg.prompts_per_update,
                           math.ceil(self.cfg.prompts_per_update * self.cfg.oversample))
        needs_sft = []
        while len(groups) < self.cfg.prompts_per_update and stats["attempts"] < max_attempts:
            task = self._next_task()
            stats["attempts"] += 1
            self.model.eval()
            prompt_ids, samples = self.engine.sample(task, self.cfg.group_size)
            rewards, successes, failures = [], [], []
            for sample in samples:
                reward, success, failure = self._reward(task, sample)
                rewards.append(reward)
                successes.append(success)
                failures.append(failure)
                stats["samples"] += 1
                stats["successes"] += int(success)
                stats["tokens"] += len(sample.token_ids)
                stats["entropy"] += sample.entropy
            primary_count = sum(successes)
            success_rate = primary_count / self.cfg.group_size
            if task.capability in self.cap_history:
                self.cap_history[task.capability].append(success_rate)
            if self._last_frontier_key in self.frontier_history:
                self.frontier_history[self._last_frontier_key].append(success_rate)
            if primary_count == 0:
                needs_sft.append({"task": task.to_dict(), "failure_codes": failures})
            if 0 < primary_count < self.cfg.group_size:
                centered = np.asarray(rewards, dtype=np.float32) - float(np.mean(rewards))
                groups.append((task, prompt_ids, samples, centered.tolist()))
                stats["useful"] += 1
        self._adjust_curriculum()
        if needs_sft:
            with (self.stage_dir / "needs_sft.jsonl").open("a", encoding="utf-8") as stream:
                for row in needs_sft:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return groups, stats

    def _adjust_curriculum(self):
        for capability, history in self.cap_history.items():
            if len(history) < 20:
                continue
            rate = sum(history) / len(history)
            if rate > 0.72:
                self.difficulty[capability] = min(0.95, self.difficulty[capability] + 0.05)
            elif rate < 0.08:
                self.difficulty[capability] = max(0.05, self.difficulty[capability] - 0.05)
            history.clear()
        for key, history in self.frontier_history.items():
            if len(history) < 12:
                continue
            rate = sum(history) / len(history)
            observed_utility = max(0.005, rate * (1.0 - rate))
            base_utility = max(0.005, self.frontier_base_weights[key])
            desired = max(0.25, min(4.0, observed_utility / base_utility))
            self.frontier_scales[key] = 0.5 * self.frontier_scales[key] + 0.5 * desired
            history.clear()

    def _snapshot_control_state(self) -> dict:
        return {
            "difficulty": dict(self.difficulty),
            "cap_history": {key: list(history) for key, history in self.cap_history.items()},
            "frontier_scales": dict(self.frontier_scales),
            "frontier_history": {key: list(history)
                                 for key, history in self.frontier_history.items()},
            "kl_beta": self.kl_beta, "lr_scale": self.lr_scale,
            "kl_excursions": self.kl_excursions, "replay_index": self.replay_index,
            "replay_scale": self.replay_scale,
        }

    def _restore_control_state(self, state: dict):
        self.difficulty = dict(state["difficulty"])
        self.frontier_scales = dict(state["frontier_scales"])
        for key, values in state["cap_history"].items():
            self.cap_history[key].clear()
            self.cap_history[key].extend(values)
        for key, values in state["frontier_history"].items():
            self.frontier_history[key].clear()
            self.frontier_history[key].extend(values)
        self.kl_beta = float(state["kl_beta"])
        self.lr_scale = float(state["lr_scale"])
        self.kl_excursions = int(state["kl_excursions"])
        self.replay_index = int(state["replay_index"])
        self.replay_scale = float(state["replay_scale"])

    def _snapshot_train_state(self) -> dict:
        """Snapshot CPU exact, réservé aux updates dans la zone KL d'alerte."""
        return {
            "model": _clone_to_cpu(self.model.state_dict()),
            "optimizer": _clone_to_cpu(self.optimizer.state_dict()),
        }

    def _restore_train_state(self, state: dict):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer.zero_grad(set_to_none=True)

    def _register_kl(self, kl: float) -> bool:
        """Adapte beta et LR ; renvoie True lorsqu'une baisse de LR a eu lieu."""
        excursion = not math.isfinite(kl) or kl > self.cfg.kl_soft_max
        if excursion:
            self.kl_excursions += 1
            self.kl_beta = min(self.cfg.kl_beta_max, self.kl_beta * 1.5)
        else:
            self.kl_excursions = max(0, self.kl_excursions - 1)
            if kl < self.cfg.kl_target / 1.5:
                self.kl_beta = max(0.002, self.kl_beta / 1.1)
        if self.kl_excursions < self.cfg.kl_excursion_patience:
            return False
        previous = self.lr_scale
        self.lr_scale = max(self.cfg.min_lr_scale, self.lr_scale * 0.5)
        self.kl_excursions = 0
        return self.lr_scale < previous

    def _dev_regression(self, macro: float) -> bool:
        return self.best_score - macro > self.cfg.max_dev_drop + 1e-12

    def _capability_focus(self, capability: str) -> float:
        """Concentre le RL sur les déficits sans retirer totalement une frontière acquise."""
        score = getattr(self, "best_per_capability", {}).get(capability, 0.0)
        return max(0.02, (1.0 - score) ** 2)

    def _capability_gate(self, evaluation: dict) -> tuple[bool, dict[str, tuple[float, float]]]:
        current = evaluation.get("per_capability", {})
        failures = {}
        for capability, peak in getattr(self, "capability_peaks", {}).items():
            floor = max(0.0, peak - self.cfg.max_capability_drop)
            score = float(current.get(capability, 0.0))
            if score + 1e-12 < floor:
                failures[capability] = (score, floor)
        return not failures, failures

    def _record_best_capabilities(self, evaluation: dict):
        current = {key: float(value)
                   for key, value in evaluation.get("per_capability", {}).items()}
        self.best_per_capability = current
        if not hasattr(self, "capability_peaks"):
            self.capability_peaks = {}
        for capability, score in current.items():
            self.capability_peaks[capability] = max(
                score, self.capability_peaks.get(capability, 0.0)
            )

    def _recover_from_best(self, reason: str, failures: dict | None = None) -> bool:
        """Rollback vers le best, durcit la rétention et reprend avec un LR plus petit."""
        can_continue = self.recovery_count < self.cfg.max_auto_recoveries
        path = resolve_checkpoint(self.run_dir, self.cfg.stage_name, "best")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        previous_update = self.update
        previous_scale = self.lr_scale
        previous_beta = self.kl_beta
        self.model.load_state_dict(checkpoint["model"])
        self.update = int(checkpoint.get("accepted_updates", checkpoint.get("step", 0)))
        self.optimizer = self._make_optimizer()
        self.optimizer_start_update = self.update
        self.lr_scale = max(self.cfg.min_lr_scale, previous_scale * 0.5)
        self.kl_beta = min(self.cfg.kl_beta_max,
                           max(self.cfg.kl_beta, previous_beta * 1.5))
        self.kl_excursions = 0
        self.replay_scale = min(2.0, self.replay_scale * 1.25)
        if can_continue:
            self.recovery_count += 1
        del checkpoint
        gc.collect()
        self._rewind_future_metrics()
        _atomic_link(path, self.stage_dir / "ckpt_latest.pt")
        event = {
            "from_update": previous_update, "to_update": self.update,
            "reason": reason, "capability_failures": failures or {},
            "recovery": self.recovery_count, "lr_scale": self.lr_scale,
            "kl_beta": self.kl_beta, "replay_scale": self.replay_scale,
        }
        with (self.stage_dir / "recoveries.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"[↩] Rollback automatique {previous_update} → best {self.update} "
              f"({reason}) · reprise {self.recovery_count}/{self.cfg.max_auto_recoveries} · "
              f"lr×{self.lr_scale:.3f} · beta {self.kl_beta:.3f} · "
              f"replay×{self.replay_scale:.2f}")
        if not can_continue:
            print("[!] Budget de reprises automatiques épuisé : arrêt sur ckpt_best.")
        return can_continue

    def _next_retention_task(self) -> TaskSpec:
        """Alterne bridge des déficits et rétention équilibrée de toutes les capacités."""
        capabilities = tuple(SCHEMAS_BY_CAPABILITY)
        slot = self.replay_index
        bridge_specs = getattr(self, "bridge_specs", [])
        if bridge_specs and slot % 2 == 0:
            ordered = sorted(
                bridge_specs,
                key=lambda row: (-self._capability_focus(row["capability"]),
                                 row["capability"], row["schema_id"]),
            )
            spec = ordered[(slot // 2) % len(ordered)]
            capability, schema = spec["capability"], spec["schema_id"]
        else:
            balanced_slot = slot // 2 if bridge_specs else slot
            capability = capabilities[balanced_slot % len(capabilities)]
            schemas = SCHEMAS_BY_CAPABILITY[capability]
            schema = schemas[(balanced_slot // len(capabilities)) % len(schemas)]
        seed = self.cfg.seed + 10_000_000 + slot * 37
        self.replay_index += 1
        return make_task(seed, "train", 0.35, capability, schema)

    def _make_microbatches(self, groups):
        rows = []
        for _task, prompt_ids, samples, advantages in groups:
            for sample, advantage in zip(samples, advantages):
                rows.append((prompt_ids + sample.token_ids, len(prompt_ids), advantage))
        batches = []
        for start in range(0, len(rows), self.cfg.micro_bs):
            chunk = rows[start:start + self.cfg.micro_bs]
            width = max(len(sequence) for sequence, _, _ in chunk)
            x = torch.full((len(chunk), width), self.sp["eot"], dtype=torch.long)
            mask = torch.zeros((len(chunk), width - 1), dtype=torch.bool)
            adv = torch.empty((len(chunk), 1), dtype=torch.float32)
            for row, (sequence, prompt_len, advantage) in enumerate(chunk):
                x[row, :len(sequence)] = torch.tensor(sequence)
                mask[row, prompt_len - 1:len(sequence) - 1] = True
                adv[row, 0] = advantage
            batches.append((x, mask, adv))
        return batches, len(rows) * self.cfg.max_new_tokens

    def _replay_batch(self):
        batch = []
        retention_rows = max(1, self.cfg.micro_bs // 2)
        for index in range(self.cfg.micro_bs):
            if index >= retention_rows:
                question, answer = self.rng.choice(_CURATED_REPLAY)
                prompt = D.render_chat([{"role": "user", "text": question}]) + f"{D.IM_START}assistant\n"
                completion = f"{D.THINK}\n\n{D.THINK_END}\n{answer}{D.IM_END}\n"
            else:
                task = self._next_retention_task()
                prompt_ids, _prefill = self.engine.prompt(task)
                prompt = None
                completion = _canonical_answer(task) + f"{D.IM_END}\n"
                if task.requires_trace:
                    completion = "\n".join(task.canonical_trace) + f"\n{D.THINK_END}\n" + completion
                prefix = prompt_ids
            if prompt is not None:
                prefix = self.tok.encode(prompt).ids
            target = self.tok.encode(completion).ids
            batch.append((prefix + target, len(prefix)))
        width = max(len(sequence) for sequence, _ in batch)
        x = torch.full((len(batch), width - 1), self.sp["eot"], dtype=torch.long)
        y = torch.full_like(x, -100)
        for row, (sequence, prompt_len) in enumerate(batch):
            values = torch.tensor(sequence, dtype=torch.long)
            x[row, :len(sequence) - 1] = values[:-1]
            y[row, prompt_len - 1:len(sequence) - 1] = values[prompt_len:]
        return x, y

    @torch.inference_mode()
    def _measure_kl(self, batches, constant_denominator: int) -> float:
        self.model.eval()
        total = 0.0
        for x_cpu, mask_cpu, _adv_cpu in batches:
            x = x_cpu.to(self.device, non_blocking=self.use_cuda)
            mask = mask_cpu.to(self.device, non_blocking=self.use_cuda)
            targets = x[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda):
                logits, _, _ = self.model(x, diagnostics=False)
                ref_logits, _, _ = self.ref(x, diagnostics=False)
            vocab = logits.shape[-1]
            logp = -F.cross_entropy(logits[:, :-1].reshape(-1, vocab).float(),
                                    targets.reshape(-1), reduction="none").view_as(targets)
            ref_logp = -F.cross_entropy(ref_logits[:, :-1].reshape(-1, vocab).float(),
                                        targets.reshape(-1), reduction="none").view_as(targets)
            difference = ref_logp - logp
            kl_tokens = difference.exp() - difference - 1.0
            total += float(kl_tokens[mask].sum() / constant_denominator)
            del x, mask, targets, logits, ref_logits, logp, ref_logp
            del difference, kl_tokens
        return total

    def _train_update(self, groups) -> dict:
        batches, constant_denominator = self._make_microbatches(groups)
        self.model.train()
        warm_step = self.update - self.optimizer_start_update + 1
        lr = (self.cfg.lr * self.lr_scale
              * min(1.0, warm_step / max(1, self.cfg.warmup)))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        pg_total = kl_total = token_count = 0.0
        for x_cpu, mask_cpu, adv_cpu in batches:
            x = x_cpu.to(self.device, non_blocking=self.use_cuda)
            mask = mask_cpu.to(self.device, non_blocking=self.use_cuda)
            advantages = adv_cpu.to(self.device, non_blocking=self.use_cuda)
            targets = x[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda):
                logits, _, _ = self.model(x, diagnostics=False)
            vocab = logits.shape[-1]
            logp = -F.cross_entropy(logits[:, :-1].reshape(-1, vocab).float(),
                                    targets.reshape(-1), reduction="none").view_as(targets)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                  enabled=self.use_cuda):
                ref_logits, _, _ = self.ref(x, diagnostics=False)
            ref_logp = -F.cross_entropy(ref_logits[:, :-1].reshape(-1, vocab).float(),
                                        targets.reshape(-1), reduction="none").view_as(targets)
            difference = ref_logp - logp
            kl_tokens = difference.exp() - difference - 1.0
            pg = -(advantages * logp)[mask].sum() / constant_denominator
            kl = kl_tokens[mask].sum() / constant_denominator
            (pg + self.kl_beta * kl).backward()
            pg_total += float(pg.detach())
            kl_total += float(kl.detach())
            token_count += float(mask.sum())
            del x, mask, advantages, targets, logits, ref_logits, logp, ref_logp
            del difference, kl_tokens, pg, kl

        common = {"pg": pg_total, "kl": kl_total, "post_kl": None,
                  "replay": 0.0, "retention_kl": 0.0,
                  "grad_norm": 0.0, "lr": lr,
                  "tokens": token_count}
        if not math.isfinite(kl_total) or kl_total > self.cfg.kl_hard_max:
            self.optimizer.zero_grad(set_to_none=True)
            return {**common, "accepted": False, "reject_reason": "kl_pre_hard"}

        transaction = None
        if kl_total > self.cfg.kl_soft_max:
            try:
                transaction = self._snapshot_train_state()
            except (MemoryError, RuntimeError):
                self.optimizer.zero_grad(set_to_none=True)
                gc.collect()
                return {**common, "accepted": False,
                        "reject_reason": "transaction_snapshot_failed"}

        replay_x, replay_y = self._replay_batch()
        replay_x = replay_x.to(self.device, non_blocking=self.use_cuda)
        replay_y = replay_y.to(self.device, non_blocking=self.use_cuda)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda):
            _logits, replay_loss, _ = self.model(
                replay_x, targets=replay_y, diagnostics=False, loss_reduction="mean"
            )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                              enabled=self.use_cuda):
            replay_ref_logits, _, _ = self.ref(replay_x, diagnostics=False)
        replay_mask = replay_y.ne(-100)
        replay_targets = replay_y.clamp_min(0)
        replay_vocab = _logits.shape[-1]
        replay_logp = -F.cross_entropy(
            _logits.reshape(-1, replay_vocab).float(), replay_targets.reshape(-1),
            reduction="none",
        ).view_as(replay_targets)
        replay_ref_logp = -F.cross_entropy(
            replay_ref_logits.reshape(-1, replay_vocab).float(), replay_targets.reshape(-1),
            reduction="none",
        ).view_as(replay_targets)
        replay_difference = replay_ref_logp - replay_logp
        retention_tokens = replay_difference.exp() - replay_difference - 1.0
        retention_kl = retention_tokens[replay_mask].mean()
        (self.cfg.replay_weight * self.replay_scale * replay_loss
         + self.cfg.retention_kl_weight * self.replay_scale * retention_kl).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                         self.cfg.grad_clip))
        replay_value = float(replay_loss.detach())
        retention_value = float(retention_kl.detach())
        if (not math.isfinite(replay_value) or not math.isfinite(retention_value)
                or not math.isfinite(grad_norm)):
            self.optimizer.zero_grad(set_to_none=True)
            del replay_x, replay_y, _logits, replay_loss, replay_ref_logits
            del replay_mask, replay_targets, replay_logp, replay_ref_logp
            del replay_difference, retention_tokens, retention_kl
            if transaction is not None:
                del transaction
            gc.collect()
            return {**common, "post_kl": float("nan"), "replay": replay_value,
                    "retention_kl": retention_value,
                    "grad_norm": grad_norm, "accepted": False,
                    "reject_reason": "nonfinite_replay_or_grad"}
        self.optimizer.step()
        del replay_x, replay_y, _logits, replay_loss, replay_ref_logits
        del replay_mask, replay_targets, replay_logp, replay_ref_logp
        del replay_difference, retention_tokens, retention_kl
        post_kl = None
        if transaction is not None:
            post_kl = self._measure_kl(batches, constant_denominator)
            if not math.isfinite(post_kl) or post_kl > self.cfg.kl_hard_max:
                self._restore_train_state(transaction)
                del transaction
                gc.collect()
                return {**common, "post_kl": post_kl,
                        "replay": replay_value, "retention_kl": retention_value,
                        "grad_norm": grad_norm, "accepted": False,
                        "reject_reason": "kl_post_hard"}
            del transaction
            gc.collect()
        return {**common, "post_kl": post_kl, "replay": replay_value,
                "retention_kl": retention_value,
                "grad_norm": grad_norm, "accepted": True, "reject_reason": None}

    @torch.inference_mode()
    def _evaluate(self) -> dict:
        self.model.eval()
        scores = defaultdict(list)
        rows = []
        if self.profile is not None:
            eval_rows = self.profile["rows"][:self.cfg.eval_tasks]
            profile_seed = int(self.profile["config"]["seed"])
            tasks = [make_task(profile_seed + index * 17, "dev", row["difficulty"],
                               row["capability"])
                     for index, row in enumerate(eval_rows)]
            if any(task.task_id != row["task_id"] for task, row in zip(tasks, eval_rows)):
                raise RuntimeError("impossible de reconstruire les tâches dev du profil")
        else:
            capabilities = tuple(CAPABILITY_WEIGHTS)
            tasks = [make_task(455_700 + index * 101, "dev", 0.45,
                               capabilities[index % len(capabilities)])
                     for index in range(self.cfg.eval_tasks)]
        for task in tasks:
            capability = task.capability
            sample = self.engine.greedy(task)
            result = verify(task.answer, sample.text)
            scores[capability].append(float(result.primary_success))
            rows.append({"task_id": task.task_id, "capability": capability,
                         "success": result.primary_success,
                         "failure_code": result.failure_code,
                         "answer": final_text(sample.text)[:300]})
        per_capability = {name: sum(values) / len(values) for name, values in scores.items()}
        macro = sum(per_capability.values()) / max(1, len(per_capability))
        return {"macro": macro, "per_capability": per_capability, "rows": rows}

    def train(self):
        if self.baseline_score is None:
            baseline = self._evaluate()
            self.baseline_score = baseline["macro"]
            self.best_score = max(self.best_score, self.baseline_score)
            self._record_best_capabilities(baseline)
            (self.stage_dir / "eval_baseline.json").write_text(
                json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[i] Baseline dev macro : {self.baseline_score:.3f}")
            self._resume_revalidate = False
        elif self._resume_revalidate:
            resumed_eval = self._evaluate()
            self.best_score = resumed_eval["macro"]
            self._record_best_capabilities(resumed_eval)
            (self.stage_dir / f"eval_resume_{self.update:06d}.json").write_text(
                json.dumps(resumed_eval, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._resume_revalidate = False
            detail = ", ".join(f"{key}={value:.1f}"
                               for key, value in self.best_per_capability.items())
            print(f"[i] Checkpoint revalidé avec {VERIFIER_VERSION} : "
                  f"macro {self.best_score:.3f} · {detail}")
        print(f"[i] RLVR v4.5 local · cible {self.cfg.accepted_updates} updates acceptées · "
              f"{self.cfg.prompts_per_update}×{self.cfg.group_size} rollouts · lr {self.cfg.lr:.1e}")
        ref_meta = getattr(self, "ref_meta", {})
        print(f"[i] Ancre KL fp32 · {ref_meta.get('stage')} "
              f"step {ref_meta.get('step')} · "
              f"source {ref_meta.get('anchor', 'explicite')}")
        if self.profile is not None:
            frontier = ", ".join(sorted({row["schema_id"] for row in self.frontier_specs}))
            bridge = ", ".join(sorted({row["schema_id"] for row in self.bridge_specs}))
            print(f"[i] Curriculum phase 2 · RL sur frontières [{frontier}] · "
                  f"bridge supervisé seulement [{bridge}]")
            print(f"[i] Adaptation en ligne · {len(self.frontier_specs)} frontières pondérées · "
                  "difficulté sémantique : reasoning_program")
            focus = ", ".join(
                f"{capability}×{self._capability_focus(capability):.2f}"
                for capability in SCHEMAS_BY_CAPABILITY
            )
            print(f"[i] Focus déficit · {focus}")

        def stop(_signum, _frame):
            self.stop_requested = True
            print("\n[i] Arrêt demandé : sauvegarde après l'update courante.")

        signal.signal(signal.SIGINT, stop)
        stop_file = self.run_dir / "STOP"
        empty_batches = 0
        rejected_kl_streak = 0
        rejections_path = self.stage_dir / "rejected_updates.jsonl"
        while self.update < self.cfg.accepted_updates and not self.stop_requested:
            started = time.time()
            control_state = self._snapshot_control_state()
            groups, rollout = self._rollouts()
            if len(groups) < self.cfg.prompts_per_update:
                self._restore_control_state(control_state)
                empty_batches += 1
                print(f"[!] Lot incomplet : {len(groups)}/{self.cfg.prompts_per_update} "
                      f"groupes dynamiques sur {int(rollout['attempts'])} tirages; nouvel essai.")
                if empty_batches >= self.cfg.max_empty_batches:
                    raise RuntimeError(
                        "impossible de remplir un lot RLVR fixe après 20 essais : "
                        "raffiner le profil ou renforcer le bridge SFT"
                    )
                continue
            empty_batches = 0
            train_stats = self._train_update(groups)
            self.tokens_generated += int(rollout["tokens"])
            if not train_stats["accepted"]:
                self._restore_control_state(control_state)
                observed_kl = train_stats["post_kl"]
                if observed_kl is None or not math.isfinite(observed_kl):
                    observed_kl = train_stats["kl"]
                lr_reduced = self._register_kl(float(observed_kl))
                rejected_kl_streak += 1
                rejection = {
                    "after_update": self.update, "rollout_index": self.rollout_index,
                    "reason": train_stats["reject_reason"],
                    "kl": train_stats["kl"], "post_kl": train_stats["post_kl"],
                    "kl_beta": self.kl_beta, "lr_scale": self.lr_scale,
                    "lr_reduced": lr_reduced, "elapsed_s": time.time() - started,
                }
                with rejections_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(rejection, ensure_ascii=False) + "\n")
                print(f"[!] Update {self.update + 1} rejetée ({train_stats['reject_reason']}) · "
                      f"KL {train_stats['kl']:.4f} · post "
                      f"{train_stats['post_kl'] if train_stats['post_kl'] is not None else '-'} · "
                      f"beta {self.kl_beta:.3f} · lr×{self.lr_scale:.3f}")
                if rejected_kl_streak >= self.cfg.max_kl_rejections:
                    if not self._recover_from_best("kl_rejections"):
                        self.stop_requested = True
                    rejected_kl_streak = 0
                continue
            rejected_kl_streak = 0
            observed_kl = max(train_stats["kl"], train_stats["post_kl"] or 0.0)
            lr_reduced = self._register_kl(observed_kl)
            self.update += 1
            success_rate = rollout["successes"] / max(1, rollout["samples"])
            record = {
                "update": self.update, "rollout_index": self.rollout_index,
                "success_rate": success_rate, "useful_groups": int(rollout["useful"]),
                "attempts": int(rollout["attempts"]),
                "mean_entropy": rollout["entropy"] / max(1, rollout["samples"]),
                "mean_length": rollout["tokens"] / max(1, rollout["samples"]),
                "tokens_generated": self.tokens_generated, "kl_beta": self.kl_beta,
                "lr_scale": self.lr_scale, "lr_reduced": lr_reduced,
                "difficulty": dict(self.difficulty),
                "frontier_scales": dict(self.frontier_scales),
                "elapsed_s": time.time() - started,
                **train_stats,
            }
            is_best = False
            recovery_reason = None
            capability_failures = {}
            if self.update % self.cfg.eval_every == 0 or self.update == self.cfg.accepted_updates:
                evaluation = self._evaluate()
                record["eval"] = evaluation
                safe_kl = train_stats["kl"] <= self.cfg.kl_soft_max
                capability_ok, capability_failures = self._capability_gate(evaluation)
                if evaluation["macro"] > self.best_score and safe_kl and capability_ok:
                    self.best_score = evaluation["macro"]
                    self._record_best_capabilities(evaluation)
                    self.recovery_count = 0
                    is_best = True
                elif not capability_ok:
                    recovery_reason = "capability_floor"
                elif self._dev_regression(evaluation["macro"]):
                    recovery_reason = "dev_regression"
                if recovery_reason:
                    record["recovery_reason"] = recovery_reason
                    record["capability_failures"] = capability_failures
                (self.stage_dir / f"eval_{self.update:06d}.json").write_text(
                    json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"update {self.update:3d}/{self.cfg.accepted_updates} · "
                  f"ok {100*success_rate:4.1f}% · groupes {len(groups)}/{int(rollout['attempts'])} · "
                  f"KL {train_stats['kl']:.4f} · replay {train_stats['replay']:.3f} · "
                  f"retKL {train_stats['retention_kl']:.4f} · "
                  f"lr×{self.lr_scale:.3f} · "
                  f"{time.time()-started:.1f}s")
            if recovery_reason:
                if capability_failures:
                    detail = ", ".join(
                        f"{key}={score:.1f}<{floor:.1f}"
                        for key, (score, floor) in capability_failures.items()
                    )
                    print(f"[!] Gate capacité déclenché : {detail}")
                if not self._recover_from_best(recovery_reason, capability_failures):
                    self.stop_requested = True
                continue
            if is_best or self.update % self.cfg.save_every == 0 or stop_file.exists():
                self._save(best=is_best)
            if stop_file.exists():
                stop_file.unlink(missing_ok=True)
                self.stop_requested = True
        self._save(best=False)
        print(f"[✓] RLVR v4.5 terminé/repris à l'update {self.update} : {self.stage_dir}")


def _prepare_resume_profile(args) -> str:
    """Crée/réutilise automatiquement un pass@32 lié au checkpoint de phase 2."""
    if not args.resume or args.keep_reference:
        return args.profile_name or "profile.json"
    profile_name = args.profile_name or "profile_phase2.json"
    if args.no_refresh_profile:
        return profile_name
    run_dir = Path(args.out_dir) / args.run
    checkpoint_path = resolve_checkpoint(run_dir, args.stage_name, args.resume)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    expected_stage = checkpoint.get("stage")
    expected_step = int(checkpoint.get("accepted_updates", checkpoint.get("step", -1)))
    del checkpoint
    profile_path = run_dir / args.stage_name / profile_name
    if profile_path.is_file():
        existing = json.loads(profile_path.read_text(encoding="utf-8"))
        source = existing.get("checkpoint", {})
        profile_cfg = existing.get("config", {})
        same_recipe = (
            int(profile_cfg.get("tasks", -1)) == args.eval_tasks
            and int(profile_cfg.get("k", -1)) == 6
            and int(profile_cfg.get("frontier_k", -1)) == 32
            and int(profile_cfg.get("max_new", -1)) == args.max_new
            and int(profile_cfg.get("seed", -1)) == 455_001
        )
        if (source.get("stage") == expected_stage
                and int(source.get("step", -2)) == expected_step and same_recipe):
            print(f"[i] Profil phase 2 réutilisé : {profile_path.name} (best {expected_step})")
            return profile_name
    print(f"[i] Reprofilage automatique pass@32 du best {expected_step} avant la phase 2.")
    from frlm.rl_profile_v45 import profile
    profile(
        args.run, args.data_dir, args.out_dir, args.stage_name, args.resume,
        args.eval_tasks, 6, 32, args.max_new, 455_001, args.device,
        profile_name, "", args.stage_name,
    )
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return profile_name


def cmd_rl_v45(args):
    profile_name = _prepare_resume_profile(args)
    cfg = RLVRConfig(
        run_name=args.run, data_dir=args.data_dir, out_dir=args.out_dir,
        stage_name=args.stage_name, init_stage=args.init_stage, init_ckpt=args.init_ckpt,
        ref_stage=args.ref_stage, ref_ckpt=args.ref_ckpt,
        profile_name=profile_name, anchor_resume_reference=not args.keep_reference,
        accepted_updates=args.updates, prompts_per_update=args.prompts,
        group_size=args.group, max_new_tokens=args.max_new, micro_bs=args.micro_bs,
        lr=args.lr, kl_beta=args.kl_beta, kl_target=args.kl_target,
        kl_beta_max=args.kl_beta_max, kl_soft_max=args.kl_soft_max,
        kl_hard_max=args.kl_hard_max,
        kl_excursion_patience=args.kl_excursion_patience,
        min_lr_scale=args.min_lr_scale,
        replay_weight=args.replay_weight, retention_kl_weight=args.retention_kl_weight,
        oversample=args.oversample,
        eval_every=args.eval_every, eval_tasks=args.eval_tasks,
        save_every=args.save_every, max_kl_rejections=args.max_kl_rejections,
        max_dev_drop=args.max_dev_drop, max_capability_drop=args.max_capability_drop,
        max_auto_recoveries=args.max_auto_recoveries,
        seed=args.seed, device=args.device,
        require_profile=not args.allow_no_profile,
        reset_optimizer_on_resume=args.reset_optimizer,
    )
    RLVRTrainer(cfg, resume=args.resume).train()
