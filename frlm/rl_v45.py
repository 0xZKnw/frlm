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
from frlm.verifiers_v45 import final_text, verify


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
    accepted_updates: int = 200
    prompts_per_update: int = 3
    group_size: int = 6
    max_new_tokens: int = 112
    micro_bs: int = 2
    lr: float = 2e-6
    warmup: int = 5
    kl_beta: float = 0.018
    kl_target: float = 0.012
    replay_weight: float = 0.05
    grad_clip: float = 1.0
    oversample: float = 4.0
    eval_every: int = 10
    eval_tasks: int = 60
    save_every: int = 10
    keep_last: int = 2
    max_empty_batches: int = 20
    seed: int = 455_100
    device: str = "cuda"
    require_profile: bool = True


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
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible")
        self.device = cfg.device
        self.use_cuda = self.device.startswith("cuda")
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.stage_dir = self.run_dir / cfg.stage_name
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self.stage_dir / "profile.json"
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
        init_path = resolve_checkpoint(self.run_dir, cfg.init_stage, cfg.init_ckpt)
        self.model, self.mcfg, self.init_meta = load_policy(init_path, self.device, torch.float32)
        if cfg.ref_stage == cfg.init_stage and cfg.ref_ckpt == cfg.init_ckpt:
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
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=0.0, foreach=False,
        )
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
        self.best_score = -1.0
        self.baseline_score = None
        self.difficulty = {capability: 0.25 for capability in CAPABILITY_WEIGHTS}
        self.cap_history = {capability: deque(maxlen=40) for capability in CAPABILITY_WEIGHTS}
        self.stop_requested = False
        self.metrics_path = self.stage_dir / "metrics.jsonl"
        if resume:
            self._resume(resume)
        (self.stage_dir / "config.json").write_text(
            json.dumps({"rlvr_v45": asdict(cfg), "model": self.mcfg.to_dict(),
                        "init": self.init_meta, "reference": self.ref_meta,
                        "profile_sha256": self.profile_sha256,
                        "frontier_schemas": sorted({row["schema_id"]
                                                    for row in self.frontier_specs}),
                        "bridge_schemas": sorted({row["schema_id"]
                                                  for row in self.bridge_specs})},
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

    def _checkpoint_payload(self) -> dict:
        return {
            "model": self.model.state_dict(), "optimizers": [self.optimizer.state_dict()],
            "model_cfg": self.mcfg.to_dict(), "rlvr_cfg": asdict(self.cfg),
            "stage": self.cfg.stage_name, "step": self.update,
            "accepted_updates": self.update, "rollout_index": self.rollout_index,
            "tokens_seen": self.init_meta.get("tokens_seen", 0) + self.tokens_generated,
            "tokens_generated": self.tokens_generated, "kl_beta": self.kl_beta,
            "best_score": self.best_score, "baseline_score": self.baseline_score,
            "difficulty": self.difficulty, "profile_sha256": self.profile_sha256,
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
        if checkpoint.get("profile_sha256") not in (None, self.profile_sha256):
            raise ValueError("le profil pass@k a changé depuis ce checkpoint RLVR")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizers"][0])
        self.update = int(checkpoint["accepted_updates"])
        self.rollout_index = int(checkpoint.get("rollout_index", 0))
        self.tokens_generated = int(checkpoint.get("tokens_generated", 0))
        self.kl_beta = float(checkpoint.get("kl_beta", self.cfg.kl_beta))
        self.best_score = float(checkpoint.get("best_score", -1.0))
        self.baseline_score = checkpoint.get("baseline_score")
        self.difficulty.update(checkpoint.get("difficulty", {}))
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
        print(f"[i] Reprise RLVR v4.5 : {path.name}, update {self.update}")

    def _next_task(self) -> TaskSpec:
        if self.frontier_specs and (not self.bridge_specs or self.rng.random() >= 0.20):
            spec = self.rng.choices(
                self.frontier_specs,
                weights=[row["weight"] for row in self.frontier_specs], k=1,
            )[0]
        elif self.bridge_specs:
            spec = self.rng.choice(self.bridge_specs)
        else:
            names, weights = zip(*CAPABILITY_WEIGHTS.items())
            capability = self.rng.choices(names, weights=weights, k=1)[0]
            spec = {"capability": capability, "schema_id": None,
                    "difficulty": self.difficulty[capability]}
        capability = spec["capability"]
        seed = self.cfg.seed + 1_000_000 + self.rollout_index * 31
        self.rollout_index += 1
        offset = self.difficulty[capability] - 0.25
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
            self.cap_history[task.capability].append(primary_count / self.cfg.group_size)
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
        for index in range(self.cfg.micro_bs):
            if not self.bridge_specs or self.rng.random() >= 0.70:
                question, answer = self.rng.choice(_CURATED_REPLAY)
                prompt = D.render_chat([{"role": "user", "text": question}]) + f"{D.IM_START}assistant\n"
                completion = f"{D.THINK}\n\n{D.THINK_END}\n{answer}{D.IM_END}\n"
            else:
                spec = self.rng.choice(self.bridge_specs)
                capability = spec["capability"]
                task = make_task(self.cfg.seed + 9_000_000 + self.rollout_index + index,
                                 "train", min(0.45, float(spec["difficulty"])), capability,
                                 spec["schema_id"])
                prompt_ids, prefill = self.engine.prompt(task)
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

    def _train_update(self, groups) -> dict:
        batches, constant_denominator = self._make_microbatches(groups)
        self.model.train()
        lr = self.cfg.lr * min(1.0, (self.update + 1) / max(1, self.cfg.warmup))
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
            del x, mask, targets, logits, ref_logits, logp, ref_logp, kl_tokens
        replay_x, replay_y = self._replay_batch()
        replay_x = replay_x.to(self.device, non_blocking=self.use_cuda)
        replay_y = replay_y.to(self.device, non_blocking=self.use_cuda)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda):
            _logits, replay_loss, _ = self.model(
                replay_x, targets=replay_y, diagnostics=False, loss_reduction="mean"
            )
        (self.cfg.replay_weight * replay_loss).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip))
        self.optimizer.step()
        if kl_total > self.cfg.kl_target * 1.5:
            self.kl_beta = min(0.10, self.kl_beta * 1.2)
        elif kl_total < self.cfg.kl_target / 1.5:
            self.kl_beta = max(0.002, self.kl_beta / 1.2)
        return {"pg": pg_total, "kl": kl_total, "replay": float(replay_loss.detach()),
                "grad_norm": grad_norm, "lr": lr, "tokens": token_count}

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
            (self.stage_dir / "eval_baseline.json").write_text(
                json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[i] Baseline dev macro : {self.baseline_score:.3f}")
        print(f"[i] RLVR v4.5 local · cible {self.cfg.accepted_updates} updates acceptées · "
              f"{self.cfg.prompts_per_update}×{self.cfg.group_size} rollouts · lr {self.cfg.lr:.1e}")
        if self.profile is not None:
            frontier = ", ".join(sorted({row["schema_id"] for row in self.frontier_specs}))
            bridge = ", ".join(sorted({row["schema_id"] for row in self.bridge_specs}))
            print(f"[i] Curriculum profilé · 80 % frontière [{frontier}] · "
                  f"20 % exploration bridge [{bridge}]")

        def stop(_signum, _frame):
            self.stop_requested = True
            print("\n[i] Arrêt demandé : sauvegarde après l'update courante.")

        signal.signal(signal.SIGINT, stop)
        stop_file = self.run_dir / "STOP"
        empty_batches = 0
        while self.update < self.cfg.accepted_updates and not self.stop_requested:
            started = time.time()
            groups, rollout = self._rollouts()
            if len(groups) < self.cfg.prompts_per_update:
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
            self.update += 1
            self.tokens_generated += int(rollout["tokens"])
            success_rate = rollout["successes"] / max(1, rollout["samples"])
            record = {
                "update": self.update, "rollout_index": self.rollout_index,
                "success_rate": success_rate, "useful_groups": int(rollout["useful"]),
                "attempts": int(rollout["attempts"]),
                "mean_entropy": rollout["entropy"] / max(1, rollout["samples"]),
                "mean_length": rollout["tokens"] / max(1, rollout["samples"]),
                "tokens_generated": self.tokens_generated, "kl_beta": self.kl_beta,
                "difficulty": dict(self.difficulty), "elapsed_s": time.time() - started,
                **train_stats,
            }
            is_best = False
            if self.update % self.cfg.eval_every == 0 or self.update == self.cfg.accepted_updates:
                evaluation = self._evaluate()
                record["eval"] = evaluation
                safe_kl = train_stats["kl"] <= max(0.05, self.cfg.kl_target * 3)
                if evaluation["macro"] > self.best_score and safe_kl:
                    self.best_score = evaluation["macro"]
                    is_best = True
                (self.stage_dir / f"eval_{self.update:06d}.json").write_text(
                    json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"update {self.update:3d}/{self.cfg.accepted_updates} · "
                  f"ok {100*success_rate:4.1f}% · groupes {len(groups)}/{int(rollout['attempts'])} · "
                  f"KL {train_stats['kl']:.4f} · replay {train_stats['replay']:.3f} · "
                  f"{time.time()-started:.1f}s")
            if is_best or self.update % self.cfg.save_every == 0 or stop_file.exists():
                self._save(best=is_best)
            if stop_file.exists():
                stop_file.unlink(missing_ok=True)
                self.stop_requested = True
        self._save(best=False)
        print(f"[✓] RLVR v4.5 terminé/repris à l'update {self.update} : {self.stage_dir}")


def cmd_rl_v45(args):
    cfg = RLVRConfig(
        run_name=args.run, data_dir=args.data_dir, out_dir=args.out_dir,
        stage_name=args.stage_name, init_stage=args.init_stage, init_ckpt=args.init_ckpt,
        ref_stage=args.ref_stage, ref_ckpt=args.ref_ckpt,
        accepted_updates=args.updates, prompts_per_update=args.prompts,
        group_size=args.group, max_new_tokens=args.max_new, micro_bs=args.micro_bs,
        lr=args.lr, kl_beta=args.kl_beta, kl_target=args.kl_target,
        replay_weight=args.replay_weight, oversample=args.oversample,
        eval_every=args.eval_every, eval_tasks=args.eval_tasks,
        save_every=args.save_every, seed=args.seed, device=args.device,
        require_profile=not args.allow_no_profile,
    )
    RLVRTrainer(cfg, resume=args.resume).train()
