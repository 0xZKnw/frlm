"""DPO local v4.5 sur les paires RLAIF scellées et auditées."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import signal
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from frlm import data as D
from frlm.rl_engine_v45 import clone_reference, load_policy, resolve_checkpoint, resolve_tokenizer
from frlm.rl_v45 import _CURATED_REPLAY, _atomic_link, _atomic_torch_save


@dataclass
class DPOConfig:
    run_name: str = "fr-v4-v45-sft"
    data_dir: str = "data-v4"
    out_dir: str = "runs"
    stage_name: str = "dpo-v45"
    init_stage: str = "rlvr-v45"
    init_ckpt: str = "best"
    ref_stage: str = "rlvr-v45"
    ref_ckpt: str = "best"
    pairs_path: str = ""
    epochs: int = 1
    grad_accum: int = 8
    max_steps: int = 120
    seq_len: int = 512
    lr: float = 5e-7
    beta: float = 0.10
    replay_weight: float = 0.03
    grad_clip: float = 1.0
    eval_every: int = 10
    save_every: int = 10
    seed: int = 455_300
    device: str = "cuda"


def _load_pairs(path: Path, seed: int) -> tuple[list[dict], list[dict]]:
    rows, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("pair_id", "prompt_id", "prompt", "chosen", "rejected")
            if any(not str(row.get(key) or "").strip() for key in required):
                raise ValueError(f"paire DPO incomplète : {row.get('pair_id')}")
            if row["pair_id"] in seen or row["chosen"] == row["rejected"]:
                continue
            seen.add(row["pair_id"])
            rows.append(row)
    if len(rows) < 10:
        raise ValueError("au moins 10 paires distinctes sont requises pour le DPO")
    by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)
    if len(by_prompt) < 5:
        raise ValueError("au moins 5 prompts distincts sont requis pour isoler la validation DPO")
    rng = random.Random(seed)
    prompt_ids = list(by_prompt)
    rng.shuffle(prompt_ids)
    val_prompt_count = max(1, min(len(prompt_ids) // 5, 8))
    val_ids = set(prompt_ids[:val_prompt_count])
    train = [row for row in rows if row["prompt_id"] not in val_ids]
    val = [row for row in rows if row["prompt_id"] in val_ids]
    rng.shuffle(train)
    rng.shuffle(val)
    if {row["prompt_id"] for row in train} & {row["prompt_id"] for row in val}:
        raise AssertionError("fuite de prompt entre train et validation DPO")
    return train, val


class DPOTrainer:
    def __init__(self, cfg: DPOConfig, resume: str | None = None):
        self.cfg = cfg
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible")
        self.device = cfg.device
        self.use_cuda = cfg.device.startswith("cuda")
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.stage_dir = self.run_dir / cfg.stage_name
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        pairs_path = Path(cfg.pairs_path) if cfg.pairs_path else \
            self.run_dir / "rlaif-v45" / "pairs.sealed.jsonl"
        manifest = pairs_path.with_name("pairs.manifest.json")
        if not manifest.is_file():
            raise FileNotFoundError("manifest scellé absent : importe d'abord les jugements")
        expected = json.loads(manifest.read_text(encoding="utf-8")).get("sha256")
        digest = hashlib.sha256()
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest.update(line.encode("utf-8"))
        if expected != digest.hexdigest():
            raise ValueError("le fichier de paires a changé depuis son scellement")
        self.train_pairs, self.val_pairs = _load_pairs(pairs_path, cfg.seed)
        self.tok = D.load_tokenizer(resolve_tokenizer(self.run_dir, Path(cfg.data_dir)))
        init_path = resolve_checkpoint(self.run_dir, cfg.init_stage, cfg.init_ckpt)
        self.model, self.mcfg, self.init_meta = load_policy(init_path, self.device, torch.float32)
        if cfg.init_stage == cfg.ref_stage and cfg.init_ckpt == cfg.ref_ckpt:
            self.ref = clone_reference(self.model, self.mcfg, self.device)
        else:
            ref_path = resolve_checkpoint(self.run_dir, cfg.ref_stage, cfg.ref_ckpt)
            self.ref, ref_cfg, _ = load_policy(ref_path, self.device, torch.float32)
            if ref_cfg.to_dict() != self.mcfg.to_dict():
                raise ValueError("architecture de référence DPO incompatible")
            self.ref.eval()
            for parameter in self.ref.parameters():
                parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
            weight_decay=0.0, foreach=False,
        )
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(cfg.seed)
        self.step = 0
        self.best_val = -1.0
        self.stop_requested = False
        self.order = list(range(len(self.train_pairs)))
        if resume:
            self._resume(resume)
        (self.stage_dir / "config.json").write_text(
            json.dumps({"dpo_v45": asdict(cfg), "model": self.mcfg.to_dict(),
                        "pairs_train": len(self.train_pairs), "pairs_val": len(self.val_pairs),
                        "prompts_train": len({row["prompt_id"] for row in self.train_pairs}),
                        "prompts_val": len({row["prompt_id"] for row in self.val_pairs})},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _encode(self, pair: dict):
        prefix = D.render_chat([{"role": "user", "text": pair["prompt"]}])
        prefix += f"{D.IM_START}assistant\n{D.THINK}\n\n{D.THINK_END}\n"
        prefix_ids = self.tok.encode(prefix).ids
        rows = []
        for key in ("chosen", "rejected"):
            completion = self.tok.encode(str(pair[key]).strip() + D.IM_END).ids
            room = max(1, self.cfg.seq_len - len(prefix_ids))
            completion = completion[:room]
            sequence = prefix_ids + completion
            rows.append((sequence, len(prefix_ids)))
        width = max(len(sequence) for sequence, _ in rows)
        x = torch.zeros((2, width), dtype=torch.long)
        mask = torch.zeros((2, width - 1), dtype=torch.bool)
        for index, (sequence, prefix_len) in enumerate(rows):
            x[index, :len(sequence)] = torch.tensor(sequence)
            mask[index, prefix_len - 1:len(sequence) - 1] = True
        return x, mask

    def _sequence_logps(self, model, x, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda):
            logits, _, _ = model(x, diagnostics=False)
        targets = x[:, 1:]
        vocab = logits.shape[-1]
        logps = -F.cross_entropy(logits[:, :-1].reshape(-1, vocab).float(),
                                 targets.reshape(-1), reduction="none").view_as(targets)
        return (logps * mask).sum(-1), mask.sum(-1).clamp_min(1)

    def _dpo_loss(self, pair: dict, train: bool):
        x_cpu, mask_cpu = self._encode(pair)
        x = x_cpu.to(self.device, non_blocking=self.use_cuda)
        mask = mask_cpu.to(self.device, non_blocking=self.use_cuda)
        policy_logps, lengths = self._sequence_logps(self.model, x, mask)
        with torch.no_grad():
            ref_logps, _ = self._sequence_logps(self.ref, x, mask)
        logits = self.cfg.beta * ((policy_logps[0] - ref_logps[0])
                                  - (policy_logps[1] - ref_logps[1]))
        loss = -F.logsigmoid(logits)
        accuracy = float(logits.detach() > 0)
        margin = float(logits.detach())
        if train:
            (loss / self.cfg.grad_accum).backward()
        return float(loss.detach()), accuracy, margin, [int(v) for v in lengths.tolist()]

    def _replay_loss(self):
        question, answer = self.rng.choice(_CURATED_REPLAY)
        prefix = D.render_chat([{"role": "user", "text": question}])
        prefix += f"{D.IM_START}assistant\n{D.THINK}\n\n{D.THINK_END}\n"
        prefix_ids = self.tok.encode(prefix).ids
        target_ids = self.tok.encode(answer + D.IM_END).ids
        sequence = (prefix_ids + target_ids)[:self.cfg.seq_len]
        x = torch.tensor([sequence[:-1]], device=self.device)
        y = torch.full_like(x, -100)
        y[0, len(prefix_ids) - 1:] = torch.tensor(
            sequence[len(prefix_ids):], device=self.device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda):
            _logits, loss, _ = self.model(x, targets=y, diagnostics=False)
        return loss

    @torch.inference_mode()
    def _evaluate(self):
        self.model.eval()
        losses, accuracies, margins = [], [], []
        for pair in self.val_pairs:
            loss, accuracy, margin, _ = self._dpo_loss(pair, train=False)
            losses.append(loss)
            accuracies.append(accuracy)
            margins.append(margin)
        self.model.train()
        return {"loss": sum(losses) / len(losses),
                "pair_accuracy": sum(accuracies) / len(accuracies),
                "margin": sum(margins) / len(margins)}

    def _payload(self):
        return {"model": self.model.state_dict(), "optimizers": [self.optimizer.state_dict()],
                "model_cfg": self.mcfg.to_dict(), "dpo_cfg": asdict(self.cfg),
                "stage": self.cfg.stage_name, "step": self.step,
                "tokens_seen": self.init_meta.get("tokens_seen", 0),
                "best_val": self.best_val,
                "rng": {"python": self.rng.getstate(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if self.use_cuda else None}}

    def _save(self, best=False):
        numbered = self.stage_dir / f"ckpt_{self.step:06d}.pt"
        _atomic_torch_save(self._payload(), numbered)
        _atomic_link(numbered, self.stage_dir / "ckpt_latest.pt")
        if best:
            _atomic_link(numbered, self.stage_dir / "ckpt_best.pt")
        numbered_files = sorted(self.stage_dir.glob("ckpt_[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
        for old in numbered_files[:-2]:
            old.unlink(missing_ok=True)

    def _resume(self, spec: str):
        path = resolve_checkpoint(self.run_dir, self.cfg.stage_name, spec)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizers"][0])
        self.step = int(checkpoint["step"])
        self.best_val = float(checkpoint.get("best_val", -1.0))
        rng = checkpoint.get("rng", {})
        if rng.get("python") is not None:
            self.rng.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if self.use_cuda and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        del checkpoint
        gc.collect()

    def train(self):
        baseline = self._evaluate()
        self.best_val = max(self.best_val, baseline["pair_accuracy"])
        (self.stage_dir / "eval_baseline.json").write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[i] DPO v4.5 · {len(self.train_pairs)} train / {len(self.val_pairs)} val · "
              f"accuracy initiale {baseline['pair_accuracy']:.1%}")

        def stop(_signum, _frame):
            self.stop_requested = True
        signal.signal(signal.SIGINT, stop)
        max_possible = math.ceil(len(self.train_pairs) * self.cfg.epochs / self.cfg.grad_accum)
        target_steps = min(self.cfg.max_steps, max_possible)
        self.model.train()
        while self.step < target_steps and not self.stop_requested:
            self.optimizer.zero_grad(set_to_none=True)
            losses, accuracies = [], []
            for _ in range(self.cfg.grad_accum):
                index = (self.step * self.cfg.grad_accum + len(losses)) % len(self.train_pairs)
                loss, accuracy, _margin, _lengths = self._dpo_loss(
                    self.train_pairs[self.order[index]], train=True
                )
                losses.append(loss)
                accuracies.append(accuracy)
            replay = self._replay_loss()
            (self.cfg.replay_weight * replay).backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                             self.cfg.grad_clip))
            self.optimizer.step()
            self.step += 1
            record = {"step": self.step, "loss": sum(losses) / len(losses),
                      "pair_accuracy": sum(accuracies) / len(accuracies),
                      "replay_loss": float(replay.detach()), "grad_norm": grad_norm}
            best = False
            if self.step % self.cfg.eval_every == 0 or self.step == target_steps:
                evaluation = self._evaluate()
                record["eval"] = evaluation
                if evaluation["pair_accuracy"] > self.best_val:
                    self.best_val = evaluation["pair_accuracy"]
                    best = True
            with (self.stage_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"step {self.step:3d}/{target_steps} · loss {record['loss']:.3f} · "
                  f"acc {record['pair_accuracy']:.1%} · replay {record['replay_loss']:.3f}")
            if best or self.step % self.cfg.save_every == 0:
                self._save(best)
        self._save(False)
        print(f"[✓] DPO v4.5 terminé : {self.stage_dir}")


def cmd_dpo(args):
    cfg = DPOConfig(
        run_name=args.run, data_dir=args.data_dir, out_dir=args.out_dir,
        stage_name=args.stage_name, init_stage=args.init_stage, init_ckpt=args.init_ckpt,
        ref_stage=args.ref_stage, ref_ckpt=args.ref_ckpt, pairs_path=args.pairs,
        epochs=args.epochs, grad_accum=args.grad_accum, max_steps=args.max_steps,
        seq_len=args.seq_len, lr=args.lr, beta=args.beta,
        replay_weight=args.replay_weight, eval_every=args.eval_every,
        save_every=args.save_every, seed=args.seed, device=args.device,
    )
    DPOTrainer(cfg, resume=args.resume).train()
