"""Briques communes au profilage et au RLVR local de FRLM v4.5."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from frlm import config_from_dict, model_from_cfg
from frlm import data as D
from frlm.rl_tasks_v45 import TaskSpec


@dataclass
class Sample:
    token_ids: list[int]
    text: str
    stopped: bool
    entropy: float


def resolve_checkpoint(run_dir: Path, stage: str, spec: str) -> Path:
    explicit = Path(spec)
    if explicit.is_file():
        return explicit
    stage_dir = run_dir / stage
    names = {
        "best": ("ckpt_best.pt", "ckpt_latest.pt"),
        "latest": ("ckpt_latest.pt", "ckpt_best.pt"),
    }.get(spec, (spec,))
    for name in names:
        path = stage_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"checkpoint introuvable : {stage_dir} / {spec}")


def resolve_tokenizer(run_dir: Path, data_dir: Path) -> Path:
    for path in (run_dir / "tokenizer.json", data_dir / "tokenizer.json"):
        if path.is_file():
            return path
    raise FileNotFoundError("tokenizer.json absent du run et de --data-dir")


def load_policy(path: Path, device: str = "cuda", dtype: torch.dtype = torch.float32):
    """Charge seulement les poids utiles, sans restaurer l'optimiseur SFT massif."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(checkpoint["model_cfg"])
    model = model_from_cfg(cfg)
    model.load_state_dict(checkpoint["model"])
    metadata = {
        "path": str(path), "step": int(checkpoint.get("step", 0)),
        "stage": checkpoint.get("stage"), "val_loss": checkpoint.get("val_loss"),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
    }
    del checkpoint
    gc.collect()
    model.to(device=device, dtype=dtype)
    return model, cfg, metadata


def clone_reference(policy, cfg, device: str):
    """Crée l'ancre gelée fp32 après que le checkpoint CPU a été libéré.

    Une ancre bf16 donnait une KL initiale artificielle proche de 1 sur model_v3
    (contre < 1e-3 en fp32), malgré des poids source identiques. La précision de
    l'ancre est donc un invariant de correction, pas une option de performance.
    """
    state = {name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()}
    ref = model_from_cfg(cfg)
    ref.load_state_dict(state)
    del state
    gc.collect()
    ref.to(device=device, dtype=torch.float32)
    ref.eval()
    for parameter in ref.parameters():
        parameter.requires_grad_(False)
    return ref


class RolloutEngine:
    def __init__(self, model, tokenizer, device: str, max_new_tokens: int = 112):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.use_cuda = device.startswith("cuda")
        self.max_new_tokens = max_new_tokens
        self.sp = D.special_ids(tokenizer)
        self.think = tokenizer.token_to_id(D.THINK)
        self.think_end = tokenizer.token_to_id(D.THINK_END)

    def prompt(self, task: TaskSpec) -> tuple[list[int], str]:
        base = D.render_chat([{"role": "user", "text": task.prompt}])
        if task.requires_trace:
            prefill = f"{D.THINK}\n"
        else:
            prefill = f"{D.THINK}\n\n{D.THINK_END}\n"
        text = base + f"{D.IM_START}assistant\n" + prefill
        ids = self.tok.encode(text).ids
        room = self.model.cfg.max_seq_len - self.max_new_tokens
        if len(ids) > room:
            raise ValueError(f"prompt trop long ({len(ids)} tokens, plafond {room})")
        return ids, prefill

    @torch.inference_mode()
    def sample(self, task: TaskSpec, group_size: int, temperature: float = 1.0,
               top_p: float = 1.0) -> tuple[list[int], list[Sample]]:
        if not math.isclose(temperature, 1.0) or not math.isclose(top_p, 1.0):
            raise ValueError("RLVR v4.5 est strictement on-policy : T=1 et top_p=1")
        prompt_ids, prefill = self.prompt(task)
        prefix = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prefix = prefix.repeat(group_size, 1)
        max_len = min(self.model.cfg.max_seq_len, len(prompt_ids) + self.max_new_tokens)
        cache_dtype = torch.bfloat16 if self.use_cuda else next(self.model.parameters()).dtype
        caches = self.model._alloc_caches(group_size, max_len, self.device, cache_dtype)
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda)
        with amp:
            logits = self.model._forward_cached(prefix, caches, 0)
        outputs: list[list[int]] = [[] for _ in range(group_size)]
        entropy_sum = [0.0] * group_size
        active_steps = [0] * group_size
        finished = [False] * group_size
        stop_ids = (self.sp["im_end"], self.sp["eot"])
        suppress = [self.think]
        if not task.requires_trace:
            suppress.append(self.think_end)
        position = len(prompt_ids)
        for _ in range(self.max_new_tokens):
            scores = logits[:, -1, :].float()
            scores[:, suppress] = float("-inf")
            probs = torch.softmax(scores, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
            next_ids = torch.multinomial(probs, num_samples=1)
            for index, token in enumerate(next_ids.squeeze(1).tolist()):
                if finished[index]:
                    next_ids[index, 0] = self.sp["eot"]
                    continue
                outputs[index].append(token)
                entropy_sum[index] += float(entropy[index])
                active_steps[index] += 1
                if token in stop_ids:
                    finished[index] = True
            if all(finished) or position + 1 >= max_len:
                break
            with amp:
                logits = self.model._forward_cached(next_ids, caches, position)
            position += 1
        samples = []
        for index, ids in enumerate(outputs):
            decoded = self.tok.decode(ids, skip_special_tokens=False)
            samples.append(Sample(
                token_ids=ids,
                text=prefill + decoded,
                stopped=bool(ids and ids[-1] in stop_ids),
                entropy=entropy_sum[index] / max(1, active_steps[index]),
            ))
        del caches, logits
        return prompt_ids, samples

    @torch.inference_mode()
    def greedy(self, task: TaskSpec) -> Sample:
        prompt_ids, prefill = self.prompt(task)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        suppress = (self.think,) if task.requires_trace else (self.think, self.think_end)
        amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_cuda)
        with amp:
            out = self.model.generate(
                ids, max_new_tokens=self.max_new_tokens, temperature=0.0,
                top_k=0, top_p=1.0, repetition_penalty=1.0,
                stop_ids=(self.sp["im_end"], self.sp["eot"]), suppress_ids=suppress,
            )
        generated = out[0, ids.shape[1]:].tolist()
        return Sample(generated, prefill + self.tok.decode(generated, False),
                      bool(generated and generated[-1] in (self.sp["im_end"], self.sp["eot"])),
                      0.0)
