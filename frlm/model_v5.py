"""Qwen4-exp réduit à 350M : implémentation Transformers épinglée, poids neufs.

Le contexte reste borné à 2048 : le budget QSA couvre toute la fenêtre. Aucun
gain de sparsité ni entraînement de l'indexeur long contexte n'est revendiqué.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from frlm.model import QwenLikeLM


PRESETS_V5 = {"v5-qwen4exp-350m": {}}


def validate_resume(ck, cfg, model_cfg, data_hash, tokenizer_hash, sft_hash,
                    weights_only=False):
    """Même contrat au préflight CPU et à la reprise effective."""
    if ck.get("model_cfg") != model_cfg:
        raise ValueError("configuration v5 différente du checkpoint")
    if ck.get("data_manifest_sha256") != data_hash or ck.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("données/tokenizer v5 différents du checkpoint")
    if ck.get("stage") != cfg.stage:
        return
    if weights_only:
        raise ValueError("reprise v5 : état complet obligatoire dans la même phase")
    if ck.get("sft_manifest_sha256") != sft_hash:
        raise ValueError("manifest SFT différent du checkpoint")
    stable = ("batch_size", "grad_accum", "seq_len", "optimizer", "lr", "adam_lr",
              "beta1", "beta2", "weight_decay", "grad_clip", "schedule", "warmup",
              "min_lr_frac", "decay_frac", "z_loss", "max_steps", "seed", "dtype",
              "replay_frac", "replay_mix", "replay_val", "sft_recipe")
    changed = [k for k in stable if ck.get("train_cfg", {}).get(k) != getattr(cfg, k)]
    if changed or not ck.get("optimizers") or not {"torch", "cuda", "numpy"} <= ck.get("rng", {}).keys():
        raise ValueError(f"reprise v5 non exacte : réglages modifiés {changed} ou état absent")


@dataclass
class ModelConfigV5:
    vocab_size: int = 32768
    d_model: int = 512
    n_layer: int = 20
    n_head: int = 8
    n_kv_head: int = 2
    head_dim: int = 64
    d_ff: int = 1024
    max_seq_len: int = 2048
    eos_id: int = 0
    bos_id: int = 0
    tie_embeddings: bool = True
    num_experts: int = 8
    experts_per_token: int = 2
    linear_heads: int = 8
    linear_key_heads: int = 4
    ngram_vocab: int = 32768
    router_aux_loss_coef: float = 0.001

    @property
    def rope_dims(self):
        return self.head_dim // 2

    def to_dict(self):
        return {"arch": "v5-qwen4exp", **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k != "arch"})

    def hf_config(self):
        from transformers import Qwen4ExpTextConfig

        if not 1 <= self.max_seq_len <= 2048 or self.n_layer < 4:
            raise ValueError("Qwen4-exp v5 exige au moins 4 couches et un contexte <= 2048")
        if self.head_dim % 4 or self.d_model % 8:
            raise ValueError("head_dim doit être divisible par 4, d_model par 8")
        sections = [self.rope_dims // 6] * 3
        sections[0] += self.rope_dims // 2 - sum(sections)
        return Qwen4ExpTextConfig(
            vocab_size=self.vocab_size, hidden_size=self.d_model,
            num_hidden_layers=self.n_layer, num_attention_heads=self.n_head,
            num_key_value_heads=self.n_kv_head, head_dim=self.head_dim,
            linear_num_key_heads=self.linear_key_heads,
            linear_num_value_heads=self.linear_heads,
            linear_key_head_dim=self.head_dim, linear_value_head_dim=self.head_dim,
            num_experts=self.num_experts, num_experts_per_tok=self.experts_per_token,
            moe_intermediate_size=self.d_ff, shared_expert_intermediate_size=self.d_ff,
            hc_count=4, hc_lowrank=self.d_model // 8,
            ple_layer_ids=[2], ple_embed_dim=self.d_model,
            ngram_vocab_size_base=self.ngram_vocab, ngram_size=3, heads_per_ngram=4,
            make_ngram_vocab_size_divisible_by=128, split_ngram_parts=1,
            eos_token_id=self.eos_id, bos_token_id=self.bos_id, pad_token_id=self.eos_id,
            tie_word_embeddings=self.tie_embeddings,
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=self.head_dim // 2,
            indexer_budget=2048, indexer_compress_ratio=4,
            max_position_embeddings=self.max_seq_len,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "partial_rotary_factor": 0.5, "mrope_section": sections},
            output_gate_type="sigmoid", router_aux_loss_coef=self.router_aux_loss_coef,
        )


class Qwen4ExpLM(QwenLikeLM):
    def __init__(self, cfg: ModelConfigV5):
        from transformers import Qwen4ExpForCausalLM

        nn.Module.__init__(self)
        self.cfg = cfg
        self.hf = Qwen4ExpForCausalLM(cfg.hf_config())
        for module in self.hf.modules():
            if type(module).__name__ == "Qwen4ExpTextRMSNorm":
                module.zero_centered = True

    def num_params(self, non_embedding=False):
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Embedding))
            if not self.cfg.tie_embeddings:
                total -= self.hf.lm_head.weight.numel()
        return total

    def flops_per_token(self):
        # Estimation matricielle seulement ; les scans/routages ne sont pas un MFU exact.
        inactive = sum(p.numel() for n, p in self.named_parameters() if ".experts." in n)
        inactive *= 1 - self.cfg.experts_per_token / self.cfg.num_experts
        output = self.hf.lm_head.weight.numel()
        return 6 * (self.num_params(non_embedding=True) - inactive + output)

    def describe(self):
        return "Qwen4-exp · GDN 3:1 · MoE 2/8 + partagé · GR4 · PLE"

    def forward(self, idx, targets=None, loss_mask=None, z_loss=0.,
                diagnostics=True, loss_reduction="mean"):
        if idx.shape[1] > self.cfg.max_seq_len:
            raise ValueError("séquence supérieure au contexte v5")
        if loss_reduction not in ("mean", "sum"):
            raise ValueError("loss_reduction doit être mean ou sum")
        out = self.hf(idx, use_cache=False, output_router_logits=targets is not None)
        logits = out.logits
        if targets is None:
            return logits, None, {}
        # Les cibles frlm sont déjà décalées d'un token, contrairement aux labels HF.
        valid = targets.ne(-100)
        if loss_mask is not None:
            valid = valid & loss_mask.bool()
        losses = F.cross_entropy(logits.float().flatten(0, 1),
                                 targets.masked_fill(~valid, -100).flatten(),
                                 reduction="none").view_as(targets)
        if z_loss:
            losses = losses + z_loss * logits.float().logsumexp(-1).square() * valid
        count = valid.sum()
        loss = losses.sum()
        if self.training and out.aux_loss is not None:
            loss = loss + self.cfg.router_aux_loss_coef * out.aux_loss * count
        if loss_reduction == "mean":
            loss = loss / count.clamp_min(1)
        return logits, loss, {}

    def _alloc_caches(self, batch, max_len, device, dtype):
        from transformers.cache_utils import DynamicCache
        return DynamicCache(config=self.hf.config)

    def _forward_cached(self, idx, caches, pos):
        return self.hf(idx, past_key_values=caches, use_cache=True,
                       logits_to_keep=1, output_router_logits=False).logits
