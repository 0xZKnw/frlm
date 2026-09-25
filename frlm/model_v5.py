"""V5 Qwen4-exp dense et lecture des anciens checkpoints Qwen4-exp MoE.

Implémentations Transformers épinglées, poids neufs, contexte borné à 2048.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from frlm.model import QwenLikeLM


PRESETS_V5 = {
    "v5-qwen35-230m": {"arch": "v5-qwen35", "d_model": 768, "n_head": 12, "d_ff": 2816},
    "v5-qwen35-350m": {"arch": "v5-qwen35"},
    "v5-dense-350m": {"arch": "v5-dense"},
    "v5-qwen4exp-350m": {"arch": "v5-qwen4exp"},
}


def full_context_indexer(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """Si tous les blocs tiennent dans le budget, QSA garde le masque causal entier."""
    if attention_mask.shape[-1] > self.token_budget:
        return type(self).forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)
    if past_key_values is not None:
        # Garder les clés pour une éventuelle continuation au-delà du budget QSA.
        keys = self.index_qk_proj(hidden_states)[..., self.index_n_heads * self.index_head_dim:]
        keys = keys.reshape(*hidden_states.shape[:2], -1, self.index_head_dim).squeeze(2)
        past_key_values.update_indexer(keys, self.layer_idx)
    if attention_mask.dtype == torch.bool:
        return attention_mask
    return torch.where(attention_mask == 0, attention_mask.new_zeros(()),
                       torch.finfo(attention_mask.dtype).min)


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


@dataclass
class ModelConfigV5Dense(ModelConfigV5):
    d_model: int = 768
    n_layer: int = 24
    n_head: int = 12
    n_kv_head: int = 3
    d_ff: int = 3840
    linear_heads: int = 12
    linear_key_heads: int = 6
    num_experts: int = 0
    experts_per_token: int = 0
    router_aux_loss_coef: float = 0.0

    def to_dict(self):
        return {"arch": "v5-dense", **asdict(self)}

    def validate(self):
        dimensions = (self.vocab_size, self.d_model, self.n_layer, self.n_head,
                      self.n_kv_head, self.head_dim, self.d_ff, self.linear_heads,
                      self.linear_key_heads, self.ngram_vocab)
        if any(v <= 0 for v in dimensions):
            raise ValueError("dimensions v5 dense strictement positives")
        if not 1 <= self.max_seq_len <= 2048 or self.n_layer < 4:
            raise ValueError("v5 dense : au moins 4 couches, contexte entre 1 et 2048")
        if self.d_model != self.n_head * self.head_dim or self.n_head % self.n_kv_head:
            raise ValueError("v5 dense : dimensions des têtes GQA incompatibles")
        if self.head_dim % 4 or self.d_model % 8:
            raise ValueError("v5 dense : head_dim divisible par 4, d_model par 8")
        if self.linear_heads % self.linear_key_heads:
            raise ValueError("v5 dense : têtes Gated DeltaNet incompatibles")
        if self.num_experts != 0 or self.experts_per_token != 0 or self.router_aux_loss_coef != 0:
            raise ValueError("v5 dense : aucun expert ni routeur")

    def hf_config(self):
        from frlm.qwen4_dense import Qwen4ExpDenseConfig

        self.validate()
        # Réutiliser exactement les réglages GDN/GR4/PLE/QSA du Qwen4-exp existant.
        common = ModelConfigV5(**(asdict(self) | {"num_experts": 1, "experts_per_token": 1}))
        return Qwen4ExpDenseConfig.from_dict(common.hf_config().to_dict() | {
            "model_type": Qwen4ExpDenseConfig.model_type, "intermediate_size": self.d_ff,
            "num_experts": 0, "num_experts_per_tok": 0,
            "moe_intermediate_size": 1, "shared_expert_intermediate_size": 1,
            "router_aux_loss_coef": 0.0, "output_router_logits": False,
        })


@dataclass
class ModelConfigV5Qwen35:
    """Backbone texte Qwen3.5 dense, sans vision ni MTP."""
    vocab_size: int = 32768
    d_model: int = 896
    n_layer: int = 24
    n_head: int = 14
    n_kv_head: int = 2
    head_dim: int = 64
    d_ff: int = 4096
    linear_heads: int = 8
    linear_key_heads: int = 8
    linear_head_dim: int = 64
    max_seq_len: int = 2048
    eos_id: int = 0
    bos_id: int = 0
    tie_embeddings: bool = True

    @property
    def rope_dims(self):
        return self.head_dim // 4

    def to_dict(self):
        return {"arch": "v5-qwen35", **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k != "arch"})

    def validate(self):
        dims = (self.vocab_size, self.d_model, self.n_layer, self.n_head,
                self.n_kv_head, self.head_dim, self.d_ff, self.linear_heads,
                self.linear_key_heads, self.linear_head_dim)
        if any(v <= 0 for v in dims):
            raise ValueError("dimensions Qwen3.5 strictement positives")
        if self.n_layer < 4 or self.n_layer % 4 or not 1 <= self.max_seq_len <= 2048:
            raise ValueError("Qwen3.5 : couches multiples de 4, contexte <= 2048")
        if self.d_model != self.n_head * self.head_dim or self.n_head % self.n_kv_head:
            raise ValueError("Qwen3.5 : dimensions GQA incompatibles")
        if self.d_model % 8 or self.linear_heads % self.linear_key_heads:
            raise ValueError("Qwen3.5 : dimensions GDN/RoPE incompatibles")
        if self.head_dim % 8:
            raise ValueError("Qwen3.5 : MRoPE exige head_dim divisible par 8")

    def hf_config(self):
        from transformers import Qwen3_5TextConfig

        self.validate()
        rope_quarter = self.head_dim // 8
        sections = [rope_quarter // 3] * 3
        sections[0] += rope_quarter - sum(sections)
        return Qwen3_5TextConfig(
            vocab_size=self.vocab_size, hidden_size=self.d_model,
            num_hidden_layers=self.n_layer, num_attention_heads=self.n_head,
            num_key_value_heads=self.n_kv_head, head_dim=self.head_dim,
            intermediate_size=self.d_ff, linear_num_key_heads=self.linear_key_heads,
            linear_num_value_heads=self.linear_heads,
            linear_key_head_dim=self.linear_head_dim,
            linear_value_head_dim=self.linear_head_dim,
            layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                         for i in range(self.n_layer)],
            max_position_embeddings=self.max_seq_len,
            eos_token_id=self.eos_id, bos_token_id=self.bos_id, pad_token_id=self.eos_id,
            tie_word_embeddings=self.tie_embeddings,
            rope_parameters={"rope_type": "default", "rope_theta": 10000000.,
                             "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                             "mrope_section": sections},
        )


class TransformersLM(QwenLikeLM):
    """Interface frlm commune : cibles déjà décalées, masques SFT et cache HF."""
    has_router = False

    def _configure_qwen4(self):
        for module in self.hf.modules():
            if type(module).__name__ == "Qwen4ExpTextRMSNorm":
                module.zero_centered = True
            elif type(module).__name__ == "Qwen4ExpTextQSAIndexer":
                module.forward = MethodType(full_context_indexer, module)

    def num_params(self, non_embedding=False):
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Embedding))
            if not self.cfg.tie_embeddings:
                total -= self.hf.lm_head.weight.numel()
        return total

    def forward(self, idx, targets=None, loss_mask=None, z_loss=0.,
                diagnostics=True, loss_reduction="mean"):
        if idx.shape[1] > self.cfg.max_seq_len:
            raise ValueError("séquence supérieure au contexte v5")
        if loss_reduction not in ("mean", "sum"):
            raise ValueError("loss_reduction doit être mean ou sum")
        router_args = {"output_router_logits": targets is not None} if self.has_router else {}
        out = self.hf(idx, use_cache=False, **router_args)
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
        if self.has_router and self.training and out.aux_loss is not None:
            loss = loss + self.cfg.router_aux_loss_coef * out.aux_loss * count
        if loss_reduction == "mean":
            loss = loss / count.clamp_min(1)
        return logits, loss, {}

    def _alloc_caches(self, batch, max_len, device, dtype):
        from transformers.cache_utils import DynamicCache
        return DynamicCache(config=self.hf.config)

    def _forward_cached(self, idx, caches, pos):
        router_args = {"output_router_logits": False} if self.has_router else {}
        return self.hf(idx, past_key_values=caches, use_cache=True,
                       logits_to_keep=1, **router_args).logits


class DenseLM(TransformersLM):
    def __init__(self, cfg: ModelConfigV5Dense):
        from frlm.qwen4_dense import Qwen4ExpDenseForCausalLM

        nn.Module.__init__(self)
        self.cfg = cfg
        self.hf = Qwen4ExpDenseForCausalLM(cfg.hf_config())
        self._configure_qwen4()

    def flops_per_token(self):
        # Estimation matricielle uniquement, hors scans GDN et produits QK/AV.
        return 6 * (self.num_params(non_embedding=True) + self.hf.lm_head.weight.numel())

    def describe(self):
        return "Qwen4-exp dense · GDN 3:1 · SwiGLU · GR4 · PLE"


class Qwen35LM(TransformersLM):
    def __init__(self, cfg: ModelConfigV5Qwen35):
        from transformers import Qwen3_5ForCausalLM

        nn.Module.__init__(self)
        self.cfg = cfg
        self.hf = Qwen3_5ForCausalLM(cfg.hf_config())
        for module in self.hf.modules():
            if type(module).__name__ == "Qwen3_5RMSNorm":
                module.zero_centered = True

    def flops_per_token(self):
        # Approximation matricielle ; les scans GDN ne sont pas comptés.
        return 6 * (self.num_params(non_embedding=True) + self.hf.lm_head.weight.numel())

    def describe(self):
        return "Qwen3.5 texte dense · GDN 3:1 · SwiGLU · GQA · RMSNorm"


class Qwen4ExpLM(TransformersLM):
    has_router = True

    def __init__(self, cfg: ModelConfigV5):
        from transformers import Qwen4ExpForCausalLM

        nn.Module.__init__(self)
        self.cfg = cfg
        self.hf = Qwen4ExpForCausalLM(cfg.hf_config())
        self._configure_qwen4()

    def flops_per_token(self):
        # Estimation matricielle seulement ; les scans/routages ne sont pas un MFU exact.
        inactive = sum(p.numel() for n, p in self.named_parameters() if ".experts." in n)
        inactive *= 1 - self.cfg.experts_per_token / self.cfg.num_experts
        output = self.hf.lm_head.weight.numel()
        return 6 * (self.num_params(non_embedding=True) - inactive + output)

    def describe(self):
        return "Qwen4-exp · GDN 3:1 · MoE 2/8 + partagé · GR4 · PLE"
