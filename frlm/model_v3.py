"""
model_v3.py — Architecture "speedrun" : le stack modded-nanoGPT + Canon layers,
sans contrainte de compatibilité llama.cpp. Objectif : intelligence maximale par
GPU-heure sur la RTX 4060, pour un modèle de RAISONNEMENT français.

Ce qu'on garde de v2 (model.py) : RMSNorm zéro-centrée, QK-Norm, RoPE partiel,
GQA, Muon côté optim. Ce qu'on ajoute, chaque brique validée empiriquement :

  Canon layers (PhysicsLM 4.1, arXiv:2512.17351)
      Une conv causale depthwise de noyau 4 ajoutée en résiduel AVANT l'attention
      et AVANT le MLP de chaque bloc : x = x + conv(x). Trois nombres par canal,
      coût FLOP quasi nul, et le papier mesure ~2x de profondeur de raisonnement.
      Init à zéro -> au départ le réseau est exactement le transformer de base.

  Value embeddings (speedrun modded-nanoGPT)
      Des tables d'embedding supplémentaires injectées directement dans le V de
      l'attention (v = v + lambda * ve[token]) sur les 3 premières et 3 dernières
      couches. De la capacité gratuite : des paramètres en plus, zéro FLOP de
      calcul en plus (un lookup).

  Skips U-net (speedrun)
      La couche L-1-i reçoit en résiduel la sortie de la couche i (scalaire
      appris). Les gradients traversent le réseau par des raccourcis -> converge
      plus vite en profondeur.

  ReLU² dans le MLP (speedrun ; Primer, arXiv:2109.08668)
      MLP à 2 matmuls (au lieu de 3 pour SwiGLU) avec activation relu(x)².
      À paramètres égaux, matmuls plus gros -> meilleur MFU, et les ablations
      speedrun le donnent gagnant à cette échelle.

  Attention à fenêtre glissante 3:1 (speedrun ; Gemma)
      3 couches sur 4 ne regardent que les 512 derniers tokens, la 4e est
      globale. À ctx 1024 ça réduit le coût attention, et le local est déjà
      géré par les canon layers.

  Logit softcap (Gemma 2)
      logits = 15 * tanh(logits / 15) : borne la tête de sortie, stabilise le
      bf16 et remplace en douceur ce que z-loss corrige brutalement.

  Embeddings NON liés + tête init à zéro (speedrun)
      La tête démarre à zéro (loss = ln(V) exactement, gradients propres) et
      apprend séparément de l'embedding d'entrée. +10M params, gain mesuré.

L'interface est IDENTIQUE à QwenLikeLM (forward/targets/loss_mask/z_loss,
_alloc_caches, _forward_cached, generate hérité) : run.py, chat, bench et rl.py
fonctionnent sans modification autre que le routage v2/v3 au chargement.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from frlm.model import (RMSNorm, QwenLikeLM, apply_partial_rope,
                        build_rope_cache, repeat_kv)

# FlexAttention pour les couches fenêtrées : un masque dense attn_mask prive SDPA
# de flash-attention et matérialise les matrices T×T (mesuré : ~39 Go à bs32
# seq2048 sur v4-base, MFU 18-21% sur A100/H100). FlexAttention calcule la même
# chose par blocs, sans jamais matérialiser. Repli : FRLM_ATTN=sdpa (env) ou
# torch trop vieux -> ancien chemin masqué, numériquement équivalent.
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX = True
except Exception:                                   # torch < 2.5
    _HAS_FLEX = False
_USE_FLEX = _HAS_FLEX and os.environ.get("FRLM_ATTN", "flex").lower() == "flex"

if _USE_FLEX:
    # pytorch#148827 : l'optimisation de layout d'inductor (pensée pour les convs)
    # permute q/k/v et fait échouer la compilation de flex_attention ("Query must
    # be contiguous"). On la coupe : seules nos canon (depthwise k=4) la perdraient,
    # pour un coût nul mesuré.
    try:
        import torch._inductor.config as _icfg
        _icfg.layout_optimization = False
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfigV3:
    vocab_size: int = 16384
    n_layer: int = 12
    n_head: int = 10
    n_kv_head: int = 2
    d_model: int = 640
    head_dim: int = 64
    d_ff: int = 2560               # MLP ReLU² : 2 matmuls d->ff->d (pas de gate)
    max_seq_len: int = 1024
    rope_theta: float = 100_000.0
    rms_eps: float = 1e-6
    dropout: float = 0.0

    # --- briques speedrun ---
    rope_frac: float = 0.5         # RoPE partiel (canon fournit déjà l'ordre local)
    window: int = 512              # fenêtre des couches locales
    global_every: int = 4          # 1 couche globale toutes les N (les autres = fenêtre)
    canon_kernel: int = 4          # noyau des canon layers (0 = désactivé)
    n_value_embeds: int = 3        # couches du début ET de la fin qui reçoivent un VE
    unet_skips: bool = True
    softcap: float = 15.0
    zero_centered: bool = True

    # ids de tokens spéciaux (remplis par le tokenizer)
    bos_id: int = 0
    eos_id: int = 0
    pad_id: int = 0

    @property
    def rope_dims(self) -> int:
        r = int(self.head_dim * self.rope_frac)
        return max(2, r - r % 2)

    def layer_is_global(self, i: int) -> bool:
        # la couche globale ferme chaque groupe (elle relit ce que les locales ont posé)
        return i % self.global_every == self.global_every - 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["arch"] = "v3"           # routage au chargement des checkpoints
        return d

    @staticmethod
    def from_dict(d: dict) -> "ModelConfigV3":
        known = set(ModelConfigV3.__dataclass_fields__)
        return ModelConfigV3(**{k: v for k, v in d.items() if k in known})


# Presets calibrés 8 Go (les params "gratuits" — VE, tête déliée — gonflent le
# compte total : regarder surtout le "hors embeddings" affiché au lancement).
PRESETS_V3: dict[str, dict] = {
    # ~20M hors emb. : débogage rapide du pipeline
    "v3-nano": dict(n_layer=6, n_head=6, n_kv_head=2, d_model=384, head_dim=64,
                    d_ff=1536, max_seq_len=1024),
    # ~28M hors emb. : le plus de tokens par heure (~1,3B en 8h)
    "v3-mini": dict(n_layer=10, n_head=8, n_kv_head=2, d_model=512, head_dim=64,
                    d_ff=2048, max_seq_len=1024),
    # ~42M hors emb. : le compromis cerveau/VRAM taillé pour 8 Go (bench 2026-08-20)
    "v3-mid": dict(n_layer=12, n_head=9, n_kv_head=3, d_model=576, head_dim=64,
                   d_ff=2304, max_seq_len=1024),
    # ~53M hors emb. : le plus gros cerveau — nécessite --batch-size 12 (VRAM)
    "v3-base": dict(n_layer=12, n_head=10, n_kv_head=2, d_model=640, head_dim=64,
                    d_ff=2560, max_seq_len=1024),
    # ~159M hors emb. (~205M total) : cible v4 sur A100 Modal — profond (canon
    # double la profondeur effective), MLP aminci à 3,5× pour payer les 16 couches
    "v4-base": dict(n_layer=16, n_head=16, n_kv_head=4, d_model=1024, head_dim=64,
                    d_ff=3584, max_seq_len=2048),
    # ~154M hors emb. (~200M total) : variante 14 couches / MLP 4× — à départager
    # au bench_speed (même budget, profondeur vs largeur MLP)
    "v4-alt": dict(n_layer=14, n_head=16, n_kv_head=4, d_model=1024, head_dim=64,
                   d_ff=4096, max_seq_len=2048),
}


# --------------------------------------------------------------------------------------
# Canon layer : conv causale depthwise en résiduel, init zéro
# --------------------------------------------------------------------------------------
class Canon(nn.Module):
    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)
        nn.init.zeros_(self.conv.weight)       # départ = identité (x + 0)

    def forward(self, x: torch.Tensor, cache: dict | None = None, key: str = "") -> torch.Tensor:
        xt = x.transpose(1, 2)                          # (B, d, T)
        if cache is not None:
            padded = torch.cat([cache[key].to(xt.dtype), xt], dim=2)
            cache[key] = padded[:, :, -(self.kernel - 1):]
        else:
            padded = F.pad(xt, (self.kernel - 1, 0))    # causal
        return x + self.conv(padded).transpose(1, 2)


# --------------------------------------------------------------------------------------
# Attention : GQA + QK-Norm + RoPE partiel + fenêtre glissante + value embeddings
# --------------------------------------------------------------------------------------
class WindowedAttention(nn.Module):
    def __init__(self, cfg: ModelConfigV3, layer_idx: int):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.rope_dims = cfg.rope_dims
        self.window = None if cfg.layer_is_global(layer_idx) else cfg.window
        assert cfg.n_head % cfg.n_kv_head == 0

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps, cfg.zero_centered)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps, cfg.zero_centered)
        self.dropout_p = cfg.dropout

    def alloc_cache(self, batch: int, max_len: int, device, dtype) -> dict:
        shape = (batch, self.n_kv_head, max_len, self.head_dim)
        return {"kind": "attn",
                "k": torch.zeros(shape, device=device, dtype=dtype),
                "v": torch.zeros(shape, device=device, dtype=dtype)}

    def forward(self, x, cos, sin, cache=None, pos: int = 0,
                mask=None, vembed=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        if vembed is not None:                      # value embedding : capacité gratuite
            v = v + vembed.view(B, T, self.n_kv_head, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_partial_rope(q, k, cos, sin, self.rope_dims)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if cache is not None:
            cache["k"][:, :, pos:pos + T] = k
            cache["v"][:, :, pos:pos + T] = v
            fin = pos + T
            # fenêtre coulissante appliquée en tranchant le cache SEULEMENT en
            # génération token par token ; au préremplissage (T > 1, pos = 0) on
            # garde tout le cache et on réutilise le masque d'entraînement, sinon
            # les requêtes anciennes verraient le futur (le crop désaligne is_causal)
            deb = max(0, fin - self.window) if (self.window is not None and T == 1) else 0
            k = cache["k"][:, :, deb:fin]
            v = cache["v"][:, :, deb:fin]

        drop = self.dropout_p if self.training else 0.0
        if self.window is not None and T > 1:
            if mask is not None and not isinstance(mask, torch.Tensor):
                # BlockMask FlexAttention : fenêtre par blocs, rien de matérialisé,
                # GQA natif (pas de repeat_kv). Choisi par _backbone. Sous autocast
                # le value embedding (lookup fp32) contamine v : flex exige un dtype
                # unique là où SDPA convertissait en silence.
                # .contiguous() : inductor exige la dernière dim contiguë après nos
                # transposes (coût mineur, sans commune mesure avec le gain flash)
                out = flex_attention(q.contiguous(), k.to(q.dtype).contiguous(),
                                     v.to(q.dtype).contiguous(),
                                     block_mask=mask, enable_gqa=True)
            else:
                # couche locale, entraînement ou préremplissage : masque causal fenêtré
                out = F.scaled_dot_product_attention(
                    q, repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep),
                    attn_mask=mask[:T, :T], dropout_p=drop)
        else:
            causal = q.shape[2] == k.shape[2] and q.shape[2] > 1
            out = F.scaled_dot_product_attention(
                q, repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep),
                is_causal=causal, dropout_p=drop)
        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.o_proj(out)


class ReluSquaredMLP(nn.Module):
    def __init__(self, cfg: ModelConfigV3):
        super().__init__()
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)).square())


class BlockV3(nn.Module):
    def __init__(self, cfg: ModelConfigV3, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps, cfg.zero_centered)
        self.attn = WindowedAttention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps, cfg.zero_centered)
        self.mlp = ReluSquaredMLP(cfg)
        k = cfg.canon_kernel
        self.canon_a = Canon(cfg.d_model, k) if k else None   # avant attention
        self.canon_c = Canon(cfg.d_model, k) if k else None   # avant MLP

    def forward(self, x, cos, sin, cache=None, pos: int = 0, mask=None, vembed=None):
        if self.canon_a is not None:
            x = self.canon_a(x, cache, "ca")
        x = x + self.attn(self.input_layernorm(x), cos, sin, cache, pos, mask, vembed)
        if self.canon_c is not None:
            x = self.canon_c(x, cache, "cc")
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# --------------------------------------------------------------------------------------
# Modèle complet — hérite de QwenLikeLM pour generate() (même boucle d'échantillonnage)
# --------------------------------------------------------------------------------------
class SpeedLM(QwenLikeLM):
    def __init__(self, cfg: ModelConfigV3):        # noqa : on N'appelle PAS super().__init__
        nn.Module.__init__(self)
        self.cfg = cfg
        L = cfg.n_layer
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(BlockV3(cfg, i) for i in range(L))
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps, cfg.zero_centered)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # value embeddings : n premières couches + n dernières, avec PARTAGE MIROIR
        # (modded-nanoGPT) : la couche j et la couche L-1-j utilisent la même table,
        # chacune avec son propre lambda -> moitié moins de tables pour le même effet.
        n_ve = min(cfg.n_value_embeds, L // 2)
        self.ve_map = {}                            # couche -> (table, lambda)
        for j in range(n_ve):
            self.ve_map[j] = (j, j)
            self.ve_map[L - 1 - j] = (j, n_ve + j)
        kv_dim = cfg.n_kv_head * cfg.head_dim
        self.value_embeds = nn.ModuleList(nn.Embedding(cfg.vocab_size, kv_dim)
                                          for _ in range(n_ve))
        self.ve_lambdas = nn.Parameter(torch.full((2 * n_ve,), 0.5))

        # skips U-net : la couche L-1-i reçoit la sortie de la couche i
        n_skip = L // 2 if cfg.unet_skips else 0
        self.skip_lambdas = nn.Parameter(torch.ones(n_skip)) if n_skip else None

        cos, sin = build_rope_cache(cfg.rope_dims, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # masque causal fenêtré des couches locales (bool, 1 Mo à ctx 1024)
        idx = torch.arange(cfg.max_seq_len)
        win = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] < cfg.window)
        self.register_buffer("win_mask", win, persistent=False)
        # BlockMasks FlexAttention, construits à la demande par (T, device) —
        # simple attribut (pas un buffer) : absent du state_dict, ckpts inchangés
        self._flex_masks: dict = {}

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * L))
        nn.init.zeros_(self.lm_head.weight)        # loss initiale = ln(V) pile
        for canon in self.modules():
            if isinstance(canon, Canon):
                nn.init.zeros_(canon.conv.weight)  # re-zéro (après apply)
        self.ve_lambdas.data.fill_(0.5)
        if self.skip_lambdas is not None:
            self.skip_lambdas.data.fill_(1.0)

    # ---- stats -------------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel() + self.lm_head.weight.numel()
            n -= sum(ve.weight.numel() for ve in self.value_embeds)
        return n

    def flops_per_token(self) -> float:
        c = self.cfg
        base = 6 * self.num_params(non_embedding=True)
        attn = 0
        for i in range(c.n_layer):
            span = c.max_seq_len if c.layer_is_global(i) else min(c.window, c.max_seq_len)
            attn += 12 * c.n_head * c.head_dim * span
        return base + attn

    def describe(self) -> str:
        c = self.cfg
        pattern = "".join("G" if c.layer_is_global(i) else "s" for i in range(c.n_layer))
        extras = []
        if c.canon_kernel:
            extras.append("canon")
        if self.value_embeds:
            extras.append(f"ve{len(self.value_embeds)}×2")
        if self.skip_lambdas is not None:
            extras.append("unet")
        return pattern + " +" + "+".join(extras)

    # ---- passes ------------------------------------------------------------------
    def _flex_mask(self, T: int, device):
        cle = (T, str(device))
        bm = self._flex_masks.get(cle)
        if bm is None:
            w = self.cfg.window

            def fenetre(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (q_idx - kv_idx < w)

            # hors inference_mode : le masque est mis en cache et resservi aux passes
            # avec gradient (RL : rollouts inference_mode puis backward sur le même T),
            # or autograd ne peut pas sauvegarder un inference tensor pour le backward
            with torch.inference_mode(False):
                bm = create_block_mask(fenetre, None, None, T, T, device=device)
            self._flex_masks[cle] = bm
        return bm

    def _backbone(self, idx, caches=None, pos: int = 0):
        B, T = idx.shape
        x = F.rms_norm(self.embed_tokens(idx), (self.cfg.d_model,))   # embeddings normés
        cos = self.rope_cos[pos:pos + T]
        sin = self.rope_sin[pos:pos + T]
        mask = self.win_mask          # utilisé dès que T > 1 (train ET préremplissage)
        # flex ne gère pas le dropout d'attention (0.0 partout chez nous) ni T = 1
        if _USE_FLEX and T > 1 and idx.is_cuda and self.cfg.dropout == 0.0:
            mask = self._flex_mask(T, idx.device)

        L = self.cfg.n_layer
        n_skip = 0 if self.skip_lambdas is None else self.skip_lambdas.numel()
        skips = []
        for i, layer in enumerate(self.layers):
            if n_skip and i >= L - n_skip:
                x = x + self.skip_lambdas[L - 1 - i] * skips[L - 1 - i]
            ve = None
            if i in self.ve_map:
                t, lam = self.ve_map[i]
                ve = self.ve_lambdas[lam] * self.value_embeds[t](idx)
            cache = caches[i] if caches is not None else None
            x = layer(x, cos, sin, cache, pos, mask, ve)
            if n_skip and i < n_skip:
                skips.append(x)
        return self.norm(x)

    def _logits(self, x):
        logits = self.lm_head(x)
        cap = self.cfg.softcap
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits

    def forward(self, idx, targets=None, loss_mask=None, z_loss: float = 0.0,
                diagnostics: bool = True):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"séquence {T} > max_seq_len {self.cfg.max_seq_len}"
        x = self._backbone(idx)

        if targets is None:
            return self._logits(x), None, {}

        logits = self._logits(x)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        if loss_mask is not None:
            flat_targets = flat_targets.masked_fill(loss_mask.reshape(-1) == 0, -100)
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
        if z_loss > 0:      # avec le softcap, quasi inactif — gardé par cohérence
            lse = torch.logsumexp(flat_logits, dim=-1).float()
            loss = loss + z_loss * lse.pow(2).mean()

        if not diagnostics:
            return logits, loss, {}
        with torch.no_grad():
            n = flat_logits.shape[0]
            kk = min(n, 2048)
            sel = torch.arange(0, n, max(1, n // kk), device=flat_logits.device)[:kk]
            sl = flat_logits[sel].float()
            st = flat_targets[sel]
            keep = st != -100
            if not bool(keep.any()):
                keep = torch.ones_like(st, dtype=torch.bool)
                st = st.clamp(min=0)
            sl, st = sl[keep], st[keep]
            correct = (sl.argmax(-1) == st).float()
            logp = F.log_softmax(sl, dim=-1)
            entropy = -(logp.exp() * logp).sum(-1)
            stats = {"acc_top1": correct.mean().detach(),
                     "entropy": entropy.mean().detach(),
                     "logit_rms": sl.pow(2).mean().sqrt().detach()}
        return logits, loss, stats

    # ---- génération (generate() est hérité de QwenLikeLM) -------------------------
    def _alloc_caches(self, batch: int, max_len: int, device, dtype):
        caches = []
        k = self.cfg.canon_kernel
        for layer in self.layers:
            c = layer.attn.alloc_cache(batch, max_len, device, dtype)
            if k:
                c["ca"] = torch.zeros(batch, self.cfg.d_model, k - 1, device=device, dtype=dtype)
                c["cc"] = torch.zeros(batch, self.cfg.d_model, k - 1, device=device, dtype=dtype)
            caches.append(c)
        return caches

    def _forward_cached(self, idx, caches, pos: int):
        x = self._backbone(idx, caches, pos)
        return self._logits(x[:, -1:])


def build_model_v3(cfg: ModelConfigV3) -> SpeedLM:
    return SpeedLM(cfg)
