"""
model.py — Architecture type Qwen3.5 (2026) implémentée from scratch en PyTorch.

Qwen3.5 est une architecture HYBRIDE : les couches alternent deux mélangeurs de
tokens dans un ratio 3:1 —

  Gated DeltaNet (3/4 des couches) : attention *linéaire* à état récurrent.
      Au lieu de stocker tous les K/V passés, la couche maintient une matrice
      d'état S (dk x dv) mise à jour token par token par la "règle delta" :
          S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
          o_t = S_t^T q_t
      alpha_t (0..1) = combien on oublie, beta_t (0..1) = force d'écriture.
      C'est un mécanisme de mémoire associative : "remplace ce que S prédisait
      pour la clé k_t par la valeur v_t". Coût O(1) par token en génération,
      O(T) en entraînement via un algorithme par blocs (voir gated_delta_scan).

  Attention complète "gated" (1/4 des couches) : GQA + les 3 raffinements Qwen3.5
      - gate de sortie : out * sigmoid(W_g x). Non-linéarité par tête qui
        supprime les "attention sinks" (NeurIPS 2025 oral, équipe Qwen).
      - QK-Norm zéro-centrée : le gain de la RMSNorm est stocké comme 1 + w
        (w init 0) et w subit le weight decay -> le gain est tiré vers 1,
        jamais vers 0. Corrige les poids de norme qui explosent dans Qwen3.
      - RoPE partiel : seulement 1/4 des dimensions de tête sont tournées.
        Les autres dimensions sont libres de coder de l'information sans
        position (les couches DeltaNet fournissent déjà de l'ordre local).

  Le reste (RMSNorm pre-norm, SwiGLU, zéro biais, embeddings partagés) est
  hérité de Qwen3. Le MoE de Qwen3.5 (397 Md params) n'a pas de sens à notre
  échelle : un petit modèle dense est strictement meilleur à budget mémoire égal.

Réfs : Qwen3 Technical Report (arXiv:2505.09388), blog Qwen3-Next (alibabacloud
.com/blog/602580), Gated DeltaNet (arXiv:2412.06464), Gated Attention
(arXiv:2505.06708).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_FUSED_RMSNORM = hasattr(F, "rms_norm")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 16384
    n_layer: int = 8
    n_head: int = 8            # têtes de query (attention) et têtes DeltaNet
    n_kv_head: int = 2         # têtes de key/value des couches d'attention (GQA)
    d_model: int = 512
    head_dim: int = 64
    d_ff: int = 1408           # dimension cachée du SwiGLU
    max_seq_len: int = 1024
    rope_theta: float = 100_000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0

    # --- spécifique Qwen3.5 ---
    # hybrid=False par défaut : le Gated DeltaNet est implémenté et exact, mais sans
    # les kernels Triton de `flash-linear-attention` il entraîne ~3,5x plus lentement
    # (mesuré : 8,7k vs 31k tok/s sur la 4060) — et à contexte 1024 son avantage
    # (mémoire O(1) sur le très long contexte) ne joue pas. Actif via --hybrid.
    hybrid: bool = False       # True = DeltaNet:attention en 3:1 ; False = attention partout
    attn_gate: bool = True     # gate de sortie sigmoïde sur les couches d'attention
    rope_frac: float = 0.25    # fraction des dims de tête qui reçoivent RoPE
    zero_centered: bool = True # RMSNorm zéro-centrées (gain = 1 + w, w décayé)
    conv_kernel: int = 4       # conv causale courte des couches DeltaNet
    scan_chunk: int = 128      # bloc du scan DeltaNet (128 = optimum mesuré sur 4060)

    # ids de tokens spéciaux (remplis par le tokenizer)
    bos_id: int = 0
    eos_id: int = 0
    pad_id: int = 0

    @property
    def layer_kinds(self) -> list[str]:
        """Disposition des couches. Hybride = motif [delta, delta, delta, attn].

        Comme Qwen3.5/Qwen3-Next : 3 couches linéaires pour 1 couche d'attention
        complète, l'attention en dernière position de chaque groupe (elle relit
        globalement ce que les couches récurrentes ont résumé).
        """
        if not self.hybrid:
            return ["attn"] * self.n_layer
        kinds = [("attn" if i % 4 == 3 else "delta") for i in range(self.n_layer)]
        if kinds[-1] != "attn" and self.n_layer % 4 != 0:
            kinds[-1] = "attn"             # on termine toujours par une attention complète
        return kinds

    @property
    def rope_dims(self) -> int:
        r = int(self.head_dim * self.rope_frac)
        return max(2, r - r % 2)           # pair, au moins 2

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        known = set(ModelConfig.__dataclass_fields__)
        return ModelConfig(**{k: v for k, v in d.items() if k in known})


# Presets pensés pour 8 Go de VRAM (RTX 4060).
PRESETS: dict[str, dict] = {
    # ~15M : itère très vite, bon pour débugger le pipeline
    "nano": dict(n_layer=6, n_head=6, n_kv_head=2, d_model=384, head_dim=64, d_ff=1024, max_seq_len=512),
    # ~30M : défaut — le bon compromis discussion FR / temps d'entraînement
    "micro": dict(n_layer=8, n_head=8, n_kv_head=2, d_model=512, head_dim=64, d_ff=1408, max_seq_len=1024),
    # ~56M
    "mini": dict(n_layer=12, n_head=9, n_kv_head=3, d_model=576, head_dim=64, d_ff=1536, max_seq_len=1024),
    # ~54M mais PROFOND (16 couches) : le raisonnement multi-étapes aime la profondeur
    # (chaque couche ≈ un pas de calcul). Candidat par défaut de la v2, à départager
    # au benchmark contre mini.
    "deep": dict(n_layer=16, n_head=8, n_kv_head=2, d_model=512, head_dim=64, d_ff=1408, max_seq_len=1024),
    # ~123M : il faut beaucoup plus de tokens et de temps
    "small": dict(n_layer=16, n_head=12, n_kv_head=4, d_model=768, head_dim=64, d_ff=2048, max_seq_len=1024),
}


# --------------------------------------------------------------------------------------
# Normes
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm, en version zéro-centrée par défaut (Qwen3.5).

    Zéro-centrée = le paramètre appris est un OFFSET w autour de 1 (gain = 1+w).
    Combiné au weight decay (voir optim.build_optimizers), le gain est rappelé
    vers 1 : on ne voit plus les poids de norme dériver vers des valeurs
    extrêmes comme dans Qwen3.
    """

    def __init__(self, dim: int, eps: float = 1e-6, zero_centered: bool = True):
        super().__init__()
        self.eps = eps
        self.dim = (dim,)
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.zeros(dim) if zero_centered else torch.ones(dim))

    def _gain(self, dtype):
        w = self.weight.to(dtype)
        return 1.0 + w if self.zero_centered else w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_FUSED_RMSNORM:
            return F.rms_norm(x, self.dim, None, self.eps) * self._gain(x.dtype)
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self._gain(dtype)


# --------------------------------------------------------------------------------------
# RoPE (partiel)
# --------------------------------------------------------------------------------------
def build_rope_cache(rope_dims: int, max_seq_len: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dims, 2, dtype=torch.float32, device=device) / rope_dims))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)        # (T, rope_dims)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(q, k, cos, sin, rope_dims: int):
    """Ne tourne que les `rope_dims` premières dimensions de chaque tête."""
    cos = cos[None, :, None, :].to(q.dtype)
    sin = sin[None, :, None, :].to(q.dtype)
    qr, qp = q[..., :rope_dims], q[..., rope_dims:]
    kr, kp = k[..., :rope_dims], k[..., rope_dims:]
    qr = qr * cos + _rotate_half(qr) * sin
    kr = kr * cos + _rotate_half(kr) * sin
    return torch.cat((qr, qp), dim=-1), torch.cat((kr, kp), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, KV, T, D) -> (B, KV*n_rep, T, D) pour la GQA.

    ATTENTION : ne PAS utiliser `enable_gqa=True` de SDPA à la place. Il fait
    silencieusement tomber PyTorch sur le backend "math" qui matérialise la
    matrice d'attention (B,H,T,T) : mesuré à 1,1 Go / 35 ms par couche contre
    72 Mo / 3,2 ms en dupliquant les KV à la main.
    """
    if n_rep == 1:
        return x
    B, KV, T, D = x.shape
    return x.unsqueeze(2).expand(B, KV, n_rep, T, D).reshape(B, KV * n_rep, T, D)


# --------------------------------------------------------------------------------------
# Attention complète "gated" (le 1/4 de couches plein-contexte)
# --------------------------------------------------------------------------------------
class GatedAttention(nn.Module):
    """GQA + QK-Norm zéro-centrée + RoPE partiel + gate de sortie sigmoïde."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.rope_dims = cfg.rope_dims
        assert cfg.n_head % cfg.n_kv_head == 0

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)
        # gate de sortie : "quelle part de ce que l'attention a lu je laisse passer"
        self.gate_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False) if cfg.attn_gate else None

        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps, cfg.zero_centered)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps, cfg.zero_centered)
        self.dropout_p = cfg.dropout

    def alloc_cache(self, batch: int, max_len: int, device, dtype) -> dict:
        shape = (batch, self.n_kv_head, max_len, self.head_dim)
        return {"kind": "attn",
                "k": torch.zeros(shape, device=device, dtype=dtype),
                "v": torch.zeros(shape, device=device, dtype=dtype)}

    def forward(self, x, cos, sin, cache=None, pos: int = 0):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_partial_rope(q, k, cos, sin, self.rope_dims)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if cache is not None:
            cache["k"][:, :, pos:pos + T] = k
            cache["v"][:, :, pos:pos + T] = v
            k = cache["k"][:, :, : pos + T]
            v = cache["v"][:, :, : pos + T]

        causal = q.shape[2] == k.shape[2] and q.shape[2] > 1
        drop = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep),
            is_causal=causal, dropout_p=drop,
        )
        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out)


# --------------------------------------------------------------------------------------
# Gated DeltaNet (les 3/4 de couches récurrentes)
# --------------------------------------------------------------------------------------
@torch.compiler.disable
def gated_delta_scan(q, k, v, log_alpha, beta, S0, chunk: int):
    """Règle delta "gated" par blocs — mathématiquement EXACTE (testée contre la
    récurrence pas-à-pas à 1e-15 près), parallèle à l'intérieur de chaque bloc.

    q,k,v : (N, T, d)   log_alpha, beta : (N, T)   S0 : (N, dk, dv)
    Renvoie (sorties (N,T,dv), état final).

    Idée : dans un bloc, on résout d'un coup les "pseudo-valeurs" U (le système
    est triangulaire -> une seule solve_triangular), puis toutes les sorties du
    bloc s'écrivent comme des produits de matrices. Tous les facteurs de décroissance
    apparaissent sous forme exp(G_t - G_s) avec G_t - G_s <= 0 : jamais d'overflow.
    """
    N, T, dk = k.shape
    dv = v.shape[-1]
    S = S0
    outs = []
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    for c0 in range(0, T, chunk):
        Q, K, V = q[:, c0:c0 + chunk], k[:, c0:c0 + chunk], v[:, c0:c0 + chunk]
        la, b = log_alpha[:, c0:c0 + chunk], beta[:, c0:c0 + chunk]
        n = Q.shape[1]
        G = torch.cumsum(la, dim=1)                        # (N, n)
        D = G[:, :, None] - G[:, None, :]                  # G_t - G_s  (<= 0 sous la diag)
        decay = torch.exp(D.tril())                        # partie utile seulement
        A = (b[:, :, None] * decay * (K @ K.mT)).tril(-1)
        rhs = b[:, :, None] * V - (b * torch.exp(G))[:, :, None] * (K @ S)
        I = eye[:n, :n] if n != chunk else eye
        U = torch.linalg.solve_triangular(I + A, rhs, upper=False, unitriangular=True)
        QK = ((Q @ K.mT) * decay).tril()
        outs.append(torch.exp(G)[:, :, None] * (Q @ S) + QK @ U)
        gl = G[:, -1]
        S = torch.exp(gl)[:, None, None] * S + K.mT @ (torch.exp(gl[:, None] - G)[:, :, None] * U)
    return torch.cat(outs, dim=1), S


class GatedDeltaNet(nn.Module):
    """Couche d'attention linéaire de Qwen3.5.

    Pipeline : projection q/k/v fusionnée -> conv causale courte (kernel 4,
    depthwise : donne un ordre local aux tokens, crucial pour les modèles
    récurrents) -> SiLU -> q,k normalisés L2 -> règle delta avec oubli alpha et
    force d'écriture beta -> RMSNorm de sortie modulée par un gate SiLU -> o_proj.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H, dh = cfg.n_head, cfg.head_dim
        self.n_head, self.head_dim = H, dh
        self.chunk = cfg.scan_chunk
        self.kernel = cfg.conv_kernel
        inner = H * dh

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * inner, bias=False)
        self.conv = nn.Conv1d(3 * inner, 3 * inner, cfg.conv_kernel,
                              groups=3 * inner, bias=False)
        self.a_proj = nn.Linear(cfg.d_model, H, bias=False)   # -> alpha (oubli)
        self.b_proj = nn.Linear(cfg.d_model, H, bias=False)   # -> beta (écriture)
        self.g_proj = nn.Linear(cfg.d_model, inner, bias=False)  # gate de sortie
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.out_norm = RMSNorm(dh, cfg.rms_eps, cfg.zero_centered)

        # échelles de temps d'oubli variées par tête : alpha = exp(-exp(A_log)*softplus(a))
        # A_log réparti entre têtes "mémoire longue" (alpha~0.98) et "mémoire courte" (~0.1)
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(0.02, 4.0)))

    def alloc_cache(self, batch: int, max_len: int, device, dtype) -> dict:
        inner3 = 3 * self.n_head * self.head_dim
        return {"kind": "delta",
                "conv": torch.zeros(batch, inner3, self.kernel - 1, device=device, dtype=dtype),
                "S": torch.zeros(batch, self.n_head, self.head_dim, self.head_dim,
                                 device=device, dtype=torch.float32)}

    def _gates(self, x):
        """log_alpha (B,T,H) et beta (B,T,H), calculés en fp32."""
        log_alpha = -torch.exp(self.A_log.float())[None, None] * F.softplus(self.a_proj(x).float())
        beta = torch.sigmoid(self.b_proj(x).float())
        return log_alpha, beta

    def forward(self, x, cos=None, sin=None, cache=None, pos: int = 0):
        B, T, _ = x.shape
        H, dh = self.n_head, self.head_dim

        qkv = self.qkv_proj(x).transpose(1, 2)             # (B, 3*H*dh, T)
        if cache is not None:
            padded = torch.cat([cache["conv"].to(qkv.dtype), qkv], dim=2)
            cache["conv"] = padded[:, :, -(self.kernel - 1):]
        else:
            padded = F.pad(qkv, (self.kernel - 1, 0))      # conv causale
        qkv = F.silu(self.conv(padded))                    # (B, 3*H*dh, T)

        q, k, v = qkv.transpose(1, 2).view(B, T, 3, H, dh).unbind(2)
        q = F.normalize(q, dim=-1)                         # la règle delta suppose |k|=1
        k = F.normalize(k, dim=-1)
        log_alpha, beta = self._gates(x)

        # La dynamique d'état se calcule TOUJOURS en fp32 (autocast désactivé) :
        # une récurrence multiplicative en bf16 dérive au fil des tokens.
        with torch.autocast(x.device.type, enabled=False):
            if cache is not None and T == 1:
                # génération : récurrence directe, O(1) — LE point fort de DeltaNet
                S = cache["S"]                             # (B, H, dk, dv) fp32
                qf, kf, vf = (t.float().squeeze(1) for t in (q, k, v))   # (B,H,dh)
                a = torch.exp(log_alpha.float()).view(B, H, 1, 1)
                bta = beta.view(B, H, 1, 1)
                kS = torch.einsum("bhk,bhkv->bhv", kf, S)  # ce que S prédit pour k
                S = a * (S - bta * kf.unsqueeze(-1) * kS.unsqueeze(-2)) \
                    + bta * kf.unsqueeze(-1) * vf.unsqueeze(-2)
                cache["S"] = S
                o = torch.einsum("bhk,bhkv->bhv", qf, S).view(B, 1, H, dh)
            else:
                # entraînement / préremplissage : scan par blocs en fp32
                qf = q.float().transpose(1, 2).reshape(B * H, T, dh)
                kf = k.float().transpose(1, 2).reshape(B * H, T, dh)
                vf = v.float().transpose(1, 2).reshape(B * H, T, dh)
                la = log_alpha.transpose(1, 2).reshape(B * H, T)
                bt = beta.transpose(1, 2).reshape(B * H, T)
                S0 = cache["S"].view(B * H, dh, dh) if cache is not None \
                    else torch.zeros(B * H, dh, dh, device=x.device, dtype=torch.float32)
                o, S = gated_delta_scan(qf, kf, vf, la, bt, S0, self.chunk)
                if cache is not None:
                    cache["S"] = S.view(B, H, dh, dh)
                o = o.view(B, H, T, dh).transpose(1, 2)    # (B, T, H, dh)

        o = self.out_norm(o.to(x.dtype)) * F.silu(self.g_proj(x)).view(B, T, H, dh)
        return self.o_proj(o.reshape(B, T, H * dh))


# --------------------------------------------------------------------------------------
# Bloc transformer et modèle complet
# --------------------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps, cfg.zero_centered)
        self.mixer = GatedDeltaNet(cfg) if kind == "delta" else GatedAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps, cfg.zero_centered)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, pos: int = 0):
        x = x + self.mixer(self.input_layernorm(x), cos, sin, cache, pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class QwenLikeLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg, kind) for kind in cfg.layer_kinds)
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps, cfg.zero_centered)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = build_rope_cache(cfg.rope_dims, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ---- stats du modèle -------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """6*N (fwd+bwd) + terme quadratique pour les seules couches d'attention."""
        c = self.cfg
        n_attn = sum(1 for k in c.layer_kinds if k == "attn")
        n_delta = c.n_layer - n_attn
        base = 6 * self.num_params(non_embedding=True)
        attn = 12 * n_attn * c.n_head * c.head_dim * c.max_seq_len
        delta = 12 * n_delta * c.n_head * c.head_dim * (c.head_dim + 2 * c.scan_chunk)
        return base + attn + delta

    def describe(self) -> str:
        kinds = self.cfg.layer_kinds
        return "".join("A" if k == "attn" else "D" for k in kinds)

    # ---- forward ---------------------------------------------------------------------
    def forward(self, idx, targets=None, loss_mask=None, z_loss: float = 0.0, diagnostics: bool = True):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"séquence {T} > max_seq_len {self.cfg.max_seq_len}"
        x = self.embed_tokens(idx)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        if targets is None:
            return self.lm_head(x), None, {}

        logits = self.lm_head(x)
        # Pas de cast fp32 des logits : +768 Mo mesurés pour des gradients identiques
        # (le noyau CUDA de cross_entropy accumule déjà en fp32 en interne).
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)

        # Masque de loss (SFT) : positions ignorées marquées -100, géré par le noyau.
        if loss_mask is not None:
            flat_targets = flat_targets.masked_fill(loss_mask.reshape(-1) == 0, -100)
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

        # z-loss : pénalise l'explosion de logsumexp(logits) — stabilité bf16.
        if z_loss > 0:
            lse = torch.logsumexp(flat_logits, dim=-1).float()
            loss = loss + z_loss * lse.pow(2).mean()

        if not diagnostics:
            return logits, loss, {}

        # --- diagnostics quasi gratuits, sur un sous-échantillon de positions ---
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
            stats = {
                "acc_top1": correct.mean().detach(),
                "entropy": entropy.mean().detach(),
                "logit_rms": sl.pow(2).mean().sqrt().detach(),
            }
        return logits, loss, stats

    # ---- génération ------------------------------------------------------------------
    def _alloc_caches(self, batch: int, max_len: int, device, dtype):
        return [layer.mixer.alloc_cache(batch, max_len, device, dtype) for layer in self.layers]

    def _forward_cached(self, idx, caches, pos: int):
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos = self.rope_cos[pos:pos + T]
        sin = self.rope_sin[pos:pos + T]
        for layer, cache in zip(self.layers, caches):
            x = layer(x, cos, sin, cache, pos)
        return self.lm_head(self.norm(x[:, -1:]))

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop_ids: tuple[int, ...] = (),
        on_token=None,
    ):
        self.eval()
        device = idx.device
        dtype = next(self.parameters()).dtype
        B, T = idx.shape
        max_len = min(self.cfg.max_seq_len, T + max_new_tokens)
        if T >= max_len:
            idx = idx[:, -(max_len - 1):]
            T = idx.shape[1]

        caches = self._alloc_caches(B, max_len, device, dtype)
        logits = self._forward_cached(idx, caches, 0)
        pos = T
        out = idx

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                recent = torch.unique(out[0, -128:])
                sel = logits[0, recent]
                logits[0, recent] = torch.where(sel > 0, sel / repetition_penalty,
                                                sel * repetition_penalty)

            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    cum = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
                    remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            tok = int(next_id.item())
            out = torch.cat([out, next_id], dim=1)
            if on_token is not None:
                on_token(tok)
            if tok in stop_ids:
                break
            if pos + 1 >= max_len:
                break
            logits = self._forward_cached(next_id, caches, pos)
            pos += 1

        return out


def build_model(cfg: ModelConfig) -> QwenLikeLM:
    return QwenLikeLM(cfg)
