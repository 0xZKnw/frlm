"""
optim.py — Muon (MomentUm Orthogonalized by Newton-Schulz) + schedules de learning rate.

Muon (Keller Jordan, 2024) est l'optimiseur qui détient les records de vitesse
d'entraînement sur les petits GPT. L'idée :

  Adam met à l'échelle chaque coefficient indépendamment. Mais pour une matrice de
  poids, ce qui compte c'est la *direction* de la mise à jour dans l'espace des
  matrices. Muon prend le gradient avec momentum, puis l'**orthogonalise**
  (le remplace par la matrice orthogonale la plus proche, via 5 itérations de
  Newton-Schulz — uniquement des matmuls, donc ultra rapide sur GPU). Résultat :
  toutes les directions singulières avancent à la même vitesse, on n'a plus une
  poignée de directions dominantes qui saturent le pas.

  Ça ne marche que sur les matrices 2D des blocs transformer. Les embeddings,
  la tête de sortie et les gains 1D des RMSNorm restent sur AdamW.

En pratique sur un modèle de 60M : ~1.3-1.5x moins de steps qu'AdamW pour la
même loss, pour ~2% de surcoût par step.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approxime UV^T de la SVD de G (= l'orthogonalisation) sans jamais faire de SVD.

    Le polynôme (a,b,c) est choisi pour converger très vite depuis un spectre
    normalisé, quitte à ne pas être exact : on veut la direction, pas la précision.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon avec deux optimisations qui comptent beaucoup sur un petit modèle :

    1. le momentum est appliqué avec les opérations `_foreach_*` (un seul noyau
       CUDA pour toutes les matrices au lieu d'un par matrice) ;
    2. les matrices **de même forme** sont empilées et orthogonalisées d'un seul
       coup — dans un transformer, les N couches ont exactement les mêmes formes,
       donc on passe de ~90 matrices × 15 matmuls à ~7 lots × 15 matmuls.

    Mesuré sur le preset `mini` (RTX 4060) : 92 ms -> ~20 ms par step.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]

            bufs = []
            for p in params:
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                bufs.append(st["momentum_buffer"])

            torch._foreach_lerp_(bufs, grads, 1 - mom)
            updates = torch._foreach_lerp(grads, bufs, mom) if group["nesterov"] else list(bufs)

            if wd:
                torch._foreach_mul_(params, 1 - lr * wd)

            # regroupement par forme -> orthogonalisation par lots
            buckets: dict[tuple, list[int]] = {}
            for i, p in enumerate(params):
                buckets.setdefault(tuple(p.shape), []).append(i)

            for shape, idxs in buckets.items():
                stacked = torch.stack([updates[i] for i in idxs]) if len(idxs) > 1 else updates[idxs[0]][None]
                # une seule conversion de dtype pour tout le lot (et non une par
                # matrice : 70 petits noyaux de cast coûtaient plus cher que
                # l'orthogonalisation elle-même)
                ortho = zeropower_via_newtonschulz5(stacked, group["ns_steps"]).to(params[idxs[0]].dtype)
                # correction de forme : une matrice très rectangulaire a besoin
                # d'un pas plus grand pour un effet équivalent
                scale = -lr * max(1.0, shape[-2] / shape[-1]) ** 0.5
                torch._foreach_add_([params[i] for i in idxs], list(ortho.unbind(0)), alpha=scale)
        return loss


# --------------------------------------------------------------------------------------
def _owner_module(model, param_name: str):
    """Renvoie le module qui possède le paramètre `a.b.c.weight` -> module a.b.c."""
    path = param_name.rsplit(".", 1)[0]
    mod = model
    for part in path.split("."):
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    return mod


def build_optimizers(model, args) -> tuple[list[torch.optim.Optimizer], dict]:
    """Sépare les paramètres en groupes et construit le/les optimiseurs.

    - matrices 2D des blocs      -> Muon (ou AdamW avec weight decay)
    - embeddings / lm_head       -> AdamW (Muon n'a pas de sens sur une table de lookup)
    - offsets RMSNorm zéro-centrées -> AdamW AVEC weight decay : c'est tout l'intérêt
      du zéro-centré (Qwen3.5) — le decay tire le gain vers 1, pas vers 0
    - conv depthwise DeltaNet (3D), A_log, gains de norme classiques -> AdamW sans decay
    """
    muon_params, adam_decay, adam_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed_tokens" in name or "lm_head" in name:
            adam_decay.append(p)
        elif p.ndim == 2 and "conv" not in name:
            muon_params.append(p)
        elif p.ndim == 1 and "norm" in name.lower() and getattr(
                _owner_module(model, name), "zero_centered", False):
            adam_decay.append(p)
        else:
            adam_nodecay.append(p)

    info = {
        "muon_params": sum(p.numel() for p in muon_params),
        "adam_params": sum(p.numel() for p in adam_decay + adam_nodecay),
        "n_muon_tensors": len(muon_params),
    }

    if args.optimizer == "muon":
        opts = [
            Muon(muon_params, lr=args.lr, momentum=0.95, ns_steps=5, weight_decay=args.weight_decay),
            torch.optim.AdamW(
                [
                    {"params": adam_decay, "weight_decay": args.weight_decay},
                    {"params": adam_nodecay, "weight_decay": 0.0},
                ],
                lr=args.adam_lr, betas=(args.beta1, args.beta2), eps=1e-8, fused=torch.cuda.is_available(),
            ),
        ]
        info["lrs"] = {"muon": args.lr, "adamw": args.adam_lr}
    else:
        opts = [
            torch.optim.AdamW(
                [
                    {"params": muon_params + adam_decay, "weight_decay": args.weight_decay},
                    {"params": adam_nodecay, "weight_decay": 0.0},
                ],
                lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8, fused=torch.cuda.is_available(),
            )
        ]
        info["lrs"] = {"adamw": args.lr}
    return opts, info


def lr_multiplier(step: int, max_steps: int, warmup: int, schedule: str = "wsd",
                  min_frac: float = 0.05, decay_frac: float = 0.2) -> float:
    """Renvoie un facteur multiplicatif dans [min_frac, 1] appliqué au LR de base.

    - "cosine" : le classique. Décroît en cos sur toute la durée.
    - "wsd"    : Warmup-Stable-Decay. Plateau constant puis chute sur les derniers
                 `decay_frac` du run. Avantage énorme ici : pendant le plateau, un
                 checkpoint vaut n'importe quel autre, donc on peut allonger ou
                 raccourcir l'entraînement sans casser le schedule.
    """
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if schedule == "cosine":
        prog = (step - warmup) / max(1, max_steps - warmup)
        prog = min(1.0, prog)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog))
    # wsd
    decay_start = max_steps - int(max_steps * decay_frac)
    if step < decay_start:
        return 1.0
    prog = (step - decay_start) / max(1, max_steps - decay_start)
    prog = min(1.0, prog)
    return min_frac + (1 - min_frac) * (1 - math.sqrt(prog))
