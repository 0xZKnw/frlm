"""frlm — petit LLM français de raisonnement, entraîné sur une RTX 4060.

Modules :
    model     l'architecture v2 type Qwen3.5 (attention gated, GQA, RoPE partiel)
    model_v3  l'architecture v3 "speedrun" (canon layers, value embeds, U-net,
              ReLU², fenêtre glissante, softcap) — voir son docstring
    data      corpus FR, tokenizer BPE à chiffres séparés, binarisation
    synth     générateur de problèmes maths/logique FR (solutions calculées)
    distill   distillation filtrée depuis un gros teacher (API Kimi)
    optim     Muon + schedules de LR
    rl        GRPO : renforcement à récompenses vérifiables

Point d'entrée : run.py à la racine du dépôt.
"""


def config_from_dict(d: dict):
    """Reconstruit la config modèle depuis un dict de checkpoint, v2 ou v3.

    Les checkpoints v3 portent {"arch": "v3"} (écrit par ModelConfigV3.to_dict) ;
    les checkpoints v2 antérieurs n'ont pas de clé arch.
    """
    if d.get("arch") == "v3":
        from frlm.model_v3 import ModelConfigV3
        return ModelConfigV3.from_dict(d)
    from frlm.model import ModelConfig
    return ModelConfig.from_dict(d)


def model_from_cfg(mcfg):
    """Construit le bon modèle (v2 ou v3) depuis une config déjà typée."""
    from frlm.model_v3 import ModelConfigV3, build_model_v3
    if isinstance(mcfg, ModelConfigV3):
        return build_model_v3(mcfg)
    from frlm.model import build_model
    return build_model(mcfg)
